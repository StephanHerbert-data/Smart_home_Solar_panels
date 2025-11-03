import math
from dataclasses import dataclass, replace
from typing import Optional, Dict
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# -----------------------------
# Domain model
# -----------------------------
@dataclass
class Params:
    start_year: int
    horizon_years: int
    capex: float
    discount_rate: float

    pv_year1_kwh: float
    pv_degradation: float            # fraction, e.g. 0.005 for 0.5%/yr

    imported_kwh_year: float         # user-entered imported energy (kWh/yr)
    exported_kwh_year: float         # user-entered exported energy (kWh/yr)
    battery_loss_rate: float         # fraction, e.g. 0.03 for 3% round-trip loss

    grid_price_eur_kwh: float
    grid_price_escalation: float     # 0..1 per year (applies to both scenarios)

    feedin_eur_kwh: float            # constant over horizon (no escalation)

    opex_eur_year: float
    inverter_swap_year: Optional[int] = None
    inverter_swap_cost: float = 0.0

# -----------------------------
# Simulation core
# -----------------------------
def simulate_cashflows(p: Params) -> pd.DataFrame:
    """
    Build a year-by-year cashflow table for the PV system.
    Year index starts at 1; t0 is the investment moment (capex).
    """
    years = np.arange(1, p.horizon_years + 1)

    # PV production with degradation
    pv_kwh = p.pv_year1_kwh * (1 - p.pv_degradation) ** (years - 1)

    # Prices (grid escalates; feed-in constant)
    grid_price = p.grid_price_eur_kwh * (1 + p.grid_price_escalation) ** (years - 1)
    feedin_price = np.full_like(years, p.feedin_eur_kwh, dtype=float)

    # User-entered flows (cap export by PV)
    exported_kwh = np.minimum(p.exported_kwh_year, pv_kwh)

    # Battery losses (assumed to apply to PV after export)
    battery_losses_kwh = np.maximum((pv_kwh - exported_kwh) * p.battery_loss_rate, 0.0)

    # Self-consumed PV after battery losses
    self_consumed = np.maximum(pv_kwh - exported_kwh - battery_losses_kwh, 0.0)

    # Imported energy taken as entered (note: if you want constant total demand,
    # you could auto-adjust imports = baseline_demand - self_consumed)
    imported_kwh = np.full_like(years, p.imported_kwh_year, dtype=float)

    # Derived total household consumption (for info)
    consumption_kwh = imported_kwh + self_consumed

    # Cash components
    avoided_import_cost = self_consumed * grid_price
    feedin_revenue = exported_kwh * feedin_price
    opex = np.full_like(years, fill_value=p.opex_eur_year, dtype=float)

    inverter_cost = np.zeros_like(years, dtype=float)
    if p.inverter_swap_year is not None and 1 <= p.inverter_swap_year <= p.horizon_years:
        inverter_cost[p.inverter_swap_year - 1] = p.inverter_swap_cost

    cashflow = avoided_import_cost + feedin_revenue - opex - inverter_cost

    # Discounted cashflow relative to t0
    discount_factors = 1.0 / (1.0 + p.discount_rate) ** years
    dcf = cashflow * discount_factors

    df = pd.DataFrame({
        "year_index": years,
        "calendar_year": p.start_year + years - 1,
        "pv_kwh": pv_kwh,
        "exported_kwh": exported_kwh,
        "battery_losses_kwh": battery_losses_kwh,
        "self_consumed_kwh": self_consumed,
        "imported_kwh": imported_kwh,
        "consumption_kwh": consumption_kwh,
        "grid_price_eur_kwh": grid_price,
        "feedin_eur_kwh": feedin_price,
        "avoided_import_eur": avoided_import_cost,
        "feedin_revenue_eur": feedin_revenue,
        "opex_eur": opex,
        "inverter_cost_eur": inverter_cost,
        "cashflow_eur": cashflow,
        "discount_factor": discount_factors,
        "dcf_eur": dcf
    })

    # Cumulative sums (CAPEX at t0)
    df["cum_cashflow_eur"] = -p.capex + df["cashflow_eur"].cumsum()
    df["cum_dcf_eur"] = -p.capex + df["dcf_eur"].cumsum()
    return df

def find_payback_year(cum_series: pd.Series) -> Optional[float]:
    if cum_series.iloc[-1] < 0:
        return None
    if cum_series.iloc[0] >= 0:
        return 0.0
    for i in range(1, len(cum_series)):
        prev_val = cum_series.iloc[i - 1]
        curr_val = cum_series.iloc[i]
        if prev_val < 0 <= curr_val:
            frac = (0 - prev_val) / (curr_val - prev_val + 1e-12)
            return (i - 1) + frac + 1e-12
    return None

def compute_kpis(df: pd.DataFrame, capex: float) -> Dict[str, Optional[float]]:
    simple_pb = find_payback_year(df["cum_cashflow_eur"])
    disc_pb = find_payback_year(df["cum_dcf_eur"])
    npv = df["cum_dcf_eur"].iloc[-1]
    cash = np.r_[-capex, df["cashflow_eur"].values]
    try:
        irr = np.irr(cash)
    except Exception:
        irr = None
    return {
        "simple_payback_years": simple_pb,
        "discounted_payback_years": disc_pb,
        "npv_eur": npv,
        "irr": irr
    }

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="PV Payback (battery losses + sidebar)", layout="wide")
st.title("☀️ PV Payback: Baseline vs Optimized")
st.caption("Sidebar controls. Battery round-trip losses included. X-axis shows integer calendar years; axes labeled.")

with st.sidebar:
    st.header("General Parameters")
    start_year = st.number_input("Start year (commissioning)", value=2025, step=1)
    horizon = st.slider("Horizon (years)", min_value=10, max_value=35, value=25, step=1)
    capex = st.number_input("CAPEX (EUR)", value=18000.0, min_value=0.0, step=500.0, format="%.2f")
    discount_rate = st.number_input("Discount rate (per year)", value=0.04, min_value=0.0, max_value=0.2, step=0.005, format="%.3f")

    st.markdown("---")
    st.header("PV & Energy Flows")
    pv_y1 = st.number_input("PV production in Year 1 (kWh)", value=9000.0, min_value=0.0, step=100.0)
    degr_pct = st.number_input("PV degradation per year (%)", value=0.5, min_value=0.0, max_value=5.0, step=0.1, format="%.1f")
    degr = degr_pct / 100.0
    imported_kwh = st.number_input("Imported energy (kWh/yr)", value=1200.0, min_value=0.0, step=100.0)
    exported_kwh = st.number_input("Exported energy (kWh/yr)", value=3500.0, min_value=0.0, step=100.0)
    battery_loss_pct = st.number_input("Battery round-trip loss (%)", value=3.0, min_value=0.0, max_value=50.0, step=0.5, format="%.1f")
    battery_loss_rate = battery_loss_pct / 100.0

    st.markdown("---")
    st.header("Prices")
    base_grid_price = st.number_input("Baseline grid price (EUR/kWh)", value=0.32, min_value=0.0, step=0.01, format="%.3f")
    opt_grid_price = st.number_input("Optimized grid price (EUR/kWh)", value=0.28, min_value=0.0, step=0.01, format="%.3f")
    grid_escal = st.number_input("Grid price escalation (per year, both scenarios)", value=0.02, min_value=0.0, max_value=0.2, step=0.005, format="%.3f")
    feedin = st.number_input("Feed-in tariff (EUR/kWh, constant)", value=0.08, min_value=0.0, step=0.005, format="%.3f")

    st.markdown("---")
    st.header("Costs")
    opex = st.number_input("O&M (EUR/year)", value=150.0, min_value=0.0, step=10.0, format="%.2f")
    inv_swap_enable = st.checkbox("Add inverter replacement", value=False)
    swap_year = None
    swap_cost = 0.0
    if inv_swap_enable:
        swap_year = st.number_input("Inverter replacement year (relative, 1..N)", value=12, min_value=1, max_value=horizon, step=1)
        swap_cost = st.number_input("Inverter replacement cost (EUR)", value=1200.0, min_value=0.0, step=50.0, format="%.2f")

# Build scenarios (only grid price differs)
shared = Params(
    start_year=start_year,
    horizon_years=horizon,
    capex=capex,
    discount_rate=discount_rate,
    pv_year1_kwh=pv_y1,
    pv_degradation=degr,
    imported_kwh_year=imported_kwh,
    exported_kwh_year=exported_kwh,
    battery_loss_rate=battery_loss_rate,
    grid_price_eur_kwh=base_grid_price,  # placeholder
    grid_price_escalation=grid_escal,
    feedin_eur_kwh=feedin,
    opex_eur_year=opex,
    inverter_swap_year=swap_year,
    inverter_swap_cost=swap_cost
)

params_base = replace(shared, grid_price_eur_kwh=base_grid_price)
params_opt  = replace(shared, grid_price_eur_kwh=opt_grid_price)

df_base = simulate_cashflows(params_base)
df_opt  = simulate_cashflows(params_opt)

kpi_base = compute_kpis(df_base, params_base.capex)
kpi_opt  = compute_kpis(df_opt,  params_opt.capex)

# -----------------------------
# KPIs
# -----------------------------
st.subheader("KPIs")
c1, c2 = st.columns(2)

def pretty_years(y):
    if y is None:
        return "No payback"
    return f"{y:.1f} yrs"

with c1:
    st.markdown("**Baseline**")
    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Simple PB", pretty_years(kpi_base["simple_payback_years"]))
    cc2.metric("Disc. PB", pretty_years(kpi_base["discounted_payback_years"]))
    cc3.metric("NPV (EUR)", f"{kpi_base['npv_eur']:,.0f}")
    irr_b = kpi_base["irr"]
    cc4.metric("IRR", f"{(irr_b*100):.1f}%" if irr_b is not None and not math.isnan(irr_b) else "n/a")

with c2:
    st.markdown("**Optimized**")
    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Simple PB", pretty_years(kpi_opt["simple_payback_years"]))
    cc2.metric("Disc. PB", pretty_years(kpi_opt["discounted_payback_years"]))
    cc3.metric("NPV (EUR)", f"{kpi_opt['npv_eur']:,.0f}")
    irr_o = kpi_opt["irr"]
    cc4.metric("IRR", f"{(irr_o*100):.1f}%" if irr_o is not None and not math.isnan(irr_o) else "n/a")

# -----------------------------
# Chart: labeled axes + integer years + zero line + crossing markers
# -----------------------------
st.markdown("### Cumulative cashflow (nominal)")
plot_base = df_base[["calendar_year", "cum_cashflow_eur"]].rename(columns={"cum_cashflow_eur": "value"})
plot_base["Scenario"] = "Baseline"
plot_opt  = df_opt[["calendar_year", "cum_cashflow_eur"]].rename(columns={"cum_cashflow_eur": "value"})
plot_opt["Scenario"]  = "Optimized"
plot_df = pd.concat([plot_base, plot_opt], ignore_index=True)

zero_rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[6,4], color="#888").encode(y="y:Q")

line = alt.Chart(plot_df).mark_line().encode(
    x=alt.X("calendar_year:Q", title="year", axis=alt.Axis(format="d")),
    y=alt.Y("value:Q", title="money [€]"),
    color=alt.Color("Scenario:N", legend=alt.Legend(title="Scenario"))
)

def crossing_calendar_year(cum_series: pd.Series, years: pd.Series) -> Optional[float]:
    if cum_series.iloc[-1] < 0:
        return None
    if cum_series.iloc[0] >= 0:
        return float(years.iloc[0])
    for i in range(1, len(cum_series)):
        a, b = cum_series.iloc[i-1], cum_series.iloc[i]
        if a < 0 <= b:
            y0, y1 = years.iloc[i-1], years.iloc[i]
            frac = (0 - a) / (b - a + 1e-12)
            return float(y0 + (y1 - y0) * frac)
    return None

p_base_cal = crossing_calendar_year(df_base["cum_cashflow_eur"].reset_index(drop=True),
                                    df_base["calendar_year"].reset_index(drop=True))
p_opt_cal  = crossing_calendar_year(df_opt["cum_cashflow_eur"].reset_index(drop=True),
                                    df_opt["calendar_year"].reset_index(drop=True))

layers = [zero_rule, line]
for cal, label in [(p_base_cal, "Baseline"), (p_opt_cal, "Optimized")]:
    if cal is not None:
        pt = alt.Chart(pd.DataFrame({"calendar_year": [cal], "value": [0.0], "Scenario": [label]}))\
            .mark_point(size=100).encode(
                x=alt.X("calendar_year:Q", axis=alt.Axis(format="d"), title="year"),
                y=alt.Y("value:Q", title="money [€]"),
                color="Scenario:N"
            )
        vr = alt.Chart(pd.DataFrame({"calendar_year": [cal]})).mark_rule(strokeDash=[3,3]).encode(
            x=alt.X("calendar_year:Q", axis=alt.Axis(format="d"))
        )
        layers += [pt, vr]

chart = layers[0]
for lyr in layers[1:]:
    chart = chart + lyr

st.altair_chart(chart.properties(height=360), use_container_width=True)

# -----------------------------
# Annual details
# -----------------------------
st.markdown("### Annual details")
tab1, tab2 = st.tabs(["Baseline", "Optimized"])

show_cols = [
    "calendar_year", "pv_kwh", "exported_kwh", "battery_losses_kwh", "self_consumed_kwh",
    "imported_kwh", "consumption_kwh", "grid_price_eur_kwh", "feedin_eur_kwh",
    "cashflow_eur", "cum_cashflow_eur", "dcf_eur", "cum_dcf_eur"
]

with tab1:
    st.dataframe(df_base[show_cols].round(3), use_container_width=True)
    csv_base = df_base.to_csv(index=False).encode("utf-8")
    st.download_button("Download Baseline results (CSV)", data=csv_base, file_name="pv_payback_baseline.csv", mime="text/csv")

with tab2:
    st.dataframe(df_opt[show_cols].round(3), use_container_width=True)
    csv_opt = df_opt.to_csv(index=False).encode("utf-8")
    st.download_button("Download Optimized results (CSV)", data=csv_opt, file_name="pv_payback_optimized.csv", mime="text/csv")

st.markdown("—")
st.caption("Self-consumed = PV − Export − Battery losses. Avoided cost = Self-consumed × Grid price. Feed-in revenue = Export × Feed-in tariff.")
