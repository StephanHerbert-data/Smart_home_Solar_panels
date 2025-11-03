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
    capex: float  # Initial invest (EUR)

    pv_year1_kwh: float
    pv_degradation: float            # fraction per year, e.g. 0.005 for 0.5%

    imported_kwh_year: float         # Imported energy (kWh/yr)
    consumed_kwh_year: float         # Consumed energy (kWh/yr)
    battery_loss_rate: float         # fraction, e.g. 0.07 for 7% loss

    grid_price_eur_kwh: float
    grid_price_escalation: float     # 0..1 per year (applies to both scenarios)

    feedin_eur_kwh: float            # constant over horizon (no escalation)

    opex_eur_year: float             # Maintenance costs (EUR/year)
    inverter_swap_year: Optional[int] = None
    inverter_swap_cost: float = 0.0

# -----------------------------
# Simulation core (no discounting)
# -----------------------------
def simulate_cashflows(p: Params) -> pd.DataFrame:
    """
    Year-by-year nominal cashflows for the PV system.

    Energy balance with battery losses L:
      Consumed = PV + Imported - Exported - L*(PV - Exported)
    -> Exported = [Imported + (1-L)*PV - Consumed] / (1-L), clipped to [0, PV]
    """
    years = np.arange(1, p.horizon_years + 1)

    # PV production with degradation
    pv_kwh = p.pv_year1_kwh * (1 - p.pv_degradation) ** (years - 1)

    # Prices (grid escalates; feed-in constant)
    grid_price = p.grid_price_eur_kwh * (1 + p.grid_price_escalation) ** (years - 1)
    feedin_price = np.full_like(years, p.feedin_eur_kwh, dtype=float)

    # User-entered constants per year
    imported_kwh  = np.full_like(years, p.imported_kwh_year, dtype=float)
    consumed_kwh  = np.full_like(years, p.consumed_kwh_year, dtype=float)
    L = np.full_like(years, p.battery_loss_rate, dtype=float)

    # Avoid division by zero if L -> 1
    denom = np.maximum(1.0 - L, 1e-9)

    # Exported from energy balance, then clipped to [0, PV]
    exported_raw = (imported_kwh + (1.0 - L) * pv_kwh - consumed_kwh) / denom
    exported_kwh = np.clip(exported_raw, 0.0, pv_kwh)

    # Battery losses (apply to PV used on-site): L * (PV - Exported)
    battery_losses_kwh = L * (pv_kwh - exported_kwh)

    # Self-consumed PV after battery losses
    self_consumed_kwh = np.maximum(pv_kwh - exported_kwh - battery_losses_kwh, 0.0)

    # Avoided imports (purely informational): Consumed - Imported (>=0)
    avoided_import_kwh = np.maximum(consumed_kwh - imported_kwh, 0.0)

    # Cash components (nominal)
    avoided_import_eur = avoided_import_kwh * grid_price
    feedin_revenue_eur = exported_kwh * feedin_price
    opex_eur = np.full_like(years, fill_value=p.opex_eur_year, dtype=float)

    inverter_cost_eur = np.zeros_like(years, dtype=float)
    if p.inverter_swap_year is not None and 1 <= p.inverter_swap_year <= p.horizon_years:
        inverter_cost_eur[p.inverter_swap_year - 1] = p.inverter_swap_cost

    cashflow_eur = avoided_import_eur + feedin_revenue_eur - opex_eur - inverter_cost_eur

    # Annual user cost = Import cost − Export credit + OPEX + Inverter
    import_cost_eur   = imported_kwh * grid_price
    export_credit_eur = exported_kwh * feedin_price
    annual_user_cost_eur = import_cost_eur - export_credit_eur + opex_eur + inverter_cost_eur

    df = pd.DataFrame({
        "year_index": years,
        "calendar_year": p.start_year + years - 1,
        "pv_kwh": pv_kwh,
        "imported_kwh": imported_kwh,
        "consumed_kwh": consumed_kwh,
        "battery_loss_rate": L,
        "battery_losses_kwh": battery_losses_kwh,
        "exported_kwh": exported_kwh,
        "self_consumed_kwh": self_consumed_kwh,
        "avoided_import_kwh": avoided_import_kwh,
        "grid_price_eur_kwh": grid_price,
        "feedin_eur_kwh": feedin_price,
        "avoided_import_eur": avoided_import_eur,
        "feedin_revenue_eur": feedin_revenue_eur,
        "import_cost_eur": import_cost_eur,
        "export_credit_eur": export_credit_eur,
        "annual_user_cost_eur": annual_user_cost_eur,
        "opex_eur": opex_eur,
        "inverter_cost_eur": inverter_cost_eur,
        "cashflow_eur": cashflow_eur
    })

    # Cumulative nominal cashflow (CAPEX at t0)
    df["cum_cashflow_eur"] = -p.capex + df["cashflow_eur"].cumsum()
    return df

# --- Payback helpers (calendar year & duration from t0) ---
def payback_calendar_year(df: pd.DataFrame) -> Optional[float]:
    """Return fractional calendar year where cum_cashflow crosses >= 0."""
    cum = df["cum_cashflow_eur"].reset_index(drop=True)
    years = df["calendar_year"].reset_index(drop=True)
    if cum.iloc[-1] < 0: return None
    if cum.iloc[0] >= 0: return float(years.iloc[0])
    for i in range(1, len(cum)):
        a, b = cum.iloc[i-1], cum.iloc[i]
        if a < 0 <= b:
            y0, y1 = years.iloc[i-1], years.iloc[i]
            frac = (0 - a) / (b - a + 1e-12)
            return float(y0 + (y1 - y0) * frac)
    return None

def payback_duration_years(df: pd.DataFrame) -> Optional[float]:
    """Return fractional years from t0 to break-even (1st year ends at 1.0)."""
    cum = df["cum_cashflow_eur"].reset_index(drop=True)
    if cum.iloc[-1] < 0: return None
    if cum.iloc[0] >= 0: return 0.0
    for i in range(1, len(cum)):
        a, b = cum.iloc[i-1], cum.iloc[i]
        if a < 0 <= b:
            frac = (0 - a) / (b - a + 1e-12)
            return (i - 1) + frac
    return None

def kpis(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    annual_savings_y1 = float(df["cashflow_eur"].iloc[0])
    pb_cal = payback_calendar_year(df)
    pb_dur = payback_duration_years(df)
    return {
        "annual_savings_y1": annual_savings_y1,
        "payback_calendar_year": pb_cal,
        "payback_duration_years": pb_dur
    }

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="PV Payback (battery loss %)", layout="wide")
st.title("☀️ PV Payback: Baseline vs Optimized")
st.write(
    "This app compares two scenarios (Baseline vs. Optimized) for a residential PV system.\n\n "
    "Enter annual PV production (with degradation), imported and consumed energy, and **battery loss (%)**. "
    "Exported energy is derived from the energy balance and is clipped to [0, PV].\n\n "
    "Grid price escalation is applied. The app shows annual savings, cumulative cashflows, annual user cost, payback, "
    "and exported energy over time."
)

with st.sidebar:
    st.header("General Parameters")
    start_year = st.number_input("Start year (commissioning)", value=2023, step=1)
    horizon = st.slider("Horizon (years)", min_value=10, max_value=25, value=20, step=1)
    capex = st.number_input("Initial invest (EUR)", value=19000.0, min_value=0.0, step=500.0, format="%.2f")

    st.markdown("---")
    st.header("PV & Energy Flows")
    pv_y1 = st.number_input("PV production in Year 1 (kWh)", value=6000.0, min_value=0.0, step=100.0)
    degr_pct = st.number_input("PV degradation per year (%)", value=0.5, min_value=0.0, max_value=5.0, step=0.1, format="%.1f")
    degr = degr_pct / 100.0

    imported_kwh = st.number_input("Imported energy (kWh/yr)", value=1200.0, min_value=0.0, step=100.0)
    consumed_kwh = st.number_input("Consumed energy (kWh/yr)", value=3500.0, min_value=0.0, step=100.0)
    battery_loss_pct = st.number_input("Battery loss (%)", value=7.0, min_value=0.0, max_value=50.0, step=0.5, format="%.1f")
    battery_loss_rate = battery_loss_pct / 100.0

    st.markdown("---")
    st.header("Prices")
    base_grid_price = st.number_input("Baseline grid price (EUR/kWh)", value=0.40, min_value=0.0, step=0.01, format="%.3f")
    opt_grid_price = st.number_input("Optimized grid price (EUR/kWh)", value=0.32, min_value=0.0, step=0.01, format="%.3f")
    feedin = st.number_input("Feed-in tariff (EUR/kWh, constant)", value=0.082, min_value=0.0, step=0.005, format="%.3f")
    price_escalation_enable = st.checkbox("Add price escalation", value=False)
    grid_escal = 0.0
    if price_escalation_enable:
        grid_escal = st.number_input("Grid price escalation (per year, both scenarios)", value=0.02, min_value=0.0, max_value=0.2, step=0.005, format="%.3f")
    

    st.markdown("---")
    st.header("Costs")
    opex_enable = st.checkbox("Add Maintenance costs", value=False)
    opex = 0.0
    if opex_enable:
        opex = st.number_input("Maintenance costs (EUR/year)", value=150.0, min_value=0.0, step=10.0, format="%.2f")

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
    pv_year1_kwh=pv_y1,
    pv_degradation=degr,
    imported_kwh_year=imported_kwh,
    consumed_kwh_year=consumed_kwh,
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

kpi_base = kpis(df_base)
kpi_opt  = kpis(df_opt)

# -----------------------------
# KPIs (3 fields each)
# -----------------------------
st.subheader("KPIs")

def fmt_payback_year(y):  return "No payback" if y is None else f"{int(round(y)):d}"
def fmt_payback_duration(y): return "No payback" if y is None else f"{y:.1f} years"

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Baseline**")
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Payback year", fmt_payback_year(kpi_base["payback_calendar_year"]))
    cc2.metric("Dauer bis Payback", fmt_payback_duration(kpi_base["payback_duration_years"]))
    cc3.metric("Annual savings (Year 1)", f"{kpi_base['annual_savings_y1']:,.0f} €")
with c2:
    st.markdown("**Optimized**")
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Payback year", fmt_payback_year(kpi_opt["payback_calendar_year"]))
    cc2.metric("Dauer bis Payback", fmt_payback_duration(kpi_opt["payback_duration_years"]))
    cc3.metric("Annual savings (Year 1)", f"{kpi_opt['annual_savings_y1']:,.0f} €")

# -----------------------------
# Cumulative cashflow line chart
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
    if cum_series.iloc[-1] < 0: return None
    if cum_series.iloc[0] >= 0: return float(years.iloc[0])
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
# Annual user cost: grouped bars
# -----------------------------
st.markdown("### Annual user cost (import − export credit + maintenance + inverter)")

def build_user_cost_df(df: pd.DataFrame, label: str) -> pd.DataFrame:
    return pd.DataFrame({
        "calendar_year": df["calendar_year"],
        "annual_user_cost_eur": df["annual_user_cost_eur"],
        "Scenario": label
    })

# compute annual_user_cost_eur columns if not present (safety)
for _df in (df_base, df_opt):
    if "annual_user_cost_eur" not in _df.columns:
        _df["annual_user_cost_eur"] = (
            _df["imported_kwh"] * _df["grid_price_eur_kwh"]
            - _df["exported_kwh"] * _df["feedin_eur_kwh"]
            + _df["opex_eur"] + _df["inverter_cost_eur"]
        )

bar_base = build_user_cost_df(df_base, "Baseline")
bar_opt  = build_user_cost_df(df_opt,  "Optimized")
bar_df = pd.concat([bar_base, bar_opt], ignore_index=True)

bars = alt.Chart(bar_df).mark_bar().encode(
    x=alt.X("calendar_year:O", title="year", axis=alt.Axis(labelAngle=0, format="d")),
    y=alt.Y("annual_user_cost_eur:Q", title="money [€]"),
    color=alt.Color("Scenario:N", legend=alt.Legend(title="Scenario")),
    xOffset="Scenario:N"
)
st.altair_chart(bars.properties(height=320), use_container_width=True)

# -----------------------------
# Exported Energy line chart
# -----------------------------
st.markdown("### Exported Energy (kWh)")
exp_plot = pd.concat([
    df_base[["calendar_year", "exported_kwh"]]    
], ignore_index=True)

exp_line = alt.Chart(exp_plot).mark_line().encode(
    x=alt.X("calendar_year:Q", title="year", axis=alt.Axis(format="d")),
    y=alt.Y("exported_kwh:Q", title="exported energy [kWh]"),
    color=alt.Color("Scenario:N", legend=alt.Legend(title="Scenario"))
)
st.altair_chart(exp_line.properties(height=300), use_container_width=True)

# -----------------------------
# Annual details
# -----------------------------
st.markdown("### Annual details")
tab1, tab2 = st.tabs(["Baseline", "Optimized"])

show_cols = [
    "calendar_year", "pv_kwh", "consumed_kwh", "imported_kwh",
    "battery_loss_rate", "battery_losses_kwh", "exported_kwh",
    "grid_price_eur_kwh", "feedin_eur_kwh",
    "import_cost_eur", "export_credit_eur",
    "avoided_import_eur", "opex_eur", "inverter_cost_eur",
    "cashflow_eur", "annual_user_cost_eur", "cum_cashflow_eur"
]

with tab1:
    st.dataframe(df_base[show_cols].round(3), use_container_width=True)
with tab2:
    st.dataframe(df_opt[show_cols].round(3), use_container_width=True)

st.markdown("—")
st.caption("Battery loss (%) reduces PV used on-site. Exported energy is solved from the energy balance and clipped to [0, PV]. Costs include grid price escalation.")
