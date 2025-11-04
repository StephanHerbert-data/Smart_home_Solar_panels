# -----------------------------
# START APP via Terminal
# python -m streamlit run app.py
# ======================================
# If needed: Install Streamlit
# python -m pip install streamlit

# -----------------------------

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from dataclasses import dataclass, replace
from typing import Optional, Dict

# =============================
# Axis styling helper (unified)
# =============================
def styled_axis(title: str,
                fmt: str | None = None,
                label_size: int = 14,
                title_size: int = 16):
    """
    Create a consistent Altair Axis with unified typography.
    If fmt is None, the 'format' key is omitted to avoid schema errors.
    """
    axis_kwargs = {
        "title": title,
        "labelFontSize": label_size,
        "titleFontSize": title_size,
        "labelFont": "Source Sans Pro",  
        "titleFont": "Source Sans Pro",
        "labelColor": "#222",
        "titleColor": "#222",
        "labelAngle": 0,
    }
    if fmt is not None:  # only set when we have a real string
        axis_kwargs["format"] = fmt

    return alt.Axis(**axis_kwargs)

# =============================
# Domain model
# =============================
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

    # NEW: base fee per contract year (added only to "annual user cost")
    base_fee_eur_year: float = 0.0

# =============================
# Simulation core (no discounting)
# =============================
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

    # Informational: avoided imports = max(Consumed - Imported, 0)
    avoided_import_kwh = np.maximum(consumed_kwh - imported_kwh, 0.0)

    # Cash components (nominal, PV-related)
    avoided_import_eur = avoided_import_kwh * grid_price
    feedin_revenue_eur = exported_kwh * feedin_price
    opex_eur = np.full_like(years, fill_value=p.opex_eur_year, dtype=float)

    inverter_cost_eur = np.zeros_like(years, dtype=float)
    if p.inverter_swap_year is not None and 1 <= p.inverter_swap_year <= p.horizon_years:
        inverter_cost_eur[p.inverter_swap_year - 1] = p.inverter_swap_cost

    # PV-driven cashflow (do NOT include base fee here)
    cashflow_eur = avoided_import_eur + feedin_revenue_eur - opex_eur - inverter_cost_eur

    # User-facing annual cost incl. base fee if set
    import_cost_eur   = imported_kwh * grid_price
    export_credit_eur = exported_kwh * feedin_price
    base_fee_eur      = np.full_like(years, p.base_fee_eur_year, dtype=float)
    annual_user_cost_eur = import_cost_eur - export_credit_eur + opex_eur + inverter_cost_eur + base_fee_eur

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
        "base_fee_eur": base_fee_eur,
        "annual_user_cost_eur": annual_user_cost_eur,
        "opex_eur": opex_eur,
        "inverter_cost_eur": inverter_cost_eur,
        "cashflow_eur": cashflow_eur
    })

    # Cumulative nominal cashflow (CAPEX at t0)
    df["cum_cashflow_eur"] = -p.capex + df["cashflow_eur"].cumsum()
    return df

# =============================
# Payback helpers & KPIs
# =============================
def payback_calendar_year(df: pd.DataFrame) -> Optional[float]:
    """Return fractional calendar year where cum_cashflow crosses >= 0."""
    cum = df["cum_cashflow_eur"].reset_index(drop=True)
    years = df["calendar_year"].reset_index(drop=True)
    if cum.iloc[-1] < 0:
        return None
    if cum.iloc[0] >= 0:
        return float(years.iloc[0])
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
    if cum.iloc[-1] < 0:
        return None
    if cum.iloc[0] >= 0:
        return 0.0
    for i in range(1, len(cum)):
        a, b = cum.iloc[i-1], cum.iloc[i]
        if a < 0 <= b:
            frac = (0 - a) / (b - a + 1e-12)
            return (i - 1) + frac
    return None

def kpis(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    return {
        "annual_savings_y1": float(df["cashflow_eur"].iloc[0]),
        "payback_calendar_year": payback_calendar_year(df),
        "payback_duration_years": payback_duration_years(df),
        # Grid price KPI = Year-1 grid price of the scenario
        "grid_price_y1": float(df["grid_price_eur_kwh"].iloc[0])
    }

# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="PV Payback (battery loss %, base fee, unified axes)", layout="wide")
st.title("☀️ PV Payback: Baseline vs Optimized grid prices")

# Description with paragraphs
st.markdown(
    "This app compares **two scenarios (Baseline vs. Optimized)** for a residential PV system. Please change the 2 **grid prices** accordingly. \n\n"
    "It computes annual savings, cumulative cashflows, payback, annual user costs, and exported energy over time.\n\n"
    "**Exported energy** is derived from the energy balance and clipped to [0, PV].\n\n "
    "**Please enter** PV production (with degradation), imported and consumed energy, battery loss (%), energy prices "
    "(with escalation), maintenance & inverter costs, and optionally a contract base fee."
    
)

# Sidebar inputs
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
    base_grid_price = st.number_input("Baseline grid price (EUR/kWh)", value=0.40, min_value=0.0, step=0.01, format="%.2f")
    opt_grid_price = st.number_input("Optimized grid price (EUR/kWh)", value=0.32, min_value=0.0, step=0.01, format="%.2f")
    feedin = st.number_input("Feed-in tariff (EUR/kWh, constant)", value=0.082, min_value=0.0, step=0.005, format="%.3f")
    price_escalation_enable = st.checkbox("Add price escalation", value=False)
    grid_escal = 0.0
    if price_escalation_enable:
        grid_escal = st.number_input("Grid price escalation (per year, both scenarios)", value=0.02, min_value=0.0, max_value=0.2, step=0.005, format="%.3f")
    
    st.markdown("---")
    st.header("Contract base fee")
    include_base_fee = st.checkbox("Include base fee", value=False)
    base_fee = 0.0
    if include_base_fee:
        base_fee = st.number_input("Base fee (EUR/year)", value=154.0, min_value=0.0, step=5.0, format="%.2f")

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


# Shared params (only grid price differs per scenario)
shared = Params(
    start_year=start_year,
    horizon_years=horizon,
    capex=capex,
    pv_year1_kwh=pv_y1,
    pv_degradation=degr,
    imported_kwh_year=imported_kwh,
    consumed_kwh_year=consumed_kwh,
    battery_loss_rate=battery_loss_rate,
    grid_price_eur_kwh=base_grid_price,  # placeholder; overridden per scenario
    grid_price_escalation=grid_escal,
    feedin_eur_kwh=feedin,
    opex_eur_year=opex,
    inverter_swap_year=swap_year,
    inverter_swap_cost=swap_cost,
    base_fee_eur_year=base_fee
)

params_base = replace(shared, grid_price_eur_kwh=base_grid_price)
params_opt  = replace(shared, grid_price_eur_kwh=opt_grid_price)

df_base = simulate_cashflows(params_base)
df_opt  = simulate_cashflows(params_opt)

kpi_base = kpis(df_base)
kpi_opt  = kpis(df_opt)

# =============================
# KPIs (4 fields each)
# =============================
st.subheader("KPIs")

def fmt_payback_year(y):   return "No payback" if y is None else f"{int(round(y)):d}"
def fmt_payback_dur(y):    return "No payback" if y is None else f"{y:.1f} years"
def fmt_grid_price(v):     return f"{v:.2f} €/kWh"

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Baseline**")
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Payback year", fmt_payback_year(kpi_base["payback_calendar_year"]))
    k2.metric("Years until payback", fmt_payback_dur(kpi_base["payback_duration_years"]))
    k3.metric("Annual savings (Year 1)", f"{kpi_base['annual_savings_y1']:,.0f} €")
    k4.metric("Grid price (Y1)", fmt_grid_price(kpi_base["grid_price_y1"]))

with c2:
    st.markdown("**Optimized**")
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Payback year", fmt_payback_year(kpi_opt["payback_calendar_year"]))
    k2.metric("Years until payback", fmt_payback_dur(kpi_opt["payback_duration_years"]))
    k3.metric("Annual savings (Year 1)", f"{kpi_opt['annual_savings_y1']:,.0f} €")
    k4.metric("Grid price (Y1)", fmt_grid_price(kpi_opt["grid_price_y1"]))

# =============================
# Chart 1: Cumulative cashflow
# =============================
st.markdown("### Cumulative cashflow (nominal)")

plot_df = pd.concat([
    df_base[["calendar_year", "cum_cashflow_eur"]].rename(columns={"cum_cashflow_eur":"value"}).assign(Scenario="Baseline"),
    df_opt[ ["calendar_year", "cum_cashflow_eur"]].rename(columns={"cum_cashflow_eur":"value"}).assign(Scenario="Optimized"),
], ignore_index=True)

x_axis_year  = styled_axis("year", fmt="d")
y_axis_money = styled_axis("money [€]")

zero_rule = alt.Chart(pd.DataFrame({"y":[0]})).mark_rule(strokeDash=[6,4], color="#888").encode(y="y:Q")
line = alt.Chart(plot_df).mark_line().encode(
    x=alt.X("calendar_year:Q", axis=x_axis_year),
    y=alt.Y("value:Q",         axis=y_axis_money),
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
        pt = alt.Chart(pd.DataFrame({"calendar_year":[cal], "value":[0.0], "Scenario":[label]}))\
              .mark_point(size=100).encode(x="calendar_year:Q", y="value:Q", color="Scenario:N")
        vr = alt.Chart(pd.DataFrame({"calendar_year":[cal]})).mark_rule(strokeDash=[3,3]).encode(x="calendar_year:Q")
        layers += [pt, vr]

chart = layers[0]
for lyr in layers[1:]:
    chart = chart + lyr
st.altair_chart(chart.properties(height=360), use_container_width=True)

# =============================
# Chart 2: Annual user cost (grouped bars)
# =============================
st.markdown("### Annual user cost (import − export credit + maintenance + inverter + base fee)")

bar_df = pd.concat([
    df_base[["calendar_year","annual_user_cost_eur"]].assign(Scenario="Baseline"),
    df_opt[ ["calendar_year","annual_user_cost_eur"]].assign(Scenario="Optimized"),
], ignore_index=True)

x_axis_year  = styled_axis("year", fmt="d")
y_axis_money = styled_axis("money [€]")

bars = alt.Chart(bar_df).mark_bar().encode(
    x=alt.X("calendar_year:O", axis=x_axis_year),
    y=alt.Y("annual_user_cost_eur:Q", axis=y_axis_money),
    color=alt.Color("Scenario:N", legend=alt.Legend(title="Scenario")),
    xOffset="Scenario:N"
)
st.altair_chart(bars.properties(height=320), use_container_width=True)

# =============================
# Chart 3: Exported Energy line chart
# =============================
st.markdown("### Exported Energy (kWh) over year")

exp_plot = pd.concat([
    df_base[["calendar_year","exported_kwh"]].assign(Scenario="Baseline"),
    df_opt[ ["calendar_year","exported_kwh"]].assign(Scenario="Optimized"),
], ignore_index=True)

x_axis_year   = styled_axis("year", fmt="d")
y_axis_export = styled_axis("exported energy [kWh]")

exp_line = alt.Chart(exp_plot).mark_line().encode(
    x=alt.X("calendar_year:Q", axis=x_axis_year),
    y=alt.Y("exported_kwh:Q",  axis=y_axis_export),
    color=alt.Color("Scenario:N", legend=alt.Legend(title="Scenario"))
)
st.altair_chart(exp_line.properties(height=300), use_container_width=True)

# =============================
# Annual details
# =============================
st.markdown("### Annual details")
tab1, tab2 = st.tabs(["Baseline", "Optimized"])

show_cols = [
    "calendar_year", "pv_kwh", "consumed_kwh", "imported_kwh",
    "battery_loss_rate", "battery_losses_kwh", "exported_kwh",
    "grid_price_eur_kwh", "feedin_eur_kwh",
    "import_cost_eur", "export_credit_eur", "base_fee_eur",
    "avoided_import_eur", "opex_eur", "inverter_cost_eur",
    "cashflow_eur", "annual_user_cost_eur", "cum_cashflow_eur"
]

with tab1:
    st.dataframe(df_base[show_cols].round(3), use_container_width=True)
with tab2:
    st.dataframe(df_opt[show_cols].round(3), use_container_width=True)

st.caption("KPIs include the scenario's Year-1 grid price. Base fee is included only in Annual user cost (not in PV cashflow/payback).")
