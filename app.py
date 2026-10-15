"""Streamlit dashboard for the ecom growth projector.

A visitor can upload their own ad_spend and orders CSVs or use built in sample
data, adjust the model knobs with sliders, and see the projection as interactive
Plotly charts plus a downloadable table. All modeling is done by engine.py, which
is deterministic arithmetic (a log-log least squares fit and pooled cohort
averages). There is no machine learning anywhere.
"""

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import engine
import generate_sample_data as sample


# ----------------------------------------------------------------------------
# Page setup.
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Ecom Growth Projector", layout="wide")
st.title("Ecom Growth Projector")
st.caption(
    "Forecast revenue and gross profit from ad spend and orders. "
    "All math is deterministic arithmetic (a log-log least squares fit plus "
    "pooled cohort averages), with no machine learning."
)

# Colors used across the charts.
C_REVENUE = "#2563eb"
C_PROFIT = "#16a34a"
C_SPEND = "#94a3b8"
C_FIT = "#dc2626"
C_POINTS = "#2563eb"
C_CUM = "#dc2626"


@st.cache_data(show_spinner=False)
def get_sample_data():
    """Generate the sample ad_spend and orders DataFrames (cached)."""
    return sample.generate()


def missing_columns(df, required):
    """Return the required columns that are absent from df."""
    return [c for c in required if c not in df.columns]


def prepare_ad_spend(df):
    """Validate and clean an ad_spend frame.

    Returns (clean_df, dropped_count, has_funnel). Calls st.error and st.stop
    when required columns are missing.
    """
    miss = missing_columns(df, ["date", "spend"])
    if miss:
        st.error(
            "Ad spend CSV is missing required column(s): "
            + ", ".join(miss)
            + ". It needs at least date and spend (impressions, clicks, "
            "platform are optional)."
        )
        st.stop()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["spend"] = pd.to_numeric(df["spend"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["date", "spend"])
    dropped = before - len(df)

    # Normalize the date to a plain YYYY-MM-DD string so the SQL strftime works.
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    has_funnel = ("impressions" in df.columns) and ("clicks" in df.columns)
    if has_funnel:
        df["impressions"] = pd.to_numeric(df["impressions"], errors="coerce")
        df["clicks"] = pd.to_numeric(df["clicks"], errors="coerce")
    else:
        # The SQL layer references these columns, so provide them as NULL. With
        # all NULLs the funnel metrics come back NULL and are simply not shown.
        df["impressions"] = pd.NA
        df["clicks"] = pd.NA

    return df, dropped, has_funnel


def prepare_orders(df):
    """Validate and clean an orders frame.

    Returns (clean_df, dropped_count). Calls st.error and st.stop when required
    columns are missing.
    """
    miss = missing_columns(df, ["customer_id", "order_date", "revenue"])
    if miss:
        st.error(
            "Orders CSV is missing required column(s): "
            + ", ".join(miss)
            + ". It needs at least customer_id, order_date and revenue "
            "(order_id is optional)."
        )
        st.stop()

    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["customer_id", "order_date", "revenue"])
    dropped = before - len(df)

    df["order_date"] = df["order_date"].dt.strftime("%Y-%m-%d")
    return df, dropped


def money(value):
    """Format a number as a dollar string, handling None and NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"${value:,.0f}"


# ----------------------------------------------------------------------------
# Sidebar controls.
# ----------------------------------------------------------------------------
st.sidebar.header("Model knobs")

gross_margin = st.sidebar.slider(
    "Gross margin", min_value=0.10, max_value=0.90, value=0.45, step=0.01,
    format="%.0f%%",
    help="Product margin after COGS, before ad spend. Break-even ROAS is 1 divided by this.",
)
projection_months = st.sidebar.slider(
    "Projection horizon (months)", min_value=3, max_value=24, value=12, step=1,
)
ltv_cap_months = st.sidebar.slider(
    "LTV cap (months)", min_value=6, max_value=18, value=12, step=1,
    help="Cap the repeat curve here. Older ages have thin data.",
)

st.sidebar.subheader("Scenario monthly spend growth")
g_cons = st.sidebar.number_input("Conservative (%)", value=0.0, step=1.0) / 100.0
g_base = st.sidebar.number_input("Base (%)", value=8.0, step=1.0) / 100.0
g_aggr = st.sidebar.number_input("Aggressive (%)", value=15.0, step=1.0) / 100.0
scenarios = {"conservative": g_cons, "base": g_base, "aggressive": g_aggr}

selected_scenario = st.sidebar.selectbox(
    "Scenario for chart and table", options=list(scenarios.keys()), index=1,
)


# ----------------------------------------------------------------------------
# Data input.
# ----------------------------------------------------------------------------
st.header("1. Data")

use_sample = st.checkbox("Use sample data instead", value=True)

col_up1, col_up2 = st.columns(2)
with col_up1:
    ad_file = st.file_uploader("Ad spend CSV", type="csv", disabled=use_sample)
with col_up2:
    orders_file = st.file_uploader("Orders CSV", type="csv", disabled=use_sample)

if use_sample:
    ad_raw, orders_raw = get_sample_data()
    st.info("Using built in sample data. Uncheck the box above to upload your own CSVs.")
else:
    if ad_file is None or orders_file is None:
        st.warning("Upload both an ad spend CSV and an orders CSV, or check "
                   "\"Use sample data instead\".")
        st.stop()
    ad_raw = pd.read_csv(ad_file)
    orders_raw = pd.read_csv(orders_file)

ad_clean, ad_dropped, has_funnel = prepare_ad_spend(ad_raw)
orders_clean, orders_dropped = prepare_orders(orders_raw)

if ad_dropped:
    st.warning(f"Dropped {ad_dropped} ad spend row(s) with an unparseable date or spend.")
if orders_dropped:
    st.warning(f"Dropped {orders_dropped} order row(s) with an unparseable "
               "customer, date or revenue.")
if not has_funnel:
    st.info("No impressions or clicks columns found, so the cpm, cpc and ctr "
            "funnel metrics are skipped.")

# ----------------------------------------------------------------------------
# Run the engine.
# ----------------------------------------------------------------------------
con = engine.load_connection(ad_clean, orders_clean)
ad = engine.monthly_ad_metrics(con)
acq = engine.monthly_acquisition(con)
cohorts = engine.cohort_orders(con)
con.close()

# Guard: the acquisition curve is fit on months with spend and customers.
fit_rows = ad.merge(acq, on="month", how="inner")
fit_rows = fit_rows[(fit_rows["spend"] > 0) & (fit_rows["new_customers"] > 0)]
if len(fit_rows) < 2:
    st.error("Not enough history to fit the acquisition curve. Need at least 2 "
             "months with spend greater than 0 and new customers greater than 0.")
    st.stop()
if len(fit_rows) < 3:
    st.warning("Only 2 months of usable history. The acquisition curve fit needs "
               "more history to be reliable.")

curve_fit = engine.fit_acquisition_curve(ad, acq)
repeat_curve = engine.fit_repeat_curve(cohorts, acq, ltv_cap_months)
econ = engine.unit_economics(ad, acq, repeat_curve, gross_margin)

projections = {
    name: engine.project(ad, acq, curve_fit, repeat_curve, growth,
                         projection_months, ltv_cap_months, gross_margin)
    for name, growth in scenarios.items()
}
selected_proj = projections[selected_scenario]


# ----------------------------------------------------------------------------
# 1. KPI row.
# ----------------------------------------------------------------------------
st.header("2. Unit economics")
payback = econ["payback_months"]
row1 = st.columns(3)
row1[0].metric("Blended CAC", money(econ["blended_cac"]))
row1[1].metric("AOV (first order)", money(econ["aov"]))
row1[2].metric("Break-even ROAS", f"{econ['breakeven_roas']:.2f}x")
row2 = st.columns(3)
row2[0].metric("Margin-adjusted LTV:CAC", f"{econ['ltv_cac']:.2f}")
row2[1].metric("Payback",
               f"{payback} mo" if payback is not None else "not reached")
row2[2].metric("Elasticity b", f"{curve_fit['b']:.3f}")


# ----------------------------------------------------------------------------
# 2. Acquisition response curve.
# ----------------------------------------------------------------------------
st.header("3. Acquisition response curve")
st.caption("Diminishing returns: each extra dollar of spend buys fewer customers "
           "because b is less than 1.")

a = curve_fit["a"]
b = curve_fit["b"]
spend_min = float(fit_rows["spend"].min())
spend_max = float(fit_rows["spend"].max())
spend_line = np.linspace(spend_min, spend_max, 100)
cust_line = a * spend_line ** b

fig_acq = go.Figure()
fig_acq.add_trace(go.Scatter(
    x=fit_rows["spend"], y=fit_rows["new_customers"], mode="markers",
    name="Historical months", marker=dict(size=10, color=C_POINTS),
))
fig_acq.add_trace(go.Scatter(
    x=spend_line, y=cust_line, mode="lines",
    name="Fitted curve", line=dict(color=C_FIT, width=3),
))
fig_acq.update_layout(
    title=f"customers = {a:.3f} * spend^{b:.3f}   (R^2 = {curve_fit['r2']:.3f})",
    xaxis_title="Monthly ad spend ($)",
    yaxis_title="New customers",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=420,
)
st.plotly_chart(fig_acq, width="stretch")


# ----------------------------------------------------------------------------
# 3. LTV / repeat curve.
# ----------------------------------------------------------------------------
st.header("4. LTV and repeat curve")
st.caption("Pooled revenue per customer by age in months, with cumulative LTV.")

ages = list(repeat_curve.index)
per_customer = repeat_curve.values
cumulative = repeat_curve.cumsum().values

fig_ltv = make_subplots(specs=[[{"secondary_y": True}]])
fig_ltv.add_trace(go.Bar(
    x=ages, y=per_customer, name="Per customer revenue",
    marker_color=C_REVENUE,
), secondary_y=False)
fig_ltv.add_trace(go.Scatter(
    x=ages, y=cumulative, name="Cumulative LTV", mode="lines+markers",
    line=dict(color=C_CUM, width=3),
), secondary_y=True)
fig_ltv.update_layout(
    xaxis_title="Months since acquisition",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=420,
)
fig_ltv.update_yaxes(title_text="Per customer revenue ($)", secondary_y=False)
fig_ltv.update_yaxes(title_text="Cumulative LTV ($)", secondary_y=True)
st.plotly_chart(fig_ltv, width="stretch")


# ----------------------------------------------------------------------------
# 4. Projection chart for the selected scenario.
# ----------------------------------------------------------------------------
st.header(f"5. Projection ({selected_scenario}, {scenarios[selected_scenario]:+.0%} per month)")

fig_proj = go.Figure()
fig_proj.add_trace(go.Bar(
    x=selected_proj["month"], y=selected_proj["spend"], name="Ad spend",
    marker_color=C_SPEND, opacity=0.35,
))
fig_proj.add_trace(go.Scatter(
    x=selected_proj["month"], y=selected_proj["revenue"], name="Revenue",
    mode="lines+markers", line=dict(color=C_REVENUE, width=3),
))
fig_proj.add_trace(go.Scatter(
    x=selected_proj["month"], y=selected_proj["gross_profit"], name="Gross profit",
    mode="lines+markers", line=dict(color=C_PROFIT, width=3),
))
fig_proj.update_layout(
    xaxis_title="Month",
    yaxis_title="Dollars",
    barmode="overlay",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=440,
)
st.plotly_chart(fig_proj, width="stretch")


# ----------------------------------------------------------------------------
# 5. Scenario comparison (centerpiece).
# ----------------------------------------------------------------------------
st.header("6. Scenario comparison")
st.caption("Scaling spend raises revenue, but diminishing returns and rising CAC "
           "flatten (or collapse) gross profit.")

comp_names = list(scenarios.keys())
comp_rev = [projections[n]["revenue"].sum() for n in comp_names]
comp_gp = [projections[n]["gross_profit"].sum() for n in comp_names]

fig_comp = go.Figure()
fig_comp.add_trace(go.Bar(
    x=comp_names, y=comp_rev, name="Total revenue", marker_color=C_REVENUE,
    text=[money(v) for v in comp_rev], textposition="outside",
))
fig_comp.add_trace(go.Bar(
    x=comp_names, y=comp_gp, name="Total gross profit", marker_color=C_PROFIT,
    text=[money(v) for v in comp_gp], textposition="outside",
))
fig_comp.update_layout(
    barmode="group",
    xaxis_title="Scenario",
    yaxis_title=f"{projection_months} month total ($)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=460,
)
st.plotly_chart(fig_comp, width="stretch")


# ----------------------------------------------------------------------------
# 6. Historical monthly summary table.
# ----------------------------------------------------------------------------
st.header("7. Historical monthly summary")
hist = ad.merge(acq, on="month", how="left")
hist["cac"] = hist["spend"] / hist["new_customers"]
base_cols = ["month", "spend", "new_customers", "acq_revenue", "cac"]
funnel_cols = ["impressions", "clicks", "cpm", "cpc", "ctr_pct"]
show_cols = ["month", "spend"]
if has_funnel:
    show_cols += funnel_cols
show_cols += ["new_customers", "acq_revenue", "cac"]
st.dataframe(hist[show_cols], width="stretch", hide_index=True)


# ----------------------------------------------------------------------------
# 7. Projection table plus download.
# ----------------------------------------------------------------------------
st.header(f"8. Projection table ({selected_scenario})")
st.dataframe(selected_proj, width="stretch", hide_index=True)

csv_buffer = io.StringIO()
selected_proj.to_csv(csv_buffer, index=False)
st.download_button(
    "Download projection as CSV",
    data=csv_buffer.getvalue(),
    file_name=f"projection_{selected_scenario}.csv",
    mime="text/csv",
)
