"""Ecommerce growth projector (command line interface).

Loads ad_spend.csv and orders.csv, runs the modeling engine, and prints the
historical summary, acquisition curve, repeat/LTV curve, unit economics, the
base projection, and a scenario comparison. It also writes projection.csv.

All modeling lives in engine.py. This file is only the CLI wrapper: it reads the
CSVs, calls the engine, and formats the output. No machine learning.
"""

import pandas as pd

import engine

# ----------------------------------------------------------------------------
# Config (edit these).
# ----------------------------------------------------------------------------
GROSS_MARGIN = 0.45        # product margin after COGS, before ad spend.
PROJECTION_MONTHS = 12     # how many months to project forward.
LTV_CAP_MONTHS = 12        # cap the repeat curve here (older ages have thin data).

# Monthly percent change applied to the last historical month's spend.
SCENARIOS = {"conservative": 0.00, "base": 0.08, "aggressive": 0.15}
BASE_SCENARIO = "base"


# ----------------------------------------------------------------------------
# Output.
# ----------------------------------------------------------------------------
def rule(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def main():
    ad_spend = pd.read_csv("ad_spend.csv")
    orders = pd.read_csv("orders.csv")

    con = engine.load_connection(ad_spend, orders)

    ad = engine.monthly_ad_metrics(con)
    acq = engine.monthly_acquisition(con)
    cohorts = engine.cohort_orders(con)

    curve_fit = engine.fit_acquisition_curve(ad, acq)
    repeat_curve = engine.fit_repeat_curve(cohorts, acq, LTV_CAP_MONTHS)
    econ = engine.unit_economics(ad, acq, repeat_curve, GROSS_MARGIN)

    # Historical monthly summary.
    rule("HISTORICAL MONTHLY SUMMARY")
    hist = ad.merge(acq, on="month", how="left")
    hist["cac"] = hist["spend"] / hist["new_customers"]
    show = hist[
        ["month", "spend", "impressions", "clicks", "cpm", "cpc", "ctr_pct",
         "new_customers", "acq_revenue", "cac"]
    ].copy()
    with pd.option_context("display.float_format", lambda v: f"{v:,.2f}"):
        print(show.to_string(index=False))

    # Acquisition curve.
    rule("ACQUISITION RESPONSE CURVE  (customers = a * spend^b)")
    print(f"  a (scale):            {curve_fit['a']:.5f}")
    print(f"  b (elasticity):       {curve_fit['b']:.4f}   (b < 1 means diminishing returns)")
    print(f"  R^2 (log scale):      {curve_fit['r2']:.4f}")
    print(f"  mean monthly spend:   {curve_fit['mean_spend']:,.2f}")
    print(f"  marginal CAC at mean: {curve_fit['marginal_cac']:,.2f}   (cost of the next customer)")

    # Repeat / LTV curve.
    rule("REPEAT / LTV CURVE  (pooled revenue per customer by age in months)")
    cum = repeat_curve.cumsum()
    for age in repeat_curve.index:
        print(f"  age {age:>2}:  per-customer {repeat_curve[age]:7.2f}   cumulative {cum[age]:8.2f}")

    # Unit economics.
    rule("UNIT ECONOMICS")
    print(f"  Blended CAC (last 3 mo):     {econ['blended_cac']:,.2f}")
    print(f"  AOV (first order value):     {econ['aov']:,.2f}")
    print(f"  Full LTV (sum of curve):     {econ['ltv']:,.2f}")
    print(f"  Gross margin:                {GROSS_MARGIN:.0%}")
    print(f"  Break-even ROAS (1/margin):  {econ['breakeven_roas']:.2f}x")
    print(f"  Margin-adjusted LTV:CAC:     {econ['ltv_cac']:.2f}")
    payback = econ["payback_months"]
    print(f"  Payback period:              {payback} months" if payback is not None
          else "  Payback period:              not reached within cap")

    # Base projection.
    base = engine.project(
        ad, acq, curve_fit, repeat_curve, SCENARIOS[BASE_SCENARIO],
        PROJECTION_MONTHS, LTV_CAP_MONTHS, GROSS_MARGIN,
    )
    rule(f"PROJECTION ({BASE_SCENARIO} scenario, {SCENARIOS[BASE_SCENARIO]:+.0%} monthly spend)")
    with pd.option_context("display.float_format", lambda v: f"{v:,.2f}"):
        print(base.to_string(index=False))
    base.to_csv("projection.csv", index=False)
    print("\nWrote projection.csv")

    # Scenario comparison (12 month totals).
    rule("SCENARIO COMPARISON  (12-month totals)")
    comp_rows = []
    for name, growth in SCENARIOS.items():
        proj = engine.project(
            ad, acq, curve_fit, repeat_curve, growth,
            PROJECTION_MONTHS, LTV_CAP_MONTHS, GROSS_MARGIN,
        )
        total_spend = proj["spend"].sum()
        total_revenue = proj["revenue"].sum()
        total_gp = proj["gross_profit"].sum()
        exit_roas = proj.iloc[-1]["revenue"] / proj.iloc[-1]["spend"]
        comp_rows.append(
            {
                "scenario": name,
                "monthly_growth": f"{growth:+.0%}",
                "total_spend": total_spend,
                "total_revenue": total_revenue,
                "total_gross_profit": total_gp,
                "exit_roas": exit_roas,
            }
        )
    comp = pd.DataFrame(comp_rows)
    with pd.option_context("display.float_format", lambda v: f"{v:,.2f}"):
        print(comp.to_string(index=False))

    con.close()


if __name__ == "__main__":
    main()
