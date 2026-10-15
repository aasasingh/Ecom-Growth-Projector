"""Modeling engine for the ecom growth projector.

Pure, importable functions: they accept DataFrames and config values and return
results. No file reads, no printing. SQL owns attribution and aggregation (in an
in-memory SQLite database). The models are plain pandas and numpy: a least
squares fit of an acquisition response curve on a known functional form, and a
pooled cohort repeat curve. No machine learning.

The math here is identical to the original growth_projector.py. The only change
is that the three config values (gross_margin, projection_months, ltv_cap_months)
are function parameters instead of module constants, so a UI can vary them.
"""

import sqlite3
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Data loading.
# ----------------------------------------------------------------------------
def load_connection(ad_spend, orders):
    """Load two DataFrames into an in-memory SQLite database and add indexes."""
    con = sqlite3.connect(":memory:")
    ad_spend.to_sql("ad_spend", con, index=False, if_exists="replace")
    orders.to_sql("orders", con, index=False, if_exists="replace")

    con.execute("CREATE INDEX idx_orders_customer ON orders(customer_id)")
    con.execute("CREATE INDEX idx_orders_date ON orders(order_date)")
    return con


# ----------------------------------------------------------------------------
# SQL layer: attribution and aggregation.
# ----------------------------------------------------------------------------
def monthly_ad_metrics(con):
    """Monthly ad spend, impressions, clicks and derived cpm, cpc, ctr."""
    q = """
        SELECT
            strftime('%Y-%m', date)                 AS month,
            SUM(spend)                              AS spend,
            SUM(impressions)                        AS impressions,
            SUM(clicks)                             AS clicks,
            SUM(spend) / SUM(impressions) * 1000.0  AS cpm,
            SUM(spend) / SUM(clicks)                AS cpc,
            SUM(clicks) * 100.0 / SUM(impressions)  AS ctr_pct
        FROM ad_spend
        GROUP BY month
        ORDER BY month
    """
    return pd.read_sql(q, con)


def monthly_acquisition(con):
    """New customers and first-order revenue per acquisition month."""
    q = """
        WITH firsts AS (
            SELECT customer_id, MIN(order_date) AS first_date
            FROM orders
            GROUP BY customer_id
        )
        SELECT
            strftime('%Y-%m', f.first_date) AS month,
            COUNT(DISTINCT f.customer_id)   AS new_customers,
            SUM(o.revenue)                  AS acq_revenue
        FROM firsts f
        JOIN orders o
            ON o.customer_id = f.customer_id
            AND o.order_date = f.first_date
        GROUP BY month
        ORDER BY month
    """
    return pd.read_sql(q, con)


def cohort_orders(con):
    """Every order tagged with its customer's cohort (acquisition) month.

    months_since is computed in pandas via PeriodIndex, not in SQL.
    """
    q = """
        WITH firsts AS (
            SELECT customer_id, MIN(order_date) AS cohort_date
            FROM orders
            GROUP BY customer_id
        )
        SELECT
            o.customer_id                        AS customer_id,
            strftime('%Y-%m', f.cohort_date)     AS cohort_month,
            strftime('%Y-%m', o.order_date)      AS order_month,
            o.revenue                            AS revenue
        FROM orders o
        JOIN firsts f ON o.customer_id = f.customer_id
    """
    df = pd.read_sql(q, con)
    cohort_p = pd.PeriodIndex(df["cohort_month"], freq="M")
    order_p = pd.PeriodIndex(df["order_month"], freq="M")
    df["months_since"] = [(o - c).n for o, c in zip(order_p, cohort_p)]
    return df


# ----------------------------------------------------------------------------
# Models: plain least squares and pooled averages.
# ----------------------------------------------------------------------------
def fit_acquisition_curve(ad, acq):
    """Fit customers = a * spend^b by ordinary least squares on log-log data.

    Returns a dict with a, b, r2 (on the log scale), mean_spend and the
    marginal CAC at mean spend (cost of the next customer vs the average).
    """
    merged = ad.merge(acq, on="month", how="inner")
    rows = merged[(merged["spend"] > 0) & (merged["new_customers"] > 0)]

    x = np.log(rows["spend"].to_numpy())
    y = np.log(rows["new_customers"].to_numpy())

    b, ln_a = np.polyfit(x, y, 1)
    a = float(np.exp(ln_a))

    # R squared on the log scale.
    y_hat = b * x + ln_a
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    mean_spend = float(rows["spend"].mean())
    # d(customers)/d(spend) = a * b * spend^(b-1); marginal CAC is its inverse.
    marginal_cac = 1.0 / (a * b * mean_spend ** (b - 1.0))

    return {
        "a": a,
        "b": float(b),
        "r2": r2,
        "mean_spend": mean_spend,
        "marginal_cac": float(marginal_cac),
    }


def fit_repeat_curve(cohorts, acq, ltv_cap_months):
    """Pooled revenue per customer by age (months since acquisition).

    For each age k the numerator is total revenue across all orders at that age,
    and the denominator is the summed size of cohorts old enough to have reached
    age k. This properly handles that recent cohorts have not aged yet.
    """
    cohort_sizes = (
        acq.set_index(pd.PeriodIndex(acq["month"], freq="M"))["new_customers"]
    )
    latest_month = pd.PeriodIndex(cohorts["order_month"], freq="M").max()

    rev_by_age = cohorts.groupby("months_since")["revenue"].sum()

    values = {}
    for k in range(0, ltv_cap_months + 1):
        numerator = float(rev_by_age.get(k, 0.0))
        eligible = cohort_sizes[[(latest_month - m).n >= k for m in cohort_sizes.index]]
        denominator = float(eligible.sum())
        values[k] = numerator / denominator if denominator > 0 else 0.0

    return pd.Series(values, name="revenue_per_customer")


# ----------------------------------------------------------------------------
# Projection: cohort stacking.
# ----------------------------------------------------------------------------
def project(ad, acq, curve_fit, repeat_curve, growth,
            projection_months, ltv_cap_months, gross_margin):
    """Project one scenario month by month using a cohort ledger.

    Seeds the ledger with historical cohorts (tagged with their current age) so
    their repeat tail keeps arriving in the early projected months.
    """
    a = curve_fit["a"]
    b = curve_fit["b"]
    r = repeat_curve  # Series indexed 0..ltv_cap_months
    cap = ltv_cap_months

    acq_p = acq.set_index(pd.PeriodIndex(acq["month"], freq="M"))
    last_month = acq_p.index.max()

    # Seed the ledger with historical cohorts still within the cap.
    ledger = []
    for month, row in acq_p.iterrows():
        age = (last_month - month).n
        if age <= cap:
            ledger.append([float(row["new_customers"]), age])

    last_spend = float(ad.set_index("month").loc[last_month.strftime("%Y-%m"), "spend"])

    records = []
    spend = last_spend
    cum_revenue = 0.0
    cum_gross_profit = 0.0

    for step in range(1, projection_months + 1):
        month = (last_month + step).strftime("%Y-%m")
        spend = spend * (1.0 + growth)

        new_customers = a * spend ** b

        # Age every existing cohort, dropping any past the cap.
        revenue = new_customers * r[0]
        aged = []
        for customers, age in ledger:
            age += 1
            if age <= cap:
                revenue += customers * r[age]
                aged.append([customers, age])
        aged.append([new_customers, 0])
        ledger = aged

        gross_profit = revenue * gross_margin - spend
        cum_revenue += revenue
        cum_gross_profit += gross_profit

        records.append(
            {
                "month": month,
                "spend": spend,
                "new_customers": new_customers,
                "revenue": revenue,
                "gross_profit": gross_profit,
                "cum_revenue": cum_revenue,
                "cum_gross_profit": cum_gross_profit,
                "roas": revenue / spend if spend > 0 else float("nan"),
            }
        )

    return pd.DataFrame(records)


# ----------------------------------------------------------------------------
# Unit economics.
# ----------------------------------------------------------------------------
def unit_economics(ad, acq, repeat_curve, gross_margin):
    """Blended CAC, AOV, LTV, break-even ROAS, LTV:CAC and payback period."""
    last3_spend = ad.sort_values("month").tail(3)["spend"].sum()
    last3_new = acq.sort_values("month").tail(3)["new_customers"].sum()
    blended_cac = last3_spend / last3_new

    aov = float(repeat_curve[0])
    ltv = float(repeat_curve.sum())
    breakeven_roas = 1.0 / gross_margin
    ltv_cac = ltv * gross_margin / blended_cac

    # Payback: first age where cumulative revenue times margin covers CAC.
    cumulative = repeat_curve.cumsum() * gross_margin
    covered = cumulative[cumulative >= blended_cac]
    payback = int(covered.index[0]) if len(covered) > 0 else None

    return {
        "blended_cac": blended_cac,
        "aov": aov,
        "ltv": ltv,
        "breakeven_roas": breakeven_roas,
        "ltv_cac": ltv_cac,
        "payback_months": payback,
    }
