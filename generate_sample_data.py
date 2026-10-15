"""Generate realistic synthetic ecommerce data for the growth projector.

Writes two CSVs (ad_spend.csv and orders.csv) with diminishing returns baked
into acquisition and a decaying cohort repeat curve, so the projector has real
structure to recover. Uses a fixed seed for reproducibility.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta

# Reproducible generator (fixed seed).
SEED = 9

# Calendar: 365 days of daily data.
START = date(2025, 7, 1)
DAYS = 365
LAST_DAY = START + timedelta(days=DAYS - 1)

# Ad spend ramp (dollars per day, before weekend factor and noise).
SPEND_START = 250.0
SPEND_END = 750.0
WEEKEND_FACTOR = 0.80
SPEND_FLOOR = 50.0

# Media math.
CPM = 12.0           # cost per thousand impressions
CTR = 0.021          # click through rate (about 2.1 percent)

# Acquisition response: new customers = Poisson(A * spend^B).
# B < 1 produces diminishing returns. A is calibrated so blended CAC lands
# in a realistic 25 to 35 dollar band (lower A means fewer, pricier customers).
A_ACQ = 0.175
B_ACQ = 0.72

# Order value (lognormal, expressed via median so exp(mu) = median).
ACQ_MEDIAN = 62.0
ACQ_SIGMA = 0.50
REPEAT_MEDIAN = 59.0
REPEAT_SIGMA = 0.50

# Cohort repeat behavior: for months 1..8 after acquisition a customer places
# a repeat order with probability p, and p decays each month.
REPEAT_MONTHS = 8
REPEAT_P0 = 0.26
REPEAT_DECAY = 0.68

# Platform split (customer generation is driven by the daily total spend).
PLATFORMS = ["google", "meta"]


def lognormal_median(rng, median, sigma):
    """Draw one lognormal value with the given median (exp(mu) = median)."""
    return float(rng.lognormal(mean=np.log(median), sigma=sigma))


def generate(seed=SEED):
    """Build the ad_spend and orders DataFrames for a given seed."""
    rng = np.random.default_rng(seed)

    ad_rows = []
    order_rows = []

    customer_id = 0
    order_id = 0

    for i in range(DAYS):
        day = START + timedelta(days=i)

        # Linear ramp across the year.
        ramp = SPEND_START + (SPEND_END - SPEND_START) * (i / (DAYS - 1))

        # Weekends spend less.
        weekend = WEEKEND_FACTOR if day.weekday() >= 5 else 1.0

        # Multiplicative gaussian noise, floored.
        noise = rng.normal(1.0, 0.10)
        spend_total = max(SPEND_FLOOR, ramp * weekend * noise)

        # Split the daily spend across platforms and record per platform rows.
        split = rng.uniform(0.40, 0.60)
        shares = [split, 1.0 - split]
        for platform, share in zip(PLATFORMS, shares):
            p_spend = spend_total * share
            impressions = p_spend / CPM * 1000.0
            clicks = impressions * CTR * rng.normal(1.0, 0.08)
            ad_rows.append(
                {
                    "date": day.isoformat(),
                    "platform": platform,
                    "spend": round(p_spend, 2),
                    "impressions": int(round(impressions)),
                    "clicks": int(round(max(0.0, clicks))),
                }
            )

        # New customers acquired today from the total daily spend.
        n_new = rng.poisson(A_ACQ * spend_total ** B_ACQ)

        for _ in range(n_new):
            customer_id += 1

            # Acquisition order (placed on the acquisition day).
            order_id += 1
            order_rows.append(
                {
                    "order_id": order_id,
                    "customer_id": customer_id,
                    "order_date": day.isoformat(),
                    "revenue": round(lognormal_median(rng, ACQ_MEDIAN, ACQ_SIGMA), 2),
                }
            )

            # Repeat orders with decaying propensity, censored to the window.
            p = REPEAT_P0
            for m in range(1, REPEAT_MONTHS + 1):
                if rng.random() < p:
                    offset = int(rng.integers(-8, 9))
                    r_date = day + timedelta(days=m * 30 + offset)
                    if r_date <= LAST_DAY:
                        order_id += 1
                        order_rows.append(
                            {
                                "order_id": order_id,
                                "customer_id": customer_id,
                                "order_date": r_date.isoformat(),
                                "revenue": round(
                                    lognormal_median(rng, REPEAT_MEDIAN, REPEAT_SIGMA), 2
                                ),
                            }
                        )
                p *= REPEAT_DECAY

    ad_df = pd.DataFrame(ad_rows, columns=["date", "platform", "spend", "impressions", "clicks"])
    orders_df = pd.DataFrame(
        order_rows, columns=["order_id", "customer_id", "order_date", "revenue"]
    )
    return ad_df, orders_df


def main():
    ad_df, orders_df = generate(SEED)

    ad_df.to_csv("ad_spend.csv", index=False)
    orders_df.to_csv("orders.csv", index=False)

    # Quick calibration check so we can see the ranges without the full tool.
    total_spend = ad_df["spend"].sum()
    n_customers = orders_df["customer_id"].nunique()
    firsts = orders_df.groupby("customer_id")["order_date"].min()
    acq_orders = orders_df.merge(
        firsts.rename("first_date"), on="customer_id"
    )
    acq_orders = acq_orders[acq_orders["order_date"] == acq_orders["first_date"]]
    aov = acq_orders["revenue"].mean()
    blended_cac = total_spend / n_customers
    blended_roas = orders_df["revenue"].sum() / total_spend

    print("Generated ad_spend.csv rows:", len(ad_df))
    print("Generated orders.csv rows:  ", len(orders_df))
    print("Unique customers:           ", n_customers)
    print("Total ad spend:             ", round(total_spend, 2))
    print("Blended CAC (should be 25-35):", round(blended_cac, 2))
    print("AOV (should be 65-70):        ", round(aov, 2))
    print("Blended ROAS (should be ~2.5-3.5x):", round(blended_roas, 2))


if __name__ == "__main__":
    main()
