# Ecom Growth Projector

An ecommerce growth projector with a Streamlit web dashboard. It forecasts
revenue and gross profit from ad-spend and order data. SQL (in an in-memory
SQLite database) owns attribution and aggregation. The modeling is plain pandas
and numpy: a least-squares fit of an acquisition response curve on a known
functional form, plus a pooled cohort repeat curve. There is no machine learning
anywhere, and every number is traceable back to the data and the knobs.

## What it does

- Attributes each customer to the acquisition month of their first order and
  aggregates monthly ad metrics with SQL.
- Fits a diminishing-returns acquisition curve, `customers = a * spend^b`, by
  ordinary least squares on the log-log monthly data (b less than 1 means each
  extra dollar buys fewer customers).
- Builds an LTV / repeat curve: pooled revenue per customer by age in months,
  using only cohorts old enough to have reached each age.
- Projects revenue and gross profit forward month by month with a cohort ledger,
  under three spend-growth scenarios.
- Reports unit economics: blended CAC, AOV, break-even ROAS, margin-adjusted
  LTV:CAC, and payback period.

## Project layout

- `engine.py` pure modeling functions (SQL aggregation, curve fits, projection,
  unit economics). Accepts DataFrames and config values, returns results.
- `generate_sample_data.py` reproducible synthetic data. `generate()` returns
  the two DataFrames, and running it as a script also writes the CSVs.
- `growth_projector.py` command line interface that prints the full report and
  writes `projection.csv`.
- `app.py` the Streamlit dashboard.

## How to run

Requires Python 3.10 or newer. Only streamlit, pandas, numpy, plotly and the
standard library (sqlite3) are used.

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt # macOS / Linux
```

Run the web dashboard:

```
.venv\Scripts\streamlit run app.py     # Windows
# streamlit run app.py                 # macOS / Linux
```

The dashboard opens with sample data already loaded, so it is never empty on
first load. Uncheck "Use sample data instead" to upload your own CSVs:

- Ad spend CSV needs at least `date` and `spend` (`impressions`, `clicks`,
  `platform` are optional; without impressions and clicks the cpm, cpc and ctr
  funnel metrics are skipped).
- Orders CSV needs at least `customer_id`, `order_date` and `revenue`
  (`order_id` is optional).

Run the command line report instead (reads `ad_spend.csv` and `orders.csv` from
the current directory, so run the generator first):

```
.venv\Scripts\python.exe generate_sample_data.py
.venv\Scripts\python.exe growth_projector.py
```

## What the sidebar knobs control

- **Gross margin** (10% to 90%, default 45%) is the product margin after COGS,
  before ad spend. It sets the break-even ROAS (`1 / gross_margin`), scales gross
  profit in the projection (`revenue * gross_margin - spend`), and drives the
  margin-adjusted LTV:CAC and payback figures. The KPI metrics recompute live as
  you move it.
- **Projection horizon** (3 to 24 months, default 12) is how many months to
  forecast.
- **LTV cap** (6 to 18 months, default 12) is where the repeat curve is capped,
  since older ages have thin data.
- **Scenario monthly spend growth** three number inputs (conservative, base,
  aggressive) set the monthly percent change in ad spend applied to the last
  historical month. Because acquisition has diminishing returns, scaling spend
  harder buys customers at a rising marginal cost, so revenue keeps climbing
  while gross profit flattens or collapses.
- **Scenario for chart and table** picks which scenario drives the projection
  chart and the downloadable table. All three always appear in the scenario
  comparison chart.

## Deploying for free

The dashboard can be hosted at no cost on Streamlit Community Cloud. Push this
repository to GitHub, sign in at share.streamlit.io, connect the repository, and
point it at `app.py`. Streamlit installs `requirements.txt` and serves the app.
