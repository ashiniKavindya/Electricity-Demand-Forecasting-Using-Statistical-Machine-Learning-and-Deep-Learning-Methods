# Real-Time Electricity Demand Forecasting (AEMO NEM)

A real-time electricity-demand forecasting system built on AEMO's public NEM
(National Electricity Market, Australia) grid data — genuine operational
demand, not a synthetic dataset.

This replaces an earlier version of this project that used a Kaggle "Sri
Lanka" dataset which turned out to be synthetically generated with a
leakage bug (its `temperature` column was an exact affine transform of
demand, `temperature = 0.01 * demand + 13`, which made the ML models look
artificially perfect). That project is preserved at git tag
[`pre-aemo-pivot`](../../tree/pre-aemo-pivot); see
[`docs/archive/proposal_sri_lanka_v1.txt`](docs/archive/proposal_sri_lanka_v1.txt)
for how the leakage was diagnosed.

## Status: all 5 stages complete

| Stage | What | Status |
|---|---|---|
| 1 | Historical fetch → clean → features → seasonal-naive + XGBoost → evaluate | Done |
| 2 | Collector polling AEMO for new intervals into a SQLite database | Done |
| 3 | Live inference on each new observation | Done |
| 4 | FastAPI + React dashboard serving live data | Done |
| 5 | Rolling-accuracy / collector-health monitoring | Done (folded into Stage 4's API + dashboard - see `/monitoring/*` below) |

## Data source

[AEMO](https://aemo.com.au) (Australian Energy Market Operator) publishes
NEM operational demand data publicly, no registration required. Both
historical and live data come from the **same NEMWeb operational-demand
feed and metric** (`OPERATIONAL_DEMAND`), deliberately — AEMO's other
public live endpoint (`ELEC_NEM_SUMMARY`) reports `TOTALDEMAND`, a
different figure, and mixing the two would create a discontinuity right at
the historical/live boundary that would corrupt lag features spanning it:

- **Historical** (Stage 1): weekly archive zips from
  `https://nemweb.com.au/Reports/Archive/Operational_Demand/ACTUAL_HH/`,
  each a zip of half-hourly per-interval CSVs in AEMO's classic `C:`/`I:`/`D:`
  MMS report format, one row per NEM region (`NSW1`, `QLD1`, `SA1`, `TAS1`,
  `VIC1`). Rolling ~12-month retention window.
- **Live** (Stage 2): the same feed's `Current` directory
  (`https://nemweb.com.au/Reports/Current/Operational_Demand/ACTUAL_HH/`),
  a rolling ~60-day window of the same per-interval files, each published
  within minutes of its half-hour interval ending.

NEM market time is a **fixed UTC+10 offset with no daylight saving** — the
pipeline treats timestamps as naive datetimes in that fixed offset; never
localize them as `Australia/Sydney` (which observes DST).

The Stage 1 target is **NEM-wide total demand** (sum of all 5 regions).
Per-region demand is also kept in the processed dataset for future
per-region modeling, but per-region columns are deliberately excluded from
XGBoost's feature set — they sum to the target itself, so including them
would be the same shape of leakage bug as the old project's `temperature`
column.

## Project structure

```
data/
  raw/aemo/         # downloaded weekly AEMO archive zips (gitignored)
  processed/        # aemo_nem_demand_hourly.csv
  live/aemo.db      # SQLite database the collector writes to (gitignored)
scripts/
  fetch_historical_aemo.py     # download weekly archives
  build_historical_dataset.py  # parse + aggregate + resample to hourly
  train_baseline.py            # naive / seasonal-naive baselines
  train_xgboost.py             # XGBoost forecaster (in production - inference/ serves this one)
  train_lstm.py                # LSTM forecaster (Stage 1 only - not yet served live)
  evaluate_stage1.py           # comparison + Diebold-Mariano tests (baselines, XGBoost, LSTM)
notebooks/            # walkthrough notebooks - see "Notebooks" below
collector/
  nemweb_client.py    # list/download NEMWeb's "Current" interval files
  poller.py           # poll_once(): fetch + store whatever isn't in the DB yet
  seed_from_archives.py  # one-time: load Stage 1's downloaded archives into the DB
  scheduler.py        # APScheduler loop calling poll_once on an interval
  run.py              # entrypoint: python -m collector.run
inference/
  features.py         # load_hourly_demand(): DB observations -> hourly NEM demand series
  predictor.py         # predict_once(): build next-hour features + predict + store
  scheduler.py         # APScheduler loop calling predict_once on an interval
  run.py               # entrypoint: python -m inference.run
src/
  data/             # load_data.py, clean_data.py (generic), aemo.py (AEMO parsing), db.py (SQLite)
  features/         # lag/rolling/calendar/cyclical feature engineering (generic)
  models/           # baseline.py, machine_learning.py, deep_learning.py (in use); statistical.py (kept, deferred)
  evaluation/        # metrics + Diebold-Mariano significance test
tests/              # pytest coverage for AEMO parsing, the database layer, feature engineering, inference, and the API
models/aemo/        # trained model artifacts (gitignored)
reports/            # metrics/predictions CSVs from Stage 1
backend/            # FastAPI service: read-only views over aemo.db for the dashboard (Stage 4/5)
dashboard/          # React dashboard: live forecast, demand chart, accuracy, collector/inference health (Stage 4/5)
docs/archive/       # the superseded Sri Lanka project's proposal
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Stage 1 pipeline

```bash
python -m scripts.fetch_historical_aemo        # downloads ~50 weekly archive zips into data/raw/aemo/
python -m scripts.build_historical_dataset     # writes data/processed/aemo_nem_demand_hourly.csv
python -m scripts.train_baseline               # naive + seasonal-naive baselines -> reports/baseline_metrics.csv
python -m scripts.train_xgboost                # trains XGBoost -> models/aemo/, reports/xgboost_*.csv
python -m scripts.train_lstm                   # trains LSTM -> models/aemo/, reports/lstm_*.csv
python -m scripts.evaluate_stage1              # comparison + significance tests -> reports/model_comparison.csv
```

## Notebooks

`notebooks/` walks through the whole project narratively, for anyone who wants to see
the work rather than run the scripts - each one imports the same production code
(`src/`, `scripts/`, `inference/`, `backend/`) rather than re-implementing any of it,
so what's shown is guaranteed to match what actually runs, not a notebook-only
approximation of it:

- **`01_historical_data_pipeline.ipynb`** - the processed dataset, missing-data check,
  demand over time, per-region series, daily/weekly seasonality, the leakage-avoidance
  reasoning behind excluding per-region columns, and the feature-engineering step.
- **`02_modeling_and_evaluation.ipynb`** - retrains the naive/seasonal-naive baselines,
  XGBoost (same code path, same `random_state=42` as the committed model), and LSTM
  (same code path as `scripts/train_lstm.py`); actual-vs-predicted and residual plots,
  feature importance, training curves, and Diebold-Mariano significance tests for
  each model against the baseline and head-to-head against each other.
- **`03_live_system_demo.ipynb`** - reads `data/live/aemo.db` directly (read-only, so
  always safe to run) to show what the collector has stored, reconstructs hourly
  demand the way `inference/` does, inspects a live forecast's input features, and
  reports Stage 5's rolling live-accuracy - plus a screenshot of the dashboard
  rendering the same data.

Run the test suite with:

```bash
pytest tests/
```

## Running the live collector

```bash
python -m collector.seed_from_archives   # one-time: load Stage 1's downloaded archives into the database
python -m collector.run                  # runs an immediate poll, then polls every 5 min forever (Ctrl+C to stop)
```

The collector is a standalone process by design (not embedded in the
FastAPI backend), so its uptime doesn't depend on the API being up.
`poll_once()` is idempotent and self-catching-up: it always fetches
whatever's newer than the database's latest stored interval, so it's safe
to run cold (it'll backfill from NEMWeb's ~60-day retention window), after
downtime (it'll just fetch the gap), or on a fixed schedule (steady state
is usually 0-1 new files per poll).

Verified end-to-end: seeding loaded 85,680 rows from Stage 1's 51 archived
weeks, a first poll correctly backfilled the ~11-day gap since (575 files,
2,875 rows), and a second poll immediately after found exactly 1 new file
(5 rows) - the steady-state case.

## Running live inference

```bash
python -m inference.run   # runs an immediate forecast, then re-checks every 5 min forever (Ctrl+C to stop)
```

Also a standalone process, for the same reason as the collector: the
architecture is collector → database → everything else reads from the
database, so inference's uptime doesn't depend on the collector or a future
API being up. Each run:

1. Reads every observation currently in `data/live/aemo.db`, aggregates the
   5 regions into hourly NEM-wide demand (`inference/features.py` - the
   live-data mirror of `scripts/build_historical_dataset.py`), and drops the
   most recent hour if it's not yet backed by a full pair of half-hourly
   readings.
2. Extends that series one hour past the last complete observation and
   engineers features for it with the exact same function training uses
   (`build_all_features` in `src/features/build_features.py`, shared by both
   so they can't drift apart) - calendar features of the target hour are
   known in advance, lag/rolling features fall out of shifting the real
   series, so this is a genuine one-step-ahead forecast, not a same-hour
   nowcast.
3. Loads `models/aemo/xgboost_v1.pkl`, predicts, and stores the result in
   `data/live/aemo.db`'s `predictions` table
   (`target_timestamp`, `based_on_timestamp`, `predicted_demand`,
   `model_version`) - `INSERT OR IGNORE`-idempotent per
   `(target_timestamp, model_version)`, so re-running before the next hour's
   data lands is a safe no-op.

If there isn't yet enough trailing history for the longest lag (168h / one
week), or the model artifacts haven't been trained yet, the run logs a clear
error to `inference_health` and stores nothing - it never guesses.

Verified end-to-end against the live database: a first run correctly
forecast the next hour from the latest complete observation, and an
immediate second run found the same target hour already stored and made no
duplicate insert.

## Running the backend + dashboard

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000   # from the repo root

cd dashboard
npm install
npm run dev                                     # http://localhost:5173
```

The API (`backend/`) is deliberately **read-only and model-free**: it never loads a
model or runs inference itself, only shapes what `collector/` and `inference/` have
already written to `data/live/aemo.db` (see `backend/README.md` for the full endpoint
list). The dashboard (`dashboard/`) renders, per section, independently:

- **Live forecast** — the most recent stored prediction.
- **Actual vs predicted demand** — a chart of recent hourly demand with predictions
  overlaid on their now-actual hour, plus the newest still-future prediction trailing
  off the end.
- **Live forecast accuracy** (Stage 5) — rolling MAE/RMSE/MAPE of past predictions
  against the actual demand their target hour turned out to have, once the collector
  has caught up to it. This is the true measure of live accuracy, distinct from Stage
  1's offline backtest below.
- **Collector & inference health** (Stage 5) — each standalone process's most recent
  run and any error, so a silent failure in either is visible without reading logs.
- **Stage 1 offline evaluation** — the historical backtest comparison table, labeled
  separately so it's never confused with live accuracy.

Every section fetches independently and fails independently: if the backend isn't
running, or a particular table is still empty (e.g. no prediction has matured into an
observation yet), that one section shows a plain-language reason instead of the whole
page breaking.

## Results (most recent run)

| Model | MAE (MW) | RMSE | MAPE |
|---|---|---|---|
| Naive (previous hour) | ~1216 | ~1464 | ~5.0% |
| Seasonal naive (168h) | ~984 | ~1277 | ~4.2% |
| XGBoost | ~394 | ~534 | ~1.7% |
| LSTM | ~324 | ~440 | ~1.4% |

Both XGBoost and LSTM beat the seasonal-naive baseline with a Diebold-Mariano test
p-value effectively 0 each - a genuine, explainable improvement (not the
near-zero-error red flag the old leaky dataset produced). LSTM also beats XGBoost
head-to-head, also with p-value effectively 0 (see `reports/dm_test_results.csv`).
XGBoost remains the model `inference/` actually serves live (Stage 3) - it's cheaper
to retrain/redeploy and doesn't require carrying a fitted scaler alongside the model
artifact - so LSTM's edge here is a Stage 1 finding, not yet reflected in the live
system.
