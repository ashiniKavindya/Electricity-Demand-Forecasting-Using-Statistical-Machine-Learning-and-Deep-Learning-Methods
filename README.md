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

## Status: Stage 1 complete (offline historical pipeline)

The build is staged. **Stage 1 — historical data → features → baseline +
XGBoost → evaluation — is done and verified.** Stages 2–5 (live collector,
database, streaming backend, live dashboard, monitoring) are designed but
not yet built; `backend/` and `dashboard/` still reflect the old project's
schema and are **not currently functional** against the new data until that
work lands.

| Stage | What | Status |
|---|---|---|
| 1 | Historical fetch → clean → features → seasonal-naive + XGBoost → evaluate | Done |
| 2 | Live collector polling AEMO every 5 min into a database | Not started |
| 3 | Live inference on each new observation | Not started |
| 4 | FastAPI + React dashboard serving live data | Not started (current `backend/`/`dashboard/` are stale) |
| 5 | Rolling-accuracy / collector-health monitoring | Not started |

## Data source

[AEMO](https://aemo.com.au) (Australian Energy Market Operator) publishes
NEM operational demand data publicly, no registration required:

- **Historical** (used by Stage 1): weekly archive zips from NEMWeb
  (`https://nemweb.com.au/Reports/Archive/Operational_Demand/ACTUAL_HH/`),
  each a zip of half-hourly per-interval CSVs in AEMO's classic `C:`/`I:`/`D:`
  MMS report format, one row per NEM region (`NSW1`, `QLD1`, `SA1`, `TAS1`,
  `VIC1`). Rolling ~12-month retention window.
- **Live** (for Stage 2+): `https://visualisations.aemo.com.au/aemo/apps/api/report/ELEC_NEM_SUMMARY`,
  updated every 5 minutes.

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
scripts/
  fetch_historical_aemo.py     # download weekly archives
  build_historical_dataset.py  # parse + aggregate + resample to hourly
  train_baseline.py            # naive / seasonal-naive baselines
  train_xgboost.py             # XGBoost forecaster
  evaluate_stage1.py           # comparison + Diebold-Mariano test
src/
  data/             # load_data.py, clean_data.py (generic), aemo.py (AEMO-specific parsing)
  features/         # lag/rolling/calendar/cyclical feature engineering (generic)
  models/           # baseline.py, machine_learning.py (in use); statistical.py, deep_learning.py (kept, deferred)
  evaluation/        # metrics + Diebold-Mariano significance test
tests/              # pytest coverage for AEMO parsing + leakage-safety of feature engineering
models/aemo/        # trained model artifacts (gitignored)
reports/            # metrics/predictions CSVs from Stage 1
backend/            # STALE - FastAPI service built for the old schema, pending Stage 3/4 rewrite
dashboard/          # STALE - React dashboard shell built for the old schema, pending Stage 4 rewrite
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
python -m scripts.evaluate_stage1              # comparison + significance test -> reports/model_comparison.csv
```

Run the test suite with:

```bash
pytest tests/
```

## Results (most recent run)

| Model | MAE (MW) | RMSE | MAPE |
|---|---|---|---|
| Naive (previous hour) | ~1216 | ~1464 | ~5.0% |
| Seasonal naive (168h) | ~984 | ~1277 | ~4.2% |
| XGBoost | ~394 | ~534 | ~1.7% |

XGBoost beats the seasonal-naive baseline with a Diebold-Mariano test
p-value effectively 0 — a genuine, explainable improvement (not the
near-zero-error red flag the old leaky dataset produced).
