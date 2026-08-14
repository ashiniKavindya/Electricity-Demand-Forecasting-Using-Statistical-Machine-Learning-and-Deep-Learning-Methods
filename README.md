# Real-Time Electricity Demand Forecasting

This project forecasts electricity demand across Australia's National Electricity
Market (NEM). It covers the full workflow: collecting operational data from AEMO,
building forecasting features, comparing multiple models, serving live predictions,
and monitoring accuracy through a web dashboard.

The data is real operational demand published by the Australian Energy Market
Operator (AEMO), not a synthetic or pre-cleaned teaching dataset.

## What this project does

- Downloads and prepares historical demand data from all five NEM regions.
- Compares seasonal-naive, XGBoost, and LSTM forecasts using chronological testing.
- Polls AEMO for new half-hourly observations and stores them in SQLite.
- Generates a new one-hour-ahead forecast whenever enough data is available.
- Serves forecasts, actual demand, accuracy, and system health through FastAPI.
- Displays the live system in a React dashboard.

All five planned stages are complete.

| Stage | Deliverable | Status |
|---|---|---|
| 1 | Historical pipeline and model evaluation | Complete |
| 2 | Live AEMO data collector | Complete |
| 3 | Live one-hour-ahead inference | Complete |
| 4 | FastAPI backend and React dashboard | Complete |
| 5 | Rolling accuracy and service-health monitoring | Complete |

## Why AEMO data?

[AEMO](https://aemo.com.au) publishes operational demand for the NEM without
requiring registration. Both the training pipeline and the live collector use the
same `OPERATIONAL_DEMAND` feed, which avoids a change in measurement at the point
where historical data meets live data.

- **Historical data:** weekly archives from AEMO's `ACTUAL_HH` operational-demand
  feed.
- **Live data:** recent interval files from the same feed's `Current` directory.
- **Forecast target:** total NEM demand, calculated by summing `NSW1`, `QLD1`,
  `SA1`, `TAS1`, and `VIC1`.

AEMO's market timestamps use a fixed UTC+10 offset and do not follow daylight
saving. The pipeline preserves that convention throughout.

The regional demand columns are retained for analysis but excluded from model
features because they add up to the target. Including them would leak the answer
into the model.

## Results

Results from the latest historical backtest:

| Model | MAE (MW) | RMSE (MW) | MAPE |
|---|---:|---:|---:|
| Naive (previous hour) | ~1,216 | ~1,464 | ~5.0% |
| Seasonal naive (previous week) | ~984 | ~1,277 | ~4.2% |
| XGBoost | ~394 | ~534 | ~1.7% |
| LSTM | ~324 | ~440 | ~1.4% |

XGBoost and LSTM both improved significantly on the seasonal-naive baseline. The
Diebold-Mariano test produced p-values effectively equal to zero for both
comparisons. LSTM also outperformed XGBoost in the offline backtest.

The live system currently serves XGBoost. It is simpler and cheaper to retrain and
deploy, while still giving strong accuracy. Full metrics and significance-test
results are available in [`reports/`](reports/).

## Architecture

```text
AEMO NEMWeb
    |
    v
Live collector ---> SQLite database <--- Inference service
                          |
                          v
                   FastAPI backend
                          |
                          v
                    React dashboard
```

The collector, inference service, and API run independently. A temporary failure in
one process does not bring down the others, and both collection and inference are
idempotent, so they can safely catch up after downtime.

## Project structure

```text
backend/       FastAPI endpoints for forecasts, demand, accuracy, and health
collector/     Live AEMO polling, archive seeding, and scheduling
dashboard/     React dashboard
inference/     Feature preparation and live XGBoost inference
notebooks/     Narrative walkthroughs of the historical and live workflows
scripts/       Historical data, training, and evaluation entry points
src/           Shared data, feature, model, evaluation, and plotting code
tests/         Tests for parsing, features, storage, inference, and the API
data/          Raw, processed, and live data (generated data is ignored by Git)
models/        Trained model artifacts (ignored by Git)
reports/       Committed evaluation metrics and selected figures
```

## Getting started

Create a virtual environment and install the Python dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

For the shortest path to running the project, see
[`QUICKSTART.md`](QUICKSTART.md).

## Historical training pipeline

Run these commands from the repository root:

```bash
python -m scripts.fetch_historical_aemo
python -m scripts.build_historical_dataset
python -m scripts.train_baseline
python -m scripts.train_xgboost
python -m scripts.train_lstm
python -m scripts.evaluate_stage1
```

The pipeline downloads roughly 50 weekly archives, builds an hourly dataset, trains
the forecasting models, and writes evaluation output to `reports/`. Generated data
and model artifacts are kept out of Git.

## Running the live system

Seed the database once from the downloaded historical archives, then start the
collector:

```bash
python -m collector.seed_from_archives
python -m collector.run
```

Start live inference in a second terminal:

```bash
python -m inference.run
```

Each inference run finds the latest complete hour, builds the same features used
during training, forecasts the following hour, and stores the result in
`data/live/aemo.db`. If there is not enough history or the model artifact is
missing, it records a clear health error instead of producing a guess.

Start the API and dashboard in separate terminals:

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

```bash
cd dashboard
npm install
npm run dev
```

The dashboard is then available at <http://localhost:5173>.

It shows:

- the latest one-hour-ahead forecast;
- recent actual demand with predictions overlaid;
- rolling live MAE, RMSE, and MAPE;
- collector and inference health; and
- the historical model comparison.

Each section loads independently, so an empty table or unavailable service produces
a useful message instead of breaking the entire page.

## Notebooks

The notebooks use the same production modules as the scripts and services. They are
intended as readable walkthroughs rather than separate implementations.

- `01_historical_data_pipeline.ipynb` explores the dataset, seasonality, data
  quality, leakage prevention, and feature engineering.
- `02_modeling_and_evaluation.ipynb` trains and compares the baseline, XGBoost, and
  LSTM models, including residual analysis and significance tests.
- `03_live_system_demo.ipynb` inspects the live database, reconstructs inference
  features, and demonstrates live accuracy reporting.

## Tests

Run the complete test suite from the repository root:

```bash
pytest tests/
```

## Design decisions worth noting

- Training and live inference share the same feature-engineering code to prevent
  training-serving drift.
- The collector inserts only unseen AEMO intervals and automatically catches up
  after downtime.
- Predictions are unique by target timestamp and model version, which prevents
  duplicate forecasts.
- The API is read-only and never loads a model; it only presents data already stored
  by the collector and inference services.
- Live accuracy is kept separate from historical backtest accuracy so the dashboard
  does not blur offline evaluation with real-world performance.

Deployment instructions are available in [`DEPLOYMENT.md`](DEPLOYMENT.md).
