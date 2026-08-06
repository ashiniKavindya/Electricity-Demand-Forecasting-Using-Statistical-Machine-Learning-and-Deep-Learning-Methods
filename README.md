# Electricity Demand Forecasting Using Statistical, Machine Learning, and Deep Learning Methods

Short-term electricity demand forecasting system comparing baseline, statistical (ARIMA/SARIMA), machine learning (Random Forest/XGBoost), and deep learning (LSTM) approaches. See [proposal.txt](proposal.txt) for the full project proposal.

## Dataset

[Load Forecasting Dataset](https://www.kaggle.com/datasets/isuranga/load-forecasting-dataset) (Sri Lanka, Jan 2020 – May 2025, 15-minute resolution resampled to hourly), Kaggle. Place the downloaded CSV in `data/raw/` as `load_forecasting_dataset_corrected.csv`.

Despite being marketed as Ceylon Electricity Board (CEB) sourced, inspection of the data (see `notebooks/01_data_understanding.ipynb` and proposal.txt Section 7.1) indicates it is synthetically generated rather than authentic CEB telemetry. It's used here as a realistic Sri Lanka-style stand-in for demonstrating the forecasting methodology, not as validated real-world demand figures — see proposal.txt Section 28 for how this affects interpretation of results.

## Project Structure

```
data/
  raw/          # untouched source data
  processed/    # cleaned, resampled, feature-engineered data
  external/     # any supplementary data (holidays, events)
notebooks/      # analysis pipeline, one notebook per project stage
src/
  data/         # loading and cleaning
  features/     # lag/rolling/calendar/cyclical feature engineering
  models/       # baseline, statistical, ML, and DL model code
  evaluation/   # metrics and significance testing
  visualization/# plotting utilities
models/         # saved trained model artifacts
reports/
  figures/      # exported charts
  final_report/ # written report and slides
backend/        # optional local-only FastAPI service for live predictions
dashboard/      # optional React + Vite + TypeScript app
```

## Setup

### Python (data, notebooks, models)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Dashboard (optional)

```bash
cd dashboard
npm install
npm run dev
```

By default the dashboard shows precomputed results exported from Kaggle
(see `src/evaluation/export_dashboard_data.py` and `dashboard/public/data/`).

### Local live-prediction backend (optional)

For an actual live forecast in the dashboard (not just precomputed charts),
run the FastAPI backend locally alongside the dashboard — see
[backend/README.md](backend/README.md) for what model artifacts it expects:

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

This is local-only by design (no hosting cost) — the dashboard's "Live
Forecast" panel calls `http://localhost:8000` and simply shows an error if
the backend isn't running.

## Workflow

Notebooks are numbered in the order they should be run, following the pipeline described in the proposal (Sections 9–21): data understanding → cleaning → EDA → decomposition/stationarity → feature engineering → baseline → statistical models → ML models → deep learning → comparison → peak-demand analysis.
