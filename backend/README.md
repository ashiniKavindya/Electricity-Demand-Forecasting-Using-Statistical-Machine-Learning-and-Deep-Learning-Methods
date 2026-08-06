# Local Forecast API

A FastAPI service that loads your trained models and serves live multi-step
forecasts to the dashboard. **Local-only** — nothing here is deployed
publicly; run it on your own machine alongside the dashboard's dev server.

## Run

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

Run this from the repository root so the `src` package is importable.
Check `http://localhost:8000/health` to see which models were found.

## What it needs from you

The API doesn't train anything — it loads artifacts that your notebooks
must save in specific places, and reads cleaned historical data to build
the input features for forecasting. Nothing is required until you export
these; missing pieces just make `/predict` return a clear 503 instead of
a wrong answer.

### 1. Processed data — `data/processed/demand_clean.csv`

The cleaned, timestamp-sorted hourly series (post-cleaning, pre-feature-
engineering — the API rebuilds lag/rolling/calendar/cyclical features
itself using `src/features/build_features.py`). Must contain a timestamp
column plus the target column and any weather columns referenced by your
saved `feature_cols` lists, with names matching `config.yaml`.

### 2. XGBoost — `models/xgboost_model.pkl` + `models/xgboost_feature_columns.json`

```python
import joblib, json
joblib.dump(model, "models/xgboost_model.pkl")
json.dump(feature_cols, open("models/xgboost_feature_columns.json", "w"))
```

`feature_cols` is the exact list (and order) of predictor columns the
model was trained on — lag/rolling/calendar/cyclical/weather columns.
It must **not** include the raw target column itself.

### 3. SARIMA — `models/sarima_model.pkl`

```python
import joblib
results = model.fit()
joblib.dump(results, "models/sarima_model.pkl")
```

The series the model was fit on must have a `DatetimeIndex` with `freq`
set, so `results.get_forecast(steps=horizon)` produces correctly-spaced
future timestamps.

### 4. LSTM — `models/lstm_model.h5` + `models/lstm_scaler.pkl` + `models/lstm_feature_columns.json`

```python
import joblib, json
model.save("models/lstm_model.h5")
joblib.dump(scaler, "models/lstm_scaler.pkl")
json.dump(feature_cols, open("models/lstm_feature_columns.json", "w"))
```

Here `feature_cols` is the **full** column set the scaler was fit on,
including the target column (needed to invert the scaling on a single
predicted value).

## Known simplification

Future weather values aren't known at inference time, so multi-step
forecasts hold each weather column constant at its last observed value
for the whole horizon. This is a demo simplification — see proposal.txt
Section 28 (Limitations) — not a substitute for real weather forecasts.
