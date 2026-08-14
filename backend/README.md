# Live Demand API

A FastAPI service serving live AEMO NEM demand data and forecasts to the dashboard.
**Local-only** — nothing here is deployed publicly; run it on your own machine alongside
the dashboard's dev server.

## Run

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

Run this from the repository root so `src`/`inference` are importable. Check
`http://localhost:8000/health` to see whether the database and trained model were found.

## What it needs from you

Nothing beyond what Stages 2 and 3 already produce:

1. **`data/live/aemo.db`** — written by `collector/` (`python -m collector.run`).
2. **`models/aemo/xgboost_v1.pkl`** — trained by `python -m scripts.train_xgboost`.
3. Predictions in the `predictions` table — written by `inference/`
   (`python -m inference.run`).

Missing pieces make the relevant endpoint return a clear 503 instead of guessing.

## Design

This API is deliberately **read-only and model-free**: `collector/` and `inference/` are
the only processes that write to `data/live/aemo.db`, and this service just shapes what
they've already written for the dashboard (collector → database → everything else reads
from the database). It never loads a model or runs inference itself, so it can't ever
serve a forecast the standalone `inference` job didn't already produce and validate.

## Endpoints

| Endpoint | What |
|---|---|
| `GET /health` | DB/model presence, latest observation & prediction, last collector/inference run |
| `GET /demand/history?hours=168` | Hourly actual NEM demand for the last N hours |
| `GET /demand/predictions?limit=100` | Recent stored predictions |
| `GET /demand/chart?hours=168` | Actual + predicted merged into one series, for the dashboard's time-series chart |
| `GET /forecast/latest` | The most recent (next-hour) forecast |
| `GET /monitoring/collector?limit=20` | Recent collector poll outcomes |
| `GET /monitoring/inference?limit=20` | Recent inference run outcomes |
| `GET /monitoring/accuracy?limit=200` | Rolling MAE/RMSE/MAPE of past predictions against the actual demand their target hour turned out to have |
| `GET /metrics/offline` | Stage 1's offline backtest comparison (`reports/model_comparison.csv`) |
