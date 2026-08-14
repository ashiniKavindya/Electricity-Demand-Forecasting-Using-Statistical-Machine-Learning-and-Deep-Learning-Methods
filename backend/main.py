"""Local-only FastAPI service serving live AEMO NEM demand data and forecasts.

Run with: uvicorn backend.main:app --reload --port 8000
(from the repository root, so `src`/`inference` are importable)

Read-only against data/live/aemo.db: collector/ and inference/ are the sole writers,
running as their own standalone processes (collector -> database -> everything else
reads from the database). This API adds no model logic of its own - it only shapes
what's already in the database for the dashboard, plus Stage 1's offline backtest
metrics from reports/model_comparison.csv.
"""
import sys
import os
from pathlib import Path

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend import db  # noqa: E402

with open(REPO_ROOT / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

DB_PATH = str(REPO_ROOT / CONFIG["data"]["db_path"])
MODEL_VERSION = CONFIG["models"]["version"]

app = FastAPI(title="AEMO NEM Live Demand API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:4173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _connect():
    try:
        return db.connect(DB_PATH)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health():
    model_path = REPO_ROOT / CONFIG["models"]["xgboost_path"]
    if not os.getenv("DATABASE_URL") and not Path(DB_PATH).exists():
        return {"db_found": False, "model_trained": model_path.exists()}

    with db.connect(DB_PATH) as conn:
        latest_obs = db.latest_observation_timestamp(conn)
        latest_pred = db.latest_prediction(conn)
        last_collector_run = db.recent_health(conn, "collector_health", limit=1)
        last_inference_run = db.recent_health(conn, "inference_health", limit=1)

    return {
        "db_found": True,
        "model_trained": model_path.exists(),
        "model_version": MODEL_VERSION,
        "latest_observation": latest_obs,
        "latest_prediction": latest_pred,
        "last_collector_run": last_collector_run[0] if last_collector_run else None,
        "last_inference_run": last_inference_run[0] if last_inference_run else None,
    }


@app.get("/demand/history")
def demand_history(hours: int = 168):
    if not 1 <= hours <= 24 * 90:
        raise HTTPException(status_code=400, detail="hours must be between 1 and 2160 (90 days).")
    with _connect() as conn:
        hourly = db.hourly_demand_history(conn, hours)
    return [{"timestamp": ts.isoformat(), "demand": float(row["demand"])} for ts, row in hourly.iterrows()]


@app.get("/demand/predictions")
def demand_predictions(limit: int = 100):
    if not 1 <= limit <= 1000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 1000.")
    with _connect() as conn:
        return db.recent_predictions(conn, limit).to_dict(orient="records")


@app.get("/demand/chart")
def demand_chart(hours: int = 168):
    """Actual + predicted demand merged into one series, shaped for the dashboard's
    time-series chart: predictions overlay their now-actual hour where one exists, and
    the newest still-future prediction trails off the end as the live forecast point.
    """
    if not 1 <= hours <= 24 * 90:
        raise HTTPException(status_code=400, detail="hours must be between 1 and 2160 (90 days).")
    with _connect() as conn:
        hourly = db.hourly_demand_history(conn, hours)
        preds = db.recent_predictions(conn, limit=hours + 5)
    return db.demand_chart_series(hourly, preds)


@app.get("/forecast/latest")
def forecast_latest():
    with _connect() as conn:
        latest = db.latest_prediction(conn)
    if latest is None:
        raise HTTPException(status_code=503, detail="No predictions stored yet - has inference/run.py run at least once?")
    return latest


@app.get("/monitoring/collector")
def monitoring_collector(limit: int = 20):
    if not 1 <= limit <= 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500.")
    with _connect() as conn:
        return db.recent_health(conn, "collector_health", limit)


@app.get("/monitoring/inference")
def monitoring_inference(limit: int = 20):
    if not 1 <= limit <= 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500.")
    with _connect() as conn:
        return db.recent_health(conn, "inference_health", limit)


@app.get("/monitoring/accuracy")
def monitoring_accuracy(limit: int = 200):
    if not 1 <= limit <= 2000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 2000.")
    with _connect() as conn:
        return db.rolling_accuracy(conn, limit)


@app.get("/metrics/offline")
def metrics_offline():
    path = REPO_ROOT / "reports" / "model_comparison.csv"
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"{path} not found - run scripts/evaluate_stage1.py first.")
    return pd.read_csv(path).to_dict(orient="records")
