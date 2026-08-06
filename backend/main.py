"""Local-only FastAPI backend serving live predictions from trained models.

Run with: uvicorn backend.main:app --reload --port 8000
(from the repository root, so the `src` package is importable)

Intended for local demos only. Nothing here is deployed publicly; the
dashboard talks to http://localhost:8000 and falls back to precomputed
JSON exports (see src/evaluation/export_dashboard_data.py) when this
API isn't running.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend import forecasting  # noqa: E402

with open(REPO_ROOT / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

app = FastAPI(title="Electricity Demand Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

_models: dict = {}


def _load_if_exists(path: Path):
    return path if path.exists() else None


def _try_load_xgboost():
    model_path = REPO_ROOT / CONFIG["models"]["xgboost_path"]
    cols_path = REPO_ROOT / CONFIG["models"]["xgboost_feature_columns"]
    if not (model_path.exists() and cols_path.exists()):
        return None
    return {
        "model": joblib.load(model_path),
        "feature_cols": json.loads(cols_path.read_text()),
    }


def _try_load_sarima():
    model_path = REPO_ROOT / CONFIG["models"]["sarima_path"]
    if not model_path.exists():
        return None
    return {"results": joblib.load(model_path)}


def _try_load_lstm():
    model_path = REPO_ROOT / CONFIG["models"]["lstm_path"]
    scaler_path = REPO_ROOT / CONFIG["models"]["lstm_scaler_path"]
    cols_path = REPO_ROOT / CONFIG["models"]["lstm_feature_columns"]
    if not (model_path.exists() and scaler_path.exists() and cols_path.exists()):
        return None
    from tensorflow import keras

    return {
        "model": keras.models.load_model(model_path),
        "scaler": joblib.load(scaler_path),
        "feature_cols": json.loads(cols_path.read_text()),
    }


@app.on_event("startup")
def load_models():
    _models["xgboost"] = _try_load_xgboost()
    _models["sarima"] = _try_load_sarima()
    _models["lstm"] = _try_load_lstm()


def _load_history() -> pd.DataFrame:
    processed_path = REPO_ROOT / CONFIG["data"]["processed_file"]
    if not processed_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Processed data not found at {processed_path}. Run the cleaning/feature notebooks first.",
        )
    timestamp_col = CONFIG["data"]["timestamp_col"]
    df = pd.read_csv(processed_path, parse_dates=[timestamp_col])
    return df.set_index(timestamp_col).sort_index()


@app.get("/health")
def health():
    return {"models_loaded": {name: bool(m) for name, m in _models.items()}}


@app.get("/predict")
def predict(model: str, horizon: int = 24):
    if model not in _models:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'. Choose from xgboost, sarima, lstm.")
    loaded = _models[model]
    if loaded is None:
        raise HTTPException(
            status_code=503,
            detail=f"No trained '{model}' artifacts found in models/. Train and export it first (see backend/README.md).",
        )
    if horizon < 1 or horizon > 168:
        raise HTTPException(status_code=400, detail="horizon must be between 1 and 168 hours.")

    history = _load_history()
    target_col = CONFIG["data"]["target_col"]
    freq = CONFIG["data"]["frequency"]

    if model == "xgboost":
        predictions = forecasting.recursive_xgboost_forecast(
            model=loaded["model"],
            feature_cols=loaded["feature_cols"],
            history=history,
            target_col=target_col,
            lags=CONFIG["features"]["lags"],
            rolling_windows=CONFIG["features"]["rolling_windows"],
            horizon=horizon,
            freq=freq,
        )
    elif model == "lstm":
        predictions = forecasting.recursive_lstm_forecast(
            model=loaded["model"],
            scaler=loaded["scaler"],
            feature_cols=loaded["feature_cols"],
            history=history,
            target_col=target_col,
            sequence_length=CONFIG["lstm"]["sequence_length"],
            horizon=horizon,
            freq=freq,
        )
    else:
        predictions = forecasting.sarima_forecast(loaded["results"], horizon=horizon)

    return {"model": model, "horizon": horizon, "predictions": predictions}
