"""Predict next-hour NEM demand from whatever's currently in the database and store the
forecast. Mirrors collector/poller.py's poll_once(): safe to call repeatedly, always
targets whatever hour comes next after the latest complete observation.
"""
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from inference.features import load_hourly_demand
from src.data.db import insert_prediction, log_inference_health
from src.features.build_features import build_all_features


def load_model(config: dict) -> tuple:
    model_path = Path(config["models"]["xgboost_path"])
    feature_cols_path = Path(config["models"]["xgboost_feature_columns"])
    if not model_path.exists() or not feature_cols_path.exists():
        raise FileNotFoundError(
            f"model artifacts not found ({model_path}, {feature_cols_path}) - run scripts/train_xgboost.py first"
        )
    model = joblib.load(model_path)
    feature_cols = json.loads(feature_cols_path.read_text())
    return model, feature_cols


def build_forecast_row(
    hourly: pd.DataFrame, target_col: str, lags: list[int], rolling_windows: list[int]
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Extend the hourly series one hour past its last complete observation and engineer
    features for that new hour. Calendar features of the target hour are legitimately
    known in advance; lag/rolling features fall out of shifting the real, observed
    series - so this is exactly the row the trained model expects for a genuine
    one-step-ahead forecast (not a same-hour nowcast using the target's own demand).
    """
    if hourly.empty:
        raise ValueError("no observations in the database yet")

    based_on = hourly.index.max()
    target_ts = based_on + pd.Timedelta(hours=1)
    extended = hourly.reindex(hourly.index.union([target_ts]))

    engineered = build_all_features(extended, target_col, lags, rolling_windows)
    row = engineered.loc[[target_ts]]
    return row, target_ts, based_on


def predict_once(conn, config: dict) -> dict:
    """Forecast the next hour and store it. Returns a result dict; on failure (no data
    yet, not enough trailing history, missing model artifacts) `error` is set and
    nothing is written to `predictions`, matching collector poll_once()'s error shape.
    """
    run_at = datetime.now().isoformat()
    try:
        target_col = config["data"]["target_col"]
        lags = config["features"]["lags"]
        rolling_windows = config["features"]["rolling_windows"]

        hourly = load_hourly_demand(conn)
        row, target_ts, based_on = build_forecast_row(hourly, target_col, lags, rolling_windows)

        model, feature_cols = load_model(config)
        X = row[feature_cols]
        if X.isna().any().any():
            missing = X.columns[X.isna().any()].tolist()
            raise ValueError(
                f"not enough trailing history to forecast {target_ts.isoformat()} - "
                f"missing {missing} (need up to {max(lags)}h of history)"
            )

        predicted_demand = float(model.predict(X)[0])
        model_version = config["models"]["version"]

        inserted = insert_prediction(
            conn,
            target_timestamp=target_ts.isoformat(),
            based_on_timestamp=based_on.isoformat(),
            predicted_demand=predicted_demand,
            model_version=model_version,
            predicted_at=run_at,
        )
        log_inference_health(conn, run_at, predicted=True)
        return {
            "target_timestamp": target_ts.isoformat(),
            "based_on_timestamp": based_on.isoformat(),
            "predicted_demand": predicted_demand,
            "inserted": inserted,
            "error": None,
        }
    except Exception as exc:
        log_inference_health(conn, run_at, predicted=False, error=str(exc))
        return {
            "target_timestamp": None,
            "based_on_timestamp": None,
            "predicted_demand": None,
            "inserted": False,
            "error": str(exc),
        }
