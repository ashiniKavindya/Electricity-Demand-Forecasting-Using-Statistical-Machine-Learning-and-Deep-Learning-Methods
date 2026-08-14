from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from inference.features import load_hourly_demand
from inference.predictor import build_forecast_row, predict_once
from src.data.db import get_connection, insert_observations

REGIONS = ["NSW1", "QLD1", "SA1", "TAS1", "VIC1"]


def _seed_half_hourly(conn, hours: int, start: str = "2026-08-01T00:00:00") -> None:
    """Insert `hours` worth of complete half-hourly observations across all 5 regions."""
    start_ts = pd.Timestamp(start)
    records = []
    for i in range(hours * 2):
        ts = (start_ts + pd.Timedelta(minutes=30 * i)).isoformat()
        records.extend((ts, region, 1000.0 + i) for region in REGIONS)
    insert_observations(conn, records, source="test", ingested_at=datetime.now().isoformat())


@pytest.fixture
def conn(tmp_path):
    return get_connection(str(tmp_path / "test.db"))


def _config(tmp_path, lags=(1, 2), rolling_windows=(3,)):
    return {
        "data": {"target_col": "demand"},
        "features": {"lags": list(lags), "rolling_windows": list(rolling_windows)},
        "models": {
            "xgboost_path": str(tmp_path / "model.pkl"),
            "xgboost_feature_columns": str(tmp_path / "cols.json"),
            "version": "v1",
        },
    }


class _StubModel:
    def predict(self, X):
        return np.array([9999.0])


# --- inference/features.py ---


def test_load_hourly_demand_drops_incomplete_trailing_hour(conn):
    _seed_half_hourly(conn, hours=3)
    lone_point_ts = pd.Timestamp("2026-08-01T03:00:00").isoformat()
    insert_observations(
        conn, [(lone_point_ts, r, 2000.0) for r in REGIONS], source="test", ingested_at=datetime.now().isoformat()
    )

    hourly = load_hourly_demand(conn)

    assert len(hourly) == 3
    assert hourly.index.max() == pd.Timestamp("2026-08-01T02:00:00")


def test_load_hourly_demand_empty_database(conn):
    assert load_hourly_demand(conn).empty


# --- inference/predictor.py: build_forecast_row ---


def test_build_forecast_row_uses_only_past_demand_for_the_target_hour():
    index = pd.date_range("2026-08-01", periods=5, freq="h")
    hourly = pd.DataFrame({"demand": [100.0, 200.0, 300.0, 400.0, 500.0]}, index=index)

    row, target_ts, based_on = build_forecast_row(hourly, "demand", lags=[1, 2], rolling_windows=[3])

    assert based_on == index[-1]
    assert target_ts == index[-1] + pd.Timedelta(hours=1)
    assert row["demand_lag_1"].iloc[0] == 500.0
    assert row["demand_lag_2"].iloc[0] == 400.0
    # Calendar features must reflect the target hour, not the last observed one.
    assert row["hour"].iloc[0] == target_ts.hour


def test_build_forecast_row_raises_on_empty_series():
    with pytest.raises(ValueError):
        build_forecast_row(pd.DataFrame(columns=["demand"]), "demand", lags=[1], rolling_windows=[3])


# --- inference/predictor.py: predict_once ---


def test_predict_once_stores_a_prediction(conn, tmp_path):
    _seed_half_hourly(conn, hours=6)
    config = _config(tmp_path)
    feature_cols = [
        "hour", "day_of_week", "month", "is_weekend",
        "demand_lag_1", "demand_lag_2", "demand_roll_mean_3", "demand_roll_std_3",
        "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos",
    ]

    with patch("inference.predictor.load_model", return_value=(_StubModel(), feature_cols)):
        result = predict_once(conn, config)

    assert result["error"] is None
    assert result["predicted_demand"] == 9999.0
    assert result["inserted"] is True

    row = conn.execute("SELECT predicted_demand, model_version FROM predictions").fetchone()
    assert row == (9999.0, "v1")


def test_predict_once_is_idempotent_for_the_same_target_hour(conn, tmp_path):
    _seed_half_hourly(conn, hours=6)
    config = _config(tmp_path)
    feature_cols = ["demand_lag_1", "demand_lag_2", "demand_roll_mean_3", "demand_roll_std_3"]

    with patch("inference.predictor.load_model", return_value=(_StubModel(), feature_cols)):
        first = predict_once(conn, config)
        second = predict_once(conn, config)

    assert first["inserted"] is True
    assert second["inserted"] is False
    assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 1


def test_predict_once_reports_error_when_history_too_short(conn, tmp_path):
    _seed_half_hourly(conn, hours=1)
    config = _config(tmp_path, lags=(1, 24), rolling_windows=(3,))

    with patch("inference.predictor.load_model", return_value=(_StubModel(), ["demand_lag_24"])):
        result = predict_once(conn, config)

    assert result["error"] is not None
    assert result["inserted"] is False
    assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 0


def test_predict_once_reports_error_on_empty_database(conn, tmp_path):
    config = _config(tmp_path)

    with patch("inference.predictor.load_model", return_value=(_StubModel(), ["demand_lag_1"])):
        result = predict_once(conn, config)

    assert result["error"] is not None
    assert "no observations" in result["error"]


def test_load_model_raises_a_clear_error_when_artifacts_are_missing(tmp_path):
    from inference.predictor import load_model

    with pytest.raises(FileNotFoundError):
        load_model(_config(tmp_path))
