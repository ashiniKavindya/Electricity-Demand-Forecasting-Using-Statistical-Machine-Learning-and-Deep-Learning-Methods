from datetime import datetime

import pandas as pd
import pytest

from backend.db import demand_chart_series, rolling_accuracy
from src.data.db import get_connection, insert_observations, insert_prediction

REGIONS = ["NSW1", "QLD1", "SA1", "TAS1", "VIC1"]


def _seed_hour(conn, hour_start: str, demand_per_region: float) -> None:
    """Insert a complete pair of half-hourly observations for one hourly bin."""
    ts0 = pd.Timestamp(hour_start)
    for offset in (0, 30):
        ts = (ts0 + pd.Timedelta(minutes=offset)).isoformat()
        insert_observations(
            conn, [(ts, r, demand_per_region) for r in REGIONS], source="test", ingested_at=datetime.now().isoformat()
        )


@pytest.fixture
def conn(tmp_path):
    return get_connection(str(tmp_path / "test.db"))


# --- demand_chart_series ---


def test_demand_chart_series_overlays_prediction_on_matching_actual_hour():
    hourly = pd.DataFrame(
        {"demand": [100.0, 200.0]}, index=pd.to_datetime(["2026-08-01T00:00:00", "2026-08-01T01:00:00"])
    ).rename_axis("timestamp")
    preds = pd.DataFrame({"target_timestamp": ["2026-08-01T01:00:00"], "predicted_demand": [210.0]})

    series = demand_chart_series(hourly, preds)

    assert series[0] == {"timestamp": "2026-08-01T00:00:00", "actual": 100.0, "predicted": None}
    assert series[1] == {"timestamp": "2026-08-01T01:00:00", "actual": 200.0, "predicted": 210.0}


def test_demand_chart_series_trails_a_future_only_prediction():
    hourly = pd.DataFrame({"demand": [100.0]}, index=pd.to_datetime(["2026-08-01T00:00:00"])).rename_axis("timestamp")
    preds = pd.DataFrame({"target_timestamp": ["2026-08-01T01:00:00"], "predicted_demand": [150.0]})

    series = demand_chart_series(hourly, preds)

    assert len(series) == 2
    assert series[1] == {"timestamp": "2026-08-01T01:00:00", "actual": None, "predicted": 150.0}


def test_demand_chart_series_empty_when_nothing_stored():
    assert demand_chart_series(pd.DataFrame(columns=["demand"]), pd.DataFrame(columns=["target_timestamp", "predicted_demand"])) == []


# --- rolling_accuracy ---


def test_rolling_accuracy_only_scores_predictions_with_a_known_outcome(conn):
    _seed_hour(conn, "2026-08-01T00:00:00", 1000.0)  # -> hourly demand 5000.0
    _seed_hour(conn, "2026-08-01T01:00:00", 1010.0)  # -> hourly demand 5050.0

    insert_prediction(conn, "2026-08-01T01:00:00", "2026-08-01T00:00:00", 5100.0, "v1", "2026-08-01T00:05:00")
    insert_prediction(conn, "2026-08-01T05:00:00", "2026-08-01T04:00:00", 9999.0, "v1", "2026-08-01T04:05:00")

    result = rolling_accuracy(conn, limit=200)

    assert result["n"] == 1
    assert result["mae"] == pytest.approx(50.0)
    assert result["points"][0]["actual"] == pytest.approx(5050.0)


def test_rolling_accuracy_empty_when_no_predictions_have_matured(conn):
    _seed_hour(conn, "2026-08-01T00:00:00", 1000.0)
    insert_prediction(conn, "2026-08-01T05:00:00", "2026-08-01T04:00:00", 9999.0, "v1", "2026-08-01T04:05:00")

    result = rolling_accuracy(conn, limit=200)

    assert result == {"n": 0, "mae": None, "rmse": None, "mape": None, "points": []}


def test_rolling_accuracy_empty_on_a_fresh_database(conn):
    result = rolling_accuracy(conn, limit=200)
    assert result == {"n": 0, "mae": None, "rmse": None, "mape": None, "points": []}
