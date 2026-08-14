import sqlite3

import pytest

from src.data.db import (
    get_connection,
    insert_observations,
    insert_prediction,
    latest_interval_datetime,
    latest_prediction_target,
    log_collector_health,
    log_inference_health,
)


@pytest.fixture
def conn(tmp_path):
    return get_connection(str(tmp_path / "test.db"))


def test_get_connection_enables_wal(conn):
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_insert_observations_counts_new_rows(conn):
    records = [("2026-08-06T00:00:00", "NSW1", 8000.0), ("2026-08-06T00:00:00", "QLD1", 6000.0)]
    inserted = insert_observations(conn, records, source="live_poll", ingested_at="2026-08-06T00:05:00")
    assert inserted == 2


def test_insert_observations_ignores_duplicates(conn):
    records = [("2026-08-06T00:00:00", "NSW1", 8000.0)]
    insert_observations(conn, records, source="live_poll", ingested_at="2026-08-06T00:05:00")
    second_insert = insert_observations(conn, records, source="live_poll", ingested_at="2026-08-06T00:10:00")
    assert second_insert == 0

    count = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    assert count == 1


def test_latest_interval_datetime_returns_max(conn):
    records = [
        ("2026-08-06T00:00:00", "NSW1", 8000.0),
        ("2026-08-06T00:30:00", "NSW1", 8100.0),
        ("2026-08-06T00:15:00", "NSW1", 8050.0),
    ]
    insert_observations(conn, records, source="live_poll", ingested_at="2026-08-06T00:35:00")
    assert latest_interval_datetime(conn) == "2026-08-06T00:30:00"


def test_latest_interval_datetime_none_when_empty(conn):
    assert latest_interval_datetime(conn) is None


def test_log_collector_health_records_a_row(conn):
    log_collector_health(conn, polled_at="2026-08-06T00:05:00", files_checked=3, rows_inserted=15)
    row = conn.execute("SELECT files_checked, rows_inserted, error FROM collector_health").fetchone()
    assert row == (3, 15, None)


def test_insert_prediction_returns_true_when_new(conn):
    inserted = insert_prediction(
        conn,
        target_timestamp="2026-08-06T01:00:00",
        based_on_timestamp="2026-08-06T00:00:00",
        predicted_demand=8500.0,
        model_version="v1",
        predicted_at="2026-08-06T00:05:00",
    )
    assert inserted is True


def test_insert_prediction_ignores_duplicate_target_and_version(conn):
    kwargs = dict(
        target_timestamp="2026-08-06T01:00:00",
        based_on_timestamp="2026-08-06T00:00:00",
        predicted_demand=8500.0,
        model_version="v1",
        predicted_at="2026-08-06T00:05:00",
    )
    insert_prediction(conn, **kwargs)
    second = insert_prediction(conn, **{**kwargs, "predicted_demand": 8600.0, "predicted_at": "2026-08-06T00:10:00"})
    assert second is False

    count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    assert count == 1


def test_latest_prediction_target_scoped_to_model_version(conn):
    insert_prediction(conn, "2026-08-06T01:00:00", "2026-08-06T00:00:00", 8500.0, "v1", "2026-08-06T00:05:00")
    insert_prediction(conn, "2026-08-06T02:00:00", "2026-08-06T01:00:00", 8600.0, "v1", "2026-08-06T01:05:00")
    insert_prediction(conn, "2026-08-06T03:00:00", "2026-08-06T02:00:00", 8700.0, "v2", "2026-08-06T02:05:00")

    assert latest_prediction_target(conn, "v1") == "2026-08-06T02:00:00"
    assert latest_prediction_target(conn, "v2") == "2026-08-06T03:00:00"


def test_latest_prediction_target_none_when_empty(conn):
    assert latest_prediction_target(conn, "v1") is None


def test_log_inference_health_records_a_row(conn):
    log_inference_health(conn, run_at="2026-08-06T00:05:00", predicted=True)
    row = conn.execute("SELECT predicted, error FROM inference_health").fetchone()
    assert row == (1, None)
