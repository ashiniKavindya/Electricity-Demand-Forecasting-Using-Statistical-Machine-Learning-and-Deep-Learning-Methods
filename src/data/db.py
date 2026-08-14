"""Database storage for AEMO observations, collector health, and predictions.

When DATABASE_URL is set, this module delegates to the PostgreSQL adapter used by
the Docker/AWS deployment. Otherwise it uses the local SQLite database. Keeping
the public functions here stable lets collector and inference run unchanged in
both modes.
"""
import os
import sqlite3
from pathlib import Path

from src.data import db_postgres

SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interval_datetime TEXT NOT NULL,
    region_id TEXT NOT NULL,
    operational_demand REAL NOT NULL,
    source TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    UNIQUE(interval_datetime, region_id)
);

CREATE TABLE IF NOT EXISTS collector_health (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    polled_at TEXT NOT NULL,
    files_checked INTEGER NOT NULL,
    rows_inserted INTEGER NOT NULL,
    error TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_timestamp TEXT NOT NULL,
    based_on_timestamp TEXT NOT NULL,
    predicted_demand REAL NOT NULL,
    model_version TEXT NOT NULL,
    predicted_at TEXT NOT NULL,
    UNIQUE(target_timestamp, model_version)
);

CREATE TABLE IF NOT EXISTS inference_health (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT NOT NULL,
    predicted BOOLEAN NOT NULL,
    error TEXT
);
"""


def get_connection(db_path: str) -> sqlite3.Connection:
    if os.getenv("DATABASE_URL"):
        return db_postgres.get_connection()

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def insert_observations(conn: sqlite3.Connection, records: list[tuple], source: str, ingested_at: str) -> int:
    """Bulk-insert (interval_datetime, region_id, operational_demand) tuples, ignoring duplicates."""
    if db_postgres.is_postgres_connection(conn):
        return db_postgres.insert_observations(conn, records, source, ingested_at)

    before = conn.total_changes
    conn.executemany(
        "INSERT OR IGNORE INTO observations (interval_datetime, region_id, operational_demand, source, ingested_at) "
        "VALUES (?, ?, ?, ?, ?)",
        [(interval_datetime, region_id, demand, source, ingested_at) for interval_datetime, region_id, demand in records],
    )
    conn.commit()
    return conn.total_changes - before


def latest_interval_datetime(conn: sqlite3.Connection) -> str | None:
    if db_postgres.is_postgres_connection(conn):
        return db_postgres.latest_interval_datetime(conn)

    row = conn.execute("SELECT MAX(interval_datetime) FROM observations").fetchone()
    return row[0] if row else None


def log_collector_health(
    conn: sqlite3.Connection, polled_at: str, files_checked: int, rows_inserted: int, error: str | None = None
) -> None:
    if db_postgres.is_postgres_connection(conn):
        db_postgres.log_collector_health(conn, polled_at, files_checked, rows_inserted, error)
        return

    conn.execute(
        "INSERT INTO collector_health (polled_at, files_checked, rows_inserted, error) VALUES (?, ?, ?, ?)",
        (polled_at, files_checked, rows_inserted, error),
    )
    conn.commit()


def insert_prediction(
    conn: sqlite3.Connection,
    target_timestamp: str,
    based_on_timestamp: str,
    predicted_demand: float,
    model_version: str,
    predicted_at: str,
) -> bool:
    """Store a forecast for `target_timestamp`, ignoring it if one already exists for
    this (target_timestamp, model_version) pair. Returns whether a row was inserted.
    """
    if db_postgres.is_postgres_connection(conn):
        return db_postgres.insert_prediction(
            conn, target_timestamp, based_on_timestamp, predicted_demand, model_version, predicted_at
        )

    before = conn.total_changes
    conn.execute(
        "INSERT OR IGNORE INTO predictions "
        "(target_timestamp, based_on_timestamp, predicted_demand, model_version, predicted_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (target_timestamp, based_on_timestamp, predicted_demand, model_version, predicted_at),
    )
    conn.commit()
    return conn.total_changes > before


def latest_prediction_target(conn: sqlite3.Connection, model_version: str) -> str | None:
    if db_postgres.is_postgres_connection(conn):
        return db_postgres.latest_prediction_target(conn, model_version)

    row = conn.execute(
        "SELECT MAX(target_timestamp) FROM predictions WHERE model_version = ?", (model_version,)
    ).fetchone()
    return row[0] if row else None


def log_inference_health(conn: sqlite3.Connection, run_at: str, predicted: bool, error: str | None = None) -> None:
    if db_postgres.is_postgres_connection(conn):
        db_postgres.log_inference_health(conn, run_at, predicted, error)
        return

    conn.execute(
        "INSERT INTO inference_health (run_at, predicted, error) VALUES (?, ?, ?)",
        (run_at, predicted, error),
    )
    conn.commit()
