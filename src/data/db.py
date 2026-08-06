"""SQLite storage for AEMO observations, shared by the collector and (later) inference.

WAL mode is enabled on every connection since the collector (writer, every
few minutes) and any later reader (inference job, API) touch this file
concurrently.
"""
import sqlite3
from pathlib import Path

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
"""


def get_connection(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def insert_observations(conn: sqlite3.Connection, records: list[tuple], source: str, ingested_at: str) -> int:
    """Bulk-insert (interval_datetime, region_id, operational_demand) tuples, ignoring duplicates."""
    before = conn.total_changes
    conn.executemany(
        "INSERT OR IGNORE INTO observations (interval_datetime, region_id, operational_demand, source, ingested_at) "
        "VALUES (?, ?, ?, ?, ?)",
        [(interval_datetime, region_id, demand, source, ingested_at) for interval_datetime, region_id, demand in records],
    )
    conn.commit()
    return conn.total_changes - before


def latest_interval_datetime(conn: sqlite3.Connection) -> str | None:
    row = conn.execute("SELECT MAX(interval_datetime) FROM observations").fetchone()
    return row[0] if row else None


def log_collector_health(
    conn: sqlite3.Connection, polled_at: str, files_checked: int, rows_inserted: int, error: str | None = None
) -> None:
    conn.execute(
        "INSERT INTO collector_health (polled_at, files_checked, rows_inserted, error) VALUES (?, ?, ?, ?)",
        (polled_at, files_checked, rows_inserted, error),
    )
    conn.commit()
