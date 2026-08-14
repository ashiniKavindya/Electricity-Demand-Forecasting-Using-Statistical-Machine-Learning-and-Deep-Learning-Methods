"""PostgreSQL storage for AEMO observations, shared by collector and inference.

This module provides a PostgreSQL-compatible database interface for the AEMO
demand forecasting system. It's designed to work with the containerized setup.
"""
import os
from datetime import datetime
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    id SERIAL PRIMARY KEY,
    interval_datetime TIMESTAMP NOT NULL,
    region_id VARCHAR(10) NOT NULL,
    operational_demand REAL NOT NULL,
    source VARCHAR(50) NOT NULL,
    ingested_at TIMESTAMP NOT NULL,
    UNIQUE(interval_datetime, region_id)
);

CREATE INDEX IF NOT EXISTS idx_observations_interval_datetime 
    ON observations(interval_datetime DESC);

CREATE TABLE IF NOT EXISTS collector_health (
    id SERIAL PRIMARY KEY,
    polled_at TIMESTAMP NOT NULL,
    files_checked INTEGER NOT NULL,
    rows_inserted INTEGER NOT NULL,
    error TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    target_timestamp TIMESTAMP NOT NULL,
    based_on_timestamp TIMESTAMP NOT NULL,
    predicted_demand REAL NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    predicted_at TIMESTAMP NOT NULL,
    UNIQUE(target_timestamp, model_version)
);

CREATE INDEX IF NOT EXISTS idx_predictions_target_timestamp 
    ON predictions(target_timestamp DESC);

CREATE TABLE IF NOT EXISTS inference_health (
    id SERIAL PRIMARY KEY,
    run_at TIMESTAMP NOT NULL,
    predicted BOOLEAN NOT NULL,
    error TEXT
);
"""


def is_postgres_connection(conn) -> bool:
    return conn.__class__.__module__.startswith("psycopg2.")


def has_database_url() -> bool:
    return bool(os.getenv("DATABASE_URL"))


def get_connection():
    """Get a PostgreSQL connection from DATABASE_URL environment variable."""
    import psycopg2

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    conn = psycopg2.connect(database_url)
    
    # Initialize schema
    with conn.cursor() as cur:
        cur.execute(SCHEMA)
    conn.commit()
    
    return conn


def insert_observations(conn, records: list[tuple], source: str, ingested_at: str) -> int:
    """Bulk-insert (interval_datetime, region_id, operational_demand) tuples, ignoring duplicates."""
    from psycopg2.extras import execute_values

    if not records:
        return 0
    
    # Convert to ISO format strings for PostgreSQL
    ingested_at_ts = datetime.fromisoformat(ingested_at)
    
    with conn.cursor() as cur:
        # Use execute_values with ON CONFLICT for upsert
        query = """
            INSERT INTO observations 
            (interval_datetime, region_id, operational_demand, source, ingested_at)
            VALUES %s
            ON CONFLICT (interval_datetime, region_id) DO NOTHING
        """
        
        data = [
            (
                datetime.fromisoformat(interval_datetime),
                region_id,
                demand,
                source,
                ingested_at_ts
            )
            for interval_datetime, region_id, demand in records
        ]
        
        execute_values(cur, query, data)
        rows_inserted = cur.rowcount
        conn.commit()
    
    return rows_inserted


def latest_interval_datetime(conn) -> Optional[str]:
    """Get the latest interval_datetime from observations."""
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(interval_datetime) FROM observations")
        row = cur.fetchone()
    
    return row[0].isoformat() if row and row[0] else None


def latest_prediction_target(conn, model_version: str) -> Optional[str]:
    """Get the newest stored prediction target for one model version."""
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(target_timestamp) FROM predictions WHERE model_version = %s", (model_version,))
        row = cur.fetchone()

    return row[0].isoformat() if row and row[0] else None


def log_collector_health(
    conn, polled_at: str, files_checked: int, rows_inserted: int, error: Optional[str] = None
) -> None:
    """Log collector run status."""
    polled_at_ts = datetime.fromisoformat(polled_at)
    
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO collector_health (polled_at, files_checked, rows_inserted, error) VALUES (%s, %s, %s, %s)",
            (polled_at_ts, files_checked, rows_inserted, error)
        )
    conn.commit()


def insert_prediction(
    conn,
    target_timestamp: str,
    based_on_timestamp: str,
    predicted_demand: float,
    model_version: str,
    predicted_at: str,
) -> bool:
    """Store a forecast, ignoring it if one already exists for this
    (target_timestamp, model_version) pair. Returns whether a row was inserted.
    """
    target_ts = datetime.fromisoformat(target_timestamp)
    based_on_ts = datetime.fromisoformat(based_on_timestamp)
    predicted_at_ts = datetime.fromisoformat(predicted_at)
    
    with conn.cursor() as cur:
        # Try to insert; ON CONFLICT does nothing
        cur.execute(
            """
            INSERT INTO predictions 
            (target_timestamp, based_on_timestamp, predicted_demand, model_version, predicted_at)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (target_timestamp, model_version) DO NOTHING
            """,
            (target_ts, based_on_ts, predicted_demand, model_version, predicted_at_ts)
        )
        rows_inserted = cur.rowcount
        conn.commit()
    
    return rows_inserted > 0


def log_inference_health(conn, run_at: str, predicted: bool, error: Optional[str] = None) -> None:
    """Log inference run status."""
    run_at_ts = datetime.fromisoformat(run_at)
    
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO inference_health (run_at, predicted, error) VALUES (%s, %s, %s)",
            (run_at_ts, predicted, error)
        )
    conn.commit()


def query_demand_history(conn, hours: int = 168):
    """Query recent demand observations."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT interval_datetime, region_id, operational_demand
            FROM observations
            WHERE interval_datetime > NOW() - INTERVAL '%s hours'
            ORDER BY interval_datetime DESC
            """,
            (hours,)
        )
        return cur.fetchall()


def query_recent_predictions(conn, limit: int = 100):
    """Query recent predictions."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT target_timestamp, predicted_demand, model_version, predicted_at
            FROM predictions
            ORDER BY predicted_at DESC
            LIMIT %s
            """,
            (limit,)
        )
        return cur.fetchall()
