"""Read-only DB access for the API.

The API never runs its own model inference or writes to the database - it only reads
what collector/ and inference/ have already produced (collector -> database ->
everything else reads from the database). Keeps the API's uptime independent of theirs
and guarantees the dashboard can never show a forecast the standalone inference job
didn't itself produce and validate against the trained model artifacts.
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from inference.features import load_hourly_demand
from src.data import db_postgres

HEALTH_COLUMNS = {
    "collector_health": ["polled_at", "files_checked", "rows_inserted", "error"],
    "inference_health": ["run_at", "predicted", "error"],
}

PREDICTION_COLUMNS = ["target_timestamp", "based_on_timestamp", "predicted_demand", "model_version", "predicted_at"]


@contextmanager
def connect(db_path: str):
    if db_postgres.has_database_url():
        conn = db_postgres.get_connection()
        try:
            yield conn
        finally:
            conn.close()
        return

    if not Path(db_path).exists():
        raise FileNotFoundError(f"database not found at {db_path} - has the collector run yet?")
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def hourly_demand_history(conn, hours: int) -> pd.DataFrame:
    return load_hourly_demand(conn).tail(hours)


def latest_observation_timestamp(conn) -> str | None:
    if db_postgres.is_postgres_connection(conn):
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(interval_datetime) FROM observations")
            row = cur.fetchone()
        return row[0].isoformat() if row and row[0] else None

    row = conn.execute("SELECT MAX(interval_datetime) FROM observations").fetchone()
    return row[0] if row else None


def recent_predictions(conn, limit: int) -> pd.DataFrame:
    if db_postgres.is_postgres_connection(conn):
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {', '.join(PREDICTION_COLUMNS)} FROM predictions ORDER BY target_timestamp DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
        for col in ["target_timestamp", "based_on_timestamp", "predicted_at"]:
            if col in df:
                df[col] = df[col].map(lambda value: value.isoformat() if hasattr(value, "isoformat") else value)
        return df

    return pd.read_sql_query(
        f"SELECT {', '.join(PREDICTION_COLUMNS)} FROM predictions ORDER BY target_timestamp DESC LIMIT ?",
        conn,
        params=(limit,),
    )


def latest_prediction(conn) -> dict | None:
    if db_postgres.is_postgres_connection(conn):
        with conn.cursor() as cur:
            cur.execute(f"SELECT {', '.join(PREDICTION_COLUMNS)} FROM predictions ORDER BY target_timestamp DESC LIMIT 1")
            row = cur.fetchone()
        if not row:
            return None
        result = dict(zip(PREDICTION_COLUMNS, row))
        for col in ["target_timestamp", "based_on_timestamp", "predicted_at"]:
            if hasattr(result[col], "isoformat"):
                result[col] = result[col].isoformat()
        return result

    row = conn.execute(
        f"SELECT {', '.join(PREDICTION_COLUMNS)} FROM predictions ORDER BY target_timestamp DESC LIMIT 1"
    ).fetchone()
    return dict(zip(PREDICTION_COLUMNS, row)) if row else None


def recent_health(conn, table: str, limit: int) -> list[dict]:
    columns = HEALTH_COLUMNS[table]
    order_col = columns[0]
    if db_postgres.is_postgres_connection(conn):
        with conn.cursor() as cur:
            cur.execute(f"SELECT {', '.join(columns)} FROM {table} ORDER BY {order_col} DESC LIMIT %s", (limit,))
            rows = cur.fetchall()
        results = []
        for row in rows:
            item = dict(zip(columns, row))
            if hasattr(item[order_col], "isoformat"):
                item[order_col] = item[order_col].isoformat()
            results.append(item)
        return results

    rows = conn.execute(
        f"SELECT {', '.join(columns)} FROM {table} ORDER BY {order_col} DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(zip(columns, row)) for row in rows]


def demand_chart_series(hourly: pd.DataFrame, preds: pd.DataFrame) -> list[dict]:
    """Merge actual hourly demand with stored predictions into one timestamp-ordered
    series for charting: past predictions overlay their now-actual hour, and the
    newest not-yet-observed prediction trails off the end of the actual series as
    the live forecast point.
    """
    series: dict[str, dict] = {
        ts.isoformat(): {"timestamp": ts.isoformat(), "actual": float(row["demand"]), "predicted": None}
        for ts, row in hourly.iterrows()
    }
    for _, row in preds.iterrows():
        entry = series.setdefault(row["target_timestamp"], {"timestamp": row["target_timestamp"], "actual": None, "predicted": None})
        entry["predicted"] = float(row["predicted_demand"])
    return [series[key] for key in sorted(series)]


def rolling_accuracy(conn, limit: int) -> dict:
    """Compare stored predictions against the actual demand their target hour turned
    out to have - the only true measure of live forecast accuracy (as opposed to
    Stage 1's offline backtest on historical data the model never had to forecast
    into the future for).
    """
    hourly = load_hourly_demand(conn)
    preds = recent_predictions(conn, limit=100000)
    if hourly.empty or preds.empty:
        return {"n": 0, "mae": None, "rmse": None, "mape": None, "points": []}

    preds = preds.copy()
    preds["target_timestamp"] = pd.to_datetime(preds["target_timestamp"])
    actual = hourly[["demand"]].reset_index().rename(columns={"timestamp": "target_timestamp", "demand": "actual_demand"})
    matched = preds.merge(actual, on="target_timestamp", how="inner").sort_values("target_timestamp").tail(limit)

    if matched.empty:
        return {"n": 0, "mae": None, "rmse": None, "mape": None, "points": []}

    errors = matched["predicted_demand"] - matched["actual_demand"]
    mae = float(errors.abs().mean())
    rmse = float((errors**2).mean() ** 0.5)
    mape = float((errors.abs() / matched["actual_demand"]).mean() * 100)
    points = [
        {"timestamp": row.target_timestamp.isoformat(), "actual": row.actual_demand, "predicted": row.predicted_demand}
        for row in matched.itertuples()
    ]
    return {"n": len(matched), "mae": mae, "rmse": rmse, "mape": mape, "points": points}
