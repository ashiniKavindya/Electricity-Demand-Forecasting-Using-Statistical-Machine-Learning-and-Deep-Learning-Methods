"""Turn the collector's raw per-region observations into an hourly NEM-wide demand
series ready for feature engineering - the live-data mirror of
scripts/build_historical_dataset.py, reading from the database instead of archive zips.
"""
import pandas as pd

from src.data.aemo import aggregate_regions_to_nem_total, resample_to_hourly
from src.data.db_postgres import is_postgres_connection


def _load_observations(conn) -> pd.DataFrame:
    query = "SELECT interval_datetime, region_id, operational_demand FROM observations"
    if is_postgres_connection(conn):
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=["interval_datetime", "region_id", "operational_demand"])
    else:
        df = pd.read_sql_query(query, conn)
    df["interval_datetime"] = pd.to_datetime(df["interval_datetime"])
    return df


def _drop_incomplete_last_hour(wide_indexed: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    """Drop the most recent hourly bin if it isn't backed by a full pair of half-hourly
    readings yet. AEMO publishes on the half-hour, so a bin the collector caught
    mid-fill would average to a partial-hour figure - treating that as the settled
    value for the "based on" side of a forecast would quietly bias every prediction.
    """
    if hourly.empty:
        return hourly
    last_hour_start = hourly["timestamp"].iloc[-1]
    last_hour_end = last_hour_start + pd.Timedelta(hours=1)
    points_in_last_hour = wide_indexed.index[(wide_indexed.index >= last_hour_start) & (wide_indexed.index < last_hour_end)]
    if len(points_in_last_hour) < 2:
        return hourly.iloc[:-1]
    return hourly


def load_hourly_demand(conn) -> pd.DataFrame:
    """Return a timestamp-indexed hourly NEM demand series (plus per-region columns)
    built from every observation currently in the database, with any still-filling
    final hour dropped. Empty if the collector hasn't stored anything yet.
    """
    long_df = _load_observations(conn)
    if long_df.empty:
        return pd.DataFrame(columns=["demand"]).rename_axis("timestamp")

    wide = aggregate_regions_to_nem_total(long_df)
    hourly = resample_to_hourly(wide)
    hourly = _drop_incomplete_last_hour(wide.set_index("timestamp"), hourly)

    return hourly.set_index("timestamp").sort_index()
