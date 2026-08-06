"""Poll NEMWeb's Current directory for new operational-demand intervals and store them."""
import sqlite3
from datetime import datetime

from collector.nemweb_client import download_interval_zip, list_available_intervals
from src.data.aemo import parse_single_interval_zip
from src.data.db import insert_observations, latest_interval_datetime, log_collector_health


def _normalize(records: list[tuple[str, str, float]]) -> list[tuple[str, str, float]]:
    """Convert AEMO's "YYYY/MM/DD HH:MM:SS" timestamps to sortable ISO format."""
    return [
        (datetime.strptime(raw_ts, "%Y/%m/%d %H:%M:%S").isoformat(), region, demand)
        for raw_ts, region, demand in records
    ]


def poll_once(conn: sqlite3.Connection) -> dict:
    """Fetch and store every interval not yet in the database. Safe to call repeatedly -
    on an empty database this backfills from NEMWeb's ~60-day retention window; on a
    warm database it only fetches what's new since the last poll.
    """
    polled_at = datetime.now().isoformat()
    try:
        available = list_available_intervals()
        watermark = latest_interval_datetime(conn)
        new_intervals = [
            (filename, interval_start)
            for filename, interval_start in available
            if watermark is None or interval_start.isoformat() > watermark
        ]

        rows_inserted = 0
        for filename, _ in new_intervals:
            records = _normalize(parse_single_interval_zip(download_interval_zip(filename)))
            rows_inserted += insert_observations(conn, records, source="live_poll", ingested_at=polled_at)

        log_collector_health(conn, polled_at, files_checked=len(new_intervals), rows_inserted=rows_inserted)
        return {"files_checked": len(new_intervals), "rows_inserted": rows_inserted, "error": None}
    except Exception as exc:
        log_collector_health(conn, polled_at, files_checked=0, rows_inserted=0, error=str(exc))
        return {"files_checked": 0, "rows_inserted": 0, "error": str(exc)}
