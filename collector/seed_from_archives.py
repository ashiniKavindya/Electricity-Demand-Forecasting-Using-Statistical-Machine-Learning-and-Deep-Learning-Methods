"""One-time seed: load Stage 1's already-downloaded weekly AEMO archives into the database.

Run once before starting the collector, so the database has continuous
history back to whatever scripts/fetch_historical_aemo.py already
downloaded, instead of relying solely on NEMWeb's ~60-day "Current"
retention window for everything.
"""
from datetime import datetime
from pathlib import Path

import yaml

from src.data.aemo import parse_weekly_archive
from src.data.db import get_connection, insert_observations

if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    conn = get_connection(config["data"]["db_path"])
    raw_dir = Path(config["data"]["raw_dir"])
    zip_paths = sorted(raw_dir.glob("PUBLIC_ACTUAL_OPERATIONAL_DEMAND_HH_*.zip"))
    if not zip_paths:
        raise FileNotFoundError(f"No archive zips found in {raw_dir}. Run scripts/fetch_historical_aemo.py first.")

    ingested_at = datetime.now().isoformat()
    total_inserted = 0
    for zip_path in zip_paths:
        df = parse_weekly_archive(zip_path)
        records = [(ts.isoformat(), region, demand) for ts, region, demand in df.itertuples(index=False, name=None)]
        inserted = insert_observations(conn, records, source="historical_backfill", ingested_at=ingested_at)
        total_inserted += inserted
        print(f"{zip_path.name}: {inserted} new row(s)")

    print(f"seeded {total_inserted} observation(s) from {len(zip_paths)} archive file(s)")
