"""Download AEMO's weekly NEM operational-demand archive zips.

Source: NEMWeb's public Archive directory (no auth needed):
    https://nemweb.com.au/Reports/Archive/Operational_Demand/ACTUAL_HH/
Files are named PUBLIC_ACTUAL_OPERATIONAL_DEMAND_HH_<YYYYMMDD>.zip, one per
week (dated on the Sunday it starts). The archive's retention window is a
rolling ~12-13 months, so we probe every Sunday in the requested range and
skip ones that 404 rather than assume a fixed start date.
"""
import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import requests

ARCHIVE_URL = "https://nemweb.com.au/Reports/Archive/Operational_Demand/ACTUAL_HH/PUBLIC_ACTUAL_OPERATIONAL_DEMAND_HH_{date}.zip"
HEADERS = {"User-Agent": "Mozilla/5.0 (electricity-demand-forecasting research script)"}
REQUEST_DELAY_SECONDS = 0.3


def sundays_covering(weeks: int) -> list[date]:
    today = date.today()
    last_sunday = today - timedelta(days=(today.weekday() + 1) % 7)
    return [last_sunday - timedelta(weeks=i) for i in range(weeks)]


def download_weekly_archives(output_dir: Path, weeks: int = 53) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for sunday in sundays_covering(weeks):
        filename = f"PUBLIC_ACTUAL_OPERATIONAL_DEMAND_HH_{sunday:%Y%m%d}.zip"
        dest = output_dir / filename
        if dest.exists():
            downloaded.append(dest)
            continue

        url = ARCHIVE_URL.format(date=f"{sunday:%Y%m%d}")
        response = requests.get(url, headers=HEADERS, timeout=30)
        if response.status_code == 404:
            print(f"skip (not available): {filename}")
            continue
        response.raise_for_status()

        dest.write_bytes(response.content)
        downloaded.append(dest)
        print(f"downloaded {filename} ({len(response.content):,} bytes)")
        time.sleep(REQUEST_DELAY_SECONDS)
    return downloaded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weeks", type=int, default=53, help="How many trailing weeks to fetch")
    parser.add_argument("--output-dir", default="data/raw/aemo", help="Where to save the weekly archive zips")
    args = parser.parse_args()

    files = download_weekly_archives(Path(args.output_dir), weeks=args.weeks)
    print(f"{len(files)} weekly archive file(s) available in {args.output_dir}")
