"""Build the processed hourly NEM demand dataset from downloaded AEMO archives.

Run after scripts/fetch_historical_aemo.py. Parses every weekly archive zip
in data/raw/aemo/, aggregates the 5 NEM regions into per-region and NEM-wide
total demand, resamples from half-hourly to hourly, fills small gaps, and
writes data/processed/aemo_nem_demand_hourly.csv.
"""
import argparse
from pathlib import Path

import pandas as pd

from src.data.aemo import aggregate_regions_to_nem_total, parse_weekly_archive, resample_to_hourly
from src.data.clean_data import fill_missing_values
from src.data.load_data import detect_missing_intervals

MAX_GAP_HOURS_TO_SILENTLY_INTERPOLATE = 3


def build_long_dataset(raw_dir: Path) -> pd.DataFrame:
    zip_paths = sorted(raw_dir.glob("PUBLIC_ACTUAL_OPERATIONAL_DEMAND_HH_*.zip"))
    if not zip_paths:
        raise FileNotFoundError(f"No AEMO archive zips found in {raw_dir}. Run scripts/fetch_historical_aemo.py first.")

    frames = []
    for zip_path in zip_paths:
        print(f"parsing {zip_path.name}...")
        frames.append(parse_weekly_archive(zip_path))
    long_df = pd.concat(frames, ignore_index=True)
    return long_df.drop_duplicates(subset=["interval_datetime", "region_id"])


def largest_consecutive_gap_hours(missing: pd.DatetimeIndex) -> float:
    if len(missing) == 0:
        return 0.0
    gaps = pd.Series(missing).diff().dt.total_seconds().div(3600).fillna(1.0)
    run = best = 1.0
    for gap in gaps:
        run = run + 1.0 if gap == 1.0 else 1.0
        best = max(best, run)
    return best


def fill_hourly_gaps(hourly: pd.DataFrame) -> pd.DataFrame:
    indexed = hourly.set_index("timestamp").sort_index()
    full_range = pd.date_range(indexed.index.min(), indexed.index.max(), freq="h")
    reindexed = indexed.reindex(full_range)
    reindexed.index.name = "timestamp"

    missing = detect_missing_intervals(indexed, freq="h")
    if len(missing) > 0:
        largest_gap = largest_consecutive_gap_hours(missing)
        print(f"WARNING: {len(missing)} missing hourly interval(s); largest consecutive gap is {largest_gap:.0f}h.")
        if largest_gap > MAX_GAP_HOURS_TO_SILENTLY_INTERPOLATE:
            print(f"  Gap exceeds {MAX_GAP_HOURS_TO_SILENTLY_INTERPOLATE}h - inspect before trusting downstream models.")

    filled = fill_missing_values(reindexed, method="time")
    return filled.reset_index()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/aemo")
    parser.add_argument("--output", default="data/processed/aemo_nem_demand_hourly.csv")
    args = parser.parse_args()

    long_df = build_long_dataset(Path(args.raw_dir))
    wide = aggregate_regions_to_nem_total(long_df)
    hourly = resample_to_hourly(wide)
    hourly = fill_hourly_gaps(hourly)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(output_path, index=False)
    print(f"wrote {len(hourly)} hourly rows to {output_path}")
    print(f"range: {hourly['timestamp'].min()} to {hourly['timestamp'].max()}")
