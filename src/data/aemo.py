"""Parse and aggregate AEMO NEM operational-demand archive data.

AEMO publishes actual operational demand as weekly archive zips, each a
zip of ~336 per-half-hour-interval zips, each holding one CSV in AEMO's
classic MMS report format (C: report metadata, I: column names, D: data
rows) - not a flat CSV. Verified against a real downloaded sample:

    C,NEMP.WORLD,ACTUAL_OPERATIONAL_DEMAND_HH,AEMO,PUBLIC,2026/07/19,00:00:07,...
    I,OPERATIONAL_DEMAND,ACTUAL,3,REGIONID,INTERVAL_DATETIME,OPERATIONAL_DEMAND,...
    D,OPERATIONAL_DEMAND,ACTUAL,3,NSW1,"2026/07/19 00:00:00",8239,0,0,...
    C,"END OF REPORT",8

Interval timestamps are NEM market time: a fixed UTC+10 offset with no
daylight saving (never `Australia/Sydney`, which observes DST). This
module keeps them as naive timestamps representing that fixed offset.
"""
import csv
import io
import zipfile
from pathlib import Path

import pandas as pd

NEM_REGIONS = ["NSW1", "QLD1", "SA1", "TAS1", "VIC1"]


def parse_operational_demand_csv(csv_bytes: bytes) -> list[tuple[str, str, float]]:
    """Parse one AEMO C/I/D-format operational-demand CSV.

    Returns (interval_datetime, region_id, operational_demand) tuples, one per D row.
    """
    reader = csv.reader(io.StringIO(csv_bytes.decode("utf-8")))
    records = []
    columns = None
    for row in reader:
        if not row:
            continue
        if row[0] == "I":
            columns = row[4:]
        elif row[0] == "D" and columns is not None:
            fields = dict(zip(columns, row[4:]))
            records.append((fields["INTERVAL_DATETIME"], fields["REGIONID"], float(fields["OPERATIONAL_DEMAND"])))
    return records


def parse_weekly_archive(zip_path: Path) -> pd.DataFrame:
    """Parse one weekly archive zip (a zip of per-interval zips, each holding one CSV)."""
    records = []
    with zipfile.ZipFile(zip_path) as outer:
        for inner_name in outer.namelist():
            if not inner_name.lower().endswith(".zip"):
                continue
            with zipfile.ZipFile(io.BytesIO(outer.read(inner_name))) as inner:
                for csv_name in inner.namelist():
                    if csv_name.upper().endswith(".CSV"):
                        records.extend(parse_operational_demand_csv(inner.read(csv_name)))

    df = pd.DataFrame(records, columns=["interval_datetime", "region_id", "operational_demand"])
    df["interval_datetime"] = pd.to_datetime(df["interval_datetime"], format="%Y/%m/%d %H:%M:%S")
    return df


def aggregate_regions_to_nem_total(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot per-region rows into a wide frame: per-region columns plus a NEM-wide total."""
    wide = df.pivot_table(index="interval_datetime", columns="region_id", values="operational_demand", aggfunc="sum")
    wide = wide.reindex(columns=NEM_REGIONS)
    wide.columns = [f"demand_{c.lower()}" for c in wide.columns]
    wide["demand"] = wide.sum(axis=1)
    return wide.sort_index().reset_index().rename(columns={"interval_datetime": "timestamp"})


def resample_to_hourly(df: pd.DataFrame, timestamp_col: str = "timestamp", how: str = "mean") -> pd.DataFrame:
    """Resample a half-hourly (or finer) demand frame to hourly bins (label=left, closed=left)."""
    indexed = df.set_index(timestamp_col).sort_index()
    resampled = indexed.resample("h", label="left", closed="left").agg(how)
    return resampled.reset_index()
