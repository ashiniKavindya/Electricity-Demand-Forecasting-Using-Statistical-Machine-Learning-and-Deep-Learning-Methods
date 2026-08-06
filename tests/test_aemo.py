import pandas as pd

from src.data.aemo import aggregate_regions_to_nem_total, parse_operational_demand_csv, resample_to_hourly

SAMPLE_CSV = (
    b'C,NEMP.WORLD,ACTUAL_OPERATIONAL_DEMAND_HH,AEMO,PUBLIC,2026/07/19,00:00:07,0000000528134281,DEMAND,0000000528134281\r\n'
    b'I,OPERATIONAL_DEMAND,ACTUAL,3,REGIONID,INTERVAL_DATETIME,OPERATIONAL_DEMAND,OPERATIONAL_DEMAND_ADJUSTMENT,WDR_ESTIMATE,LASTCHANGED\r\n'
    b'D,OPERATIONAL_DEMAND,ACTUAL,3,NSW1,"2026/07/19 00:00:00",8239,0,0,"2026/07/19 00:00:01"\r\n'
    b'D,OPERATIONAL_DEMAND,ACTUAL,3,QLD1,"2026/07/19 00:00:00",6071,0,0,"2026/07/19 00:00:01"\r\n'
    b'D,OPERATIONAL_DEMAND,ACTUAL,3,SA1,"2026/07/19 00:00:00",1599,0,0,"2026/07/19 00:00:01"\r\n'
    b'D,OPERATIONAL_DEMAND,ACTUAL,3,TAS1,"2026/07/19 00:00:00",1092,0,0,"2026/07/19 00:00:01"\r\n'
    b'D,OPERATIONAL_DEMAND,ACTUAL,3,VIC1,"2026/07/19 00:00:00",6407,0,0,"2026/07/19 00:00:01"\r\n'
    b'C,"END OF REPORT",8\r\n'
)


def test_parse_operational_demand_csv_extracts_all_regions():
    records = parse_operational_demand_csv(SAMPLE_CSV)
    assert len(records) == 5
    region_ids = {region for _, region, _ in records}
    assert region_ids == {"NSW1", "QLD1", "SA1", "TAS1", "VIC1"}


def test_parse_operational_demand_csv_reads_correct_values():
    records = parse_operational_demand_csv(SAMPLE_CSV)
    by_region = {region: demand for _, region, demand in records}
    assert by_region["NSW1"] == 8239.0
    assert by_region["VIC1"] == 6407.0


def test_aggregate_regions_to_nem_total_sums_all_regions():
    df = pd.DataFrame(parse_operational_demand_csv(SAMPLE_CSV), columns=["interval_datetime", "region_id", "operational_demand"])
    df["interval_datetime"] = pd.to_datetime(df["interval_datetime"], format="%Y/%m/%d %H:%M:%S")

    wide = aggregate_regions_to_nem_total(df)

    assert len(wide) == 1
    assert wide["demand"].iloc[0] == 8239 + 6071 + 1599 + 1092 + 6407
    assert wide["demand_nsw1"].iloc[0] == 8239


def test_resample_to_hourly_averages_half_hourly_points():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-07-19 00:00:00", "2026-07-19 00:30:00", "2026-07-19 01:00:00"]),
            "demand": [100.0, 200.0, 300.0],
        }
    )

    hourly = resample_to_hourly(df)

    assert len(hourly) == 2
    assert hourly.loc[hourly["timestamp"] == pd.Timestamp("2026-07-19 00:00:00"), "demand"].iloc[0] == 150.0
    assert hourly.loc[hourly["timestamp"] == pd.Timestamp("2026-07-19 01:00:00"), "demand"].iloc[0] == 300.0
