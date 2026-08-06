"""Client for NEMWeb's "Current" operational-demand directory.

This is the same OPERATIONAL_DEMAND metric and C/I/D CSV format used by
the historical weekly archives in src/data/aemo.py (deliberately - mixing
it with AEMO's live ELEC_NEM_SUMMARY endpoint would introduce a metric
discontinuity, since ELEC_NEM_SUMMARY reports TOTALDEMAND, a different
figure). Each entry here is one already-settled half-hourly interval,
published within minutes of the interval ending - "near-real-time", not
literally instantaneous, but consistent with the training data.
"""
import re
from datetime import datetime

import requests

CURRENT_DIR_URL = "https://nemweb.com.au/Reports/Current/Operational_Demand/ACTUAL_HH/"
FILENAME_PATTERN = re.compile(r"PUBLIC_ACTUAL_OPERATIONAL_DEMAND_HH_(\d{12})_\d{14}\.zip")
HEADERS = {"User-Agent": "Mozilla/5.0 (electricity-demand-forecasting research collector)"}


def list_available_intervals() -> list[tuple[str, datetime]]:
    """Return (filename, interval_start) for every interval file currently listed, oldest first."""
    response = requests.get(CURRENT_DIR_URL, headers=HEADERS, timeout=30)
    response.raise_for_status()

    # Each filename appears twice in the raw listing (once in the href, once
    # as the link text), so dedupe before returning.
    seen = {}
    for match in FILENAME_PATTERN.finditer(response.text):
        filename = match.group(0)
        seen[filename] = datetime.strptime(match.group(1), "%Y%m%d%H%M")
    return sorted(seen.items(), key=lambda entry: entry[1])


def download_interval_zip(filename: str) -> bytes:
    response = requests.get(CURRENT_DIR_URL + filename, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.content
