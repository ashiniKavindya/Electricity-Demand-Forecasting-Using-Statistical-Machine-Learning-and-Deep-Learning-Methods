from datetime import datetime
from unittest.mock import Mock, patch

from collector.nemweb_client import list_available_intervals
from collector.poller import _normalize

SAMPLE_LISTING_HTML = """
<html><body><pre>
459 <A HREF="/Reports/Current/Operational_Demand/ACTUAL_HH/PUBLIC_ACTUAL_OPERATIONAL_DEMAND_HH_202608061730_20260806173006.zip">PUBLIC_ACTUAL_OPERATIONAL_DEMAND_HH_202608061730_20260806173006.zip</A><br>
460 <A HREF="/Reports/Current/Operational_Demand/ACTUAL_HH/PUBLIC_ACTUAL_OPERATIONAL_DEMAND_HH_202608061800_20260806180006.zip">PUBLIC_ACTUAL_OPERATIONAL_DEMAND_HH_202608061800_20260806180006.zip</A><br>
</pre></body></html>
"""


def test_list_available_intervals_dedupes_and_sorts():
    mock_response = Mock(text=SAMPLE_LISTING_HTML)
    mock_response.raise_for_status = Mock()

    with patch("collector.nemweb_client.requests.get", return_value=mock_response):
        entries = list_available_intervals()

    assert len(entries) == 2
    assert entries[0] == (
        "PUBLIC_ACTUAL_OPERATIONAL_DEMAND_HH_202608061730_20260806173006.zip",
        datetime(2026, 8, 6, 17, 30),
    )
    assert entries[1][1] == datetime(2026, 8, 6, 18, 0)


def test_normalize_converts_aemo_timestamp_to_iso():
    records = [("2026/08/06 18:00:00", "NSW1", 8500.0)]
    normalized = _normalize(records)
    assert normalized == [("2026-08-06T18:00:00", "NSW1", 8500.0)]
