"""Run the AEMO collector on a recurring schedule."""
import yaml
from apscheduler.schedulers.blocking import BlockingScheduler

from collector.poller import poll_once
from src.data.db import get_connection


def _poll_and_report(conn) -> None:
    result = poll_once(conn)
    if result["error"]:
        print(f"poll failed: {result['error']}")
    else:
        print(f"poll ok: checked {result['files_checked']} file(s), inserted {result['rows_inserted']} row(s)")


def run_forever(config_path: str = "config.yaml") -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    conn = get_connection(config["data"]["db_path"])
    interval_seconds = config["aemo"]["poll_interval_seconds"]

    print("running initial poll...")
    _poll_and_report(conn)

    scheduler = BlockingScheduler()
    scheduler.add_job(lambda: _poll_and_report(conn), "interval", seconds=interval_seconds)
    print(f"collector running, polling every {interval_seconds}s (Ctrl+C to stop)")
    scheduler.start()


if __name__ == "__main__":
    run_forever()
