"""Run the live-inference job on a recurring schedule."""
import yaml
from apscheduler.schedulers.blocking import BlockingScheduler

from inference.predictor import predict_once
from src.data.db import get_connection


def _predict_and_report(conn, config: dict) -> None:
    result = predict_once(conn, config)
    if result["error"]:
        print(f"inference failed: {result['error']}")
    elif result["inserted"]:
        print(
            f"predicted demand={result['predicted_demand']:.0f}MW for {result['target_timestamp']} "
            f"(based on {result['based_on_timestamp']})"
        )
    else:
        print(f"prediction for {result['target_timestamp']} already stored, nothing new")


def run_forever(config_path: str = "config.yaml") -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    conn = get_connection(config["data"]["db_path"])
    interval_seconds = config["inference"]["interval_seconds"]

    print("running initial inference pass...")
    _predict_and_report(conn, config)

    scheduler = BlockingScheduler()
    scheduler.add_job(lambda: _predict_and_report(conn, config), "interval", seconds=interval_seconds)
    print(f"inference running, checking every {interval_seconds}s (Ctrl+C to stop)")
    scheduler.start()


if __name__ == "__main__":
    run_forever()
