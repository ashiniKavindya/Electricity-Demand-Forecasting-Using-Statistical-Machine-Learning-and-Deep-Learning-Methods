"""Evaluate naive and seasonal-naive baseline forecasts on the AEMO NEM demand series."""
import pandas as pd
import yaml

from src.data.load_data import chronological_split, load_raw_data, set_timestamp_index
from src.evaluation.metrics import mae, mape, rmse, smape
from src.models.baseline import naive_forecast, seasonal_naive_forecast

SEASON_LENGTH = 168  # 1 week of hourly data


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate(name: str, forecast: pd.Series, actual: pd.Series) -> dict:
    valid = forecast.notna()
    y_true = actual[valid].to_numpy()
    y_pred = forecast[valid].to_numpy()
    return {
        "model": name,
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "n": len(y_true),
    }


if __name__ == "__main__":
    config = load_config()
    df = load_raw_data(config["data"]["processed_file"], timestamp_col=config["data"]["timestamp_col"])
    df = set_timestamp_index(df, timestamp_col=config["data"]["timestamp_col"])
    target = df[config["data"]["target_col"]]

    train, val, test = chronological_split(df, config["split"]["train_ratio"], config["split"]["val_ratio"])
    print(f"train={len(train)} val={len(val)} test={len(test)} rows")

    naive_full = naive_forecast(target)
    seasonal_full = seasonal_naive_forecast(target, SEASON_LENGTH)

    results = [
        evaluate("Naive (previous hour)", naive_full.loc[test.index], target.loc[test.index]),
        evaluate(f"Seasonal naive ({SEASON_LENGTH}h)", seasonal_full.loc[test.index], target.loc[test.index]),
    ]

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    results_df.to_csv("reports/baseline_metrics.csv", index=False)
    print("wrote reports/baseline_metrics.csv")
