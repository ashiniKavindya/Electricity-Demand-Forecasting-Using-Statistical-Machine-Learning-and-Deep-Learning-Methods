"""Compare baseline, XGBoost, and LSTM forecasts with Diebold-Mariano significance tests.

Run after scripts/train_baseline.py, scripts/train_xgboost.py, and (optionally)
scripts/train_lstm.py - LSTM's rows are skipped if reports/lstm_*.csv aren't present.
"""
from pathlib import Path

import pandas as pd
import yaml

from src.data.load_data import load_raw_data, set_timestamp_index
from src.evaluation.metrics import diebold_mariano_test
from src.models.baseline import seasonal_naive_forecast

MODEL_PREDICTIONS = {
    "XGBoost": "reports/xgboost_predictions.csv",
    "LSTM": "reports/lstm_predictions.csv",
}


def diebold_mariano_vs_seasonal_naive(
    model_name: str, predictions_path: str, seasonal_forecast: pd.Series, target: pd.Series
) -> dict:
    predictions = pd.read_csv(predictions_path, parse_dates=["timestamp"]).set_index("timestamp")
    model_errors = predictions["actual"] - predictions["predicted"]

    common_index = predictions.index.intersection(seasonal_forecast.dropna().index)
    seasonal_errors = (target.loc[common_index] - seasonal_forecast.loc[common_index]).to_numpy()
    model_errors_common = model_errors.loc[common_index].to_numpy()

    dm_stat, p_value = diebold_mariano_test(model_errors_common, seasonal_errors)
    return {
        "model_a": model_name,
        "model_b": "Seasonal naive (168h)",
        "n_common_obs": len(common_index),
        "dm_statistic": dm_stat,
        "p_value": p_value,
        "significant_at_0.05": p_value < 0.05,
    }


if __name__ == "__main__":
    metrics_frames = [pd.read_csv("reports/baseline_metrics.csv")]
    for model_name, predictions_path in MODEL_PREDICTIONS.items():
        metrics_path = f"reports/{model_name.lower()}_metrics.csv"
        if Path(metrics_path).exists():
            metrics_frames.append(pd.read_csv(metrics_path))
        else:
            print(f"skipping {model_name}: {metrics_path} not found (run its training script first)")

    comparison = pd.concat(metrics_frames, ignore_index=True)
    comparison.to_csv("reports/model_comparison.csv", index=False)
    print(comparison.to_string(index=False))

    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    df = load_raw_data(config["data"]["processed_file"], timestamp_col=config["data"]["timestamp_col"])
    df = set_timestamp_index(df, timestamp_col=config["data"]["timestamp_col"])
    target = df[config["data"]["target_col"]]
    seasonal_forecast = seasonal_naive_forecast(target, 168)

    dm_results = []
    for model_name, predictions_path in MODEL_PREDICTIONS.items():
        if not Path(predictions_path).exists():
            continue
        result = diebold_mariano_vs_seasonal_naive(model_name, predictions_path, seasonal_forecast, target)
        dm_results.append(result)
        print(
            f"\nDiebold-Mariano test ({model_name} vs seasonal-naive, n={result['n_common_obs']}): "
            f"dm_stat={result['dm_statistic']:.3f}, p_value={result['p_value']:.4g}, "
            f"significant_at_0.05={result['significant_at_0.05']}"
        )

    if all(Path(p).exists() for p in MODEL_PREDICTIONS.values()):
        xgb = pd.read_csv(MODEL_PREDICTIONS["XGBoost"], parse_dates=["timestamp"]).set_index("timestamp")
        lstm = pd.read_csv(MODEL_PREDICTIONS["LSTM"], parse_dates=["timestamp"]).set_index("timestamp")
        common_index = xgb.index.intersection(lstm.index)
        xgb_errors = (xgb["actual"] - xgb["predicted"]).loc[common_index].to_numpy()
        lstm_errors = (lstm["actual"] - lstm["predicted"]).loc[common_index].to_numpy()

        dm_stat, p_value = diebold_mariano_test(lstm_errors, xgb_errors)
        head_to_head = {
            "model_a": "LSTM",
            "model_b": "XGBoost",
            "n_common_obs": len(common_index),
            "dm_statistic": dm_stat,
            "p_value": p_value,
            "significant_at_0.05": p_value < 0.05,
        }
        dm_results.append(head_to_head)
        print(
            f"\nDiebold-Mariano test (LSTM vs XGBoost, n={head_to_head['n_common_obs']}): "
            f"dm_stat={dm_stat:.3f}, p_value={p_value:.4g}, significant_at_0.05={head_to_head['significant_at_0.05']}"
        )

    pd.DataFrame(dm_results).to_csv("reports/dm_test_results.csv", index=False)
    print("\nwrote reports/model_comparison.csv and reports/dm_test_results.csv")
