"""Compare baseline and XGBoost forecasts with a Diebold-Mariano significance test.

Run after scripts/train_baseline.py and scripts/train_xgboost.py.
"""
import pandas as pd
import yaml

from src.data.load_data import load_raw_data, set_timestamp_index
from src.evaluation.metrics import diebold_mariano_test
from src.models.baseline import seasonal_naive_forecast

if __name__ == "__main__":
    baseline_metrics = pd.read_csv("reports/baseline_metrics.csv")
    xgboost_metrics = pd.read_csv("reports/xgboost_metrics.csv")
    comparison = pd.concat([baseline_metrics, xgboost_metrics], ignore_index=True)
    comparison.to_csv("reports/model_comparison.csv", index=False)
    print(comparison.to_string(index=False))

    xgboost_predictions = pd.read_csv("reports/xgboost_predictions.csv", parse_dates=["timestamp"])
    xgboost_predictions = xgboost_predictions.set_index("timestamp")
    xgboost_errors = xgboost_predictions["actual"] - xgboost_predictions["predicted"]

    # Reconstruct the seasonal-naive error series over the same test window for a fair DM test.
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    df = load_raw_data(config["data"]["processed_file"], timestamp_col=config["data"]["timestamp_col"])
    df = set_timestamp_index(df, timestamp_col=config["data"]["timestamp_col"])
    target = df[config["data"]["target_col"]]
    seasonal_forecast = seasonal_naive_forecast(target, 168)

    common_index = xgboost_predictions.index.intersection(seasonal_forecast.dropna().index)
    seasonal_errors = (target.loc[common_index] - seasonal_forecast.loc[common_index]).to_numpy()
    xgboost_errors_common = xgboost_errors.loc[common_index].to_numpy()

    dm_stat, p_value = diebold_mariano_test(xgboost_errors_common, seasonal_errors)
    significant = p_value < 0.05
    print(
        f"\nDiebold-Mariano test (XGBoost vs seasonal-naive, n={len(common_index)}): "
        f"dm_stat={dm_stat:.3f}, p_value={p_value:.4g}, significant_at_0.05={significant}"
    )

    pd.DataFrame(
        [
            {
                "model_a": "XGBoost",
                "model_b": "Seasonal naive (168h)",
                "n_common_obs": len(common_index),
                "dm_statistic": dm_stat,
                "p_value": p_value,
                "significant_at_0.05": significant,
            }
        ]
    ).to_csv("reports/dm_test_results.csv", index=False)
    print("wrote reports/model_comparison.csv and reports/dm_test_results.csv")
