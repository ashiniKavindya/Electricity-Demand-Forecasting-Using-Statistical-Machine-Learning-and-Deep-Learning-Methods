"""Train and evaluate an XGBoost demand forecaster on the AEMO NEM demand series."""
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml

from src.data.load_data import chronological_split, load_raw_data, set_timestamp_index
from src.evaluation.metrics import mae, mape, rmse, smape
from src.features.build_features import (
    add_calendar_features,
    add_cyclical_features,
    add_lag_features,
    add_rolling_features,
)
from src.models.machine_learning import fit_xgboost

CYCLICAL_PERIODS = {"hour": 24, "day_of_week": 7, "month": 12}
# Per-region demand columns sum to the target itself - including them as
# features would be near-perfect leakage (the same shape of bug as the old
# project's temperature column), so they're excluded here on purpose.
REGION_DEMAND_COLUMNS = ["demand_nsw1", "demand_qld1", "demand_sa1", "demand_tas1", "demand_vic1"]


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_features(df: pd.DataFrame, target_col: str, lags: list[int], rolling_windows: list[int]) -> pd.DataFrame:
    df = add_calendar_features(df)
    df = add_lag_features(df, target_col, lags)
    df = add_rolling_features(df, target_col, rolling_windows)
    for col, period in CYCLICAL_PERIODS.items():
        df = add_cyclical_features(df, col, period)
    return df


if __name__ == "__main__":
    config = load_config()
    target_col = config["data"]["target_col"]

    df = load_raw_data(config["data"]["processed_file"], timestamp_col=config["data"]["timestamp_col"])
    df = set_timestamp_index(df, timestamp_col=config["data"]["timestamp_col"])

    features_df = build_features(df, target_col, config["features"]["lags"], config["features"]["rolling_windows"])
    exclude = {target_col, *REGION_DEMAND_COLUMNS}
    feature_cols = [c for c in features_df.columns if c not in exclude]
    features_df = features_df.dropna(subset=feature_cols + [target_col])

    train, val, test = chronological_split(features_df, config["split"]["train_ratio"], config["split"]["val_ratio"])
    print(f"train={len(train)} val={len(val)} test={len(test)} rows after dropping warm-up NaNs")

    X_train, y_train = train[feature_cols], train[target_col]
    X_test, y_test = test[feature_cols], test[target_col]

    model = fit_xgboost(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "model": "XGBoost",
        "mae": mae(y_test.to_numpy(), y_pred),
        "rmse": rmse(y_test.to_numpy(), y_pred),
        "mape": mape(y_test.to_numpy(), y_pred),
        "smape": smape(y_test.to_numpy(), y_pred),
        "n": len(y_test),
    }
    print(pd.DataFrame([metrics]).to_string(index=False))

    model_path = Path(config["models"]["xgboost_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    Path(config["models"]["xgboost_feature_columns"]).write_text(json.dumps(feature_cols, indent=1))
    print(f"saved model to {model_path}")

    pd.DataFrame([metrics]).to_csv("reports/xgboost_metrics.csv", index=False)
    pd.DataFrame(
        {"timestamp": test.index, "actual": y_test.to_numpy(), "predicted": y_pred, "model": "XGBoost"}
    ).to_csv("reports/xgboost_predictions.csv", index=False)
    print("wrote reports/xgboost_metrics.csv and reports/xgboost_predictions.csv")
