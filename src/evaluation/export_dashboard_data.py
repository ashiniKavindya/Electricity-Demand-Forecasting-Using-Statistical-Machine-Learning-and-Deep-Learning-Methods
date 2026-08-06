"""Export model results to the JSON shape the React dashboard reads.

Run this at the end of a Kaggle notebook (or locally) after generating
predictions, metrics, and feature importances. Copy the resulting files
into dashboard/public/data/ to make them appear in the dashboard.
"""
import json
import os

import pandas as pd


def export_demand_series(df: pd.DataFrame, output_dir: str, filename: str = "demand.json") -> None:
    """df must have columns: timestamp, actual, predicted (predicted may be NaN for history)."""
    records = df[["timestamp", "actual", "predicted"]].to_dict(orient="records")
    _write_json(records, output_dir, filename)


def export_metrics(df: pd.DataFrame, output_dir: str, filename: str = "metrics.json") -> None:
    """df must have columns: model, mae, rmse, mape, trainingTimeSeconds."""
    records = df[["model", "mae", "rmse", "mape", "trainingTimeSeconds"]].to_dict(orient="records")
    _write_json(records, output_dir, filename)


def export_feature_importance(df: pd.DataFrame, output_dir: str, filename: str = "feature_importance.json") -> None:
    """df must have columns: feature, importance."""
    records = df[["feature", "importance"]].to_dict(orient="records")
    _write_json(records, output_dir, filename)


def _write_json(records: list[dict], output_dir: str, filename: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, filename), "w") as f:
        json.dump(records, f, indent=1, default=str)


if __name__ == "__main__":
    import joblib

    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
    REPORTS = os.path.join(ROOT, "reports")
    MODELS = os.path.join(ROOT, "models")
    OUTPUT_DIR = os.path.join(ROOT, "dashboard", "public", "data")

    ml_predictions = pd.read_csv(os.path.join(REPORTS, "ml_predictions.csv"))
    demand_df = ml_predictions[ml_predictions["model"] == "XGBoost"]
    export_demand_series(demand_df, OUTPUT_DIR)

    metrics_df = pd.read_csv(os.path.join(REPORTS, "model_comparison.csv"))
    metrics_df = metrics_df.rename(columns={"training_time_seconds": "trainingTimeSeconds"})
    metrics_df["trainingTimeSeconds"] = metrics_df["trainingTimeSeconds"].fillna(0)
    export_metrics(metrics_df, OUTPUT_DIR)

    xgb_model = joblib.load(os.path.join(MODELS, "xgboost_model.pkl"))
    with open(os.path.join(MODELS, "xgboost_feature_columns.json")) as f:
        feature_columns = json.load(f)
    importance_df = pd.DataFrame(
        {"feature": feature_columns, "importance": xgb_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    export_feature_importance(importance_df, OUTPUT_DIR)

    print(
        f"Exported demand.json ({len(demand_df)} rows), metrics.json ({len(metrics_df)} rows), "
        f"feature_importance.json ({len(importance_df)} rows) to {OUTPUT_DIR}"
    )
