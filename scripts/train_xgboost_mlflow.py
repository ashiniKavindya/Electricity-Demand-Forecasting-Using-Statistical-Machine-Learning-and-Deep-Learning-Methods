"""Train and evaluate an XGBoost demand forecaster with MLflow experiment tracking."""
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from src.data.load_data import chronological_split, load_raw_data, set_timestamp_index
from src.evaluation.metrics import mae, mape, rmse, smape
from src.features.build_features import build_all_features
from src.models.machine_learning import fit_xgboost

# Per-region demand columns sum to the target itself - including them as
# features would be near-perfect leakage, so they're excluded here on purpose.
REGION_DEMAND_COLUMNS = ["demand_nsw1", "demand_qld1", "demand_sa1", "demand_tas1", "demand_vic1"]


def load_config(path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def setup_mlflow(experiment_name: str = "AEMO-Demand-Forecasting"):
    """Initialize MLflow tracking."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", None)

    if tracking_uri:
        # Use remote MLflow server
        mlflow.set_tracking_uri(tracking_uri)
        print(f"MLflow tracking URI: {tracking_uri}")
    else:
        # Use a local SQLite backend instead of the deprecated file store.
        mlruns_dir = Path("./mlruns")
        mlruns_dir.mkdir(exist_ok=True)
        sqlite_path = str((mlruns_dir / "mlflow.db").resolve())
        mlflow.set_tracking_uri(f"sqlite:///{sqlite_path}")
        print("Warning: MLFLOW_TRACKING_URI not set. Using local SQLite backend.")
        print(f"MLflow database will be saved to: {sqlite_path}")

    mlflow.set_experiment(experiment_name)
    print(f"MLflow experiment: {experiment_name}")


if __name__ == "__main__":
    config = load_config()
    target_col = config["data"]["target_col"]
    
    # Initialize MLflow
    setup_mlflow()
    
    with mlflow.start_run(run_name="xgboost_v1"):
        # Log configuration parameters
        mlflow.log_params({
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "train_ratio": config["split"]["train_ratio"],
            "val_ratio": config["split"]["val_ratio"],
            "test_ratio": config["split"]["test_ratio"],
        })
        
        # Load and prepare data
        print("Loading data...")
        df = load_raw_data(config["data"]["processed_file"], timestamp_col=config["data"]["timestamp_col"])
        df = set_timestamp_index(df, timestamp_col=config["data"]["timestamp_col"])
        
        print("Building features...")
        features_df = build_all_features(
            df, 
            target_col, 
            config["features"]["lags"], 
            config["features"]["rolling_windows"]
        )
        
        exclude = {target_col, *REGION_DEMAND_COLUMNS}
        feature_cols = [c for c in features_df.columns if c not in exclude]
        features_df = features_df.dropna(subset=feature_cols + [target_col])
        
        print(f"Total features: {len(feature_cols)}")
        mlflow.log_param("num_features", len(feature_cols))
        
        # Split data
        train, val, test = chronological_split(
            features_df, 
            config["split"]["train_ratio"], 
            config["split"]["val_ratio"]
        )
        print(f"train={len(train)} val={len(val)} test={len(test)} rows after dropping warm-up NaNs")
        
        X_train, y_train = train[feature_cols], train[target_col]
        X_test, y_test = test[feature_cols], test[target_col]
        
        # Train model
        print("Training XGBoost model...")
        model = fit_xgboost(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            "mae": mae(y_test.to_numpy(), y_pred),
            "rmse": rmse(y_test.to_numpy(), y_pred),
            "mape": mape(y_test.to_numpy(), y_pred),
            "smape": smape(y_test.to_numpy(), y_pred),
            "n_test": len(y_test),
        }
        
        print("\nMetrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
            mlflow.log_metric(key, value)
        
        # Save model artifacts
        model_path = Path(config["models"]["xgboost_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        
        # Log model with MLflow. XGBoost models are not trusted by default in newer MLflow
        # versions, so we must explicitly allow the model classes used here.
        mlflow.sklearn.log_model(
            model,
            "xgboost_model",
            skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBRegressor"],
        )
        mlflow.log_artifact(str(model_path), artifact_path="models")
        
        # Save feature columns
        feature_columns_path = Path(config["models"]["xgboost_feature_columns"])
        feature_columns_path.write_text(json.dumps(feature_cols, indent=1))
        mlflow.log_artifact(str(feature_columns_path), artifact_path="models")
        
        print(f"\nModel saved to {model_path}")
        print(f"Feature columns saved to {feature_columns_path}")
        
        # Save reports
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        metrics_df = pd.DataFrame([{"model": "XGBoost", **metrics}])
        metrics_df.to_csv("reports/xgboost_metrics.csv", index=False)
        mlflow.log_artifact("reports/xgboost_metrics.csv", artifact_path="reports")
        
        predictions_df = pd.DataFrame({
            "timestamp": test.index,
            "actual": y_test.to_numpy(),
            "predicted": y_pred,
            "model": "XGBoost"
        })
        predictions_df.to_csv("reports/xgboost_predictions.csv", index=False)
        mlflow.log_artifact("reports/xgboost_predictions.csv", artifact_path="reports")
        
        print("\nMLflow run completed!")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
