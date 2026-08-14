"""Train and evaluate an LSTM demand forecaster with MLflow experiment tracking."""
import json
import os
from pathlib import Path

import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

from src.data.load_data import chronological_split, load_raw_data, set_timestamp_index
from src.evaluation.metrics import mae, mape, rmse, smape
from src.features.build_features import build_all_features
from src.models.deep_learning import build_lstm, train_lstm

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
    
    with mlflow.start_run(run_name="lstm_v1"):
        # Log LSTM hyperparameters
        lstm_config = config["lstm"]
        mlflow.log_params({
            "model": "LSTM",
            "sequence_length": lstm_config["sequence_length"],
            "units": lstm_config["units"],
            "dropout": lstm_config["dropout"],
            "learning_rate": lstm_config["learning_rate"],
            "epochs": lstm_config["epochs"],
            "batch_size": lstm_config["batch_size"],
            "early_stopping_patience": lstm_config["early_stopping_patience"],
            "train_ratio": config["split"]["train_ratio"],
            "val_ratio": config["split"]["val_ratio"],
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
        print(f"train={len(train)} val={len(val)} test={len(test)} rows")
        
        # Prepare data for LSTM
        X_train, y_train = train[feature_cols], train[target_col]
        X_test, y_test = test[feature_cols], test[target_col]
        
        # Scale features for LSTM
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train LSTM
        print("Training LSTM model...")
        model = build_lstm(
            input_shape=(X_train_scaled.shape[1],),
            units=lstm_config["units"],
            dropout=lstm_config["dropout"],
            learning_rate=lstm_config["learning_rate"]
        )
        
        history = train_lstm(
            model,
            X_train_scaled,
            y_train.to_numpy(),
            epochs=lstm_config["epochs"],
            batch_size=lstm_config["batch_size"],
            early_stopping_patience=lstm_config["early_stopping_patience"]
        )
        
        # Evaluate
        y_pred = model.predict(X_test_scaled).flatten()
        
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
        models_dir = Path(config["models"]["lstm_path"]).parent
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model.save(config["models"]["lstm_path"])
        mlflow.keras.log_model(model, "lstm_model")
        
        # Save scaler
        import joblib
        scaler_path = Path(config["models"]["lstm_scaler_path"])
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(str(scaler_path), artifact_path="models")
        
        # Save feature columns
        feature_columns_path = Path(config["models"]["lstm_feature_columns"])
        feature_columns_path.write_text(json.dumps(feature_cols, indent=1))
        mlflow.log_artifact(str(feature_columns_path), artifact_path="models")
        
        print(f"\nModel saved to {config['models']['lstm_path']}")
        print(f"Scaler saved to {scaler_path}")
        
        # Save reports
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        metrics_df = pd.DataFrame([{"model": "LSTM", **metrics}])
        metrics_df.to_csv("reports/lstm_metrics.csv", index=False)
        mlflow.log_artifact("reports/lstm_metrics.csv", artifact_path="reports")
        
        predictions_df = pd.DataFrame({
            "timestamp": test.index,
            "actual": y_test.to_numpy(),
            "predicted": y_pred,
            "model": "LSTM"
        })
        predictions_df.to_csv("reports/lstm_predictions.csv", index=False)
        mlflow.log_artifact("reports/lstm_predictions.csv", artifact_path="reports")
        
        print("\nMLflow run completed!")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
