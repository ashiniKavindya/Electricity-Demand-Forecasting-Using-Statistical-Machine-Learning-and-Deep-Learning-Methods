"""Train and evaluate an LSTM demand forecaster on the AEMO NEM demand series.

Unlike XGBoost's engineered lag/rolling features, the LSTM only gets calendar +
cyclical columns plus the raw target - its recurrence over the sequence itself is
what's supposed to learn the lag/rolling-style patterns, so re-deriving them by hand
would be redundant. Per-region demand columns are dropped for the same leakage reason
as XGBoost's REGION_DEMAND_COLUMNS exclusion (see scripts/train_xgboost.py): they sum
to the target.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

from src.data.load_data import chronological_split, load_raw_data, set_timestamp_index
from src.evaluation.metrics import mae, mape, rmse, smape
from src.features.build_features import CYCLICAL_PERIODS, add_calendar_features, add_cyclical_features
from src.models.deep_learning import build_lstm_model, create_sequences


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_lstm_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_calendar_features(df)
    for col, period in CYCLICAL_PERIODS.items():
        df = add_cyclical_features(df, col, period)
    return df


def inverse_target(scaler: MinMaxScaler, scaled_values: np.ndarray, target_idx: int, n_features: int) -> np.ndarray:
    """Invert scaling for a single column: the scaler was fit on all `n_features`
    columns jointly, so the other columns need a (unused) placeholder to inverse_transform.
    """
    dummy = np.zeros((len(scaled_values), n_features))
    dummy[:, target_idx] = scaled_values
    return scaler.inverse_transform(dummy)[:, target_idx]


if __name__ == "__main__":
    config = load_config()
    target_col = config["data"]["target_col"]
    lstm_config = config["lstm"]
    sequence_length = lstm_config["sequence_length"]

    df = load_raw_data(config["data"]["processed_file"], timestamp_col=config["data"]["timestamp_col"])
    df = set_timestamp_index(df, timestamp_col=config["data"]["timestamp_col"])

    region_cols = [c for c in df.columns if c.startswith("demand_") and c != target_col]
    features_df = build_lstm_features(df.drop(columns=region_cols))
    # target_col first, so target_idx=0 stays stable regardless of the rest of the column list
    feature_cols = [target_col] + [c for c in features_df.columns if c != target_col]
    features_df = features_df[feature_cols]

    train, val, test = chronological_split(features_df, config["split"]["train_ratio"], config["split"]["val_ratio"])
    print(f"train={len(train)} val={len(val)} test={len(test)} rows")

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    # Prepend each split's trailing `sequence_length` rows from the split before it, so
    # the first window of val/test isn't thrown away - without ever fitting the scaler
    # on val/test data.
    val_scaled = scaler.transform(pd.concat([train.iloc[-sequence_length:], val]))
    test_scaled = scaler.transform(pd.concat([val.iloc[-sequence_length:], test]))

    target_idx = feature_cols.index(target_col)
    X_train, y_train = create_sequences(train_scaled, target_idx, sequence_length)
    X_val, y_val = create_sequences(val_scaled, target_idx, sequence_length)
    X_test, y_test = create_sequences(test_scaled, target_idx, sequence_length)

    model = build_lstm_model(
        sequence_length=sequence_length,
        n_features=len(feature_cols),
        units=lstm_config["units"],
        dropout=lstm_config["dropout"],
        learning_rate=lstm_config["learning_rate"],
    )
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=lstm_config["epochs"],
        batch_size=lstm_config["batch_size"],
        callbacks=[keras.callbacks.EarlyStopping(patience=lstm_config["early_stopping_patience"], restore_best_weights=True)],
        verbose=2,
    )

    scaled_pred = model.predict(X_test, verbose=0).flatten()
    y_pred = inverse_target(scaler, scaled_pred, target_idx, len(feature_cols))
    y_true = inverse_target(scaler, y_test, target_idx, len(feature_cols))

    metrics = {
        "model": "LSTM",
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "n": len(y_true),
    }
    print(pd.DataFrame([metrics]).to_string(index=False))

    model_path = Path(config["models"]["lstm_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    joblib.dump(scaler, config["models"]["lstm_scaler_path"])
    Path(config["models"]["lstm_feature_columns"]).write_text(json.dumps(feature_cols, indent=1))
    print(f"saved model to {model_path}")

    pd.DataFrame([metrics]).to_csv("reports/lstm_metrics.csv", index=False)
    # test_scaled was padded with sequence_length rows from val's tail, so every row
    # of `test` produced a prediction - timestamps line up 1:1 with test.index.
    pd.DataFrame({"timestamp": test.index, "actual": y_true, "predicted": y_pred, "model": "LSTM"}).to_csv(
        "reports/lstm_predictions.csv", index=False
    )
    print("wrote reports/lstm_metrics.csv and reports/lstm_predictions.csv")
