"""Recursive multi-step forecasting for serving trained models via the API.

Future weather values are unknown at inference time, so each weather
column is held constant at its last observed value for the forecast
horizon (see proposal.txt Section 28, Limitations). This is a demo
simplification, not a substitute for real weather forecasts.
"""
import numpy as np
import pandas as pd

from src.features.build_features import (
    add_calendar_features,
    add_cyclical_features,
    add_lag_features,
    add_rolling_features,
)

CYCLICAL_PERIODS = {"hour": 24, "day_of_week": 7, "month": 12}


def _build_feature_frame(history: pd.DataFrame, target_col: str, lags: list[int], rolling_windows: list[int]) -> pd.DataFrame:
    df = add_calendar_features(history)
    df = add_lag_features(df, target_col, lags)
    df = add_rolling_features(df, target_col, rolling_windows)
    for col, period in CYCLICAL_PERIODS.items():
        df = add_cyclical_features(df, col, period)
    return df


def _next_timestamp(history: pd.DataFrame, freq: str) -> pd.Timestamp:
    return history.index[-1] + pd.tseries.frequencies.to_offset(freq)


def _extend_with_persisted_weather(history: pd.DataFrame, target_col: str, freq: str) -> pd.DataFrame:
    """Append a row for the next timestamp, carrying weather columns forward unchanged."""
    next_ts = _next_timestamp(history, freq)
    new_row = history.iloc[[-1]].copy()
    new_row.index = [next_ts]
    new_row[target_col] = np.nan
    return pd.concat([history, new_row])


def recursive_xgboost_forecast(
    model,
    feature_cols: list[str],
    history: pd.DataFrame,
    target_col: str,
    lags: list[int],
    rolling_windows: list[int],
    horizon: int,
    freq: str = "h",
) -> list[dict]:
    """feature_cols must be the predictor columns saved at training time (must not include target_col itself)."""
    working = history.copy()
    predictions = []
    for _ in range(horizon):
        working = _extend_with_persisted_weather(working, target_col, freq)
        features = _build_feature_frame(working, target_col, lags, rolling_windows)
        row = features.iloc[[-1]][feature_cols]
        predicted = float(model.predict(row)[0])
        working.iloc[-1, working.columns.get_loc(target_col)] = predicted
        predictions.append({"timestamp": working.index[-1].isoformat(), "predicted": predicted})
    return predictions


def recursive_lstm_forecast(
    model,
    scaler,
    feature_cols: list[str],
    history: pd.DataFrame,
    target_col: str,
    sequence_length: int,
    horizon: int,
    freq: str = "h",
) -> list[dict]:
    """feature_cols is the full column set (including target_col) the scaler was fit on."""
    working = history.copy()
    target_idx = feature_cols.index(target_col)
    predictions = []
    for _ in range(horizon):
        window = add_calendar_features(working.tail(sequence_length))
        for col, period in CYCLICAL_PERIODS.items():
            window = add_cyclical_features(window, col, period)
        recent = window[feature_cols].to_numpy()
        scaled = scaler.transform(recent)
        X = scaled.reshape(1, sequence_length, len(feature_cols))
        scaled_pred = float(model.predict(X, verbose=0)[0, 0])

        dummy = np.zeros((1, len(feature_cols)))
        dummy[0, target_idx] = scaled_pred
        predicted = float(scaler.inverse_transform(dummy)[0, target_idx])

        working = _extend_with_persisted_weather(working, target_col, freq)
        working.iloc[-1, working.columns.get_loc(target_col)] = predicted
        predictions.append({"timestamp": working.index[-1].isoformat(), "predicted": predicted})
    return predictions


def sarima_forecast(results, horizon: int) -> list[dict]:
    forecast = results.get_forecast(steps=horizon)
    predicted_mean = forecast.predicted_mean
    return [
        {"timestamp": ts.isoformat(), "predicted": float(value)}
        for ts, value in predicted_mean.items()
    ]
