"""Calendar, lag, rolling, and cyclical feature engineering."""
import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, day-of-week, month, and weekend indicator features."""
    df = df.copy()
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, target_col: str, lags: list[int]) -> pd.DataFrame:
    """Add shifted lag features for the target column (no future leakage)."""
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target_col: str, windows: list[int]) -> pd.DataFrame:
    """Add rolling mean/std features computed on shifted (past-only) values."""
    df = df.copy()
    shifted = df[target_col].shift(1)
    for window in windows:
        df[f"{target_col}_roll_mean_{window}"] = shifted.rolling(window).mean()
        df[f"{target_col}_roll_std_{window}"] = shifted.rolling(window).std()
    return df


def add_cyclical_features(df: pd.DataFrame, col: str, period: int) -> pd.DataFrame:
    """Sine/cosine encode a cyclical column (e.g. hour, day_of_week, month)."""
    df = df.copy()
    df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
    df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)
    return df
