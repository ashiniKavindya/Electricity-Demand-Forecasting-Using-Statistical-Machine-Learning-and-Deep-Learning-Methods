"""Naive baseline forecasting models."""
import pandas as pd


def naive_forecast(series: pd.Series) -> pd.Series:
    """Predict each value as the previous observed value."""
    return series.shift(1)


def seasonal_naive_forecast(series: pd.Series, season_length: int) -> pd.Series:
    """Predict each value as the observation one season length earlier."""
    return series.shift(season_length)


def moving_average_forecast(series: pd.Series, window: int) -> pd.Series:
    """Predict each value as the rolling mean of the previous `window` observations."""
    return series.shift(1).rolling(window).mean()
