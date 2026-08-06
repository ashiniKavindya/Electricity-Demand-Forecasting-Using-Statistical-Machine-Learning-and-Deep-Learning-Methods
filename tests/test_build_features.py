import numpy as np
import pandas as pd
import pytest

from src.data.load_data import chronological_split
from src.features.build_features import (
    add_calendar_features,
    add_cyclical_features,
    add_lag_features,
    add_rolling_features,
)


@pytest.fixture
def hourly_series() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=48, freq="h")
    return pd.DataFrame({"demand": np.arange(48, dtype=float)}, index=index)


def test_add_lag_features_shifts_forward_not_backward(hourly_series):
    result = add_lag_features(hourly_series, "demand", [1, 24])

    # A lag-1 feature at time t must equal the actual value at t-1, i.e. it
    # cannot see its own or any future observation.
    assert result["demand_lag_1"].iloc[5] == hourly_series["demand"].iloc[4]
    assert result["demand_lag_24"].iloc[30] == hourly_series["demand"].iloc[6]
    assert pd.isna(result["demand_lag_1"].iloc[0])


def test_add_rolling_features_excludes_current_value(hourly_series):
    result = add_rolling_features(hourly_series, "demand", [3])

    # The window ending just before t (values at t-3, t-2, t-1) must not
    # include the value at t itself.
    window_mean_at_10 = hourly_series["demand"].iloc[7:10].mean()
    assert result["demand_roll_mean_3"].iloc[10] == pytest.approx(window_mean_at_10)


def test_add_calendar_features_derives_from_index(hourly_series):
    result = add_calendar_features(hourly_series)

    assert result["hour"].iloc[0] == 0
    assert result["hour"].iloc[13] == 13
    assert set(result["is_weekend"].unique()).issubset({0, 1})


def test_add_cyclical_features_bounded(hourly_series):
    calendar = add_calendar_features(hourly_series)
    result = add_cyclical_features(calendar, "hour", 24)

    assert result["hour_sin"].between(-1, 1).all()
    assert result["hour_cos"].between(-1, 1).all()


def test_chronological_split_preserves_time_order(hourly_series):
    train, val, test = chronological_split(hourly_series, train_ratio=0.5, val_ratio=0.25)

    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()
    assert len(train) + len(val) + len(test) == len(hourly_series)
