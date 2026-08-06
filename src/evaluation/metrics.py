"""Forecast accuracy metrics and significance testing."""
import numpy as np
from scipy import stats


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(np.abs(y_true - y_pred) / denom) * 100


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, season_length: int = 1) -> float:
    """MASE scaled against the in-sample seasonal naive forecast error."""
    naive_error = np.mean(np.abs(y_train[season_length:] - y_train[:-season_length]))
    return np.mean(np.abs(y_true - y_pred)) / naive_error


def diebold_mariano_test(errors_a: np.ndarray, errors_b: np.ndarray, power: int = 2) -> tuple[float, float]:
    """Diebold-Mariano test comparing forecast error series from two models.

    Returns (dm_statistic, p_value). A significant p-value indicates the
    two models' forecast accuracy differs by more than random variation.
    """
    d = np.abs(errors_a) ** power - np.abs(errors_b) ** power
    d_mean = d.mean()
    d_var = d.var(ddof=1) / len(d)
    dm_stat = d_mean / np.sqrt(d_var)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value
