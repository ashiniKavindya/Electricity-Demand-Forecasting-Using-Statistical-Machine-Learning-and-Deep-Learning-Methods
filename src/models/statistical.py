"""ARIMA and SARIMA statistical forecasting models."""
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


def fit_arima(series: pd.Series, order: tuple[int, int, int]):
    """Fit an ARIMA(p, d, q) model on the training series."""
    return ARIMA(series, order=order).fit()


def fit_sarima(
    series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
):
    """Fit a SARIMA model with the given order and seasonal order.

    `low_memory=True` skips retaining the full smoothed-state history, which
    isn't needed downstream (only `.forecast()` is used) and reduces the
    O(state_dim^2 * n_obs) memory a seasonal_order=24 fit requires -- that
    matters on long hourly histories. Note: `simple_differencing=True` looks
    like it would help further (it moves (d, D) out of the state vector,
    shrinking state_dim), but it makes `.forecast()` return values still in
    differenced space instead of the original scale -- don't add it here.
    """
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    return model.fit(disp=False, low_memory=True)
