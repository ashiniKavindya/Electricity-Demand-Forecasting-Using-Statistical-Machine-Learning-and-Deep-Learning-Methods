"""Random Forest and XGBoost forecasting models."""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def fit_random_forest(X_train: pd.DataFrame, y_train: pd.Series, **params) -> RandomForestRegressor:
    """Fit a Random Forest regressor on engineered features."""
    model = RandomForestRegressor(random_state=42, **params)
    return model.fit(X_train, y_train)


def fit_xgboost(X_train: pd.DataFrame, y_train: pd.Series, **params) -> XGBRegressor:
    """Fit an XGBoost regressor on engineered features."""
    model = XGBRegressor(random_state=42, **params)
    return model.fit(X_train, y_train)
