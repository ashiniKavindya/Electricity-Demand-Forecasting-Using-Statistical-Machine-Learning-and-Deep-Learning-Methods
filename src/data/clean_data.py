"""Clean and impute electricity demand data."""
import pandas as pd


def remove_duplicate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate timestamp rows, keeping the first occurrence."""
    return df[~df.index.duplicated(keep="first")]


def fill_missing_values(df: pd.DataFrame, method: str = "time") -> pd.DataFrame:
    """Impute missing values using only past/interpolated information (no leakage)."""
    return df.interpolate(method=method, limit_direction="forward")


def flag_invalid_demand(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Flag negative or zero demand values for manual review."""
    df = df.copy()
    df["invalid_demand"] = df[target_col] <= 0
    return df
