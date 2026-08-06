"""Load raw electricity demand and weather data."""
import pandas as pd


def load_raw_data(path: str, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Load a CSV file and parse the timestamp column."""
    df = pd.read_csv(path, parse_dates=[timestamp_col])
    return df.sort_values(timestamp_col).reset_index(drop=True)


def set_timestamp_index(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Set the timestamp column as a sorted DatetimeIndex."""
    df = df.set_index(timestamp_col)
    return df.sort_index()


def detect_missing_intervals(df: pd.DataFrame, freq: str = "H") -> pd.DatetimeIndex:
    """Return timestamps missing from the expected regular frequency."""
    full_range = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    return full_range.difference(df.index)


def chronological_split(
    df: pd.DataFrame, train_ratio: float, val_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-indexed DataFrame into train/val/test by position, not randomly."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]
