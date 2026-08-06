"""LSTM sequence creation and model definition."""
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def create_sequences(data: np.ndarray, target_idx: int, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert a 2D array of features into (samples, sequence_length, features) windows."""
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i - sequence_length:i])
        y.append(data[i, target_idx])
    return np.array(X), np.array(y)


def build_lstm_model(
    sequence_length: int,
    n_features: int,
    units: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
) -> keras.Model:
    """Build and compile a single-layer LSTM regression model."""
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, n_features)),
        layers.LSTM(units),
        layers.Dropout(dropout),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="mse")
    return model
