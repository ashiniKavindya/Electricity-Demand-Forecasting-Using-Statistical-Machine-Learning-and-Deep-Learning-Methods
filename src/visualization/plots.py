"""Plotting utilities for demand series and forecast evaluation."""
import matplotlib.pyplot as plt
import pandas as pd


def plot_actual_vs_predicted(y_true: pd.Series, y_pred: pd.Series, title: str = "Actual vs Predicted"):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_true.index, y_true.values, label="Actual")
    ax.plot(y_pred.index, y_pred.values, label="Predicted", alpha=0.8)
    ax.set_title(title)
    ax.legend()
    return fig


def plot_residuals(y_true: pd.Series, y_pred: pd.Series, title: str = "Residuals"):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(residuals.index, residuals.values)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    return fig


def plot_training_history(history, title: str = "Training and Validation Loss"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history["loss"], label="Training loss")
    ax.plot(history.history["val_loss"], label="Validation loss")
    ax.set_title(title)
    ax.legend()
    return fig
