"""Plotting helpers for anomaly detection results."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd

from .features import SENSOR_COLUMNS
from .metrics import find_windows


def plot_overview(
    df: pd.DataFrame,
    timestamps: pd.Series,
    y_true: pd.Series,
    y_pred: pd.Series,
    score: pd.Series,
    title: str,
    output_path: str,
) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    true_windows = find_windows(y_true)
    pred_windows = find_windows(y_pred)

    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    plot_cols = SENSOR_COLUMNS[:3]

    for ax, col in zip(axes[:3], plot_cols):
        ax.plot(timestamps, df[col], linewidth=1.1)
        for start, end in true_windows:
            ax.axvspan(timestamps.iloc[start], timestamps.iloc[end - 1], color="tomato", alpha=0.15)
        for start, end in pred_windows:
            ax.axvspan(timestamps.iloc[start], timestamps.iloc[end - 1], color="steelblue", alpha=0.12)
        ax.set_ylabel(col)
        ax.grid(alpha=0.2)

    axes[3].plot(timestamps, score, color="black", linewidth=1.0)
    axes[3].set_ylabel("anomaly_score")
    axes[3].grid(alpha=0.2)
    axes[3].set_xlabel("timestamp")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
