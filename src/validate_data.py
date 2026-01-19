"""Quick sanity-check analysis and baseline model for simulated process data."""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SIGNAL_COLUMNS = [
    "T_reactor_F",
    "P_reactor_psig",
    "F_feed_bbl_per_min",
    "L_drum_pct",
    "V_valve_pct",
]


def _find_windows(flags: pd.Series) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    in_window = False
    start = 0
    for i, value in enumerate(flags.to_numpy()):
        if value and not in_window:
            in_window = True
            start = i
        elif not value and in_window:
            in_window = False
            windows.append((start, i))
    if in_window:
        windows.append((start, len(flags)))
    return windows


def _plot_signals(df: pd.DataFrame, out_dir: str) -> str:
    timestamps = df["timestamp"]
    windows = _find_windows(df["anomaly_flag"])
    # not plotting everything here
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    plot_cols = ["T_reactor_F", "P_reactor_psig", "V_valve_pct"]
    for ax, col in zip(axes, plot_cols):
        ax.plot(timestamps, df[col], linewidth=1.1)
        for start, end in windows:
            ax.axvspan(timestamps.iloc[start], timestamps.iloc[end - 1], color="tomato", alpha=0.15)
        ax.set_ylabel(col)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("timestamp")
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "signal_overview.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _build_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    features = {}
    for col in SIGNAL_COLUMNS:
        roll = df[col].rolling(window=window, min_periods=1)
        features[f"{col}_mean_{window}"] = roll.mean()
        features[f"{col}_std_{window}"] = roll.std().fillna(0.0)
    return pd.DataFrame(features)


def _time_split(n_rows: int, train_frac: float) -> tuple[np.ndarray, np.ndarray]:
    cut = int(n_rows * train_frac)
    train_idx = np.arange(0, cut)
    test_idx = np.arange(cut, n_rows)
    return train_idx, test_idx


def _window_recall(y_true: pd.Series, y_pred: np.ndarray) -> float:
    windows = _find_windows(y_true)
    if not windows:
        return 0.0
    hits = 0
    for start, end in windows:
        if y_pred[start:end].any():
            hits += 1
    return hits / len(windows)


def analyze_dataset(
    data_path: str,
    out_dir: str,
    window: int,
    train_frac: float,
) -> None:
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    plot_path = _plot_signals(df, out_dir)
    
    # TODO: try a longer rolling window to see if drift detection improves
    features = _build_features(df, window)
    target = df["anomaly_flag"].astype(int)

    train_idx, test_idx = _time_split(len(df), train_frac)
    x_train, y_train = features.iloc[train_idx], target.iloc[train_idx]
    x_test, y_test = features.iloc[test_idx], target.iloc[test_idx]

    # keeping this simple on purpose before trying anything fancier
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    normal_mask = y_test == 0
    false_positive_rate = (y_pred[normal_mask] == 1).mean() if normal_mask.any() else 0.0
    window_recall = _window_recall(y_test.reset_index(drop=True), y_pred)

    print(f"Saved plot: {plot_path}")
    print(f"False positives during normal operation: {false_positive_rate:.3f}")
    print(f"Recall on anomaly windows: {window_recall:.3f}")
    print(classification_report(y_test, y_pred, digits=3))

    if false_positive_rate > 0.05:
        fp_note = "Model is a bit jumpy on normal data."
    else:
        fp_note = "False positives look reasonable for a first pass."
    if window_recall < 0.7:
        recall_note = "Some anomaly windows are being missed."
    else:
        recall_note = "Most anomaly windows are being detected."
    print("Summary:", fp_note, recall_note)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity-check the simulated dataset.")
    parser.add_argument("--data", required=True, help="Path to simulated CSV data.")
    parser.add_argument("--out-dir", default="artifacts", help="Directory for plots and outputs.")
    parser.add_argument("--window", type=int, default=30, help="Rolling window size in samples.")
    parser.add_argument("--train-frac", type=float, default=0.7, help="Training fraction for time split.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    analyze_dataset(args.data, args.out_dir, args.window, args.train_frac)


if __name__ == "__main__":
    main()
