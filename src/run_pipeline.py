"""Entry point for the pipeline."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .baselines import combine_flags, rolling_zscore_detector


REQUIRED_COLUMNS = [
    "timestamp",
    "T_reactor_F",
    "P_reactor_psig",
    "F_feed_bbl_per_min",
    "L_drum_pct",
    "V_valve_pct",
    "anomaly_flag",
]


def _validate_data(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Found missing values in required columns.")

    # Percent columns should remain within physical limits.
    if not df["L_drum_pct"].between(0, 100).all():
        raise ValueError("L_drum_pct has values outside 0-100.")
    if not df["V_valve_pct"].between(0, 100).all():
        raise ValueError("V_valve_pct has values outside 0-100.")


def _precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    return precision, recall, f1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline anomaly detection pipeline.")
    parser.add_argument("--data", default="data/simulated_process_data.csv", help="Path to CSV data.")
    parser.add_argument("--window", type=int, default=60, help="Rolling window size.")
    parser.add_argument("--z-thresh", type=float, default=3.0, help="Z-score threshold.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.data, parse_dates=["timestamp"])
    _validate_data(df)

    # Run independent rolling z-score checks.
    t_flags = rolling_zscore_detector(df, "T_reactor_F", window=args.window, z_thresh=args.z_thresh)
    p_flags = rolling_zscore_detector(df, "P_reactor_psig", window=args.window, z_thresh=args.z_thresh)
    f_flags = rolling_zscore_detector(df, "F_feed_bbl_per_min", window=args.window, z_thresh=args.z_thresh)

    baseline_anomaly_flag = combine_flags([t_flags, p_flags, f_flags]).astype(int)
    y_true = df["anomaly_flag"].astype(int).to_numpy()
    y_pred = baseline_anomaly_flag.to_numpy()

    precision, recall, f1 = _precision_recall_f1(y_true, y_pred)
    print(f"Baseline precision: {precision:.3f}")
    print(f"Baseline recall:    {recall:.3f}")
    print(f"Baseline F1 score:  {f1:.3f}")


if __name__ == "__main__":
    main()
