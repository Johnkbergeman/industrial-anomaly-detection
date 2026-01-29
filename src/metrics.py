"""Metrics for pointwise and event-level anomaly evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class EventMetrics:
    event_recall: float
    mean_time_to_detect_min: float
    median_time_to_detect_min: float
    false_alert_minutes: float
    detected_events: int
    total_events: int


def find_windows(flags: Iterable[int]) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    in_window = False
    start = 0
    for idx, value in enumerate(flags):
        if value and not in_window:
            in_window = True
            start = idx
        elif not value and in_window:
            in_window = False
            windows.append((start, idx))
    if in_window:
        windows.append((start, idx + 1))
    return windows


def pointwise_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    return precision, recall, f1


def _infer_sample_minutes(timestamps: pd.Series) -> float:
    if len(timestamps) < 2:
        return 0.0
    diffs = timestamps.diff().dt.total_seconds().dropna()
    if diffs.empty:
        return 0.0
    return float(diffs.median() / 60.0)


def event_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.Series,
) -> EventMetrics:
    true_windows = find_windows(y_true)
    pred_windows = find_windows(y_pred)
    sample_minutes = _infer_sample_minutes(timestamps)

    detected = 0
    ttd_minutes: list[float] = []
    for start, end in true_windows:
        detected_idx = None
        for idx in range(start, end):
            if y_pred[idx] == 1:
                detected_idx = idx
                break
        if detected_idx is not None:
            detected += 1
            ttd = (timestamps.iloc[detected_idx] - timestamps.iloc[start]).total_seconds() / 60.0
            ttd_minutes.append(float(ttd))

    false_alert_points = np.sum((y_pred == 1) & (y_true == 0))
    false_alert_minutes = false_alert_points * sample_minutes

    event_recall = detected / len(true_windows) if true_windows else 0.0
    mean_ttd = float(np.mean(ttd_minutes)) if ttd_minutes else 0.0
    median_ttd = float(np.median(ttd_minutes)) if ttd_minutes else 0.0

    return EventMetrics(
        event_recall=event_recall,
        mean_time_to_detect_min=mean_ttd,
        median_time_to_detect_min=median_ttd,
        false_alert_minutes=false_alert_minutes,
        detected_events=detected,
        total_events=len(true_windows),
    )
