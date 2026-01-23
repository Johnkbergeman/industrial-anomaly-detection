"""Baseline anomaly detection utilities."""

from __future__ import annotations
import numpy as np
import pandas as pd


def rolling_zscore_detector(
    df: pd.DataFrame,
    column: str,
    window: int = 60,
    z_thresh: float = 3.0,
) -> pd.Series:
    roll = df[column].rolling(window=window, min_periods=window)
    mean = roll.mean()
    std = roll.std().replace(0, np.nan)
    z = (df[column] - mean) / std
    return (z.abs() > z_thresh).fillna(False)

def combine_flags(flag_series_list: list[pd.Series]) -> pd.Series:
    if not flag_series_list:
        raise ValueError("flag_series_list must not be empty")

    combined = flag_series_list[0]
    for flags in flag_series_list[1:]:
        combined = combined | flags

    return combined
