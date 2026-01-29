"""Feature engineering for anomaly detection models."""

from __future__ import annotations

import numpy as np
import pandas as pd


SENSOR_COLUMNS = [
    "T_reactor_F",
    "P_reactor_psig",
    "F_feed_bbl_per_min",
    "L_drum_pct",
    "V_valve_pct",
]


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    features: dict[str, pd.Series] = {}

    for col in SENSOR_COLUMNS:
        series = df[col]
        features[col] = series

        # Rolling stats capture local drift and volatility.
        for window in (15, 60):
            roll = series.rolling(window=window)
            features[f"{col}_mean_{window}"] = roll.mean()
            features[f"{col}_std_{window}"] = roll.std()

        # First difference highlights abrupt changes.
        features[f"{col}_diff"] = series.diff()

    return pd.DataFrame(features).dropna()


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    feature_frame = build_feature_frame(df)
    return feature_frame.to_numpy(), list(feature_frame.columns)
