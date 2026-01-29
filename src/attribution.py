"""Simple root-cause attribution helpers."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .features import SENSOR_COLUMNS


def rank_tags_from_zscores(
    zscores: dict[str, pd.Series],
    start: int,
    end: int,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    scores: list[tuple[str, float]] = []
    for tag, z in zscores.items():
        window_score = float(z.iloc[start:end].abs().max())
        scores.append((tag, window_score))
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:top_k]


def rank_tags_from_features(
    feature_frame: pd.DataFrame,
    feature_means: pd.Series,
    feature_stds: pd.Series,
    start: int,
    end: int,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    # Isolation Forest doesn't expose per-feature contributions,
    # so we approximate with feature-level z-score deviations.
    window = feature_frame.iloc[start:end]
    z = (window - feature_means) / feature_stds.replace(0, np.nan)
    max_abs = z.abs().max()

    tag_scores: dict[str, float] = defaultdict(float)
    for feature_name, score in max_abs.items():
        for tag in SENSOR_COLUMNS:
            if feature_name.startswith(tag):
                tag_scores[tag] = max(tag_scores[tag], float(score))
                break

    ranked = sorted(tag_scores.items(), key=lambda item: item[1], reverse=True)
    return ranked[:top_k]


def rank_tags_from_feature_window(
    window_frame: pd.DataFrame,
    feature_means: pd.Series,
    feature_stds: pd.Series,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    if window_frame.empty:
        return []
    z = (window_frame - feature_means) / feature_stds.replace(0, np.nan)
    max_abs = z.abs().max()

    tag_scores: dict[str, float] = defaultdict(float)
    for feature_name, score in max_abs.items():
        for tag in SENSOR_COLUMNS:
            if feature_name.startswith(tag):
                tag_scores[tag] = max(tag_scores[tag], float(score))
                break

    ranked = sorted(tag_scores.items(), key=lambda item: item[1], reverse=True)
    return ranked[:top_k]
