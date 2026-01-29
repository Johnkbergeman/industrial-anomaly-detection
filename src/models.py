"""Model wrappers for anomaly detection."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


def train_isolation_forest(
    X: np.ndarray,
    contamination: float = 0.02,
    random_state: int = 42,
) -> IsolationForest:
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X)
    return model


def predict_isolation_forest(
    model: IsolationForest,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # IsolationForest returns -1 for anomalies, 1 for normal.
    raw_pred = model.predict(X)
    pred_flag = (raw_pred == -1).astype(int)
    anomaly_score = -model.score_samples(X)
    return pred_flag, anomaly_score
