"""Model wrappers for anomaly detection."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


SUPPORTED_MODELS = (
    "isolation_forest",
    "oneclass_svm",
    "local_outlier_factor",
)


def _clamp_contamination(contamination: float) -> float:
    # Keep values inside the practical range expected by sklearn estimators.
    return float(min(max(contamination, 1e-4), 0.5))


def train_detector(
    model_name: str,
    X: np.ndarray,
    contamination: float = 0.02,
    random_state: int = 42,
):
    if model_name == "isolation_forest":
        return train_isolation_forest(X, contamination=contamination, random_state=random_state)
    if model_name == "oneclass_svm":
        return train_oneclass_svm(X, contamination=contamination)
    if model_name == "local_outlier_factor":
        return train_local_outlier_factor(X, contamination=contamination)
    raise ValueError(f"Unsupported model '{model_name}'. Supported: {SUPPORTED_MODELS}")


def predict_detector(model_name: str, model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if model_name == "isolation_forest":
        return predict_isolation_forest(model, X)
    if model_name == "oneclass_svm":
        return predict_oneclass_svm(model, X)
    if model_name == "local_outlier_factor":
        return predict_local_outlier_factor(model, X)
    raise ValueError(f"Unsupported model '{model_name}'. Supported: {SUPPORTED_MODELS}")


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


def train_oneclass_svm(
    X: np.ndarray,
    contamination: float = 0.02,
) -> OneClassSVM:
    model = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=_clamp_contamination(contamination),
    )
    model.fit(X)
    return model


def predict_oneclass_svm(
    model: OneClassSVM,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw_pred = model.predict(X)
    pred_flag = (raw_pred == -1).astype(int)
    anomaly_score = -model.score_samples(X)
    return pred_flag, anomaly_score


def train_local_outlier_factor(
    X: np.ndarray,
    contamination: float = 0.02,
) -> LocalOutlierFactor:
    n_samples = X.shape[0]
    n_neighbors = int(np.sqrt(n_samples))
    n_neighbors = min(max(n_neighbors, 5), 35, max(n_samples - 1, 1))

    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=_clamp_contamination(contamination),
        novelty=True,
    )
    model.fit(X)
    return model


def predict_local_outlier_factor(
    model: LocalOutlierFactor,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw_pred = model.predict(X)
    pred_flag = (raw_pred == -1).astype(int)
    anomaly_score = -model.score_samples(X)
    return pred_flag, anomaly_score
