"""Training pipeline (framework-agnostic, no Streamlit dependencies).

This module performs a simple temporal train/test split, fits a model on the
train slice, and reports basic classification metrics on both train/test sets.

It does not persist artifacts by default (you can extend with joblib/MLflow).
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from backtester.interfaces.feature_engineer import FeatureEngineer
from backtester.interfaces.model import Model


@dataclass
class TrainConfig:
    """Configuration for model training.

    Attributes
    ----------
    split_ratio : float
        Fraction of rows used for training (remainder used for testing).
    threshold : float
        Probability threshold to convert to class labels for reporting.
    """

    split_ratio: float = 0.7
    threshold: float = 0.5


def _bin_labels_from_proba(proba: np.ndarray, threshold: float) -> np.ndarray:
    """Convert probabilities to binary labels.

    Parameters
    ----------
    proba : np.ndarray
        Array of probabilities in [0, 1].
    threshold : float
        Decision threshold.

    Returns
    -------
    np.ndarray
        Binary labels (0/1).
    """
    return (proba >= threshold).astype(int)


def train(
    ohlcv: pd.DataFrame,
    feature_engineer: FeatureEngineer,
    model: Model,
    cfg: TrainConfig | None = None,
) -> Tuple[Dict[str, Any], Model]:
    """Train a model using a temporal split and report metrics.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Input OHLCV dataframe.
    feature_engineer : FeatureEngineerInterface
        Feature builder instance that returns a frame including column 'y'.
    model : ModelInterface
        Unfitted model instance to train.
    cfg : TrainConfig, optional
        Training configuration. Defaults to TrainConfig().

    Returns
    -------
    Tuple[Dict[str, Any], ModelInterface]
        (metrics_dict, fitted_model)

        metrics_dict includes:
        - 'train/size'
        - 'test/size'
        - 'train/accuracy'
        - 'test/accuracy'
        - 'threshold'
    """
    cfg = cfg or TrainConfig()
    feats = feature_engineer.make(ohlcv)
    if len(feats) < 50:
        raise ValueError("Not enough rows after feature engineering (need >= 50).")

    split_idx = int(len(feats) * cfg.split_ratio)
    X = feats.drop(columns=["y"])
    y = (feats["y"] > 0).astype(int)

    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]

    logger.info(f"Training {model.name} on {len(X_train)} rows; validating on {len(X_test)} rows")

    model.fit(X_train.fillna(0), y_train)

    # Train metrics
    p_train = model.predict_proba(X_train.fillna(0))
    yhat_train = _bin_labels_from_proba(p_train, cfg.threshold)
    train_acc = float((yhat_train == y_train.values).mean()) if len(y_train) else float("nan")

    # Test metrics
    p_test = model.predict_proba(X_test.fillna(0))
    yhat_test = _bin_labels_from_proba(p_test, cfg.threshold)
    test_acc = float((yhat_test == y_test.values).mean()) if len(y_test) else float("nan")

    metrics = {
        "train/size": int(len(X_train)),
        "test/size": int(len(X_test)),
        "train/accuracy": train_acc,
        "test/accuracy": test_acc,
        "threshold": cfg.threshold,
    }
    logger.info(f"Train metrics: {metrics}")

    return metrics, model
