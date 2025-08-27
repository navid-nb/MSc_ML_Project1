"""Buy & Hold benchmark model."""

import numpy as np
import pandas as pd

from backtester.interfaces.model import Model


class BuyHoldModel(Model):
    """Always-long benchmark."""

    name = "Buy & Hold (benchmark)"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """No-op for benchmark."""
        return

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return ones."""
        return np.ones(len(X), dtype=float)
