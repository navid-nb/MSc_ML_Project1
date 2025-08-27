"""Gaussian Naive Bayes classifier."""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backtester.interfaces.model import Model


class GaussianNBModel(Model):
    """Pipeline: StandardScaler -> GaussianNB."""

    name = "Naive Bayes (Gaussian)"

    def __init__(self) -> None:
        self.clf = make_pipeline(StandardScaler(), GaussianNB())

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.clf.fit(X.fillna(0), y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(X.fillna(0))[:, 1]
