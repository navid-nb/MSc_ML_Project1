"""Decision Tree classifier."""

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from backtester.interfaces.model import Model


class DecisionTreeModel(Model):
    """Pipeline: StandardScaler -> DecisionTreeClassifier."""

    name = "Decision Tree (clf)"

    def __init__(self, max_depth: int = 4, random_state: int = 42) -> None:
        self.clf = make_pipeline(
            StandardScaler(), DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.clf.fit(X.fillna(0), y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(X.fillna(0))[:, 1]
