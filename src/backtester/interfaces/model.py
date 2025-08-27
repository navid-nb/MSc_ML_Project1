"""Model interface (classification proba)."""

import abc

import numpy as np
import pandas as pd


class Model(abc.ABC):
    """Abstract ML model API.

    Methods
    -------
    fit(X, y) -> None
        Train model on features/labels.
    predict_proba(X) -> np.ndarray
        Probability of positive class per row (shape: [n]).

    Attributes
    ----------
    name : str
        Public name used in UI/CLI.
    """

    name: str

    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model.

        Parameters
        ----------
        X : pd.DataFrame
            Features.
        y : pd.Series
            Binary labels (0/1).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of positive class.

        Parameters
        ----------
        X : pd.DataFrame
            Features.

        Returns
        -------
        np.ndarray
            Array of probabilities in [0, 1].
        """
        raise NotImplementedError
