"""Pipeline interface (optional scaffolding for training/evaluation)."""

import abc
from typing import Any, Dict

import pandas as pd


class Pipeline(abc.ABC):
    """Abstract ML pipeline interface."""

    @abc.abstractmethod
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train a model/pipeline."""
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate a model/pipeline."""
        raise NotImplementedError
