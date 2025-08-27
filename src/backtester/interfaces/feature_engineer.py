"""Feature engineering interface."""

import abc

import pandas as pd


class FeatureEngineer(abc.ABC):
    """Abstract interface for feature engineering."""

    @abc.abstractmethod
    def make(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Build features from OHLCV dataframe.

        Parameters
        ----------
        ohlcv : pd.DataFrame
            Input OHLCV data.

        Returns
        -------
        pd.DataFrame
            Features dataframe including a supervised target column named 'y'.
        """
        raise NotImplementedError
