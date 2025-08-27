"""Asset data source interface."""

import abc
from typing import Optional

import pandas as pd


class Asset(abc.ABC):
    """Abstract interface for market data assets.

    Methods
    -------
    load(start, end, interval) -> pd.DataFrame
        Load OHLCV data within a time range and frequency.

    Notes
    -----
    Implementations must return a DataFrame indexed by datetime with
    columns: ['Open', 'High', 'Low', 'Close', 'Volume'].
    """

    symbol: str

    @abc.abstractmethod
    def load(
        self,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
        interval: Optional[str],
    ) -> pd.DataFrame:
        """Load OHLCV data.

        Parameters
        ----------
        start : Optional[pd.Timestamp]
            Inclusive start.
        end : Optional[pd.Timestamp]
            Inclusive end.
        interval : Optional[str]
            Frequency (e.g., '1d', '1h').

        Returns
        -------
        pd.DataFrame
            OHLCV dataframe.
        """
        raise NotImplementedError
