"""yfinance asset provider."""

from typing import Optional

import pandas as pd
import yfinance as yf
from loguru import logger

from backtester.interfaces.asset import Asset
from backtester.providers.adapter_utils import normalize_ohlcv


class YFinanceAsset(Asset):
    """Yahoo Finance data provider."""

    def __init__(self, symbol: str) -> None:
        """Construct the provider.

        Parameters
        ----------
        symbol : str
            Ticker (e.g., 'AAPL').
        """
        self.symbol = symbol

    def load(
        self,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
        interval: Optional[str],
    ) -> pd.DataFrame:
        """Load OHLCV using yfinance.

        Returns
        -------
        pd.DataFrame
            Normalized OHLCV.
        """
        logger.info(f"Downloading {self.symbol} interval={interval} start={start} end={end}")

        kw = dict(auto_adjust=False, progress=False, group_by="ticker")
        if interval:
            kw["interval"] = interval

        df = yf.download(
            self.symbol,
            start=str(start.date()) if start is not None else None,
            end=str(end.date()) if end is not None else None,
            **kw,
        )

        if df is None or len(df) == 0:
            raise ValueError("No data returned from yfinance.")

        return normalize_ohlcv(df, ticker_sym=self.symbol)
