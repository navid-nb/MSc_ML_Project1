"""CSV asset provider."""

from typing import Optional

import pandas as pd

from backtester.interfaces.asset import Asset
from backtester.providers.adapter_utils import normalize_ohlcv, resample_ohlcv


class CSVAsset(Asset):
    """CSV-backed asset.

    Notes
    -----
    Expects a DataFrame whose index is a datetime-like column.
    """

    def __init__(self, symbol: str, df: pd.DataFrame) -> None:
        self.symbol = symbol
        # Normalize upfront so downstream is consistent
        self._df = normalize_ohlcv(df)

    def load(
        self,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
        interval: Optional[str],
    ) -> pd.DataFrame:
        """Return sliced (and optionally resampled) OHLCV."""
        out = self._df.copy()
        if start is not None or end is not None:
            out = out.loc[slice(start, end)]
        if interval and interval.upper() not in ("", "NATIVE"):
            # Only resample for non-native frequencies
            out = resample_ohlcv(out, interval)
        return out
