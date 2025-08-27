"""Default feature engineer."""

from typing import Final

import numpy as np
import pandas as pd

from backtester.features.indicators import rsi
from backtester.interfaces.feature_engineer import FeatureEngineer


class DefaultFeatureEngineer(FeatureEngineer):
    """Lightweight, fast features suitable for demos/backtests."""

    LABEL_COL: Final[str] = "y"

    def make(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Create features + supervised target.

        Parameters
        ----------
        ohlcv : pd.DataFrame
            Input OHLCV.

        Returns
        -------
        pd.DataFrame
            Feature frame including column 'y'.
        """
        feats = pd.DataFrame(index=ohlcv.index)
        c = ohlcv["Close"]
        feats["ret_1"] = c.pct_change()
        feats["ret_5"] = c.pct_change(5)
        feats["ret_20"] = c.pct_change(20)
        feats["rsi_14"] = rsi(c, 14)
        feats["ma_10"] = c.rolling(10).mean() / c - 1
        feats["vol_10"] = feats["ret_1"].rolling(10).std()
        feats[self.LABEL_COL] = np.sign(c.shift(-1) / c - 1).replace(0, 1)
        return feats.dropna()
