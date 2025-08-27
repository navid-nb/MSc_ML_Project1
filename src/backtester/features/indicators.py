"""Technical indicators."""

import numpy as np
import pandas as pd


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    """Relative Strength Index (simple, EWM).

    Parameters
    ----------
    series : pd.Series
        Price series (Close).
    n : int
        Period.

    Returns
    -------
    pd.Series
        RSI values.
    """
    delta = series.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    gain = up.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    loss = down.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))
