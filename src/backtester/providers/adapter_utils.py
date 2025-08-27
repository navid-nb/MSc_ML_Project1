"""Provider utilities: normalization/resample."""

from typing import Optional

import pandas as pd


def normalize_ohlcv(df: pd.DataFrame, ticker_sym: Optional[str] = None) -> pd.DataFrame:
    """Normalize OHLCV columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data (possibly MultiIndex from yfinance).
    ticker_sym : Optional[str]
        If present, slice symbol level first.

    Returns
    -------
    pd.DataFrame
        dataset with ['Open','High','Low','Close','Volume'] and datetime index.
    """
    if isinstance(df.columns, pd.MultiIndex):
        if ticker_sym:
            for lvl in range(df.columns.nlevels):
                if ticker_sym in df.columns.get_level_values(lvl):
                    df = df.xs(ticker_sym, axis=1, level=lvl, drop_level=True)
                    break

        if isinstance(df.columns, pd.MultiIndex):
            chosen_lvl = None
            for lvl in range(df.columns.nlevels):
                vals = [str(x).strip().lower() for x in df.columns.get_level_values(lvl)]
                if any(k in vals for k in ("open", "high", "low", "close", "adj close", "volume")):
                    chosen_lvl = lvl
                    break
            if chosen_lvl is not None:
                other_lvls = [i for i in range(df.columns.nlevels) if i != chosen_lvl]
                df = df.droplevel(other_lvls, axis=1)
            else:
                last_vals = df.columns.get_level_values(-1)
                unique = last_vals.unique()
                df = (
                    df.droplevel(-1, axis=1)
                    if len(unique) == 1
                    else df.xs(unique[0], axis=1, level=-1, drop_level=True)
                )

    df = df.rename(columns=lambda c: str(c).strip().title())
    cols = set(df.columns)
    if "Close" not in cols and "Adj Close" in cols:
        df["Close"] = df["Adj Close"]
    for c in ["Open", "High", "Low", "Close"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    out = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    out.index = pd.to_datetime(out.index, errors="coerce").tz_localize(None)
    out = out[~out.index.isna()].sort_index()
    out.index.name = "Date"
    return out


def resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample OHLCV properly.

    Parameters
    ----------
    df : pd.DataFrame
        Input OHLCV data.
    freq : str
        Pandas offset (e.g. '1D', '1H', '5T').

    Returns
    -------
    pd.DataFrame
        Resampled OHLCV.
    """
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    return df.resample(freq).agg(agg).dropna(how="any")
