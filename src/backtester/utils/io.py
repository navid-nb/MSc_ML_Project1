"""I/O helpers: CSV ingestion."""

import pandas as pd


def read_csv_ohlcv(file_like) -> pd.DataFrame:
    """Read CSV and attempt to infer datetime index and OHLCV columns.

    Parameters
    ----------
    file_like : Any
        Path or file-like.

    Returns
    -------
    pd.DataFrame
        Dataframe with a datetime index and raw columns (not normalized).
    """
    raw = pd.read_csv(file_like)
    lower = {c.lower(): c for c in raw.columns}
    dt = next((lower[k] for k in ("datetime", "date", "timestamp") if k in lower), None)
    if dt is None:
        raise ValueError("CSV must include a Datetime/Date column.")
    raw[dt] = pd.to_datetime(raw[dt], utc=True, errors="coerce")
    raw = raw.set_index(dt)
    return raw
