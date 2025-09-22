from itertools import product
from typing import Optional, Union

import pandas as pd
import pandas_ta as ta

pd.set_option("future.no_silent_downcasting", True)


def _join_ind_result_to_out(
    df: pd.DataFrame,
    ind_result: Union[pd.Series, pd.DataFrame],
    col_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Join the result of a technical indicator calculation back into the main DataFrame.
    Handles both Series and DataFrame inputs, aligning indexes and naming columns.
    Must provide col_name in case of Series. the column label will be the col_name prefixed with "ti_".
    For DataFrames, uses existing labels and prefixes them with "ti_". no need for col_name.

    Args:
        df (pd.DataFrame): The original DataFrame with multi-index.
        ind_result (pd.Series or pd.DataFrame): Output from an indicator calculation.
        col_name (str, optional): Base name for the column if ind_result is a Series.

    Returns:
        pd.DataFrame: DataFrame with the new indicator column(s) added.
    """
    # Align indicator result index with df index
    result = ind_result.reindex(df.index)

    if isinstance(result, pd.Series):
        col = f"ti_{col_name}"
        df[col] = result
    elif isinstance(result, pd.DataFrame):
        # Prefix columns with 'ti_'
        result.columns = [f"ti_{col}" for col in result.columns]
        df = df.join(result)
    else:
        raise TypeError("Indicator function must return a Series or DataFrame")

    return df


def add_technical_indicators(
    df: pd.DataFrame,
    *,
    open_col: str = "openprc",
    high_col: str = "askhi",
    low_col: str = "bidlo",
    close_col: str = "adj_prc",
    vol_col: str = "vol",
    permno_idx: str = "permno",
    date_idx: str = "date",
    # Params
    sma_periods: list[int] = [10, 20],
    ema_periods: list[int] = [10, 20],
    rsi_periods: list[int] = [14, 28],
    atr_periods: list[int] = [14, 20, 30],
    macd_fast_list: list[int] = [12, 10],
    macd_slow_list: list[int] = [26],
    macd_signal_list: list[int] = [9],
    bb_periods: list[int] = [20],
    bb_stds: list[float] = [2.0],
    mfi_periods: list[int] = [14],
) -> pd.DataFrame:
    """
    Compute a variety of technical indicators per permno, based on OHLCV data,
    and append results to the DataFrame with column names prefixed.

    Args:
        df (pd.DataFrame): Input stock price data with MultiIndex (permno, date).
        Many optional parameters to specify indicator lengths.

    Returns:
        pd.DataFrame: DataFrame with appended technical indicator columns.
    """
    required = [open_col, high_col, low_col, close_col, vol_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"add_technical_indicators: missing required columns: {missing}")

    if (
        not isinstance(df.index, pd.MultiIndex)
        or permno_idx not in df.index.names
        or date_idx not in df.index.names
    ):
        raise KeyError("DataFrame must have a MultiIndex with permno and date as levels")

    # Sort by permno and date index levels
    df = df.sort_index(level=[permno_idx, date_idx])

    out = df.copy()
    gb = out.groupby(level=permno_idx, group_keys=False)

    # Calculate indicators and join results
    # SMA
    for period in sma_periods:
        sma_result = gb[close_col].apply(lambda x: ta.sma(x, length=period))
        out = _join_ind_result_to_out(out, sma_result, col_name=f"sma_{period}")

    # EMA
    for period in ema_periods:
        ema_result = gb[close_col].apply(lambda x: ta.ema(x, length=period))
        out = _join_ind_result_to_out(out, ema_result, col_name=f"ema_{period}")

    # RSI
    for period in rsi_periods:
        rsi_result = gb[close_col].apply(lambda x: ta.rsi(x, length=period))
        out = _join_ind_result_to_out(out, rsi_result, col_name=f"rsi_{period}")

    # ATR
    for period in atr_periods:
        atr_result = gb.apply(
            lambda g: ta.atr(high=g[high_col], low=g[low_col], close=g[close_col], length=period)
        )
        out = _join_ind_result_to_out(out, atr_result, col_name=f"atr_{period}")

    # MACD
    for macd_fast, macd_slow, macd_signal in product(
        macd_fast_list, macd_slow_list, macd_signal_list
    ):
        macd_result = gb.apply(
            lambda g: ta.macd(g[close_col], fast=macd_fast, slow=macd_slow, signal=macd_signal)
        )
        out = _join_ind_result_to_out(out, macd_result)

    for length, std in product(bb_periods, bb_stds):
        bb_result = gb[close_col].apply(lambda x: ta.bbands(x, length=length, std=std))
        out = _join_ind_result_to_out(out, bb_result, col_name=f"bbands_{length}_{std}")

    for length in mfi_periods:
        mfi_result = gb.apply(
            lambda g: ta.mfi(
                high=g[high_col],
                low=g[low_col],
                close=g[close_col],
                volume=g[vol_col],
                length=length,
            )
        )
        out = _join_ind_result_to_out(out, mfi_result, col_name=f"mfi_{length}")

    return out
