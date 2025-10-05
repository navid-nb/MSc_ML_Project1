from typing import Optional, Union

import pandas as pd
import pandas_ta as ta

pd.set_option("future.no_silent_downcasting", True)


def _join_ind_result_to_out(
    df: pd.DataFrame,
    ind_result: Union[pd.Series, pd.DataFrame],
    col_name: Optional[str] = None,
    prefix: str = 'ti_'
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
        col = f"{prefix}{col_name}"
        df[col] = result
    elif isinstance(result, pd.DataFrame):
        # Prefix columns with 'ti_'
        result.columns = [f"{prefix}{col}" for col in result.columns]
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
    prefix: str = 'ti_',
    # Params
    rsi_periods: list[int] = [14],
    atr_periods: list[int] = [14, 7],
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_period: int = 20,
    bb_std: float = 2.0,
    mfi_periods: list[int] = [14],
    adx_periods: list[int] = [14],
    psar_step: float = 0.02,
    psar_max: float = 0.2,
    cmf_periods: list[int] = [20],
    eom_periods: list[int] = [14],
    variance_periods: list[int] = [21],
    stoch_k: int = 14,
    stoch_d: int = 3,
    stoch_smooth: int = 3,
    skew_periods: list[int] = [63],
    kurtosis_periods: list[int] = [63],
    aroon_periods: list[int] = [25],
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
    # RSI
    for period in rsi_periods:
        rsi_result = gb[close_col].apply(lambda x: ta.rsi(x, length=period))
        out = _join_ind_result_to_out(out, rsi_result, col_name=f"rsi_{period}",prefix=prefix)

    # ATR
    for period in atr_periods:
        atr_result = gb.apply(
            lambda g: ta.atr(high=g[high_col], low=g[low_col], close=g[close_col], length=period)
        )
        out = _join_ind_result_to_out(out, atr_result, col_name=f"atr_{period}",prefix=prefix)

    # MACD - only keep the main MACD line (12, 26, 9)
    macd_result = gb.apply(
        lambda g: ta.macd(g[close_col], fast=macd_fast, slow=macd_slow, signal=macd_signal)[
            f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"
        ]
    )
    out = _join_ind_result_to_out(
        out, macd_result, col_name=f"MACD_{macd_fast}_{macd_slow}_{macd_signal}",prefix=prefix
    )

    # Bollinger Bands - only keep the percent B (position within bands)
    bb_result = gb[close_col].apply(
        lambda x: ta.bbands(x, length=bb_period, std=bb_std)[f"BBP_{bb_period}_{bb_std}"]
    )
    out = _join_ind_result_to_out(out, bb_result, col_name=f"bb_percent_{bb_period}_{int(bb_std)}",prefix=prefix)

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
        out = _join_ind_result_to_out(out, mfi_result, col_name=f"mfi_{length}",prefix=prefix)

    # ADX (Average Directional Index) - returns DataFrame, extract ADX column only
    for length in adx_periods:
        adx_result = gb.apply(
            lambda g: ta.adx(high=g[high_col], low=g[low_col], close=g[close_col], length=length)[
                f"ADX_{length}"
            ]
        )
        out = _join_ind_result_to_out(out, adx_result, col_name=f"adx_{length}",prefix=prefix)

    # PSAR (Parabolic SAR) - returns DataFrame, extract acceleration factor column
    psar_af_col = f"PSARaf_{psar_step}_{psar_max}"
    psar_result = gb.apply(
        lambda g: ta.psar(
            high=g[high_col], low=g[low_col], close=g[close_col], step=psar_step, max=psar_max
        )[psar_af_col]
    )
    out = _join_ind_result_to_out(out, psar_result, col_name="psar_acc",prefix=prefix)

    # CMF (Chaikin Money Flow) - returns Series
    for length in cmf_periods:
        cmf_result = gb.apply(
            lambda g: ta.cmf(
                high=g[high_col],
                low=g[low_col],
                close=g[close_col],
                volume=g[vol_col],
                open_=g[open_col],
                length=length,
            )
        )
        out = _join_ind_result_to_out(out, cmf_result, col_name=f"cmf_{length}",prefix=prefix)

    # EOM (Ease of Movement)
    for length in eom_periods:
        eom_result = gb.apply(
            lambda g: ta.eom(
                high=g[high_col],
                low=g[low_col],
                close=g[close_col],
                volume=g[vol_col],
                length=length,
            )
        )
        out = _join_ind_result_to_out(out, eom_result, col_name=f"eom_{length}",prefix=prefix)

    # Variance
    for length in variance_periods:
        variance_result = gb[close_col].apply(lambda x: ta.variance(x, length=length))
        out = _join_ind_result_to_out(out, variance_result, col_name=f"variance_{length}",prefix=prefix)

    # Stochastic Oscillator - returns DataFrame, extract STOCHk column only
    stoch_k_col = f"STOCHk_{stoch_k}_{stoch_d}_{stoch_smooth}"
    stoch_result = gb.apply(
        lambda g: ta.stoch(
            high=g[high_col],
            low=g[low_col],
            close=g[close_col],
            k=stoch_k,
            d=stoch_d,
            smooth_k=stoch_smooth,
        )[stoch_k_col]
    )
    out = _join_ind_result_to_out(
        out, stoch_result, col_name=f"stoch_k_{stoch_k}_{stoch_d}_{stoch_smooth}",prefix=prefix
    )

    # Skewness
    for length in skew_periods:
        skew_result = gb[close_col].apply(lambda x: ta.skew(x, length=length))
        out = _join_ind_result_to_out(out, skew_result, col_name=f"skew_{length}",prefix=prefix)

    # Kurtosis
    for length in kurtosis_periods:
        kurtosis_result = gb[close_col].apply(lambda x: ta.kurtosis(x, length=length))
        out = _join_ind_result_to_out(out, kurtosis_result, col_name=f"kurtosis_{length}",prefix=prefix)

    # Aroon Oscillator - returns DataFrame, extract AROONOSC column only
    for length in aroon_periods:
        aroon_osc_col = f"AROONOSC_{length}"
        aroon_result = gb.apply(
            lambda g: ta.aroon(high=g[high_col], low=g[low_col], length=length)[aroon_osc_col]
        )
        out = _join_ind_result_to_out(out, aroon_result, col_name=f"aroon_osc_{length}",prefix=prefix)

    return out



def feature_augmentaion(df):
    out=df.copy()
    #1) add technical indicators on tickers 
    out=add_technical_indicators(out)

    #2) add technical indicators on common features (indexes and etfs)
    # Collect base keys (like ^VIX, ^VXN, XLF, etc.) from columns starting with 'comm_'
    comm_keys = set()
    for col in df.columns:
        if col.startswith('comm_'):
            parts = col.split('_')
            # Join together all parts between 'comm' and the last part (which is open/high/low/close/volume)
            # Works for e.g. comm_^VIX_close or comm_XLF_volume
            if len(parts) > 2:
                key = '_'.join(parts[1:-1])
                comm_keys.add(key)

    for comm in sorted(comm_keys):
        open_col   = f'comm_{comm}_open'
        high_col   = f'comm_{comm}_high'
        low_col    = f'comm_{comm}_low'
        close_col  = f'comm_{comm}_close'
        vol_col    = f'comm_{comm}_volume'
        # Only call if all required columns are present
        if all(col in df.columns for col in [open_col, high_col, low_col, close_col, vol_col]):
            out=add_technical_indicators(
                out,
                open_col=open_col,
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
                vol_col=vol_col,
                prefix=f"comm_ti_{comm}_"  # e.g., comm_ti_^VIX_
            )
    #removing columns that are all NaN (e.g. indexes dont have volume data so eom and cmf ouputs are all NaN)
    out = out.dropna(axis=1, how='all')
    return out