from typing import Optional, Union

import numpy as np
import pandas as pd
import pandas_ta as ta

pd.set_option("future.no_silent_downcasting", True)


def _join_ind_result_to_out(
    df: pd.DataFrame,
    ind_result: Union[pd.Series, pd.DataFrame],
    col_name: Optional[str] = None,
    prefix: str = "ti_",
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


def _add_technical_indicators(
    df: pd.DataFrame,
    *,
    open_col: str = "openprc",
    high_col: str = "askhi",
    low_col: str = "bidlo",
    close_col: str = "adj_prc",
    vol_col: str = "vol",
    permno_idx: str = "permno",
    date_idx: str = "date",
    prefix: str = "ti_",
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
        out = _join_ind_result_to_out(out, rsi_result, col_name=f"rsi_{period}", prefix=prefix)

    # ATR
    for period in atr_periods:
        atr_result = gb.apply(
            lambda g: ta.atr(high=g[high_col], low=g[low_col], close=g[close_col], length=period)
        )
        out = _join_ind_result_to_out(out, atr_result, col_name=f"atr_{period}", prefix=prefix)

    # MACD - only keep the main MACD line (12, 26, 9)
    macd_result = gb.apply(
        lambda g: ta.macd(g[close_col], fast=macd_fast, slow=macd_slow, signal=macd_signal)[
            f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"
        ]
    )
    out = _join_ind_result_to_out(
        out, macd_result, col_name=f"MACD_{macd_fast}_{macd_slow}_{macd_signal}", prefix=prefix
    )

    # Bollinger Bands - only keep the percent B (position within bands)
    bb_result = gb[close_col].apply(
        lambda x: ta.bbands(x, length=bb_period, std=bb_std)[f"BBP_{bb_period}_{bb_std}"]
    )
    out = _join_ind_result_to_out(
        out, bb_result, col_name=f"bb_percent_{bb_period}_{int(bb_std)}", prefix=prefix
    )

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
        out = _join_ind_result_to_out(out, mfi_result, col_name=f"mfi_{length}", prefix=prefix)

    # ADX (Average Directional Index) - returns DataFrame, extract ADX column only
    for length in adx_periods:
        adx_result = gb.apply(
            lambda g: ta.adx(high=g[high_col], low=g[low_col], close=g[close_col], length=length)[
                f"ADX_{length}"
            ]
        )
        out = _join_ind_result_to_out(out, adx_result, col_name=f"adx_{length}", prefix=prefix)

    # PSAR (Parabolic SAR) - returns DataFrame, extract acceleration factor column
    psar_af_col = f"PSARaf_{psar_step}_{psar_max}"
    psar_result = gb.apply(
        lambda g: ta.psar(
            high=g[high_col], low=g[low_col], close=g[close_col], step=psar_step, max=psar_max
        )[psar_af_col]
    )
    out = _join_ind_result_to_out(out, psar_result, col_name="psar_acc", prefix=prefix)

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
        out = _join_ind_result_to_out(out, cmf_result, col_name=f"cmf_{length}", prefix=prefix)

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
        out = _join_ind_result_to_out(out, eom_result, col_name=f"eom_{length}", prefix=prefix)

    # Variance
    for length in variance_periods:
        variance_result = gb[close_col].apply(lambda x: ta.variance(x, length=length))
        out = _join_ind_result_to_out(
            out, variance_result, col_name=f"variance_{length}", prefix=prefix
        )

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
        out, stoch_result, col_name=f"stoch_k_{stoch_k}_{stoch_d}_{stoch_smooth}", prefix=prefix
    )

    # Skewness
    for length in skew_periods:
        skew_result = gb[close_col].apply(lambda x: ta.skew(x, length=length))
        out = _join_ind_result_to_out(out, skew_result, col_name=f"skew_{length}", prefix=prefix)

    # Kurtosis
    for length in kurtosis_periods:
        kurtosis_result = gb[close_col].apply(lambda x: ta.kurtosis(x, length=length))
        out = _join_ind_result_to_out(
            out, kurtosis_result, col_name=f"kurtosis_{length}", prefix=prefix
        )

    # Aroon Oscillator - returns DataFrame, extract AROONOSC column only
    for length in aroon_periods:
        aroon_osc_col = f"AROONOSC_{length}"
        aroon_result = gb.apply(
            lambda g: ta.aroon(high=g[high_col], low=g[low_col], length=length)[aroon_osc_col]
        )
        out = _join_ind_result_to_out(
            out, aroon_result, col_name=f"aroon_osc_{length}", prefix=prefix
        )

    return out


def _add_technical_indicators_on_common_features(
    df: pd.DataFrame, tickers: list[str]
) -> pd.DataFrame:
    """
    Add technical indicators to common features (indexes/ETFs) for a specified list of tickers.

    Parameters:
        df (pd.DataFrame): Input DataFrame with columns like 'comm_^VIX_close', etc.
        tickers (list of str): List of tickers to process (e.g., ['^VIX', '^GSPC']).

    Returns:
        pd.DataFrame: DataFrame with TI columns added.
    """
    out = df.copy()

    for ticker in tickers:
        open_col = f"comm_{ticker}_open"
        high_col = f"comm_{ticker}_high"
        low_col = f"comm_{ticker}_low"
        close_col = f"comm_{ticker}_close"
        vol_col = f"comm_{ticker}_volume"

        # Only add TIs if all price/volume columns exist
        if all(col in out.columns for col in [open_col, high_col, low_col, close_col, vol_col]):
            out = _add_technical_indicators(
                out,
                open_col=open_col,
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
                vol_col=vol_col,
                prefix=f"comm_ti_{ticker}_",  # e.g., comm_ti_^VIX_
            )

    # Remove columns that are all NaN (e.g., CMF, EOM if volume is missing)
    out = out.dropna(axis=1, how="all")

    return out


def add_lagged_columns(df: pd.DataFrame, lag_configs: dict) -> pd.DataFrame:
    """
    Add lagged versions of specified columns to the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        lag_configs (dict): Dictionary where:
            - key = column name
            - value = int or list of ints (lag(s) to create)
            Example: {'retx': 1, 'mktrf': [1, 2, 3], 'cons_mean': 1}

    Returns:
        pd.DataFrame: DataFrame with lagged columns added.
    """
    out = df.copy()

    for col, lags in lag_configs.items():
        # Normalize lags to list
        lag_list = [lags] if isinstance(lags, int) else lags

        for lag in lag_list:
            lag_col = f"{col}_lag{lag}"
            out[lag_col] = out.groupby(level="permno")[col].shift(lag)  # Respect permno groups

    return out


def feature_augmentaion(df: pd.DataFrame) -> pd.DataFrame:
    df.to_csv("df.csv")
    out = df.copy()
    # ======================================================================================
    # 1) add technical indicators on tickers
    # ======================================================================================
    out = _add_technical_indicators(out)

    # ======================================================================================
    # 2) add technical indicators on common features (indexes and etfs)
    # ======================================================================================

    # the technical indicators only for common features in tickers_list will be added
    tickers_list = ["^VIX", "^GSPC"]
    out = _add_technical_indicators_on_common_features(out, tickers=tickers_list)

    # ======================================================================================
    # 3) ADD log return columns
    # ======================================================================================
    # Columns to compute log returns for
    add_log_ret_columns = [
        "adj_prc",
        "adj_mktcap",
        "vol",
        "comm_^GSPC_close",
        "comm_^IXIC_close",
        "comm_^RUT_close",
        "comm_XLK_close",
        "comm_XLF_close",
        "comm_XLE_close",
        "comm_XLV_close",
        "comm_XLI_close",
        "comm_^VIX_close",
        "comm_^VXN_close",
        "comm_^OVX_close",
        "comm_^GVZ_close",
    ]
    for col in add_log_ret_columns:
        out[f"{col}_logret"] = np.log(out[col] / out.groupby(level="permno")[col].shift(1))

    # ======================================================================================
    # 4) ADD meaningful RATIO FEATURES (ALL COLUMNS ASSUMED PRESENT)
    # ======================================================================================

    # VIX / S&P 500 Ratio: Market Fear Relative to Price Level
    # High ratio = elevated fear relative to market level (crash risk)
    # Low ratio = complacency
    out["ratio_^VIX_^GSPC"] = out["comm_^VIX_close"] / out["comm_^GSPC_close"]

    # Sector Relative Strength: Performance of each sector vs. S&P 500
    # Rising ratio = sector outperformance (money flowing in)
    # Falling ratio = underperformance (rotation out)
    sector_etfs = ["XLK", "XLF", "XLE", "XLV", "XLI"]
    for etf in sector_etfs:
        etf_col = f"comm_{etf}_close"
        out[f"ratio_{etf}_^GSPC"] = out[etf_col] / out["comm_^GSPC_close"]

    # Implied Volatility Minus Realized Volatility (IV - RV): Option Mispricing Signal
    # Positive = options are expensive (hedging demand, fear)
    # Negative = options are cheap (complacency, low hedging)
    ret_spx = np.log(out["comm_^GSPC_close"] / out["comm_^GSPC_close"].shift(1))
    rv_30d = ret_spx.rolling(30).std() * np.sqrt(252)  # 30-day annualized realized vol
    out["ratio_volatility_premium"] = out["comm_^VIX_close"] - rv_30d

    # Market Beta Proxy: Stock sensitivity to market (risk-adjusted return)
    # High beta = volatile, low beta = defensive
    # Avoid division by zero (rare, but possible if SPX unchanged)
    out["ratio_beta_proxy"] = out["adj_prc_logret"] / (out["comm_^GSPC_close_logret"] + 1e-8)

    # ======================================================================================
    # 5) ADD lagged columns
    # ======================================================================================

    # Columns to add lagged versions of
    lag_configs = {
        # Price & Market Returns (Momentum: 1-day , 5-day = weekly)
        "adj_prc_logret": [1, 2, 3, 4, 5],
        "comm_^GSPC_close_logret": [1, 5],
        "comm_^IXIC_close_logret": [1, 5],
        "comm_^RUT_close_logret": [1, 5],
        # Sector Returns (Sector Rotation: 1-day)
        "comm_XLK_close_logret": 1,
        "comm_XLF_close_logret": 1,
        "comm_XLE_close_logret": 1,
        "comm_XLV_close_logret": 1,
        "comm_XLI_close_logret": 1,
        # Macro Ratios (Regime signals: 1-day)
        "ratio_^VIX_^GSPC": 1,
        "ratio_XLK_^GSPC": 1,
        "ratio_XLF_^GSPC": 1,
        "ratio_XLE_^GSPC": 1,
        "ratio_XLV_^GSPC": 1,
        "ratio_XLI_^GSPC": 1,
        "ratio_volatility_premium": 1,
    }
    out = add_lagged_columns(out, lag_configs=lag_configs)

    # ======================================================================================
    # 5) ADD lead returns (Target Columns)
    # ======================================================================================
    lead_periods = [1, 5]  # to forcast 1-day and 5-day(weekly) returns
    for period in lead_periods:
        lead_col = f"adj_prc_logret_lead{period}"
        out[lead_col] = out.groupby(level="permno")["adj_prc_logret"].shift(-period)

    return out


def build_final_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and return only the features to be used in the ML model.

    Removes:
    - Raw price, volume, OHLC columns
    - ETF/index high, low, open, volume
    - Technical indicators on ETFs/indexes (comm_ti_*)
    - Any non-stationary or redundant columns

    Returns a clean DataFrame ready for modeling.
    """

    # Columns to keep (explicit list to avoid accidental drops)
    feature_cols = []

    # Add all log return columns (our core stationary features)
    for col in df.columns:
        if col.endswith("_logret") and col not in feature_cols:
            feature_cols.append(col)

    # Add lagged features (momentum, sentiment, etc.)
    for col in df.columns:
        if "_lag" in col and col not in feature_cols:
            feature_cols.append(col)

    # Add ratio features (macro regime signals)
    ratio_features = [
        "ratio_^VIX_^GSPC",
        "ratio_XLK_^GSPC",
        "ratio_XLF_^GSPC",
        "ratio_XLE_^GSPC",
        "ratio_XLV_^GSPC",
        "ratio_XLI_^GSPC",
        "ratio_volatility_premium",
        "ratio_beta_proxy",
    ]
    # Fix: change 'keep_columns' to 'feature_cols'
    feature_cols.extend([f for f in ratio_features if f in df.columns])

    # Add IBES consensus and count features
    ibes_features = [
        "n_analysts",
        "n_up",
        "n_down",
        "cons_mean",
        "cons_median",
        "cons_stdev",
        "cons_high",
        "cons_low",
        "cons_cv",
        "cons_range_pct",
    ]
    # Fix: change 'keep_columns' to 'feature_cols'
    feature_cols.extend([f for f in ibes_features if f in df.columns])

    # Add Fama-French factors
    ff_factors = ["mktrf", "smb", "hml", "rf", "umd"]
    # Fix: change 'keep_columns' to 'feature_cols'
    feature_cols.extend([f for f in ff_factors if f in df.columns])

    # Add technical indicators
    for col in df.columns:
        if "ti_" in col and col not in feature_cols:
            feature_cols.append(col)

    # === WARNING: Check for duplicates ===
    seen = set()
    duplicates = []
    for col in feature_cols:
        if col in seen:
            duplicates.append(col)
        else:
            seen.add(col)

    if duplicates:
        print(f"⚠️  Warning: Duplicate columns added to feature list: {sorted(set(duplicates))}")

    # === WARNING: Check for missing columns ===
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        print(
            f"⚠️  Warning: These feature columns are not in the DataFrame and will be ignored: {sorted(missing)}"
        )

    # Now remove duplicates and missing (but user sees warning first)
    feature_cols = list(dict.fromkeys(feature_cols))  # Preserves order, removes duplicates
    feature_cols = [col for col in feature_cols if col in df.columns]

    # Final selection
    lead_cols = ["ticker"] if "ticker" in df.columns else []
    target_cols = [col for col in df.columns if "lead" in col and col.startswith("adj_prc_logret")]
    final_cols = lead_cols + target_cols + feature_cols

    dropped_columns = [col for col in df.columns if col not in final_cols]
    print(f" Dropped {len(dropped_columns)} columns when building final matrix- dropped columns:")
    print(dropped_columns)

    return df[final_cols].copy()
