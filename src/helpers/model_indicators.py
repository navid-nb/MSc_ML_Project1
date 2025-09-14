import numpy as np
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)


def _ema(s: pd.Series, span: int) -> pd.Series:
    """Exponential moving average with pandas' ewm, ignoring all-NaN safety."""
    return s.ewm(span=span, adjust=False, min_periods=1).mean()


def _sma(s: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return s.rolling(window=window, min_periods=1).mean()


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """Elementwise a/b with zero/inf protection -> NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
        out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _groupby_permno(df: pd.DataFrame):
    """Group by permno whether it's an index level or a column."""
    if "permno" in (df.index.names or []):
        return df.groupby(level="permno", group_keys=False)
    return df.groupby("permno", group_keys=False)


def _coerce_ohlcv_numeric(
    g: pd.DataFrame,
    open_col: str,
    high_col: str,
    low_col: str,
    close_col: str,
    vol_col: str,
) -> pd.DataFrame:
    """Coerce OHLCV columns to float and sanitize infs -> NaN for a group."""
    gg = g.copy()
    for c in (open_col, high_col, low_col, close_col, vol_col):
        if c in gg.columns:
            gg[c] = pd.to_numeric(gg[c], errors="coerce")
    return gg.replace([np.inf, -np.inf], np.nan)


def add_technical_indicators(
    df: pd.DataFrame,
    *,
    open_col: str = "openprc",
    high_col: str = "askhi",
    low_col: str = "bidlo",
    close_col: str = "adj_prc",
    vol_col: str = "vol",
    prefix: str = "ti_",
    # Params
    atr_len: int = 14,
    aroon_n: int = 14,
    bb_len: int = 20,
    bb_std: float = 2.0,
    clv_ema_len: int = 14,
    emv_len: int = 14,
    emv_ema: int = 9,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_sig: int = 9,
    mfi_len: int = 14,
    chvol_ema: int = 10,
    chvol_delta: int = 10,
) -> pd.DataFrame:
    """
    Compute a dependable set of indicators per PERMNO using only pandas/numpy
    and append them to the input DataFrame.

    Inputs expected to exist in df:
      open_col, high_col, low_col, close_col, vol_col

    Indicators (column names):
      - ATR (SMA):                 {prefix}atr
      - Aroon Oscillator:          {prefix}aroon_osc
      - Bollinger %B:              {prefix}bb_pctB
      - Chaikin Vol delta (1-step):{prefix}chaikin_vol_d1
      - CLV EMA:                   {prefix}clv_ema
      - EMV (EMA-smoothed):        {prefix}emv_ema
      - MACD signal:               {prefix}macd_signal
      - MFI:                       {prefix}mfi
      - Garman–Klass Vol:          {prefix}vol_gk

    Returns: df with new columns appended (index preserved).
    """
    required = [open_col, high_col, low_col, close_col, vol_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"add_technical_indicators: missing required columns: {missing}")

    out = df.copy()
    gb = _groupby_permno(out)

    # ATR (SMA)
    def _atr_group(g: pd.DataFrame) -> pd.Series:
        gg = _coerce_ohlcv_numeric(g, open_col, high_col, low_col, close_col, vol_col)
        prev_close = gg[close_col].shift(1)
        tr1 = gg[high_col] - gg[low_col]
        tr2 = (gg[high_col] - prev_close).abs()
        tr3 = (gg[low_col] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return _sma(tr, atr_len)

    out[f"{prefix}atr"] = gb.apply(_atr_group)

    # Aroon Oscillator
    def _aroon_group(g: pd.DataFrame) -> pd.Series:
        gg = _coerce_ohlcv_numeric(g, open_col, high_col, low_col, close_col, vol_col)
        n = max(aroon_n, 1)

        # rolling window operations using argmax/argmin positions:
        # "periods since highest high" and "periods since lowest low"
        def _since_last_high(x: pd.Series) -> int:
            # position of last max from the end
            idx = int(np.argmax(x.values))  # first max in the window
            return (len(x) - 1) - idx

        def _since_last_low(x: pd.Series) -> int:
            idx = int(np.argmin(x.values))
            return (len(x) - 1) - idx

        hh_dist = gg[high_col].rolling(n, min_periods=1).apply(_since_last_high, raw=False)
        ll_dist = gg[low_col].rolling(n, min_periods=1).apply(_since_last_low, raw=False)

        aroon_up = (n - hh_dist) * 100.0 / n
        aroon_dn = (n - ll_dist) * 100.0 / n
        return aroon_up - aroon_dn

    out[f"{prefix}aroon_osc"] = gb.apply(_aroon_group)

    # Bollinger %B
    def _bb_pctB_group(g: pd.DataFrame) -> pd.Series:
        gg = _coerce_ohlcv_numeric(g, open_col, high_col, low_col, close_col, vol_col)
        ma = _sma(gg[close_col], bb_len)
        sd = gg[close_col].rolling(bb_len, min_periods=1).std()
        upper = ma + bb_std * sd
        lower = ma - bb_std * sd
        width = upper - lower
        return _safe_div(gg[close_col] - lower, width)

    out[f"{prefix}bb_pctB"] = gb.apply(_bb_pctB_group)

    # Chaikin Volatility Δ1
    def _chaikin_vol_d1_group(g: pd.DataFrame) -> pd.Series:
        gg = _coerce_ohlcv_numeric(g, open_col, high_col, low_col, close_col, vol_col)
        hl_range = (gg[high_col] - gg[low_col]).abs()
        ema_hl = _ema(hl_range, span=chvol_ema)
        pct_change_n = _safe_div(ema_hl, ema_hl.shift(chvol_delta)) - 1.0
        return pct_change_n.pct_change(1)

    out[f"{prefix}chaikin_vol_d1"] = gb.apply(_chaikin_vol_d1_group)

    # CLV EMA
    def _clv_ema_group(g: pd.DataFrame) -> pd.Series:
        gg = _coerce_ohlcv_numeric(g, open_col, high_col, low_col, close_col, vol_col)
        denom = gg[high_col] - gg[low_col]
        # CLV = ((C - L) - (H - C)) / (H - L) = (2C - H - L) / (H - L)
        clv = _safe_div(2.0 * gg[close_col] - gg[high_col] - gg[low_col], denom)
        return _ema(clv, span=clv_ema_len)

    out[f"{prefix}clv_ema"] = gb.apply(_clv_ema_group)

    # EMV (EMA-smoothed)
    def _emv_ema_group(g: pd.DataFrame) -> pd.Series:
        gg = _coerce_ohlcv_numeric(g, open_col, high_col, low_col, close_col, vol_col)

        # Typical EMV (simplified):
        # Midpoint Move = ((H + L)/2 - (H_prev + L_prev)/2)
        mid = (gg[high_col] + gg[low_col]) / 2.0
        mid_move = mid - mid.shift(1)

        # Box Ratio = (Vol / 1e6) / (H - L), protect div by zero
        hl = gg[high_col] - gg[low_col]
        box_ratio = _safe_div(gg[vol_col] / 1_000_000.0, hl)
        emv = _safe_div(mid_move, box_ratio)

        # Smooth raw EMV, then a signal EMA (like many definitions)
        emv_sm = _sma(emv, emv_len)
        emv_ema_sm = _ema(emv_sm, span=emv_ema)
        return emv_ema_sm

    out[f"{prefix}emv_ema"] = gb.apply(_emv_ema_group)

    # MACD signal
    def _macd_signal_group(g: pd.DataFrame) -> pd.Series:
        gg = _coerce_ohlcv_numeric(g, open_col, high_col, low_col, close_col, vol_col)
        ema_fast = _ema(gg[close_col], span=macd_fast)
        ema_slow = _ema(gg[close_col], span=macd_slow)
        macd_line = ema_fast - ema_slow
        signal = _ema(macd_line, span=macd_sig)
        return signal

    out[f"{prefix}macd_signal"] = gb.apply(_macd_signal_group)

    # Money Flow Index
    def _mfi_group(g: pd.DataFrame) -> pd.Series:
        gg = _coerce_ohlcv_numeric(g, open_col, high_col, low_col, close_col, vol_col)
        tp = (gg[high_col] + gg[low_col] + gg[close_col]) / 3.0  # typical price
        mf = tp * gg[vol_col]  # money flow
        # Positive/Negative money flow depending on price change
        up = tp > tp.shift(1)
        pos_mf = mf.where(up).rolling(mfi_len, min_periods=1).sum()
        neg_mf = mf.where(~up).rolling(mfi_len, min_periods=1).sum()
        mr = _safe_div(pos_mf, neg_mf)  # money flow ratio
        mfi = 100.0 - (100.0 / (1.0 + mr))
        return mfi

    out[f"{prefix}mfi"] = gb.apply(_mfi_group)

    # Garman–Klass volatility
    ln2 = np.log(2.0)

    def _vol_gk_group(g: pd.DataFrame) -> pd.Series:
        gg = _coerce_ohlcv_numeric(g, open_col, high_col, low_col, close_col, vol_col)
        with np.errstate(divide="ignore", invalid="ignore"):
            hl = np.log(_safe_div(gg[high_col], gg[low_col]))
            co = np.log(_safe_div(gg[close_col], gg[open_col]))
            var = (hl**2) / 2.0 - (2 * ln2 - 1) * (co**2)
            var = var.clip(lower=0)
            return np.sqrt(var)

    out[f"{prefix}vol_gk"] = gb.apply(_vol_gk_group)

    return out
