"""Evaluation pipeline: model → signals → vectorized PnL → metrics.

This module evaluates a *trained or freshly fitted* model on a temporal
holdout and computes portfolio metrics without backtesting.py.

Metrics returned mimic the keys used in the Streamlit UI:
- 'Return [%]'
- 'Max. Drawdown [%]'
- 'Sharpe Ratio'
- 'Win Rate [%]'
- 'Trades'
- 'Exposure [%]'
- 'Equity Final [$]'
Plus an '_equity_curve' DataFrame for plotting fallbacks.
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from backtester.interfaces.feature_engineer import FeatureEngineer
from backtester.interfaces.model import Model


@dataclass
class EvalConfig:
    """Evaluation (backtest) configuration.

    Attributes
    ----------
    split_ratio : float
        Fraction of rows used for training (remainder used for testing).
    threshold : float
        Probability threshold for mapping to signals {-1, +1}.
    cash : float
        Starting equity in dollars.
    commission : float
        Proportional commission per trade (e.g., 0.0005 for 5 bps).
    """

    split_ratio: float = 0.7
    threshold: float = 0.5
    cash: float = 10_000.0
    commission: float = 0.0005


def _infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    """Infer annualization factor from a datetime index.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Time index of returns.

    Returns
    -------
    float
        Periods-per-year guess (e.g., 252 for daily).
    """
    if len(index) < 2:
        return 252.0
    # Try pandas infer first
    try:
        freq = pd.infer_freq(index)
    except Exception:
        freq = None

    if freq:
        f = freq.upper()
        if "T" in f or "MIN" in f:
            # minute bars: assume 390 mins per day * 252 days
            return 390.0 * 252.0
        if "H" in f:
            # hour bars: assume 6.5 trading hours/day
            return 6.5 * 252.0
        if "B" in f or "D" in f:
            return 252.0
        if "W" in f:
            return 52.0
        if "M" in f:
            return 12.0
    # Fallback based on median spacing in seconds
    dt = np.median(np.diff(index.view("i8"))) / 1e9  # seconds
    if dt <= 60:
        return 390.0 * 252.0
    if dt <= 3600:
        return 6.5 * 252.0
    if dt <= 60 * 60 * 24:
        return 252.0
    if dt <= 60 * 60 * 24 * 7:
        return 52.0
    if dt <= 60 * 60 * 24 * 30:
        return 12.0
    return 1.0


def _signals_from_model(
    feats: pd.DataFrame, model: Model, split_ratio: float, threshold: float
) -> Tuple[pd.Series, pd.DatetimeIndex]:
    """Fit model on train slice and produce {-1,+1} signals on test slice."""
    split_idx = int(len(feats) * split_ratio)
    X = feats.drop(columns=["y"])
    y = (feats["y"] > 0).astype(int)

    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    idx_test = X_test.index

    model.fit(X_train.fillna(0), y_train)
    proba = model.predict_proba(X_test.fillna(0))
    sig = pd.Series(np.where(proba >= threshold, 1.0, -1.0), index=idx_test, dtype=float)
    sig = sig[~sig.index.duplicated(keep="last")].sort_index()
    sig.index = pd.to_datetime(sig.index).tz_localize(None)
    return sig, idx_test


def _vectorized_pnl(
    close: pd.Series, signal: pd.Series, commission: float, cash: float
) -> Tuple[pd.Series, pd.Series, pd.Series, int, float]:
    """Compute vectorized P&L and equity curve with simple frictions.

    Assumptions:
    - Position is the signal {-1,+1}; no leverage; fully invested notionally.
    - PnL uses close-to-close returns: r_t = pos_{t-1} * pct_change(close_t).
    - Commission applied on position changes: cost = commission * abs(Δpos).

    Parameters
    ----------
    close : pd.Series
        Close prices, indexed by time.
    signal : pd.Series
        Trading signal in {-1.0, +1.0} on the same (or subset) index.
    commission : float
        Proportional commission per side.
    cash : float
        Starting equity.

    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series, int, float]
        (returns, equity, drawdown, trades_count, exposure_pct)
    """
    # Align to same index range
    idx = close.index.intersection(signal.index)
    close = close.loc[idx]
    signal = signal.loc[idx]

    # Positions and trades
    pos = signal.copy()
    pos_shift = pos.shift(1).fillna(0.0)
    trades = (pos != pos_shift).astype(int)
    trades_count = int(trades.sum())

    # Per-period return from price movement
    r_price = close.pct_change().fillna(0.0)

    # Commission on position changes (entry + exit cost counted via abs delta)
    # cost_t = commission * |pos_t - pos_{t-1}|
    cost = commission * (pos - pos_shift).abs()

    # Strategy returns
    strat_ret = (pos_shift * r_price) - cost

    # Exposure (% time in market by absolute position)
    exposure_pct = float(100.0 * (pos_shift.abs() > 0).mean())

    # Equity curve
    equity = (1.0 + strat_ret).cumprod() * cash
    equity.iloc[0] = cash  # start exactly at cash
    # Drawdown
    roll_max = equity.cummax()
    drawdown = (equity / roll_max - 1.0).fillna(0.0)

    return strat_ret, equity, drawdown, trades_count, exposure_pct


def _segment_trade_pnls(
    close: pd.Series,
    signal: pd.Series,
    commission: float,
) -> Tuple[int, float]:
    """Compute trade-level win rate using segments between signal changes.

    Parameters
    ----------
    close : pd.Series
        Close prices aligned to signal.
    signal : pd.Series
        {-1,+1} per bar.
    commission : float
        Commission per side.

    Returns
    -------
    Tuple[int, float]
        (trades_count, win_rate_pct)
    """
    idx = close.index.intersection(signal.index)
    close = close.loc[idx]
    sig = signal.loc[idx]

    # Find change points
    change = sig.ne(sig.shift(1).fillna(sig.iloc[0])).astype(int)
    # Segment ids
    seg_id = change.cumsum()

    pnls = []
    for _, seg in sig.groupby(seg_id):
        seg_idx = seg.index
        if len(seg_idx) < 2:
            continue
        s = seg.iloc[0]  # position for the segment
        p0 = close.loc[seg_idx[0]]
        p1 = close.loc[seg_idx[-1]]
        gross = (p1 - p0) / p0
        pnl = s * gross
        # Commission for enter + exit
        pnl -= 2 * commission
        pnls.append(pnl)

    if not pnls:
        return 0, 0.0

    pnls_arr = np.asarray(pnls, dtype=float)
    trades_count = int(len(pnls_arr))
    win_rate_pct = float(100.0 * (pnls_arr > 0).mean())
    return trades_count, win_rate_pct


def evaluate(
    ohlcv: pd.DataFrame,
    feature_engineer: FeatureEngineer,
    model: Model,
    cfg: EvalConfig | None = None,
) -> Dict[str, Any]:
    """Evaluate a model with vectorized backtest metrics.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Input OHLCV dataframe.
    feature_engineer : FeatureEngineerInterface
        Feature builder.
    model : ModelInterface
        Unfitted model instance (will be fitted on train slice internally).
    cfg : EvalConfig, optional
        Evaluation configuration. Defaults to EvalConfig().

    Returns
    -------
    Dict[str, Any]
        Metrics and artifacts. Keys include:
        - 'Return [%]'
        - 'Max. Drawdown [%]'
        - 'Sharpe Ratio'
        - 'Win Rate [%]'
        - 'Trades'
        - 'Exposure [%]'
        - 'Equity Final [$]'
        - '_equity_curve' (pd.DataFrame with 'Equity')
    """
    cfg = cfg or EvalConfig()

    feats = feature_engineer.make(ohlcv)
    if len(feats) < 50:
        raise ValueError("Not enough rows after feature engineering (need >= 50).")

    signal, idx_test = _signals_from_model(
        feats, model, split_ratio=cfg.split_ratio, threshold=cfg.threshold
    )

    # Slice OHLCV to test region
    df = ohlcv.loc[idx_test.min() : idx_test.max()].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    strat_ret, equity, drawdown, trades_count, exposure_pct = _vectorized_pnl(
        close=df["Close"], signal=signal, commission=cfg.commission, cash=cfg.cash
    )

    # Aggregate metrics
    total_return = float(100.0 * (equity.iloc[-1] / equity.iloc[0] - 1.0))
    max_dd = float(100.0 * drawdown.min())
    pp_year = _infer_periods_per_year(df.index)
    # Avoid division by zero
    if strat_ret.std(ddof=1) > 0:
        sharpe = float((strat_ret.mean() / strat_ret.std(ddof=1)) * np.sqrt(pp_year))
    else:
        sharpe = 0.0

    # Trade-level win rate
    t_count_trades, win_rate_pct = _segment_trade_pnls(df["Close"], signal, cfg.commission)
    # If segment-based count differs from bar-change count, prefer segment count for “Trades”
    trades_final = t_count_trades if t_count_trades > 0 else trades_count

    # Equity curve dataframe for plotting
    equity_curve = pd.DataFrame({"Equity": equity})
    equity_curve.index.name = "Date"

    stats: Dict[str, Any] = {
        "Return [%]": total_return,
        "Max. Drawdown [%]": max_dd,
        "Sharpe Ratio": sharpe,
        "Win Rate [%]": win_rate_pct,
        "Trades": trades_final,
        "Exposure [%]": exposure_pct,
        "Equity Final [$]": float(equity.iloc[-1]),
        "_equity_curve": equity_curve,
    }

    logger.info(
        f"Eval: ret={total_return:.2f}% dd={max_dd:.2f}% sharpe={sharpe:.2f} "
        f"win={win_rate_pct:.2f}% trades={trades_final} exposure={exposure_pct:.2f}% "
        f"equity=${equity.iloc[-1]:,.2f}"
    )

    return stats
