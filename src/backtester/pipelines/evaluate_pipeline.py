"""Evaluation pipeline: model(s) → signals → vectorized PnL → metrics (single & multi-asset)."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from backtester.interfaces.feature_engineer import FeatureEngineer
from backtester.interfaces.model import Model


@dataclass
class EvalConfig:
    """Evaluation (backtest) configuration."""

    split_ratio: float = 0.7
    threshold: float = 0.5
    cash: float = 10_000.0
    commission: float = 0.0005


def _infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 252.0
    try:
        freq = pd.infer_freq(index)
    except Exception:
        freq = None
    if freq:
        f = freq.upper()
        if "T" in f or "MIN" in f:
            return 390.0 * 252.0
        if "H" in f:
            return 6.5 * 252.0
        if "B" in f or "D" in f:
            return 252.0
        if "W" in f:
            return 52.0
        if "M" in f:
            return 12.0
    dt = np.median(np.diff(index.view("i8"))) / 1e9
    if dt <= 60:
        return 390.0 * 252.0
    if dt <= 3600:
        return 6.5 * 252.0
    if dt <= 86400:
        return 252.0
    if dt <= 86400 * 7:
        return 52.0
    if dt <= 86400 * 30:
        return 12.0
    return 1.0


def _signals_from_model(
    feats: pd.DataFrame, model: Model, split_ratio: float, threshold: float
) -> Tuple[pd.Series, pd.DatetimeIndex]:
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
    idx = close.index.intersection(signal.index)
    close = close.loc[idx]
    signal = signal.loc[idx]
    pos = signal.copy()
    pos_shift = pos.shift(1).fillna(0.0)
    trades = (pos != pos_shift).astype(int)
    trades_count = int(trades.sum())
    r_price = close.pct_change().fillna(0.0)
    cost = commission * (pos - pos_shift).abs()
    strat_ret = (pos_shift * r_price) - cost
    exposure_pct = float(100.0 * (pos_shift.abs() > 0).mean())
    equity = (1.0 + strat_ret).cumprod() * cash
    equity.iloc[0] = cash
    roll_max = equity.cummax()
    drawdown = (equity / roll_max - 1.0).fillna(0.0)
    return strat_ret, equity, drawdown, trades_count, exposure_pct


def _segment_trade_pnls(
    close: pd.Series, signal: pd.Series, commission: float
) -> Tuple[int, float]:
    idx = close.index.intersection(signal.index)
    close = close.loc[idx]
    sig = signal.loc[idx]
    prev = sig.shift(1).fillna(sig.iloc[0])
    change = sig.ne(prev).astype(int)
    seg_id = change.cumsum()
    pnls = []
    for _, seg in sig.groupby(seg_id):
        seg_idx = seg.index
        if len(seg_idx) < 2:
            continue
        s = seg.iloc[0]
        p0 = close.loc[seg_idx[0]]
        p1 = close.loc[seg_idx[-1]]
        gross = (p1 - p0) / p0
        pnl = s * gross
        pnl -= 2 * commission
        pnls.append(pnl)
    if not pnls:
        return 0, 0.0
    pnls_arr = np.asarray(pnls, dtype=float)
    trades_count = int(len(pnls_arr))
    win_rate_pct = float(100.0 * (pnls_arr > 0).mean())
    return trades_count, win_rate_pct


def _aggregate_metrics(
    close: pd.Series, signal: pd.Series, cfg: EvalConfig, cash_alloc: float
) -> Dict[str, Any]:
    strat_ret, equity, drawdown, trades_count, exposure_pct = _vectorized_pnl(
        close, signal, cfg.commission, cash_alloc
    )
    total_return = float(100.0 * (equity.iloc[-1] / equity.iloc[0] - 1.0))
    max_dd = float(100.0 * drawdown.min())
    pp_year = _infer_periods_per_year(close.index)
    sharpe = (
        float((strat_ret.mean() / strat_ret.std(ddof=1)) * np.sqrt(pp_year))
        if strat_ret.std(ddof=1) > 0
        else 0.0
    )
    t_count_trades, win_rate_pct = _segment_trade_pnls(close, signal, cfg.commission)
    trades_final = t_count_trades if t_count_trades > 0 else trades_count
    equity_curve = pd.DataFrame({"Equity": equity})
    equity_curve.index.name = "Date"
    return {
        "Return [%]": total_return,
        "Max. Drawdown [%]": max_dd,
        "Sharpe Ratio": sharpe,
        "Win Rate [%]": win_rate_pct,
        "Trades": trades_final,
        "Exposure [%]": exposure_pct,
        "Equity Final [$]": float(equity.iloc[-1]),
        "_equity_curve": equity_curve,
    }


def _make_cross_features_set(
    base_feats_map: Dict[str, pd.DataFrame], initial_shares: Dict[str, float]
) -> Dict[str, pd.DataFrame]:
    """Augment each symbol's features with cross-asset features + initial shares."""
    # Build a 1-bar return panel per symbol to cross-join
    one_bar = {}
    for sym, feats in base_feats_map.items():
        if "ret_1" in feats.columns:
            one_bar[sym] = feats["ret_1"]
    one_bar_df = pd.DataFrame(one_bar).dropna(how="any")
    out: Dict[str, pd.DataFrame] = {}
    for sym, feats in base_feats_map.items():
        f = feats.copy()
        # align cross asset returns to this index
        cross = one_bar_df.reindex(f.index).add_prefix("XRET1_")
        for c in cross.columns:
            if c.endswith(f"_{sym}"):  # skip self if came through add_prefix artifact
                continue
        # append cross features (excluding self column)
        xcols = [c for c in cross.columns if not c.endswith(sym)]
        f = pd.concat([f, cross[xcols]], axis=1)
        # add initial shares as a constant feature (can help models)
        f["init_shares"] = float(initial_shares.get(sym, 0.0))
        out[sym] = f.dropna()
    return out


def _orders_from_signals(
    symbol: str, close: pd.Series, signal: pd.Series, init_shares: float
) -> pd.DataFrame:
    """Generate step-by-step orders to move from current to target positions.

    Target rule (simple & explicit):
    - If signal == +1 → target_shares = max(init_shares, 1)   (go/keep long)
    - If signal == -1 → target_shares = -max(init_shares, 1)  (go short of 'unit')
    Quantity is delta from previous position. Commission handled in PnL, here we output steps.
    """
    idx = close.index.intersection(signal.index)
    sig = signal.loc[idx]
    # define unit
    unit = max(1.0, float(init_shares) if not np.isnan(init_shares) else 0.0)
    target = sig.map(lambda s: unit if s > 0 else -unit)
    prev = target.shift(1).fillna(float(init_shares or 0.0))
    qty = target - prev
    action = qty.apply(lambda q: "BUY" if q > 0 else ("SELL" if q < 0 else "HOLD"))
    orders = pd.DataFrame(
        {
            "Symbol": symbol,
            "Price": close.loc[idx],
            "Signal": sig,
            "TargetShares": target,
            "PrevShares": prev,
            "Qty": qty,
            "Action": action,
        }
    )
    orders.index.name = "Date"
    # Keep only actionable steps (BUY/SELL), but include first row if HOLD with nonzero init
    actionable = orders[(orders["Action"] != "HOLD") | (orders.index == orders.index.min())]
    return actionable.reset_index()


def evaluate(
    ohlcv: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    feature_engineer: FeatureEngineer,
    model: Union[Model, Callable[[], Model]],
    cfg: EvalConfig | None = None,
    initial_shares: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """Evaluate single OR multiple assets.

    Parameters
    ----------
    ohlcv : Union[pd.DataFrame, Dict[str, pd.DataFrame]]
        Single OHLCV df OR dict of {symbol: df}.
    feature_engineer : FeatureEngineer
        Feature builder.
    model : Model OR Callable[[], Model]
        If dict input is used, pass a constructor (factory) to build one model per symbol.
    cfg : EvalConfig, optional
        Evaluation configuration.
    initial_shares : Dict[str, float], optional
        Initial owned shares per symbol (default 0).

    Returns
    -------
    Dict[str, Any]
        If single asset: stats dict (as before).
        If multi-asset: {
            "portfolio": stats,
            "per_symbol": {sym: stats},
            "orders": pd.DataFrame([...])  # step-by-step trade plan across symbols
        }
    """
    cfg = cfg or EvalConfig()

    # --- Single-asset path (kept backwards compatible) ---
    if isinstance(ohlcv, pd.DataFrame):
        feats = feature_engineer.make(ohlcv)
        if len(feats) < 50:
            raise ValueError("Not enough rows after feature engineering (need >= 50).")
        if callable(model):
            model = model()
        signal, idx_test = _signals_from_model(feats, model, cfg.split_ratio, cfg.threshold)
        df = ohlcv.loc[idx_test.min() : idx_test.max()].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        stats = _aggregate_metrics(df["Close"], signal, cfg, cfg.cash)
        return stats

    # --- Multi-asset path ---
    initial_shares = initial_shares or {}
    symbols = sorted(ohlcv.keys())
    # build base features per symbol
    base_feats = {s: feature_engineer.make(df) for s, df in ohlcv.items()}
    for s, f in base_feats.items():
        if len(f) < 50:
            raise ValueError(f"Not enough rows for {s} after feature engineering (need >= 50).")
    # add cross features & init shares
    feats_map = _make_cross_features_set(base_feats, initial_shares)

    per_symbol_stats: Dict[str, Dict[str, Any]] = {}
    orders_out: list[pd.DataFrame] = []

    # equal cash allocation per symbol for simplicity
    cash_per_sym = cfg.cash / float(len(symbols))

    # unify test index range per symbol to compute consistent portfolio equity
    union_index = None

    for sym in symbols:
        f = feats_map[sym]
        m = model() if callable(model) else model
        signal, idx_test = _signals_from_model(f, m, cfg.split_ratio, cfg.threshold)
        df = ohlcv[sym].loc[idx_test.min() : idx_test.max()].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        # Stats per symbol on equal-allocated cash
        stats_sym = _aggregate_metrics(df["Close"], signal, cfg, cash_per_sym)
        per_symbol_stats[sym] = stats_sym
        # Orders
        orders_sym = _orders_from_signals(
            sym, df["Close"], signal, init_shares=initial_shares.get(sym, 0.0)
        )
        orders_out.append(orders_sym)
        # union index
        union_index = df.index if union_index is None else union_index.union(df.index)

    # Portfolio aggregation: sum per-period returns weighted by equal cash
    # Build aligned equity series for each symbol then sum to portfolio equity
    eq_parts = []
    for sym in symbols:
        eq = per_symbol_stats[sym]["_equity_curve"]["Equity"].reindex(union_index).ffill().bfill()
        eq_parts.append(eq)
    eq_mat = pd.concat(eq_parts, axis=1)
    eq_mat.columns = symbols
    portfolio_equity = eq_mat.sum(axis=1)
    # derive portfolio returns & drawdown
    port_ret = portfolio_equity.pct_change().fillna(0.0)
    roll_max = portfolio_equity.cummax()
    dd = (portfolio_equity / roll_max - 1.0).fillna(0.0)
    pp_year = _infer_periods_per_year(portfolio_equity.index)
    sharpe = (
        float((port_ret.mean() / port_ret.std(ddof=1)) * np.sqrt(pp_year))
        if port_ret.std(ddof=1) > 0
        else 0.0
    )
    total_return = float(100.0 * (portfolio_equity.iloc[-1] / portfolio_equity.iloc[0] - 1.0))
    max_dd = float(100.0 * dd.min())
    # approximate portfolio trades/exposure as sums/means of component metrics
    trades_total = int(sum(per_symbol_stats[s]["Trades"] for s in symbols))
    exposure_avg = float(np.mean([per_symbol_stats[s]["Exposure [%]"] for s in symbols]))
    portfolio_stats = {
        "Return [%]": total_return,
        "Max. Drawdown [%]": max_dd,
        "Sharpe Ratio": sharpe,
        "Win Rate [%]": float(np.mean([per_symbol_stats[s]["Win Rate [%]"] for s in symbols])),
        "Trades": trades_total,
        "Exposure [%]": exposure_avg,
        "Equity Final [$]": float(portfolio_equity.iloc[-1]),
        "_equity_curve": pd.DataFrame({"Equity": portfolio_equity}),
    }

    orders_df = pd.concat(orders_out, ignore_index=True)
    logger.info(
        f"Portfolio eval: ret={total_return:.2f}% dd={max_dd:.2f}% sharpe={sharpe:.2f} "
        f"trades={trades_total} symbols={len(symbols)}"
    )

    return {"portfolio": portfolio_stats, "per_symbol": per_symbol_stats, "orders": orders_df}
