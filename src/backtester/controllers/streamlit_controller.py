"""Streamlit controller: orchestrates UI interactions with services (multi-asset)."""

import re
from typing import Dict, List

import pandas as pd
import streamlit as st

from backtester.models.decision_tree import DecisionTreeModel
from backtester.models.gaussian_nb import GaussianNBModel
from backtester.pipelines.evaluate_pipeline import EvalConfig, evaluate
from backtester.providers.yfinance_asset import YFinanceAsset
from backtester.services.metrics import extract_core_metrics
from backtester.services.plotter_mpl import MatplotlibPlotter

YF_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]


def _model_factory(name: str):
    if name == DecisionTreeModel().name:
        return DecisionTreeModel()
    return GaussianNBModel()


def _parse_symbols(csv_str: str) -> List[str]:
    """Split comma/space list, keep valid tickers only."""
    syms = [s.strip() for s in re.split(r"[,\s]+", csv_str) if s.strip()]
    # de-dup but keep order
    seen, out = set(), []
    for s in syms:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def _parse_owned(owned_str: str) -> Dict[str, float]:
    """Parse 'AAPL:10, MSFT:0' → {'AAPL': 10.0, 'MSFT': 0.0}."""
    if not owned_str:
        return {}
    pairs = [p.strip() for p in owned_str.split(",") if p.strip()]
    out: Dict[str, float] = {}
    for p in pairs:
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        k, v = k.strip(), v.strip()
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out


def load_data(source: str, symbols_csv: str, start, end, interval: str) -> Dict[str, pd.DataFrame]:
    """Load multiple symbols into a dict {symbol: ohlcv_df}."""
    syms = _parse_symbols(symbols_csv)
    if not syms:
        raise ValueError("No symbols provided.")
    data: Dict[str, pd.DataFrame] = {}
    for sym in syms:
        if source == "yfinance":
            asset = YFinanceAsset(sym)
            df = asset.load(pd.to_datetime(start), pd.to_datetime(end) if end else None, interval)
            data[sym] = df
        else:
            raise NotImplementedError(source)
    return data


def render_chart_from_stats(stats: dict) -> None:
    """Render an equity chart using matplotlib (portfolio equity)."""
    fig = MatplotlibPlotter().render(backtesting_fig=None, stats=stats)
    try:
        import matplotlib.figure as mpl_figure  # noqa: F401

        st.pyplot(fig)
    except Exception:
        st.warning("Could not render chart; ensure matplotlib is available.")


def run_workflow(
    data_map: Dict[str, pd.DataFrame],
    model_name: str,
    start,
    end,
    cash: float,
    split: float,
    owned_str: str,
):
    """Run multi-asset evaluation and render results."""
    # Clip each dataset to UI range
    clipped = {s: df.loc[str(start) : str(end)].copy() for s, df in data_map.items()}
    # Sanity: need enough rows across all
    too_short = [s for s, df in clipped.items() if len(df) < 50]
    if too_short:
        st.error(f"Not enough rows for: {', '.join(too_short)}")
        return

    def model_ctor():
        return _model_factory(model_name)

    # model_ctor = lambda: _model_factory(model_name)
    owned_map = _parse_owned(owned_str)

    from backtester.features.engineer import DefaultFeatureEngineer

    stats = evaluate(
        ohlcv=clipped,  # dict[str, DataFrame]
        feature_engineer=DefaultFeatureEngineer(),
        model=model_ctor,  # factory for per-asset models
        cfg=EvalConfig(split_ratio=split, cash=float(cash)),
        initial_shares=owned_map,
    )

    # Summary (portfolio)
    from backtester.views.components import metrics_grid, orders_table, per_symbol_stats

    st.subheader("Portfolio Summary")
    metrics_grid(extract_core_metrics(stats["portfolio"]))

    # Per-symbol stats
    st.subheader("Per-Symbol Statistics")
    per_symbol_stats(stats["per_symbol"])

    # Equity
    st.subheader("Portfolio Equity")
    render_chart_from_stats(stats["portfolio"])

    # Orders
    st.subheader("Step-by-Step Trade Plan")
    orders_table(stats["orders"])

    with st.expander("What exactly did we test?", expanded=True):
        owned_map_pretty = (
            ", ".join(
                f"{k}:{int(v)}" if float(v).is_integer() else f"{k}:{v}"
                for k, v in owned_map.items()
            )
            or "(none provided)"
        )
        commission_bps = int(EvalConfig().commission * 10_000)  # 0.0005 -> 5 bps
        st.markdown(
            f"""
    **Symbols ({len(clipped)}):** {', '.join(sorted(clipped.keys()))}  
    **Data window:** {start} → {end}  
    **Frequency:** `{st.session_state.meta.get('interval', 'N/A')}`  
    **Commission:** {commission_bps} bps per trade side  
    **Signal model:** **{model_name}** (one model *per symbol*)  
    **Split:** {int(split * 100)}% train / {int((1 - split) * 100)}% test  
    **Initial owned shares:** `{owned_map_pretty}`  
    **Portfolio starting cash:** ${cash:,.0f}

    ### 1) Feature set per symbol
    For each symbol *s*, we build:
    - **Price-derived features (own-asset):**  
      `ret_1`, `ret_5`, `ret_20`, `rsi_14`, `ma_10` (price/MA − 1), `vol_10` (σ of `ret_1`).  
    - **Cross-asset features (peer context):**  
      For every other symbol *j*, we add `XRET1_j = ret_1` of *j* aligned on time.  
    - **Initial holdings:**  
      `init_shares` = starting owned shares for *s* (constant feature so the model can adapt behavior when already long/short/flat).  
    - **Target label:**  
      `y ∈ {0, 1}` indicating next-bar up move (`Close[t+1] ≥ Close[t]`).

    ### 2) Model training and outputs
    - **Training:** temporal split per symbol — first {int(split * 100)}% train, remaining test.  
    - **Inputs to model:** all features above **except** `y`.  
    - **Output:** `P(up)` for each test bar.  
    - **Signal mapping:** `signal = +1` if `P(up) ≥ 0.5`, else `-1`.

    ### 3) Position targets and orders (step-by-step)
    For each symbol independently:
    - Define a **unit size** `U = max(1, init_shares)` (so holdings influence target scale).
    - **Target shares:**  
      `+U` when `signal = +1` (long), `−U` when `signal = −1` (short).  
    - **Order qty:** Δ between current and target shares on each bar.  
    - **Action:** `BUY` if qty>0, `SELL` if qty<0, else `HOLD`.  
    - We list these in the **Step-by-Step Trade Plan** table with `Date, Symbol, Price, Signal, PrevShares, TargetShares, Qty, Action`.  
    - **Commissions** (currently {commission_bps} bps per side) are applied in P&L when positions change.

    ### 4) P&L and metrics (no backtesting.py)
    - Per symbol, we compute vectorized returns:  
      `r_t = position[t-1] * pct_change(Close[t]) − commission * |position[t] − position[t-1]|`.  
    - **Equity** = (1 + r)\_cumprod × **per-symbol cash allocation** (equal-weighted across symbols).  
    - From equity we report: **Return [%]**, **Max. Drawdown [%]**, **Sharpe Ratio**, **Win Rate [%]**, **Trades**, **Exposure [%]**, **Equity Final [$]**.
    - **Portfolio** aggregates by summing per-symbol equities over a unified timeline and recomputing return, drawdown, and Sharpe.
    """
        )
