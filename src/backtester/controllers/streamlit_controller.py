"""Streamlit controller: orchestrates UI interactions with services."""

import pandas as pd
import streamlit as st

from backtester.features.engineer import DefaultFeatureEngineer
from backtester.models.buy_hold import BuyHoldModel
from backtester.models.decision_tree import DecisionTreeModel
from backtester.models.gaussian_nb import GaussianNBModel
from backtester.pipelines.evaluate_pipeline import EvalConfig, evaluate
from backtester.providers.csv_asset import CSVAsset
from backtester.providers.yfinance_asset import YFinanceAsset
from backtester.services.metrics import extract_core_metrics
from backtester.services.plotter_mpl import MatplotlibPlotter
from backtester.utils.io import read_csv_ohlcv

YF_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]


def _model_factory(name: str):
    if name == BuyHoldModel().name:
        return BuyHoldModel()
    if name == DecisionTreeModel().name:
        return DecisionTreeModel()
    return GaussianNBModel()


def load_data(source: str, symbol: str, start, end, interval: str, uploaded) -> pd.DataFrame:
    """Load data from yfinance or CSV."""
    if source == "yfinance":
        asset = YFinanceAsset(symbol)
        return asset.load(pd.to_datetime(start), pd.to_datetime(end) if end else None, interval)
    if uploaded is None:
        raise ValueError("Upload a CSV or switch to yfinance.")
    raw = read_csv_ohlcv(uploaded)
    asset = CSVAsset("(CSV)", raw)
    return asset.load(pd.to_datetime(start), pd.to_datetime(end) if end else None, None)


def render_chart_from_stats(stats: dict) -> None:
    """Render an equity chart using matplotlib, based on evaluate() stats."""
    fig = MatplotlibPlotter().render(backtesting_fig=None, stats=stats)
    try:
        import matplotlib.figure as mpl_figure  # noqa

        if hasattr(fig, "__class__") and fig.__class__.__name__ == "Figure":
            st.pyplot(fig)
        else:
            st.warning("Could not render chart; ensure matplotlib is available.")
    except Exception:
        st.warning("Could not render chart; ensure matplotlib is available.")


def run_workflow(df: pd.DataFrame, model_name: str, start, end, cash: float, split: float):
    """Run features → model → vectorized evaluation and render results."""
    bt_range = df.loc[str(start) : str(end)].copy()
    if bt_range.empty or len(bt_range) < 50:
        st.error("Not enough rows in backtest range.")
        return

    model = _model_factory(model_name)
    stats = evaluate(
        ohlcv=bt_range,
        feature_engineer=DefaultFeatureEngineer(),
        model=model,
        cfg=EvalConfig(split_ratio=split, cash=float(cash)),
    )

    from backtester.views.components import metrics_grid

    metrics_grid(extract_core_metrics(stats))

    st.subheader("Equity & Trades")
    render_chart_from_stats(stats)

    with st.expander("What exactly did we test?", expanded=True):
        st.markdown(
            f"""
**Data window**: {start} → {end}  
**Commission**: 0.05% per trade  
**Signal model**: **{model_name}**  
**Split**: {int(split*100)}% train / {int((1-split)*100)}% test.

**Workflow**  
1) Feature engineering on selected window (returns, RSI-14, MA-10, volatility).  
2) Temporal split: first {int(split*100)}% → train, rest → test.  
3) Model outputs P(up) for each test bar.  
4) Trading rule: P(up) ≥ 0.5 → long; else short; commissions on flips.
"""
        )
