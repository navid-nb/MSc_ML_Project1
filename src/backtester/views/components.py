"""Streamlit components (rendering helpers)."""

import pandas as pd
import streamlit as st


def dataset_preview(df: pd.DataFrame) -> None:
    """Render dataset preview + stats."""
    with st.expander("Preview (first 5 rows)", expanded=True):
        st.dataframe(df.head())
    with st.expander("Dataset stats", expanded=True):
        st.json(
            {
                "rows": len(df),
                "start": str(df.index.min()),
                "end": str(df.index.max()),
                "columns": list(df.columns),
                "freq_infer": str(pd.infer_freq(df.index)),
                "has_nans": bool(df.isna().any().any()),
            }
        )


def metrics_grid(stats_dict: dict) -> None:
    """Render summary metrics as Streamlit metrics."""
    cA, cB, cC, cD = st.columns(4)
    with cA:
        st.metric("Total Return", f"{stats_dict['return_total_pct']:.2f}%")
        st.metric("Max Drawdown", f"{stats_dict['max_drawdown_pct']:.2f}%")
    with cB:
        st.metric("Sharpe", f"{stats_dict['sharpe']:.2f}")
        st.metric("Win Rate", f"{stats_dict['win_rate_pct']:.2f}%")
    with cC:
        st.metric("Trades", f"{int(stats_dict['trades'])}")
        st.metric("Exposure", f"{stats_dict['exposure_pct']:.2f}%")
    with cD:
        st.metric("Final Equity", f"${stats_dict['equity_final']:,.2f}")
