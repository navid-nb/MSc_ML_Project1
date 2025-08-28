"""Streamlit components (rendering helpers) for multi-asset."""

import pandas as pd
import streamlit as st


def dataset_stats_multi(data_map: dict[str, pd.DataFrame]) -> None:
    """Show per-symbol statistics (no row preview)."""
    rows = []
    for sym, df in data_map.items():
        rows.append(
            {
                "Symbol": sym,
                "Rows": len(df),
                "Start": str(df.index.min()),
                "End": str(df.index.max()),
                "Columns": ", ".join(df.columns),
                "Freq (infer)": str(pd.infer_freq(df.index)),
                "Has NaNs": bool(df.isna().any().any()),
            }
        )
    st.dataframe(pd.DataFrame(rows))


def metrics_grid(stats_dict: dict) -> None:
    """Render summary metrics as Streamlit metrics (portfolio-level)."""
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


def per_symbol_stats(ps: dict[str, dict]) -> None:
    """Table of per-symbol metrics."""
    rows = []
    for sym, m in ps.items():
        rows.append(
            {
                "Symbol": sym,
                "Return [%]": m.get("Return [%]", 0.0),
                "Max. Drawdown [%]": m.get("Max. Drawdown [%]", 0.0),
                "Sharpe Ratio": m.get("Sharpe Ratio", 0.0),
                "Win Rate [%]": m.get("Win Rate [%]", 0.0),
                "Trades": m.get("Trades", 0),
                "Exposure [%]": m.get("Exposure [%]", 0.0),
                "Equity Final [$]": m.get("Equity Final [$]", 0.0),
            }
        )
    st.dataframe(pd.DataFrame(rows))


def orders_table(orders_df: pd.DataFrame) -> None:
    """Render the step-by-step trade plan."""
    st.dataframe(orders_df, height=400)
