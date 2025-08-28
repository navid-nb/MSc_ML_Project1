"""Streamlit entrypoint (multi-asset)."""

from datetime import date
from random import randint, seed

import streamlit as st

from backtester.controllers.streamlit_controller import (
    YF_INTERVALS,
    load_data,
    run_workflow,
)
from backtester.utils.config import AppConfig
from backtester.views.components import dataset_stats_multi

seed(123)


def main() -> None:
    cfg = AppConfig.from_dirs("configs")

    st.set_page_config(page_title=cfg.streamlit_title, layout=cfg.streamlit_layout)  # noqa
    st.title(cfg.streamlit_title)

    for k, v in {"data_map": None, "meta": {}}.items():
        st.session_state.setdefault(k, v)

    # Step 1: Data
    st.header("Step 1 — Symbols, Range & Frequency, then Load Data")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        source = st.selectbox("Source", ["yfinance"], index=0)
    with c2:
        symbols_csv = st.text_input(
            "Symbols (comma/space separated)",
            value=cfg.data.get("yfinance", {}).get("default_symbol", "AAPL,MSFT,SPY,NVDA,INTC"),
        )
    with c3:
        interval = st.selectbox(
            "Frequency (yfinance)",
            YF_INTERVALS,
            index=max(
                0, YF_INTERVALS.index(cfg.data.get("yfinance", {}).get("default_interval", "1d"))
            ),
        )
    with c4:
        pull_btn = st.button("Pull / Validate Data", type="primary")

    c5, c6 = st.columns(2)
    with c5:
        start_date = st.date_input(
            "Start",
            value=date.fromisoformat(cfg.data.get("yfinance", {}).get("start", "2019-01-01")),
        )
    with c6:
        end_date = st.date_input("End", value=date.today())

    if pull_btn:
        try:
            data_map = load_data(source, symbols_csv, start_date, end_date, interval)
            st.session_state.data_map = data_map
            st.session_state.meta = {"source": source, "symbols": symbols_csv, "interval": interval}
            st.success("Data loaded and validated")
        except Exception as e:
            st.error(f"Load/validation failed: {e}")

    if st.session_state.data_map is not None:
        dataset_stats_multi(st.session_state.data_map)
    else:
        st.info("Load data in Step 1 to proceed.")

    st.markdown("---")

    # Step 2: Experiment
    st.header("Step 2 — Configure Experiment & Run (Multi-Asset)")
    if st.session_state.data_map is None:
        st.warning("Please load data in Step 1 first.")
        st.stop()

    data_map = st.session_state.data_map

    c1, c2, c3 = st.columns(3)
    with c1:
        model_name = st.selectbox("Model", ["Decision Tree (clf)", "Naive Bayes (Gaussian)"])
    with c2:
        bt_start = st.date_input(
            "Backtest Start",
            value=max(date(2019, 1, 1), min(df.index.min().date() for df in data_map.values())),
        )
    with c3:
        bt_end = st.date_input(
            "Backtest End",
            value=min(date.today(), max(df.index.max().date() for df in data_map.values())),
        )

    c4, c5 = st.columns(2)
    with c4:
        cash = st.number_input(
            "Initial cash (portfolio)", min_value=1000, value=int(cfg.backtest.get("cash", 10000))
        )
    with c5:
        split = st.slider(
            "Train/Test split", 0.5, 0.9, float(cfg.backtest.get("split_ratio", 0.7)), 0.05
        )

    owned = st.text_input(
        "Initial owned shares per symbol",
        value=str({s: randint(0, 100) for s in symbols_csv.split(",")}).strip("{}"),
        help="Optional. If omitted, starts at 0 for each symbol.",
    )

    run_btn = st.button("Run Backtest", type="primary")
    if run_btn:
        run_workflow(
            st.session_state.data_map,
            model_name,
            bt_start,
            bt_end,
            float(cash),
            float(split),
            owned,
        )


if __name__ == "__main__":
    main()
