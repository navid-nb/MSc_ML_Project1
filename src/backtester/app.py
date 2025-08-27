"""Streamlit entrypoint (thin UI layer)."""

from datetime import date

import streamlit as st

from backtester.controllers.streamlit_controller import (
    YF_INTERVALS,
    load_data,
    run_workflow,
)
from backtester.utils.config import AppConfig
from backtester.utils.logging import setup_logger
from backtester.views.components import dataset_preview


def main() -> None:
    """Configure and run the Streamlit UI."""
    cfg = AppConfig.from_dirs("configs")
    setup_logger(level=cfg.logging.get("level", "INFO"), fmt=cfg.logging.get("format"))

    st.set_page_config(page_title=cfg.streamlit_title, layout=cfg.streamlit_layout)
    st.title(cfg.streamlit_title)

    for k, v in {"data": None, "meta": {}}.items():
        st.session_state.setdefault(k, v)

    # Step 1: Data
    st.header("Step 1 — Choose Frequency and Date Range, then Load Data")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        source = st.selectbox(
            "Source",
            ["yfinance", "CSV"],
            index=0 if cfg.data.get("default_source", "yfinance") == "yfinance" else 1,
        )
    with c2:
        symbol = st.text_input(
            "Symbol (e.g., AAPL, SPY, BTC-USD)",
            value=cfg.data.get("yfinance", {}).get("default_symbol", "AAPL"),
        ).strip()
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
        end_def = cfg.data.get("yfinance", {}).get("end")
        end_date = st.date_input(
            "End",
            value=date.today() if end_def in (None, "", "null") else date.fromisoformat(end_def),
        )

    st.markdown("Or upload **OHLCV CSV** (`Datetime/Open/High/Low/Close[/Adj Close]/[Volume]`).")
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
    csv_resample = st.checkbox(
        "Resample uploaded CSV to a frequency", value=cfg.data.get("csv", {}).get("resample", False)
    )
    csv_freq = (
        st.text_input(
            "CSV resample freq (e.g., 1D, 1H, 5T)",
            value=cfg.data.get("csv", {}).get("resample_freq", "1D"),
        )
        if csv_resample
        else None
    )

    if pull_btn:
        try:
            data = load_data(source, symbol, start_date, end_date, interval, uploaded)
            if source == "csv" and csv_freq:
                from backtester.providers.adapter_utils import resample_ohlcv

                data = resample_ohlcv(data, csv_freq)
            st.session_state.data = data
            st.session_state.meta = {"source": source, "symbol": symbol, "interval": interval}
            st.success("Data loaded and validated ✅")
        except Exception as e:
            st.error(f"Load/validation failed: {e}")

    if st.session_state.data is not None:
        dataset_preview(st.session_state.data)
    else:
        st.info("Load data in Step 1 to proceed.")

    st.markdown("---")

    # Step 2: Experiment
    st.header("Step 2 — Configure Experiment & Run Backtest")
    if st.session_state.data is None:
        st.warning("Please load data in Step 1 first.")
        st.stop()

    data = st.session_state.data
    c1, c2, c3 = st.columns(3)
    with c1:
        model_name = st.selectbox(
            "Model", ["Buy & Hold (benchmark)", "Decision Tree (clf)", "Naive Bayes (Gaussian)"]
        )
    with c2:
        bt_start = st.date_input(
            "Backtest Start", value=max(date(2019, 1, 1), data.index.min().date())
        )
    with c3:
        bt_end = st.date_input("Backtest End", value=min(date.today(), data.index.max().date()))

    c4, c5 = st.columns(2)
    with c4:
        cash = st.number_input(
            "Initial cash", min_value=1000, value=int(cfg.backtest.get("cash", 10000)), step=1000
        )
    with c5:
        split = st.slider(
            "Train/Test split", 0.5, 0.9, float(cfg.backtest.get("split_ratio", 0.7)), 0.05
        )

    run_btn = st.button("Run Backtest", type="primary")
    if run_btn:
        run_workflow(st.session_state.data, model_name, bt_start, bt_end, float(cash), float(split))


if __name__ == "__main__":
    main()
