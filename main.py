# main.py
# Streamlit UI + backtesting.py evaluation with toy Decision Tree / Naive Bayes models
# pip install streamlit backtesting yfinance scikit-learn pandas numpy

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from backtesting import Backtest, Strategy
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------- Page setup ----------------------
st.set_page_config(page_title="Model Backtester", layout="wide")
st.title("Model Backtester (backtesting.py + Streamlit)")

# ---------------------- Session state -------------------
for k, v in {
    "data": None,  # validated OHLCV dataframe
    "data_meta": {},  # {'ticker':..., 'interval':..., 'source': 'yfinance'|'csv'}
}.items():
    st.session_state.setdefault(k, v)

# ---------------------- Helpers -------------------------
YF_INTERVALS = [
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "90m",
    "1h",
    "1d",
    "5d",
    "1wk",
    "1mo",
    "3mo",
]


def _looks_like_field_names(level_values) -> bool:
    lv = [str(x).strip().lower() for x in level_values]
    keys = {"open", "high", "low", "close", "adj close", "volume"}
    return any(k in lv for k in keys)


def normalize_ohlcv(df: pd.DataFrame, ticker_sym: str | None = None) -> pd.DataFrame:
    """
    Make df single-level columns with exactly ['Open','High','Low','Close','Volume'].
    Handles yfinance MultiIndex in any order. Falls back to 'Adj Close' for 'Close'. Fills Volume if missing.
    """
    if isinstance(df.columns, pd.MultiIndex):
        if ticker_sym:
            for lvl in range(df.columns.nlevels):
                if ticker_sym in df.columns.get_level_values(lvl):
                    df = df.xs(ticker_sym, axis=1, level=lvl, drop_level=True)
                    break
        if isinstance(df.columns, pd.MultiIndex):
            chosen_lvl = None
            for lvl in range(df.columns.nlevels):
                if _looks_like_field_names(df.columns.get_level_values(lvl)):
                    chosen_lvl = lvl
                    break
            if chosen_lvl is not None:
                other_lvls = [i for i in range(df.columns.nlevels) if i != chosen_lvl]
                df = df.droplevel(other_lvls, axis=1)
            else:
                last_vals = df.columns.get_level_values(-1)
                unique_syms = last_vals.unique()
                if len(unique_syms) == 1:
                    df = df.droplevel(-1, axis=1)
                else:
                    pick = unique_syms[0]
                    st.warning(f"Multiple symbols detected; using the first: {pick}")
                    df = df.xs(pick, axis=1, level=-1, drop_level=True)

    df = df.rename(columns=lambda c: str(c).strip().title())

    cols = set(df.columns)
    if "Close" not in cols and "Adj Close" in cols:
        df["Close"] = df["Adj Close"]

    for c in ["Open", "High", "Low", "Close"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    if "Volume" not in df.columns:
        st.warning("No 'Volume' column found. Filling zeros.")
        df["Volume"] = 0.0

    out = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    out.index = pd.to_datetime(out.index, errors="coerce").tz_localize(None)
    out = out[~out.index.isna()].sort_index()
    out.index.name = "Date"
    return out


def resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    rs = df.resample(freq).agg(agg).dropna(how="any")
    return rs


def ta_rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    gain = up.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    loss = down.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)
    close = df["Close"]
    feats["ret_1"] = close.pct_change()
    feats["ret_5"] = close.pct_change(5)
    feats["ret_20"] = close.pct_change(20)
    feats["rsi_14"] = ta_rsi(close, 14)
    feats["ma_10"] = close.rolling(10).mean() / close - 1
    feats["vol_10"] = feats["ret_1"].rolling(10).std()
    feats["y"] = np.sign(close.shift(-1) / close - 1).replace(0, 1)
    return feats.dropna()


# ---------------------- STEP 1: Load & Validate ----------------------
st.header("Step 1 — Choose Frequency and Date Range, then Load Data")

c1, c2, c3, c4 = st.columns(4)
with c1:
    instrument_type = st.selectbox(
        "Instrument", ["Stock", "ETF", "Future", "Option", "Crypto", "Other"], index=0
    )
with c2:
    ticker = st.text_input("Symbol (e.g., AAPL, SPY, BTC-USD)", value="AAPL").strip()
with c3:
    interval = st.selectbox("Frequency (yfinance)", YF_INTERVALS, index=8)  # default 1d
with c4:
    pull_btn = st.button("Pull / Validate Data", type="primary")

c5, c6 = st.columns(2)
with c5:
    start_date = st.date_input("Start", value=date(2019, 1, 1))
with c6:
    end_date = st.date_input("End", value=date.today())

st.markdown("Or upload **OHLCV CSV** (`Datetime/Open/High/Low/Close[/Adj Close]/[Volume]`).")
uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])

csv_resample = st.checkbox("Resample uploaded CSV to a frequency", value=False)
csv_freq = (
    st.text_input("CSV resample freq (e.g., 1D, 1H, 5T)", value="1D") if csv_resample else None
)

st.caption(
    "Notes: yfinance intraday has limits (e.g., 1m ~ last 30 days). We’ll try your dates but may need to narrow.\n"
    "CSV resampling aggregates OHLCV properly (first/max/min/last/sum)."
)


def load_from_yf(sym: str, start_d: date, end_d: date, itv: str) -> pd.DataFrame:
    df = yf.download(
        sym,
        start=str(start_d),
        end=str(end_d),
        interval=itv,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
    )
    if df is None or len(df) == 0:
        raise ValueError("No data returned (symbol/interval/date range may be unsupported).")
    return normalize_ohlcv(df, ticker_sym=sym)


def load_from_csv(file, resample_to: str | None) -> pd.DataFrame:
    raw = pd.read_csv(file)
    cols = {c.lower(): c for c in raw.columns}
    # Find datetime
    dt_col = None
    for cand in ["datetime", "date", "timestamp"]:
        if cand in cols:
            dt_col = cols[cand]
            break
    if dt_col is None:
        raise ValueError("CSV must include a Datetime/Date column.")

    raw[dt_col] = pd.to_datetime(raw[dt_col], utc=True, errors="coerce")
    raw = raw.rename(
        columns={
            dt_col: "Date",
            cols.get("open", "Open"): "Open",
            cols.get("high", "High"): "High",
            cols.get("low", "Low"): "Low",
            cols.get("close", "Close"): "Close",
            cols.get("adj close", "Adj Close"): (
                "Adj Close" if "adj close" in cols else "Adj Close"
            ),
            cols.get("volume", "Volume"): "Volume",
        }
    )
    keep = [
        c
        for c in ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        if c in raw.columns
    ]
    df = raw[keep].dropna(subset=["Date"]).set_index("Date")
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce").tz_convert(None)
    df = normalize_ohlcv(df)
    df = df.loc[str(start_date) : str(end_date)]
    if resample_to:
        df = resample_ohlcv(df, resample_to)
    return df


if pull_btn:
    try:
        if uploaded is not None:
            data = load_from_csv(uploaded, csv_freq if csv_resample else None)
            st.session_state.data = data
            st.session_state.data_meta = {
                "source": "csv",
                "interval": csv_freq if csv_resample else "native",
                "ticker": "(CSV)",
            }
        else:
            if not ticker:
                raise ValueError("Ticker is empty.")
            data = load_from_yf(ticker, start_date, end_date, interval)
            st.session_state.data = data
            st.session_state.data_meta = {
                "source": "yfinance",
                "interval": interval,
                "ticker": ticker,
            }

        st.success("Data loaded and validated ✅")
    except Exception as e:
        st.error(f"Load/validation failed: {e}")

# Always show preview if data is present (prevents disappearing on rerun)
if st.session_state.data is not None:
    data = st.session_state.data
    meta = st.session_state.data_meta
    with st.expander("Preview (first 5 rows)", expanded=True):
        st.dataframe(data.head())
    with st.expander("Dataset stats", expanded=True):
        info = {
            "Rows": len(data),
            "Start": str(data.index.min()),
            "End": str(data.index.max()),
            "Columns": list(data.columns),
            "Has NaNs": bool(data.isna().any().any()),
            "Freq (pandas infer)": str(pd.infer_freq(data.index)),
            "Source": meta.get("source"),
            "Ticker": meta.get("ticker"),
            "Interval": meta.get("interval"),
        }
        st.json(info)
else:
    st.info("Load data in Step 1 to proceed.")

st.markdown("---")

# ---------------------- STEP 2: Configure & Run ----------------------
st.header("Step 2 — Configure Experiment & Run Backtest")

if st.session_state.data is None:
    st.warning("Please load data in Step 1 first.")
    st.stop()

data = st.session_state.data  # local alias

c1, c2, c3 = st.columns(3)
with c1:
    model_name = st.selectbox(
        "Model",
        ["Buy & Hold (benchmark)", "Decision Tree (clf)", "Naive Bayes (Gaussian)"],
    )
with c2:
    bt_start = st.date_input("Backtest Start", value=max(date(2019, 1, 1), data.index.min().date()))
with c3:
    bt_end = st.date_input("Backtest End", value=min(date.today(), data.index.max().date()))

c4, c5 = st.columns(2)
with c4:
    cash = st.number_input("Initial cash", min_value=1000, value=10000, step=1000)
with c5:
    run_btn = st.button("Run Backtest", type="primary")


def compute_signal(model_name: str, feats: pd.DataFrame) -> pd.Series:
    split_idx = int(len(feats) * 0.7)
    X = feats.drop(columns=["y"])
    y = (feats["y"] > 0).astype(int)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_test, _ = X.iloc[split_idx:], y.iloc[split_idx:]
    index_test = X_test.index

    if model_name == "Buy & Hold (benchmark)":
        sig = pd.Series(1.0, index=index_test, dtype=float)
    else:
        if model_name == "Decision Tree (clf)":
            model = DecisionTreeClassifier(max_depth=4, random_state=42)
        else:
            model = GaussianNB()
        pipe = make_pipeline(StandardScaler(), model)
        pipe.fit(X_train.fillna(0), y_train)
        proba = pd.Series(pipe.predict_proba(X_test.fillna(0))[:, 1], index=index_test)
        sig = pd.Series(np.where(proba >= 0.5, 1.0, -1.0), index=index_test, dtype=float)

    sig = sig[~sig.index.duplicated(keep="last")].sort_index()
    sig.index = pd.to_datetime(sig.index).tz_localize(None)
    return sig


def fmt_pct(x):
    return f"{x:.2f}%"


def fmt_d(x):
    return f"{int(x)}"


def fmt_f(x):
    return f"{x:,.2f}"


# Only run backtest when clicked; keep preview on screen always
if run_btn:
    bt_range = data.loc[str(bt_start) : str(bt_end)].copy()
    if bt_range.empty or len(bt_range) < 50:
        st.error("Not enough rows in the selected backtest range. Choose a wider range.")
    else:
        feats = make_features(bt_range)
        if len(feats) < 50:
            st.error("Not enough rows after feature engineering. Choose a wider range.")
        else:
            signal = compute_signal(model_name, feats)

            bt_df = bt_range.copy()
            bt_df.index = pd.to_datetime(bt_df.index).tz_localize(None)
            bt_df["signal"] = 0.0
            common = bt_df.index.intersection(signal.index)
            bt_df.loc[common, "signal"] = signal.reindex(common).to_numpy()

            bt_slice = bt_df.loc[signal.index.min() : signal.index.max()].copy()

            try:
                assert not isinstance(bt_slice.columns, pd.MultiIndex)
                assert set(["Open", "High", "Low", "Close", "Volume"]).issubset(bt_slice.columns)
                assert len(bt_slice) > 0
            except AssertionError as e:
                st.error(f"Backtest precondition failed: {e}")
            else:

                class SignalStrategy(Strategy):
                    signal_threshold = 0.0
                    allow_short = True

                    def init(self):
                        self.signal = self.I(lambda: self.data.df["signal"].values)

                    def next(self):
                        s = self.signal[-1]
                        if s > self.signal_threshold:
                            if self.position.is_short:
                                self.position.close()
                            if not self.position.is_long:
                                self.buy()
                        elif s < -self.signal_threshold:
                            if self.position.is_long:
                                self.position.close()
                            if self.allow_short and not self.position.is_short:
                                self.sell()
                        else:
                            if self.position:
                                self.position.close()

                bt = Backtest(
                    bt_slice[["Open", "High", "Low", "Close", "Volume", "signal"]],
                    SignalStrategy,
                    cash=cash,
                    commission=0.0005,
                    exclusive_orders=True,
                    finalize_trades=True,  # <-- closes open trades for stats
                )

                stats = bt.run()
                fig = bt.plot(open_browser=False)

                # -------- Summary stats (not a table) --------
                st.subheader("Summary stats")
                # stats is a pandas Series; pull common keys defensively
                # Keys vary by backtesting.py version; use .get with defaults
                total_return = stats.get("Return [%]") or stats.get("Return (Ann.) [%]") or 0.0
                sharpe = stats.get("Sharpe Ratio") or stats.get("Sharpe") or 0.0
                win_rate = stats.get("Win Rate [%]") or 0.0
                max_dd = stats.get("Max. Drawdown [%]") or stats.get("Max Drawdown [%]") or 0.0
                trades = stats.get("Trades") or 0
                equity_final = (
                    stats.get("Equity Final [$]") or stats.get("Equity Final [$ ]") or 0.0
                )
                exposure = stats.get("Exposure [%]") or 0.0

                cA, cB, cC, cD = st.columns(4)
                with cA:
                    st.metric("Total Return", fmt_pct(total_return))
                    st.metric("Max Drawdown", fmt_pct(max_dd))
                with cB:
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    st.metric("Win Rate", fmt_pct(win_rate))
                with cC:
                    st.metric("Trades", fmt_d(trades))
                    st.metric("Exposure", fmt_pct(exposure))
                with cD:
                    st.metric("Final Equity", f"${fmt_f(equity_final)}")

                # -------- Equity & Trades plot --------
                st.subheader("Equity & Trades")
                rendered = False

                # 1) Try to render whatever backtesting.py returned as a Bokeh object
                try:
                    st.bokeh_chart(fig, use_container_width=True)
                    rendered = True
                except Exception:
                    rendered = False

                # 2) Fallback: matplotlib figure?
                if not rendered:
                    try:
                        import matplotlib.figure as mpl_figure

                        if isinstance(fig, mpl_figure.Figure):
                            st.pyplot(fig, use_container_width=True)
                            rendered = True
                    except Exception:
                        pass

                # 3) Last-resort fallback: draw equity curve from stats
                if not rendered:
                    try:
                        eq = stats.get("_equity_curve")
                        if eq is not None:
                            import matplotlib.pyplot as plt

                            if isinstance(eq, pd.DataFrame) and "Equity" in eq.columns:
                                series = eq["Equity"]
                            else:
                                series = pd.Series(eq)
                            fig_fallback, ax = plt.subplots()
                            ax.plot(series.index, series.values)
                            ax.set_title("Equity Curve (fallback)")
                            ax.set_xlabel("Time")
                            ax.set_ylabel("Equity")
                            st.pyplot(fig_fallback, use_container_width=True)
                            rendered = True
                    except Exception:
                        pass

                if not rendered:
                    st.warning(
                        "Chart could not be rendered. If you're missing Bokeh, run: "
                        "`pip install bokeh` and restart the app."
                    )

                # -------- Testing strategy explanation --------
                with st.expander("What exactly did we test?", expanded=True):
                    st.markdown(
                        f"""
**Data window**: {bt_start} → {bt_end}  
**Bar frequency**: `{st.session_state.data_meta.get('interval', 'N/A')}`  
**Commission**: 0.05% per trade  
**Signal model**: **{model_name}**

**Workflow**  
1) Feature engineering on the selected window (returns, RSI-14, MA-10, vol).  
2) **Temporal split**: first 70% → train, last 30% → test (no leakage).  
3) Model outputs a probability of up-move for each bar in the **test** segment.  
4) **Trading rule**:  
   - If prob ≥ 0.5 → go **long** (close shorts).  
   - If prob < 0.5 → go **short** (close longs).  
   - Flat when exactly at threshold.  
5) Backtest executes market orders bar-by-bar with commission and **finalizes open trades** at the end.  

You can change the model/date range/frequency in the UI to re-run.
"""
                    )

st.caption(
    "Tips: Streaming/tick data → upload CSV/Parquet and resample first. "
    "For realistic evaluation, implement walk-forward splits and slippage models."
)
