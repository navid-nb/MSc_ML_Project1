from typing import Sequence

import numpy as np
import pandas as pd

from src.helpers.data_cleanup import (
    ensure_index,
    impute_negative_crsp_factors_and_price,
    join_dsf_with_stocknames,
    join_prices_with_ff,
    join_prices_with_ibes,
    join_prices_with_ibes_actu,
    parquet_to_df,
    post_join_qa_prices,
    post_join_qa_prices_with_ff,
    post_join_qa_prices_with_ibes,
    post_join_qa_prices_with_ibes_actu,
    pre_qa_dsf,
    pre_qa_ff,
    pre_qa_ibes_actu,
    pre_qa_ibes_statsumu,
    pre_qa_stocknames,
    prepare_ibes_actu_for_daily_merge,
    prepare_ibes_for_daily_merge,
)
from src.helpers.data_extraction import wrds_extract_raw
from src.helpers.model_indicators import add_technical_indicators


def _safe_shift_by_permno(df: pd.DataFrame, cols: Sequence[str], shift: int) -> pd.DataFrame:
    """
    Group-aware shift that works whether 'permno' is in index or columns.
    Returns a *copy* with new lagged columns (suffix `_lag{shift}`).
    Silently skips requested columns that are not present.
    """
    out = df.copy()
    present = [c for c in cols if c in out.columns]
    if not present:
        return out

    if "permno" in (out.index.names or []):
        gb = out.groupby(level="permno", group_keys=False)
        for c in present:
            out[f"{c}_lag{shift}"] = gb[c].shift(shift)
    else:
        # columns path (must be sorted for time-consistent shift)
        order = out.index
        tmp = out.sort_values(["permno", "date"]).copy()
        gb = tmp.groupby("permno", group_keys=False)
        for c in present:
            tmp[f"{c}_lag{shift}"] = gb[c].shift(shift)
        out = tmp.reindex(order)

    return out


def build_model_matrix_from_df(
    df_prices: pd.DataFrame,
    *,
    lag_factors: int = 1,
    lag_actuals: int = 1,
    dropna: bool = True,
    core_required: Sequence[str] = ("adj_prc", "adj_mktcap", "retx"),
) -> pd.DataFrame:
    """
    Build a modeling matrix to predict t+1 log returns.

    TARGET (Y)
    ----------
    Y = next-day log return = log(adj_prc_{t+1} / adj_prc_t).

    LAG POLICY
    ----------
    - Fama–French daily factors (mktrf, smb, hml, rf, umd): use t-1 (lag_factors).
    - IBES actuals (act_value, act_measure, pdicity): use t-1 (lag_actuals).

    FEATURE SET
    -----------
    Kept as-is:
      adj_prc, adj_mktcap, vol, retx,
      IBES consensus fields if available: n_analysts, n_up, n_down,
      cons_mean, cons_median, cons_stdev, cons_high, cons_low, cons_cv, cons_range_pct.

    Added (lagged):
      mktrf_lag{lag_factors}, smb_lag{lag_factors}, hml_lag{lag_factors},
      rf_lag{lag_factors}, umd_lag{lag_factors},
      act_value_lag{lag_actuals}, act_measure_lag{lag_actuals}, pdicity_lag{lag_actuals}.

    Missing-data policy:
      - If dropna=True, only enforce non-null on ['Y'] + core_required (default:
        'adj_prc','adj_mktcap','retx'). Optional features are allowed to be NA.
      - If dropna=False, return all rows (you can impute later).

    Returns
    -------
    DataFrame with MultiIndex (permno, date) if present on input.
    Columns order: ['ticker' (if present), 'Y', <features...>].
    """
    out = df_prices.copy()

    # ---------- 1) Target Y = t+1 log return using adjusted price ----------
    if "adj_prc" not in out.columns:
        raise KeyError("build_model_matrix: expected 'adj_prc' in df_prices.")

    if "permno" in (out.index.names or []):
        # index-based path — ensure time order within each permno
        out = out.sort_index()
        gb = out.groupby(level="permno", group_keys=False)
        # log return at t = log(P_t / P_{t-1})
        out["log_ret"] = gb["adj_prc"].transform(lambda s: np.log(s / s.shift(1)))
        # Y at t = log_ret at t+1
        out["Y"] = gb["log_ret"].transform(lambda s: s.shift(-1))
    else:
        # column-based path
        order = out.index
        tmp = out.sort_values(["permno", "date"]).copy()
        gb = tmp.groupby("permno", group_keys=False)
        tmp["log_ret"] = gb["adj_prc"].transform(lambda s: np.log(s / s.shift(1)))
        tmp["Y"] = gb["log_ret"].transform(lambda s: s.shift(-1))
        out["log_ret"] = tmp["log_ret"].reindex(order)
        out["Y"] = tmp["Y"].reindex(order)

    # ---------- 2) Create lagged versions for factors & actuals ----------
    factor_cols = ["mktrf", "smb", "hml", "rf", "umd"]
    actual_cols = ["act_value", "act_measure", "pdicity"]

    out = _safe_shift_by_permno(out, factor_cols, lag_factors)
    out = _safe_shift_by_permno(out, actual_cols, lag_actuals)

    # Drop the raw (unlagged) variants to avoid any lookahead leakage
    to_drop = [c for c in (factor_cols + actual_cols) if c in out.columns]
    if to_drop:
        out = out.drop(columns=to_drop)

    # ---------- 3) Choose feature columns ----------
    base_features = [
        "adj_prc",
        "adj_mktcap",
        "vol",
        "retx",
        "n_analysts",
        "n_up",
        "n_down",
        "cons_mean",
        "cons_median",
        "cons_stdev",
        "cons_high",
        "cons_low",
        "cons_cv",
        "cons_range_pct",
    ]

    lagged_features = [
        f"{c}_lag{lag_factors}" for c in factor_cols if f"{c}_lag{lag_factors}" in out.columns
    ] + [f"{c}_lag{lag_actuals}" for c in actual_cols if f"{c}_lag{lag_actuals}" in out.columns]

    feature_cols = [c for c in base_features + lagged_features if c in out.columns]

    # ---------- 4) Assemble final frame ----------
    lead_cols = ["ticker"] if "ticker" in out.columns else []
    final_cols = lead_cols + ["Y"] + feature_cols
    final = out[final_cols]

    # ---------- 5) Controlled dropna ----------
    if dropna:
        # Only enforce non-null on target + core_required features (if present)
        required_now = ["Y"] + [c for c in core_required if c in final.columns]
        final = final.dropna(subset=required_now, how="any")

    return final


def build_model_matrix_from_wrds() -> pd.DataFrame:
    res = wrds_extract_raw(
        wrds_user="wboughattas",
        start="2020-01-01",
        end="2021-01-01",
        chunk_size=500_000,
        use_run="last",
        base_dir="wrds_extracts",
        artifacts=[
            ("src/migrations/001_base_extract.sql", "dsf.parquet"),
            ("src/migrations/002_crsp_names.sql", "stocknames.parquet"),
            ("src/migrations/003_ff_factors.sql", "ff.parquet"),
            ("src/migrations/004_ibes_statsumu.sql", "ibes_stats.parquet"),
            ("src/migrations/005_ibes_actu.sql", "ibes_act.parquet"),
        ],
    )
    print(res)

    # Load
    dsf = parquet_to_df(res["artifacts"], "dsf.parquet")
    stock_names = parquet_to_df(res["artifacts"], "stocknames.parquet")
    ff = parquet_to_df(res["artifacts"], "ff.parquet")
    ibes = parquet_to_df(res["artifacts"], "ibes_stats.parquet")
    ibes_act = parquet_to_df(res["artifacts"], "ibes_act.parquet")

    # Index & QA
    dsf = ensure_index(dsf, ["permno", "date"], keep_cols=False)
    pre_qa_dsf(dsf)
    pre_qa_stocknames(stock_names)
    dsf = impute_negative_crsp_factors_and_price(dsf)

    # Prices + names
    df_prices = join_dsf_with_stocknames(dsf, stock_names)
    df_prices = ensure_index(df_prices, ["permno", "date"], keep_cols=False)
    post_join_qa_prices(df_prices)

    # FF
    pre_qa_ff(ff)
    df_prices = join_prices_with_ff(df_prices, ff)
    post_join_qa_prices_with_ff(df_prices)

    # IBES statsumu (EPS)
    pre_qa_ibes_statsumu(ibes)
    ibes_daily = prepare_ibes_for_daily_merge(ibes)
    df_prices = join_prices_with_ibes(df_prices, ibes_daily)
    post_join_qa_prices_with_ibes(df_prices)

    # IBES actuals (EPS)
    pre_qa_ibes_actu(ibes_act)
    ibes_act_daily = prepare_ibes_actu_for_daily_merge(ibes_act)
    df_prices = join_prices_with_ibes_actu(df_prices, ibes_act_daily)
    post_join_qa_prices_with_ibes_actu(df_prices)

    # impute null using ffill and bfill
    df_prices = fillna_by_permno(df_prices)

    # Technical indicators
    df_prices = add_technical_indicators(df_prices)

    # Final matrix
    model_df = build_model_matrix_from_df(df_prices)

    print(f"[final] df_prices shape={df_prices.shape}")
    print(f"[final] index={list(df_prices.index.names)}")
    print(f"[final] columns={list(df_prices.columns)}")

    print(f"[model] shape={model_df.shape}")
    print(f"[model] index={list(model_df.index.names)}")
    print(f"[model] columns={list(model_df.columns)}")

    print(null_report(model_df))

    return model_df


def null_report(df: pd.DataFrame, sort: bool = True) -> pd.DataFrame:
    """
    Generate a null report (% of missing values) for each column.

    Parameters
    ----------
    df : pd.DataFrame
        The model matrix or any DataFrame.
    sort : bool, default=True
        If True, sort the report by % of nulls descending.

    Returns
    -------
    pd.DataFrame with columns:
      - column: column name
      - n_null: number of null values
      - pct_null: percentage of null values (0–100)
    """
    total = len(df)
    report = df.isna().sum().reset_index().rename(columns={"index": "column", 0: "n_null"})
    report["pct_null"] = (report["n_null"] / total * 100).round(2)
    report["has_nulls"] = report["n_null"] > 0

    if sort:
        report = report.sort_values("pct_null", ascending=False).reset_index(drop=True)

    return report


def fillna_by_permno(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """
    Fill missing values within each permno group.
    Steps:
      1. Forward fill
      2. Backward fill
      3. If still null, replace with fallback (zero or column mean)

    Parameters
    ----------
    df : DataFrame
    strategy : {"zero", "mean", "median"}
        How to handle values still missing after ffill+bfill.

    Returns
    -------
    DataFrame with nulls filled.
    """
    out = df.copy()
    if "permno" in (df.index.names or []):
        out = out.groupby(level="permno").apply(lambda g: g.ffill().bfill())
    elif "permno" in df.columns:
        out = out.groupby("permno").apply(lambda g: g.ffill().bfill()).reset_index(drop=True)
    else:
        raise KeyError("fillna_by_permno: no 'permno' found in index or columns")

    # Final fallback for columns still containing NaN
    if strategy == "zero":
        out = out.fillna(0)
    elif strategy == "mean":
        out = out.fillna(out.mean(numeric_only=True))
    elif strategy == "median":
        out = out.fillna(out.median(numeric_only=True))
    else:
        raise ValueError("Unknown strategy")

    return out
