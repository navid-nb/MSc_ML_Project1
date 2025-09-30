from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd

from src.helpers.data_cleanup import (
    clean_dsf,
    ensure_index,
    filter_by_tickers,
    filter_tickers,
    join_dsf_with_stocknames,
    join_prices_with_ff,
    join_prices_with_ibes,
    join_prices_with_ibes_actu,
    parquet_to_df,
    post_join_qa_prices_with_ff,
    post_join_qa_prices_with_ibes,
    post_join_qa_prices_with_ibes_actu,
    post_stockname_join_qa_cleaning,
    pre_qa_dsf,
    pre_qa_ff,
    pre_qa_ibes_actu,
    pre_qa_ibes_statsumu,
    pre_qa_stocknames,
    prepare_ibes_actu_for_daily_merge,
    prepare_ibes_for_daily_merge,
)
from src.helpers.data_extraction import wrds_extract_raw
from src.helpers.feature_engineering import add_technical_indicators


def _permno_level_number(df: pd.DataFrame) -> int | None:
    """
    Return the integer position of the first index level named 'permno', or None if absent.

    Args:
        df (pd.DataFrame): Input DataFrame, possibly with MultiIndex.

    Returns:
        int | None: Integer index of 'permno' level, or None if not found.
    """
    if isinstance(df.index, pd.MultiIndex):
        names = list(df.index.names)
        for i, n in enumerate(names):
            if n == "permno":
                return i
    elif getattr(df.index, "name", None) == "permno":
        return 0
    return None


def _groupby_permno(df: pd.DataFrame):
    """
    Group the DataFrame by 'permno', whether it's an index level or a column.

    Handles duplicate index level names by using the first 'permno' index level.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.core.groupby.generic.DataFrameGroupBy: Grouped DataFrame object.

    Raises:
        KeyError: If 'permno' not found as column or index level.
    """
    lvl = _permno_level_number(df)
    if lvl is not None:
        return df.groupby(level=lvl, group_keys=False)
    if "permno" in df.columns:
        return df.groupby("permno", group_keys=False)
    raise KeyError("No 'permno' found as index level or column.")


def _safe_shift_by_permno(df: pd.DataFrame, cols: Sequence[str], shift: int) -> pd.DataFrame:
    """
    Create lagged columns by shifting specified columns within each 'permno' group.

    Works whether 'permno' is an index level or a column, handles missing columns silently.

    Args:
        df (pd.DataFrame): Input DataFrame.
        cols (Sequence[str]): Columns to shift.
        shift (int): Number of periods to shift (lag).

    Returns:
        pd.DataFrame: Copy of df with additional lagged columns named '{col}_lag{shift}'.
    """
    out = df.copy()
    present = [c for c in cols if c in out.columns]
    missing = [c for c in cols if c not in out.columns]
    if missing:
        print(f"[warn] _safe_shift_by_permno: some requested columns not found in df -> {missing}")
    if not present:
        return out

    if "permno" in (out.index.names or []):
        gb = _groupby_permno(out)
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
    Build a modeling matrix to predict next-day (t+1) log returns using lagged factors and indicators.

    TARGET (adjclose_lead):
        adjclose_lead = next-day log return = log(adj_prc_{t+1} / adj_prc_t).

    FEATURE SET:
        - Market features: adj_mktcap, vol, retx
        - Return lags: adjclose_lag0 (current), adjclose_lag1, adjclose_lag2, adjclose_lag3
        - Fama-French factors: mktrf, smb, hml, rf, umd (lagged by lag_factors, default 1)
        - IBES consensus: n_analysts, n_up, n_down, cons_mean, cons_median, cons_stdev, 
          cons_high, cons_low, cons_cv, cons_range_pct
        - Technical indicators: All columns starting with 'ti_'

    EXCLUDED FEATURES (to reduce multicollinearity):
        - adj_prc (redundant with adjclose_lag0)
        - IBES actuals (act_value, act_measure, pdicity)
        - SMAs and EMAs (redundant with MACD)
        - Extra MACD components (MACDh, MACDs)
        - Bollinger Band sub-components (BBL, BBM, BBU, BBB) - keep only BBP

    Missing-data policy:
    - If dropna=True, only enforce non-null on ['adjclose_lead'] + core_required (default:
      'adj_mktcap','retx'). Optional features are allowed to be NA.
    - If dropna=False, return all rows (impute later).

    Args:
        df_prices (pd.DataFrame): Price and factor data; multi-indexed by permno and date ideally.
        lag_factors (int): Lag for factor variables (default 1).
        lag_actuals (int): Lag for actual EPS variables (not used anymore).
        dropna (bool): Whether to drop rows missing target or core features.
        core_required (Sequence[str]): Core columns that must be non-null if dropna=True.

    Returns:
        DataFrame with MultiIndex (permno, date) if present on input.
        Columns order: ['ticker' (if present), 'adjclose_lead', <features...>].
    """
    out = df_prices.copy()

    # 1) Target Y = t+1 log return using adjusted price
    if "adj_prc" not in out.columns:
        raise KeyError("build_model_matrix: expected 'adj_prc' in df_prices.")

    if "permno" in (out.index.names or []):
        out = out.sort_index()
        gb = _groupby_permno(out)
        out["log_ret"] = gb["adj_prc"].transform(lambda s: np.log(s / s.shift(1)))
        out["adjclose_lead"] = gb["log_ret"].transform(lambda s: s.shift(-1))  # Tomorrow's return (target)
        # Add lagged log returns as features (adjclose_lag0 through adjclose_lag3)
        out["adjclose_lag0"] = out["log_ret"]  # Current log return
        out["adjclose_lag1"] = gb["log_ret"].transform(lambda s: s.shift(1))
        out["adjclose_lag2"] = gb["log_ret"].transform(lambda s: s.shift(2))
        out["adjclose_lag3"] = gb["log_ret"].transform(lambda s: s.shift(3))
    else:
        order = out.index
        tmp = out.sort_values(["permno", "date"]).copy()
        gb = tmp.groupby("permno", group_keys=False)
        tmp["log_ret"] = gb["adj_prc"].transform(lambda s: np.log(s / s.shift(1)))
        tmp["adjclose_lead"] = gb["log_ret"].transform(lambda s: s.shift(-1))  # Tomorrow's return (target)
        # Add lagged log returns as features
        tmp["adjclose_lag0"] = tmp["log_ret"]
        tmp["adjclose_lag1"] = gb["log_ret"].transform(lambda s: s.shift(1))
        tmp["adjclose_lag2"] = gb["log_ret"].transform(lambda s: s.shift(2))
        tmp["adjclose_lag3"] = gb["log_ret"].transform(lambda s: s.shift(3))
        out["log_ret"] = tmp["log_ret"].reindex(order)
        out["adjclose_lead"] = tmp["adjclose_lead"].reindex(order)
        out["adjclose_lag0"] = tmp["adjclose_lag0"].reindex(order)
        out["adjclose_lag1"] = tmp["adjclose_lag1"].reindex(order)
        out["adjclose_lag2"] = tmp["adjclose_lag2"].reindex(order)
        out["adjclose_lag3"] = tmp["adjclose_lag3"].reindex(order)

    # 2) Create lagged versions for factors (rename to remove _lag1 suffix)
    factor_cols = ["mktrf", "smb", "hml", "rf", "umd"]
    
    # Create lagged factors and rename them without the lag suffix
    out = _safe_shift_by_permno(out, factor_cols, lag_factors)
    
    # Rename factor columns to remove _lag suffix (e.g., mktrf_lag1 -> mktrf)
    for col in factor_cols:
        lagged_col = f"{col}_lag{lag_factors}"
        if lagged_col in out.columns:
            out[col] = out[lagged_col]
            out = out.drop(columns=[lagged_col])

    # 3) Choose feature columns
    base_features = [
        # Price and volume features (excluding adj_prc to avoid multicollinearity with adjclose_lag0)
        "adj_mktcap",
        "vol",
        "retx",
        # IBES consensus features (analyst coverage and estimates)
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
        # Lagged returns (our main price momentum features)
        "adjclose_lag0",
        "adjclose_lag1",
        "adjclose_lag2",
        "adjclose_lag3",
        # Fama-French factors (already renamed without _lag suffix)
        "mktrf",
        "smb",
        "hml",
        "rf",
        "umd",
    ]
    # Add technical indicators as features
    for col in out.columns:
        if col.startswith("ti_"):
            base_features.append(col)

    # Only keep features that exist in the dataframe
    feature_cols = [c for c in base_features if c in out.columns]

    # 4) Assemble final frame
    lead_cols = ["ticker"] if "ticker" in out.columns else []
    final_cols = lead_cols + ["adjclose_lead"] + feature_cols
    final = out[final_cols]

    # 5) Controlled dropna
    if dropna:
        # Only enforce non-null on target + core_required features (if present)
        # Note: adj_prc is no longer in core_required since we removed it from features
        core_required_filtered = [c for c in core_required if c != "adj_prc" and c in final.columns]
        required_now = ["adjclose_lead"] + core_required_filtered
        final = final.dropna(subset=required_now, how="any")

    return final


def null_report(df: pd.DataFrame, sort: bool = True) -> pd.DataFrame:
    """
    Generate a null report (% of missing values) for each column.

    Parameters

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


def _remove_leading_nans(
    df: pd.DataFrame, remove_reason: str = "no reason indicated"
) -> pd.DataFrame:
    out = df.copy()
    mask = ~out.isna().any(axis=1)
    group_cumsum = mask.groupby(out.index.get_level_values("permno")).cumsum()
    out = out[group_cumsum > 0]
    total_rows = len(df)
    removed_pct = (total_rows - len(out)) / total_rows * 100 if total_rows else 0
    print(
        f"[INFO] percentage of rows removed due to leading NaNs : {removed_pct:.4f}%  remove reason: {remove_reason}"
    )

    return out


def forward_fill_and_remove_initial_nans(
    df: pd.DataFrame, add_fill_source_columns: bool = False
) -> pd.DataFrame:
    """
    Forward fills missing values within groups defined by 'permno' in a MultiIndex DataFrame,
    replaces columns that are entirely NaN within each group with dummy values before filling,
    and drops leading rows with NaNs after filling.

    The input DataFrame must have a MultiIndex with levels 'permno' and 'date'.
    Optionally, columns indicating the source date of the fill can be added.

    Steps performed:
    - Validates that 'permno' and 'date' exist as index levels.
    - Optionally adds columns showing the date from which forward fill values originate.
    - For each 'permno' group:
      - Replaces columns that contain only NaN values within that group with dummy values
        (-9999 for numeric columns, 'missing' for non-numeric columns) instead of dropping them,
        to maintain consistent column structure across groups.
      - Applies forward fill to propagate last valid observation forward.
    - Removes leading rows in each group that still contain NaNs after forward filling.
    - Reports statistics on how many rows were removed due to leading NaNs.
    - Issues warnings if any NaNs remain after processing.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with MultiIndex levels including 'permno' and 'date'. Data to be forward filled.
    add_fill_source_columns : bool, default False
        If True, adds additional columns indicating source dates of forward fill for each originally NaN value.

    Returns:
    --------
    pd.DataFrame
        Forward filled DataFrame with dummy-filled columns replacing all-NaN columns per group and leading NaN rows dropped.
    """
    required_levels = {"permno", "date"}
    if not required_levels.issubset(set(df.index.names)):
        missing = required_levels - set(df.index.names)
        raise ValueError(
            f"Input DataFrame must have MultiIndex levels named {required_levels}. Missing: {missing}"
        )

    out = df.copy()
    dates = out.index.get_level_values("date")

    if add_fill_source_columns:
        cols_with_nans = df.columns[df.isna().any()].tolist()
        suffix = "_ffill_source_date"
        for col in cols_with_nans:
            out[col + suffix] = pd.Series(np.where(df[col].notna(), dates, pd.NaT), index=df.index)

    # messy trick: Replace all-NaN columns(empty columns) per group with -9999 (numeric) or 'missing' (non-numeric), then forward fill
    # if we don't do this before ffill, they will remain empty and cuse issues later
    # if we drop these columns, since for some groups they are not empty there will be inconsistency in the data since we're looking at the data for all companies as one big data frame
    out = out.groupby(level="permno").apply(
        lambda g: g.assign(
            **{
                col: (-9999 if pd.api.types.is_numeric_dtype(g[col]) else "missing")
                for col in g.columns[g.isna().all()]
            }
        ).ffill()
    )

    out.index = out.index.droplevel(0)

    out = _remove_leading_nans(out, remove_reason="after forward filling empty cells")

    if out.isna().any().any():
        print("WARNING: NaN values remain after forward fill and dropping leading NaNs.")
    else:
        print("INFO: No NaN values remain after forward fill and dropping leading NaNs.")

    return out


def _get_date_index(df: pd.DataFrame, date_name: str = "date") -> pd.DatetimeIndex:
    """
    Retrieve a sorted DatetimeIndex of unique dates from the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        date_name (str): Name of the date column or index level.

    Returns:
        pd.DatetimeIndex: Sorted unique dates.
    """
    if date_name in (df.index.names or []):
        di = pd.DatetimeIndex(df.index.get_level_values(date_name))
    else:
        di = pd.to_datetime(df[date_name], errors="raise")
    return pd.DatetimeIndex(sorted(di.unique()))


def reindex_each_permno_to_global_calendar(
    df: pd.DataFrame,
    date_name: str = "date",
    id_name: str = "permno",
    fill: Literal["none", "ffill", "bfill", "both"] = "both",
    final_strategy: Literal["none", "zero", "mean", "median"] = "median",
    only_columns: Sequence[str] | None = None,
    calendar: Optional[pd.Index] = None,
) -> pd.DataFrame:
    """
    Reindex the DataFrame so that each 'permno' has rows for all dates in a global calendar,
    optionally forward/backward filling missing data and applying a final numeric fill.

    Parameters:
        df (pd.DataFrame): DataFrame to reindex.
        date_name (str): Column or index name for dates.
        id_name (str): Column or index name for entity id ('permno').
        fill (str): Fill method for missing data ('none', 'ffill', 'bfill', 'both').
        final_strategy (str): Strategy for final missing values ('none', 'zero', 'mean', 'median').
        only_columns (list|None): Columns to fill; if None, fill all except id and date.
        calendar (pd.Index|None): Calendar of dates to use; default extracts union of dates.

    Returns:
        pd.DataFrame: Reindexed and filled DataFrame.
    """
    work = df.copy()

    # Normalize index -> columns to avoid ambiguity
    if isinstance(work.index, pd.MultiIndex) or work.index.name in (id_name, date_name):
        work = work.reset_index()

    work = work.loc[:, ~work.columns.duplicated()]  # keep first if dup names
    work[date_name] = pd.to_datetime(work[date_name], errors="raise")
    work = work.sort_values([id_name, date_name])

    # Build master calendar
    if calendar is None:
        # Fallback to union of dates present (behaves like your current version)
        calendar = pd.Index(work[date_name].drop_duplicates().sort_values(), name=date_name)
    else:
        calendar = pd.Index(pd.to_datetime(calendar), name=date_name).sort_values()

    ids = pd.Index(work[id_name].unique(), name=id_name)
    full_index = pd.MultiIndex.from_product([ids, calendar], names=[id_name, date_name])

    # Select columns
    if only_columns is None:
        cols = [c for c in work.columns if c not in (id_name, date_name)]
    else:
        keep = set(only_columns)
        cols = [c for c in work.columns if c in keep and c not in (id_name, date_name)]

    out = work.set_index([id_name, date_name])[cols].reindex(full_index).sort_index()

    # Optional per-id fill
    if fill in {"ffill", "bfill", "both"}:
        gb = out.groupby(level=0, group_keys=False)
        if fill == "ffill":
            out = gb.ffill()
        elif fill == "bfill":
            out = gb.bfill()
        else:
            out = gb.ffill().groupby(level=0, group_keys=False).bfill()

    # Final numeric fallback
    if final_strategy != "none":
        if final_strategy == "zero":
            out = out.fillna(0)
        elif final_strategy == "mean":
            out = out.fillna(out.mean(numeric_only=True))
        elif final_strategy == "median":
            out = out.fillna(out.median(numeric_only=True))
        else:
            raise ValueError("final_strategy must be one of {'none','zero','mean','median'}")

    return out


def build_model_matrix_from_wrds(
    wrds_user: str,
    start: str,
    end: str,
    chunk_size: int,
    tickers: list["str"],
    use_run: str,
) -> pd.DataFrame:
    """
    Perform a full WRDS extraction with configured SQL scripts, load data, apply preprocessing,
    joins and quality checks, add technical indicators, and build the final modeling matrix.

    Parameters:
        wrds_user (str): WRDS username.
        start (str): Extraction start date.
        end (str): Extraction end date.
        chunk_size (int): Chunk size for extraction.
        tickers (list['str']): list of tickers to keep.
        use_run (str): Run folder usage ('new', 'last', or explicit).

    Returns:
        pd.DataFrame: Final model matrix ready for predictive modeling.
    """
    res = wrds_extract_raw(
        wrds_user=wrds_user,
        start=start,
        end=end,
        chunk_size=chunk_size,
        use_run=use_run,
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
    stock_names = parquet_to_df(res["artifacts"], "stocknames.parquet")
    df_filter = filter_tickers(stock_names, tickers)
    stock_names = filter_by_tickers(stock_names, df_filter)

    dsf = parquet_to_df(res["artifacts"], "dsf.parquet")
    dsf = filter_by_tickers(dsf, df_filter)

    ff = parquet_to_df(res["artifacts"], "ff.parquet")
    ff = filter_by_tickers(ff, df_filter)

    ibes = parquet_to_df(res["artifacts"], "ibes_stats.parquet")
    ibes = filter_by_tickers(ibes, df_filter)

    ibes_act = parquet_to_df(res["artifacts"], "ibes_act.parquet")
    ibes_act = filter_by_tickers(ibes_act, df_filter)

    # Index & QA
    dsf = ensure_index(dsf, ["permno", "date"], keep_cols=False)
    pre_qa_dsf(dsf)
    dsf = clean_dsf(dsf)

    # adding stock_names
    pre_qa_stocknames(stock_names)
    df_prices = join_dsf_with_stocknames(dsf, stock_names)
    df_prices = ensure_index(df_prices, ["permno", "date"], keep_cols=False)
    df_prices = post_stockname_join_qa_cleaning(df_prices, remove_unclean_permnos=True)

    # adding FF
    pre_qa_ff(ff)
    df_prices = join_prices_with_ff(df_prices, ff)
    post_join_qa_prices_with_ff(df_prices)

    # adding IBES statsumu (EPS)
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
    df_prices = forward_fill_and_remove_initial_nans(df_prices, add_fill_source_columns=False)

    # Technical indicators
    df_prices = add_technical_indicators(df_prices)
    df_prices = _remove_leading_nans(df_prices, remove_reason="after adding technical indictors")

    # Final matrix
    model_df = build_model_matrix_from_df(df_prices)
    model_df = _remove_leading_nans(model_df, remove_reason="after build_model_matrix_from_df")

    print(f"[model] shape={model_df.shape}")
    print(f"[model] index={list(model_df.index.names)}")
    print(f"[model] columns={list(model_df.columns)}")

    print(null_report(model_df))

    return model_df


def trim_to_divisible_by_252(df: pd.DataFrame) -> pd.DataFrame:
    def trim_group(g):
        n = len(g)
        remainder = n % 252
        if remainder == 0:
            return g
        # Drop rows from beginning to make length divisible by 252
        return g.iloc[remainder:]

    # Assuming df has MultiIndex with 'permno' and 'date' as levels
    # Group by 'permno' and apply trim_group, sorting by 'date' within groups
    trimmed_df = (
        df.groupby(level=0, group_keys=False)
        .apply(lambda g: g.sort_index(level=1))
        .groupby(level=0, group_keys=False)
        .apply(trim_group)
    )
    return trimmed_df


def align_and_fill_dates_across_tickers(all_stocks: pd.DataFrame) -> pd.DataFrame:
    """
    Aligns and fills missing dates in a multi-index DataFrame of stock data to ensure all stocks (permnos)
    have the same date coverage.

    The function:
    - Copies the input DataFrame to avoid modifying the original.
    - Computes the union of all unique dates present across all stocks.
    - Computes the maximum of the minimum dates for each stock (permno) group, to find the
      common earliest date where all stocks have data.
    - Trims the global union of dates to start from this common earliest date.
    - For each stock group:
        - Reindexes the data to include all dates in the trimmed union (adding missing rows).
        - Forward fills missing data within the group.
    - Returns the filled and aligned DataFrame with consistent date indices per stock.

    Parameters
    ----------
    all_stocks : pd.DataFrame
        Multi-indexed DataFrame with index levels ['permno', 'date'].

    Returns
    -------
    pd.DataFrame
        A DataFrame aligned by dates across all stocks, with missing rows added and data forward-filled.
    """
    df = all_stocks.copy()

    # Union of all dates across all stocks, sorted
    all_dates = pd.Series(df.index.get_level_values("date").unique()).sort_values()

    # Find the latest minimum date among groups (common starting date)
    group_min_dates = df.groupby(level="permno").apply(
        lambda g: g.index.get_level_values("date").min()
    )
    max_start_date = group_min_dates.max()

    # Trim dates to start from this common date
    trimmed_dates = all_dates[all_dates >= max_start_date].reset_index(drop=True)

    def fill_group(group):
        full_idx = pd.MultiIndex.from_product(
            [[group.name], trimmed_dates], names=["permno", "date"]
        )
        group = group.reindex(full_idx)
        group = group.ffill()
        return group

    filled_df = df.groupby(level="permno", group_keys=False).apply(fill_group)

    # Validation: Check identical date indexes and equal row counts for all groups
    groups = filled_df.groupby(level="permno")
    reference_dates = None
    expected_len = None
    for permno, group in groups:
        dates = group.index.get_level_values("date")
        if reference_dates is None:
            reference_dates = dates
            expected_len = len(group)
        else:
            if len(group) != expected_len:
                raise ValueError(f"Row count mismatch found in permno {permno}.")
            if not dates.equals(reference_dates):
                raise ValueError(f"Date index mismatch found in permno {permno}.")

    print(f"All groups have consistent date indices and {expected_len} rows each.")

    print(df.shape)
    print(null_report(df))  # Assuming null_report is defined elsewhere to show missing values info
    print(df.index.names, df.columns)

    return filled_df
