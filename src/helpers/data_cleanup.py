import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)


def parquet_to_df(artifacts: Dict[str, str], name: str) -> pd.DataFrame:
    """
    Load a Parquet file from the from wrds extracts as a DataFrame.

    Args:
        artifacts (Dict[str, str]): Mapping of artifact names to Parquet file paths. mapping {parquet_name -> full_path}
        name (str): The key name of the Parquet artifact to load.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    
    Raises:
        FileNotFoundError: If the artifact is missing or path does not exist.
    """
    path = artifacts.get(name)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Required artifact missing: {name}")
    return pd.read_parquet(path)


def coalesce_date_end(s: pd.Series) -> pd.Series:
    """
    Convert NULL/NaT end dates (CRSP convention for 'still valid') to a far-future timestamp.

    Args:
        s (pd.Series): Series of date-like values representing end dates.

    Returns:
        pd.Series: Series with NaT replaced by `9999-12-31`.
    """
    return s.fillna(pd.Timestamp("9999-12-31"))


def _ensure_datetime_cols(df: pd.DataFrame, cols: List[str], label: str) -> None:
    """
    Ensure specified columns exist in df and are of datetime64 dtype; convert if possible.

    Args:
        df (pd.DataFrame): DataFrame to check/convert.
        cols (List[str]): List of column names to enforce datetime on.
        label (str): Name label for error messages.

    Raises:
        KeyError: If given columns not present.
        TypeError: If conversion to datetime64 is impossible.
    """
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"{label}: '{c}' not found as a column.")
        if not np.issubdtype(df[c].dtype, np.datetime64):
            try:
                df[c] = pd.to_datetime(df[c], errors="raise")
            except Exception as e:
                raise TypeError(
                    f"{label}.{c} (column) must be datetime64 and could not be converted. "
                    f"Got {df[c].dtype}. Error: {e}"
                )


def _ensure_datetime_index(df: pd.DataFrame, levels: List[str], label: str) -> pd.DataFrame:
    """
    Ensure specified index levels are datetime64, converting if needed. Returns DataFrame with fixed index.

    Args:
        df (pd.DataFrame): DataFrame to check.
        levels (List[str]): Index level names that must be datetime64.
        label (str): Label for error messages.

    Returns:
        pd.DataFrame: DataFrame with updated datetime index levels.

    Raises:
        KeyError: If index levels missing.
        TypeError: If conversion fails.
    """
    if df.index.names is None:
        raise KeyError(f"{label}: no named index to fix.")
    idx_names = list(df.index.names)
    for lvl in levels:
        if lvl not in idx_names:
            raise KeyError(f"{label}: '{lvl}' not found as an index level.")
    # Rebuild MultiIndex safely via frame, convert, rebuild
    idx_df = df.index.to_frame(index=False)
    for lvl in levels:
        if not np.issubdtype(idx_df[lvl].dtype, np.datetime64):
            try:
                idx_df[lvl] = pd.to_datetime(idx_df[lvl], errors="raise")
            except Exception as e:
                raise TypeError(
                    f"{label}.{lvl} (index level) must be datetime64 and could not be converted. "
                    f"Got {idx_df[lvl].dtype}. Error: {e}"
                )
    df = df.copy()
    df.index = (
        pd.MultiIndex.from_frame(idx_df)
        if len(idx_df.columns) > 1
        else pd.Index(idx_df[idx_df.columns[0]])
    )
    df.index.names = idx_names
    return df


def ensure_index(
    df: pd.DataFrame, cols: List[str], *, sort: bool = True, keep_cols: bool = True
) -> pd.DataFrame:
    """
    Set the specified columns as the (multi)index of the DataFrame.
    Optionally, keep those columns as part of the DataFrame (do not drop).
    Sort the DataFrame by its index after setting the index.
    This function is idempotent: if the DataFrame's current index names match the
    requested columns, it will not reset the index but will sort it if requested.

    Args:
        df (pd.DataFrame): The DataFrame to operate on.
        cols (List[str]): List of column names to set as the index.
        sort (bool, optional): Whether to sort the DataFrame by the new index. Default True.
        keep_cols (bool, optional): Whether to keep the columns as regular columns (True) or drop them (False). Default True.

    Returns:
        pd.DataFrame: DataFrame with the specified columns as index, optionally sorted.

    Example:
        df = ensure_index(df, ['permno', 'date'], sort=True, keep_cols=False)
    """
    if list(df.index.names or []) != cols:
        df = df.set_index(cols, drop=not keep_cols)
    if sort:
        df = df.sort_index()
    return df


def _get_key_frame(df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    """
    Build a temporary DataFrame that contains only the requested key columns as normal columns,
    regardless of whether those keys are originally columns or index levels in the input DataFrame.

    Args:
        df (pd.DataFrame): The source DataFrame.
        key_cols (List[str]): Names of the key columns or index levels to extract.

    Returns:
        pd.DataFrame: A new DataFrame containing the key columns as regular columns.

    Raises:
        KeyError: If any key in key_cols is not found as either a column or an index level.

    Example:
        # If 'permno' is an index level and 'date' is a column
        keys_df = _get_key_frame(df, ['permno', 'date'])
    """
    idx_names = list(df.index.names) if df.index.names is not None else []
    data = {}
    for k in key_cols:
        if k in df.columns:
            data[k] = df[k].to_numpy()
        elif k in idx_names:
            data[k] = df.index.get_level_values(k).to_numpy()
        else:
            raise KeyError(f"Key '{k}' not found as column or index level.")
    return pd.DataFrame(data)


def check_key_dupes(df: pd.DataFrame, key_cols: List[str], label: str) -> None:
    """
    Check that specified key columns form a unique key in the DataFrame, across columns or index.

    Args:
        df (pd.DataFrame): DataFrame to check.
        key_cols (List[str]): List of columns or index levels to verify uniqueness.
        label (str): Label referring to data for error messages.

    Raises:
        AssertionError: If duplicates found, providing top offending examples.
    """
    keys_df = _get_key_frame(df, key_cols)
    dup_mask = keys_df.duplicated(keep=False)
    if dup_mask.any():
        ex = (
            keys_df[dup_mask]
            .groupby(key_cols, dropna=False)
            .size()
            .sort_values(ascending=False)
            .head(10)
        )
        raise AssertionError(f"{label}: duplicate keys on {key_cols}. Top examples (counts):\n{ex}")


def assert_no_new_rows(
    df_left: pd.DataFrame, df_joined: pd.DataFrame, *, left_name: str, join_name: str
) -> None:
    """
    Ensure that a join does not produce more rows than the left DataFrame,
    protecting against accidental many-to-many expansions.

    Args:
        df_left (pd.DataFrame): Left-side DataFrame before join.
        df_joined (pd.DataFrame): DataFrame after join.
        left_name (str): Name label for df_left for error messages.
        join_name (str): Name label for join/DataFrame after join.

    Raises:
        AssertionError: If df_joined has more rows than df_left.
    """
    if df_joined.shape[0] > df_left.shape[0]:
        raise AssertionError(
            f"{join_name}: produced {df_joined.shape[0]:,} rows > {left_name} {df_left.shape[0]:,}. "
            f"Likely interval overlap or non-unique keys."
        )


def _to_columns(df: pd.DataFrame, names: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Ensure that specified names exist as regular columns in the DataFrame,
    copying them from the index levels if necessary.

    Args:
        df (pd.DataFrame): Input DataFrame.
        names (List[str]): Names of columns or index levels to ensure as columns.

    Returns:
        Tuple[pd.DataFrame, List[str]]: A copy of df with the requested names as columns,
                                        and a list of names that were added from the index.

    Raises:
        KeyError: If any name is not found as columns or index levels.
    """
    out = df.copy()
    added: List[str] = []
    idx_names = list(out.index.names or [])
    for n in names:
        if n in out.columns:
            continue
        if n in idx_names:
            out[n] = out.index.get_level_values(n)
            added.append(n)
        else:
            raise KeyError(f"'{n}' not found as column or index.")
    return out, added


def _restore_index_if_needed(
    df: pd.DataFrame, original_index_names: List[str] | None
) -> pd.DataFrame:
    """
    Restore the original index to the DataFrame if index names are provided;
    otherwise, returns the DataFrame unchanged.

    Args:
        df (pd.DataFrame): DataFrame to adjust.
        original_index_names (List[str] | None): Original index column names.

    Returns:
        pd.DataFrame: DataFrame with restored index if names provided, else unchanged.
    """
    if original_index_names and any(original_index_names):
        return df.set_index(original_index_names)
    return df


def pre_qa_dsf(dsf: pd.DataFrame) -> None:
    """
    Perform preliminary quality assurance checks on the Daily Stock File (DSF) DataFrame.

    This function validates key aspects of the DSF data before analysis, including:
    - Ensuring the 'date' field exists and is of datetime type, either as a column or index.
    - Warning if adjustment factors ('cfacpr', 'cfacshr') contain zero values.
    - Raising an error if adjustment factors ('cfacpr', 'cfacshr') have negative values.
    - Warning if there are any negative prices in the 'prc' column.
    - Validating the uniqueness of the primary key composed of ('permno', 'date').

    Args:
        dsf (pd.DataFrame): Daily Stock File DataFrame.

    Raises:
        KeyError: If 'date' is not found.
    """
    # Handle date as column or index
    if "date" in (dsf.index.names or []):
        dsf_fixed = _ensure_datetime_index(dsf, ["date"], "dsf")
        if dsf_fixed is not dsf:
            dsf.index = dsf_fixed.index
    elif "date" in dsf.columns:
        _ensure_datetime_cols(dsf, ["date"], "dsf")
    else:
        raise KeyError("dsf: 'date' not found as column or index.")

    if (dsf["cfacpr"] < 0).any():
        raise ValueError("dsf: cfacpr < 0 (unexpected).")
    if (dsf["cfacshr"] < 0).any():
        raise ValueError("dsf: cfacshr < 0 (unexpected).")
    
    if (dsf["cfacpr"] == 0).any():
         print(f"[warn] dsf: some rows have zero cfacpr.")
    if (dsf["cfacshr"] == 0).any():
        print(f"[warn] dsf: some rows have zero cfacshr.")


    n_neg = int((dsf["prc"] < 0).sum())
    if n_neg:
        pct = round(n_neg / max(len(dsf), 1) * 100, 2)
        print(f"[warn] dsf: {n_neg:,} rows have negative prices ({pct}%).")

    check_key_dupes(dsf, ["permno", "date"], "dsf")


def _handle_zero_cfa_factors(df: pd.DataFrame, grouper) -> (pd.DataFrame):
    """
    Remove all rows where groups (by 'permno') contain any zero values 
    in 'cfacpr' or 'cfacshr' columns.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing financial data.
        grouper: GroupBy object grouping df by 'permno'.

    Returns:
        filtered_df (pd.DataFrame): DataFrame with groups containing zeros removed.
        removed_permnos (list): List of 'permno' values that were removed.

    Side Effects:
        Prints a message listing the 'permno' values removed.
    """

    # Find permnos where any row has zero in cfacpr or cfacshr
    removed_permnos = [
        permno for permno, group in grouper
        if (group["cfacpr"] == 0).any() or (group["cfacshr"] == 0).any()
    ]
    print(f"[info] Removed {len(removed_permnos)} permnos(companies) for having zero in cfacpr or cfacshr")

    # Filter out rows with those permnos
    filtered_df = df[~df.index.get_level_values('permno').isin(removed_permnos)].copy()
    return filtered_df

def _handle_negative_price(df: pd.DataFrame, grouper, max_neg_price_pct: float) -> (pd.DataFrame, list):
    """
    Remove groups where the proportion of rows with negative price exceeds max_neg_price_pct.

    Parameters:
        df (pd.DataFrame): DataFrame containing stock data.
        grouper: GroupBy object grouping df by 'permno'.
        max_neg_price_pct (float): Maximum allowed fraction (0 to 1) of negative prices per permno.

    Returns:
        filtered_df (pd.DataFrame): DataFrame after removing groups.
        removed_permnos (list): List of permnos removed.
        
    Side Effects:
        Prints [info] with the permnos removed.
    """
    removed_permnos = []
    for permno, group in grouper:
        neg_count = (group["prc"] < 0).sum()
        group_len = len(group)
        if neg_count > max_neg_price_pct * group_len:
            removed_permnos.append(permno)
    print(f"[info] Removed {len(removed_permnos)} permnos(companies) for exceeding the threshold of negative prices")
    filtered_df = df[~df.index.get_level_values('permno').isin(removed_permnos)].copy()
    filtered_df.loc[:, "prc"] = filtered_df["prc"].abs()

    return filtered_df

def clean_dsf(dsf: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and adjust the Daily Stock File DataFrame for financial analysis.

    The function performs the following steps grouped by 'permno':
    - Removes all groups containing zero values in 'cfacpr' or 'cfacshr'.
    - Removes groups where the percentage of negative prices exceeds a threshold (1%).
    - Converts all remaining prices to their absolute values.
    - Recomputes adjusted price, adjusted shares outstanding, and market capitalization.

    Args:
        dsf (pd.DataFrame): Daily Stock File DataFrame with columns including 'permno',
            'cfacpr', 'cfacshr', 'prc', and optionally 'shrout'.

    Returns:
        pd.DataFrame: A cleaned and adjusted copy of the input DataFrame ready for further analysis.

    Side Effects:
        Prints [info] messages listing permnos removed during cleaning steps.
    """
    # Work with a sorted view, regardless of index vs columns
    idx_names = list(dsf.index.names or [])
    has_permno_idx = "permno" in idx_names
    has_date_idx = "date" in idx_names

    out = dsf.copy()
    if has_permno_idx and has_date_idx:
        out = out.sort_index()
        grouper = out.groupby(level="permno", group_keys=False)
    else:
        # Ensure needed columns exist
        out, _ = _to_columns(out, ["permno", "date"])
        out = out.sort_values(["permno", "date"])
        grouper = out.groupby("permno", group_keys=False)

    # Find permnos where any row has zero in cfacpr or cfacshr columns
    out = _handle_zero_cfa_factors(out, grouper)
    out = _handle_negative_price(out, grouper, max_neg_price_pct=0.01)

    # Recompute adjusted fields
    out["adj_prc"] = out["prc"] * out["cfacpr"]
    if "shrout" in out.columns:
        out["adj_shrout"] = out["shrout"] * out["cfacshr"]
        out["adj_mktcap"] = out["adj_prc"].abs() * out["adj_shrout"]

    return out


def pre_qa_stocknames(sn: pd.DataFrame) -> None:
    """
    Preliminary quality checks on stock names file.

    Checks:
    - 'namedt' and 'nameenddt' are datetime types.
    - Warns if any 'namedt' > 'nameenddt'.
    - Warns if overlapping name windows exist for the same permno.

    Args:
        sn (pd.DataFrame): Stock names data.
    """
    # Handle dates whether columns or index levels
    if {"namedt", "nameenddt"}.issubset(sn.columns):
        _ensure_datetime_cols(sn, ["namedt", "nameenddt"], "stocknames")
    else:
        # Convert whichever are on the index
        to_fix = [n for n in ["namedt", "nameenddt"] if n in (sn.index.names or [])]
        if to_fix:
            sn_fixed = _ensure_datetime_index(sn, to_fix, "stocknames")
            if sn_fixed is not sn:
                sn.index = sn_fixed.index

    # warn if any namedt > nameenddt (data error)
    if "namedt" in sn.columns and "nameenddt" in sn.columns:
        bad = (sn["nameenddt"].notna()) & (sn["namedt"] > sn["nameenddt"])
        if bad.any():
            n = int(bad.sum())
            print(f"[warn] stocknames: {n:,} rows have namedt > nameenddt.")

    # overlap warning
    sn_lite = sn.copy()
    sn_lite["nameenddt_eff"] = (
        coalesce_date_end(
            sn_lite["nameenddt"]
            if "nameenddt" in sn_lite.columns
            else sn_lite.index.get_level_values("nameenddt")
        )
        if ("nameenddt" in sn_lite.columns or "nameenddt" in (sn_lite.index.names or []))
        else pd.NaT
    )

    overlaps = 0
    if "permno" in sn_lite.columns:
        group_iter = sn_lite.groupby("permno", sort=False)
    elif "permno" in (sn_lite.index.names or []):
        group_iter = sn_lite.groupby(level="permno", sort=False)
    else:
        group_iter = []

    for _, g in group_iter:
        # ensure columns exist for compare
        g2, _ = _to_columns(g, ["namedt"])
        g2["nameenddt_eff"] = coalesce_date_end(
            g2["nameenddt"] if "nameenddt" in g2.columns else g2["nameenddt_eff"]
        )
        g2 = g2.sort_values("namedt")
        if len(g2) > 1 and (g2["namedt"].values[1:] <= g2["nameenddt_eff"].values[:-1]).any():
            overlaps += 1

    if overlaps:
        print(
            f"[warn] stocknames: {overlaps} permno have overlapping name windows; "
            f"as-of join will pick the record with the latest namedt per (permno,date)."
        )


def join_dsf_with_stocknames(dsf: pd.DataFrame, stock_names: pd.DataFrame) -> pd.DataFrame:
    """
    Perform an as-of join between the Daily Stock File (DSF) and stock names.

    Steps:
    1. Reset indexes and work with columns for clarity.
    2. Left merge on 'permno' (unique stock identifier).
    3. Filter merged rows where DSF date falls into the valid name interval [namedt, nameenddt_eff].
    4. For overlapping intervals, keep the record with the latest 'namedt'.
    5. Ensure no row inflation after the join.
    6. Drop interval columns used for filtering.
    7. Prefer 'ncusip' over 'cusip' if both exist.
    8. Restore original index.

    Args:
        dsf (pd.DataFrame): Daily Stock File data.
        stock_names (pd.DataFrame): Stock names data with validity intervals.

    Returns:
        pd.DataFrame: Joined DataFrame with stock names merged into DSF.
    """
    """
    As-of join logic:
      1) Work in columns (reset index if needed).
      2) Left-merge on permno.
      3) Filter to rows where date ∈ [namedt, nameenddt_eff] (keep nulls for left semantics).
      4) If overlaps -> keep the record with the latest namedt per (permno,date).
      5) Safety: ensure row count didn't increase; drop interval columns; prefer ncusip.
      6) Return columns (caller can set index after).
    """
    # Remember original DSF index to restore if needed
    orig_idx = list(dsf.index.names or [])
    dsf_c = dsf.reset_index() if any(orig_idx) else dsf.copy()

    sn_c = stock_names.reset_index() if any(stock_names.index.names or []) else stock_names.copy()
    sn_c["nameenddt_eff"] = coalesce_date_end(sn_c["nameenddt"])

    _ensure_datetime_cols(dsf_c, ["date"], "dsf")
    _ensure_datetime_cols(sn_c, ["namedt"], "stocknames")

    merged = dsf_c.merge(
        sn_c[["permno", "ticker", "ncusip", "namedt", "nameenddt_eff"]],
        on="permno",
        how="left",
        suffixes=("", "_sn"),
    )

    # validity filter (or keep unmatched)
    valid = (merged["date"] >= merged["namedt"]) & (merged["date"] <= merged["nameenddt_eff"])
    merged = merged.loc[valid | merged["namedt"].isna()]

    # overlap collapse: keep latest namedt
    if {"namedt", "date", "permno"}.issubset(merged.columns):
        sentinel = pd.Timestamp("1900-01-01")  # NA last
        merged["_namedt_sort"] = merged["namedt"].fillna(sentinel)
        merged = merged.sort_values(["permno", "date", "_namedt_sort"])
        merged = merged.drop_duplicates(subset=["permno", "date"], keep="last")
        merged = merged.drop(columns=["_namedt_sort"])

    # safety checks & clean-up
    assert_no_new_rows(dsf_c, merged, left_name="dsf", join_name="dsf <- stocknames")
    merged = merged.drop(columns=["namedt", "nameenddt_eff"], errors="ignore")
    if "cusip" in merged.columns and "ncusip" in merged.columns:
        merged = merged.drop(columns=["cusip"])

    check_key_dupes(merged, ["permno", "date"], "post-join dsf")

    # restore original index if DSF had one
    return _restore_index_if_needed(merged, orig_idx)


def post_stockname_join_qa_cleaning(df: pd.DataFrame,  remove_unclean_permnos: bool = True) -> None:
    """
    Basic quality checks after joining DSF with stock names and Optionally removes unclean permnos.

    Checks:
    - Uniqueness on (permno, date).
    - Reports fraction of rows missing ticker mapping.
    - Warns on near-zero adjusted prices.
    - Warns on negative market capitalizations.
    - Optionally removes rows with permnos that have any null ticker values.

    Args:
        df (pd.DataFrame): DataFrame after DSF-stocknames join.
        remove_unclean_permnos (bool): If True, remove all rows with permnos having any null ticker row.

    Returns:
        pd.DataFrame: Possibly cleaned dataframe.
    """
    check_key_dupes(df, ["permno", "date"], "df_prices")

    if "ticker" in df.columns:
        null_ticker_rate = float(df["ticker"].isna().mean())
        if null_ticker_rate > 0:
            print(
                f"[info] df_prices: {null_ticker_rate:.2%} rows lack ticker mapping on that date."
            )
            if remove_unclean_permnos:
                unclean_permnos = df.loc[df["ticker"].isna()].index.get_level_values("permno").unique()
                mask = ~df.index.get_level_values("permno").isin(unclean_permnos)
                df = df.loc[mask].copy()
                print(f"[info] Removed {len(unclean_permnos)} permnos with null ticker rows: {unclean_permnos}")
                new_null_ticker_rate = float(df["ticker"].isna().mean())
                print(f"[info] df_prices:after cleaning there are {new_null_ticker_rate:.2%} rows lack ticker mapping on that date.")

    if "adj_prc" in df.columns:
        near_zero = int((df["adj_prc"].abs() <= 1e-8).sum())
        if near_zero:
            print(f"[warn] df_prices: {near_zero:,} rows with ~0 adjusted price.")

    if "adj_mktcap" in df.columns:
        neg_cap = int((df["adj_mktcap"] < 0).sum())
        if neg_cap:
            print(f"[warn] df_prices: {neg_cap:,} rows with negative market cap.")
    return df



def pre_qa_ff(ff: pd.DataFrame) -> None:
    """
    Basic preprocessing quality assurance for Fama-French daily factors.

    Checks required presence of columns: date, mktrf, smb, hml, rf.
    Ensures 'date' is datetime64.
    Verifies uniqueness by date.
    Warns on infinite or missing values.
    Warns on exceptionally large daily factor values.

    Args:
        ff (pd.DataFrame): Fama-French factor data.
    """
    req = {"date", "mktrf", "smb", "hml", "rf"}
    missing = req - set(ff.columns)
    if missing:
        raise AssertionError(f"ff: missing columns: {missing}")

    _ensure_datetime_cols(ff, ["date"], "ff")
    check_key_dupes(ff, ["date"], "ff")

    # Finite checks
    for col in ["mktrf", "smb", "hml", "rf", "umd"]:
        if col in ff.columns:
            n_inf = np.isinf(ff[col]).sum()
            n_nan = ff[col].isna().sum()
            if n_inf:
                print(f"[warn] ff: {col} has {n_inf:,} ±inf values.")
            if n_nan:
                share = float(n_nan / len(ff))
                print(f"[info] ff: {share:.2%} NaN in {col}.")

    # Magnitude sanity (daily factors rarely exceed ±50% in absolute terms)
    for col in ["mktrf", "smb", "hml", "umd"]:
        if col in ff.columns:
            extreme = (ff[col].abs() > 0.50).sum()
            if extreme:
                print(f"[warn] ff: {extreme:,} rows with |{col}| > 50%.")

    # Risk-free rate sanity (daily ~ few bps; warn if absurd)
    if "rf" in ff.columns:
        extreme_rf = (ff["rf"].abs() > 0.05).sum()
        if extreme_rf:
            print(f"[warn] ff: {extreme_rf:,} rows with |rf| > 5% (daily).")


def join_prices_with_ff(df_prices: pd.DataFrame, ff: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join daily prices with Fama-French daily factors on 'date'.

    Preserves original index and guarantees no row inflation.

    Args:
        df_prices (pd.DataFrame): Daily price data.
        ff (pd.DataFrame): Fama-French factor data.

    Returns:
        pd.DataFrame: Joined price and factor data.
    """
    # Preserve original index (if any)
    orig_idx = list(df_prices.index.names or [])
    p = df_prices.reset_index() if any(orig_idx) else df_prices.copy()
    f = ff.copy()

    _ensure_datetime_cols(p, ["date"], "df_prices")
    _ensure_datetime_cols(f, ["date"], "ff")

    pre_rows = int(p.shape[0])
    merged = p.merge(
        f,
        on="date",
        how="left",
        suffixes=("", "_ff"),
    )

    if merged.shape[0] != pre_rows:
        raise AssertionError(
            f"df_prices <- ff merge changed row count: {merged.shape[0]:,} vs {pre_rows:,}"
        )

    # Restore original index if there was one
    if any(orig_idx):
        merged = merged.set_index(orig_idx)

    return merged


def post_join_qa_prices_with_ff(df: pd.DataFrame) -> None:
    """
    Quality checks after adding Fama-French factors.

    Checks uniqueness on (permno, date).
    Reports missing values for key factors.

    Args:
        df (pd.DataFrame): DataFrame after joining with FF factors.
    """
    # If the frame is indexed by (permno,date) we’re good; else check via columns
    idx_names = list(df.index.names or [])
    if {"permno", "date"}.issubset(set(idx_names)):
        pass
    else:
        check_key_dupes(df, ["permno", "date"], "df_prices_ff")

    for col in ["mktrf", "smb", "hml", "rf", "umd"]:
        if col in df.columns:
            miss = float(df[col].isna().mean())
            if miss:
                print(f"[info] df_prices_ff: {miss:.2%} missing {col}.")


def pre_qa_ibes_statsumu(ibes: pd.DataFrame) -> None:
    """
    Basic hygiene for IBES statsumu (unadjusted EPS consensus):

    Checks required columns are present.
    Ensures 'stat_date' is datetime.
    Reports cases with multiple rows per (official_ticker, stat_date).

    Args:
        ibes (pd.DataFrame): IBES statsumu data.
    """
    req = {
        "official_ticker",  # oftic
        "stat_date",  # statpers
        "periodicity",  # A/Q/S
        "fpi",  # horizon indicator
        "n_analysts",  # numest
        "cons_mean",  # meanest
    }
    missing = req - set(ibes.columns)
    if missing:
        raise AssertionError(f"ibes_stats missing columns: {missing}")

    _ensure_datetime_cols(ibes, ["stat_date"], "ibes_stats")

    # Show how many rows per (official_ticker, stat_date)
    grp = ibes.groupby(["official_ticker", "stat_date"], dropna=False).size()
    multi = int((grp > 1).sum())
    if multi:
        print(
            f"[info] ibes_stats: {multi:,} (official_ticker, stat_date) pairs have >1 row "
            f"(multiple horizons/periodicities). Will collapse before join."
        )


def prepare_ibes_for_daily_merge(ibes: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse IBES data to one row per (official_ticker, stat_date) to avoid row inflation.

    Preference order:
    1) Quarterly ('Q') periodicity over others.
    2) Smallest forecast horizon (fpi).
    3) Largest number of analysts.
    4) First occurrence if ties.

    Args:
        ibes (pd.DataFrame): Raw IBES consensus data.

    Returns:
        pd.DataFrame: Deduplicated IBES data ready for merge.
    """
    ib = ibes.copy()

    # Rank preference: Q first, then S, then A (map missing to worst)
    per_rank = {"Q": 0, "S": 1, "A": 2}
    ib["_per_rank"] = ib["periodicity"].map(per_rank).fillna(3).astype(int)

    # fpi can be string in IBES; make it numeric for proper ordering when possible
    with pd.option_context("mode.chained_assignment", None):
        ib["_fpi_num"] = pd.to_numeric(ib["fpi"], errors="coerce")

    # Build a sort key
    ib = ib.sort_values(
        by=["official_ticker", "stat_date", "_per_rank", "_fpi_num", "n_analysts"],
        ascending=[True, True, True, True, False],
        kind="mergesort",  # stable
    )

    # Deduplicate to the best row per (official_ticker, stat_date)
    ib = ib.drop_duplicates(subset=["official_ticker", "stat_date"], keep="first")

    # Keep a lean set of columns (expand as needed)
    keep = [
        "official_ticker",
        "stat_date",
        "currency",
        "periodicity",
        "fpi",
        "n_analysts",
        "n_up",
        "n_down",
        "cons_mean",
        "cons_median",
        "cons_stdev",
        "cons_high",
        "cons_low",
        "fpe_date",
        "cons_cv",
        "cons_range_pct",
    ]
    keep = [c for c in keep if c in ib.columns]
    ib = ib[keep].copy()

    return ib


def join_prices_with_ibes(df_prices: pd.DataFrame, ibes_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Join daily prices with IBES consensus on (ticker, date) keys.

    Left join without row inflation.
    Restores dataframe original index.

    Args:
        df_prices (pd.DataFrame): Prices data.
        ibes_daily (pd.DataFrame): Prepared IBES consensus data.

    Returns:
        pd.DataFrame: DataFrame enriched with IBES consensus.
    """
    # Work in columns for merge clarity; remember original index
    base_index_names = list(df_prices.index.names or [])
    need_reset = any(base_index_names)
    df_tmp = df_prices.reset_index() if need_reset else df_prices.copy()

    # Ensure date columns are datetime64
    _ensure_datetime_cols(df_tmp, ["date"], "df_prices")
    _ensure_datetime_cols(ibes_daily, ["stat_date"], "ibes_stats")

    pre_rows = int(df_tmp.shape[0])
    if "ticker" not in df_tmp.columns:
        print("[warn] df_prices has no 'ticker' column; IBES match will be empty.")
    if "official_ticker" not in ibes_daily.columns:
        raise AssertionError("ibes_daily lacks 'official_ticker' after preparation.")

    merged = df_tmp.merge(
        ibes_daily,
        left_on=["ticker", "date"],
        right_on=["official_ticker", "stat_date"],
        how="left",
        suffixes=("", "_ibes"),
    )

    # Ensure no row inflation
    if merged.shape[0] != pre_rows:
        raise AssertionError(
            f"df_prices <- ibes merge changed row count: {merged.shape[0]:,} vs {pre_rows:,}"
        )

    # Drop redundant join keys from IBES side
    merged = merged.drop(columns=["official_ticker", "stat_date"], errors="ignore")

    # Restore original index
    if need_reset:
        merged = merged.set_index(base_index_names)

    return merged


def post_join_qa_prices_with_ibes(df: pd.DataFrame) -> None:
    """
    Quality assurance after merging IBES consensus.

    Checks:
    - Uniqueness on (permno, date).
    - Reports coverage (missingness) of consensus columns.

    Args:
        df (pd.DataFrame): Price data merged with IBES.
    """
    idx_names = list(df.index.names or [])
    if {"permno", "date"}.issubset(idx_names):
        pass
    else:
        check_key_dupes(df, ["permno", "date"], "df_prices(+ibes)")

    # Coverage sanity
    for col in ["n_analysts", "cons_mean", "cons_stdev", "cons_cv", "fpe_date"]:
        if col in df.columns:
            miss = float(df[col].isna().mean())
            if miss > 0:
                print(f"[info] df_prices(+ibes): {miss:.1%} missing in {col}.")


def pre_qa_ibes_actu(ibes_act: pd.DataFrame) -> None:
    """
    Hygiene checks for IBES actuals (real EPS):

    - Required columns present.
    - 'anndats' is datetime.
    - Reports multiple rows per (official_ticker, anndats), if any.

    Args:
        ibes_act (pd.DataFrame): IBES actual EPS data.
    """
    req = {"oftic", "anndats", "pdicity", "act_measure", "act_value", "usfirm"}
    missing = req - set(ibes_act.columns)
    if missing:
        raise AssertionError(f"ibes_act missing columns: {missing}")

    _ensure_datetime_cols(ibes_act, ["anndats"], "ibes_act")

    multi = (ibes_act.groupby(["oftic", "anndats"], dropna=False).size() > 1).sum()
    if multi:
        print(
            f"[info] ibes_act: {multi:,} (oftic, anndats) pairs have >1 row "
            f"(periodicity/dup loads). Will collapse before join."
        )


def prepare_ibes_actu_for_daily_merge(ibes_act: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse IBES actuals to one row per (official_ticker, anndats).

    Preference order:
    1) Quarterly ('Q') over other periodicities.
    2) Most recent activation date.
    3) First occurrence if ties.

    Args:
        ibes_act (pd.DataFrame): Raw IBES actual EPS data.

    Returns:
        pd.DataFrame: Deduplicated and cleaned IBES actuals.
    """
    ib = ibes_act.copy()

    # rename to align with prices side & consistency
    ib = ib.rename(columns={"oftic": "official_ticker", "anndats": "ann_date"})

    # periodicity rank: Q better than A; unknown worst
    per_rank = {"Q": 0, "A": 1}
    ib["_per_rank"] = ib["pdicity"].map(per_rank).fillna(2).astype(int)

    # actdats to datetime (sometimes already)
    if "actdats" in ib.columns and not np.issubdtype(ib["actdats"].dtype, np.datetime64):
        with pd.option_context("mode.chained_assignment", None):
            ib["actdats"] = pd.to_datetime(ib["actdats"], errors="coerce")

    ib = ib.sort_values(
        by=["official_ticker", "ann_date", "_per_rank", "actdats"],
        ascending=[True, True, True, True],
        kind="mergesort",
    )

    ib = ib.drop_duplicates(subset=["official_ticker", "ann_date"], keep="first")

    keep = [
        "official_ticker",
        "ann_date",
        "pdicity",
        "act_measure",
        "act_value",
        "curr_act",
        "anntims",
        "actdats",
        "acttims",
        "pends",  # fiscal period end
    ]
    keep = [c for c in keep if c in ib.columns]
    return ib[keep].copy()


def join_prices_with_ibes_actu(
    df_prices: pd.DataFrame, ibes_act_daily: pd.DataFrame
) -> pd.DataFrame:
    """
    Left-join daily prices with IBES actual EPS data on (ticker == official_ticker) and (date == ann_date).

    Guarantees:
    - No row inflation (same number of rows as df_prices).
    - Restores the original index of df_prices if it had one.

    Args:
        df_prices (pd.DataFrame): Daily prices data.
        ibes_act_daily (pd.DataFrame): Prepared IBES actual EPS data.

    Returns:
        pd.DataFrame: Prices enriched with actual EPS announcements.
    """
    base_index = list(df_prices.index.names or [])
    need_reset = any(base_index)
    left = df_prices.reset_index() if need_reset else df_prices.copy()

    _ensure_datetime_cols(left, ["date"], "df_prices")
    _ensure_datetime_cols(ibes_act_daily, ["ann_date"], "ibes_act")

    pre_rows = int(left.shape[0])

    if "ticker" not in left.columns:
        print("[warn] df_prices has no 'ticker'; IBES actuals match will be empty.")

    merged = left.merge(
        ibes_act_daily,
        left_on=["ticker", "date"],
        right_on=["official_ticker", "ann_date"],
        how="left",
        suffixes=("", "_act"),
    )

    if merged.shape[0] != pre_rows:
        raise AssertionError(
            f"df_prices <- ibes_act merge changed row count: {merged.shape[0]:,} vs {pre_rows:,}"
        )

    merged = merged.drop(columns=["official_ticker", "ann_date"], errors="ignore")
    if need_reset:
        merged = merged.set_index(base_index)
    return merged


def post_join_qa_prices_with_ibes_actu(df: pd.DataFrame) -> None:
    """
    Quality assurance checks after merging IBES actual EPS data.

    Checks:
    - Uniqueness on (permno, date).
    - Reports coverage (missingness) of the 'act_value' column.
    - Optionally reports distribution of announcement time buckets if present.

    Args:
        df (pd.DataFrame): DataFrame with prices and IBES actuals.
    """
    idx_names = list(df.index.names or [])
    if not {"permno", "date"}.issubset(idx_names):
        check_key_dupes(df, ["permno", "date"], "df_prices(+ibes_act)")

    if "act_value" in df.columns:
        miss = float(df["act_value"].isna().mean())
        print(f"[info] df_prices(+ibes_act): {miss:.1%} missing in act_value.")

    # Optional: counts by announcement time bucket (if present)
    if "anntims" in df.columns:
        vc = df["anntims"].dropna().astype(str).str.lower().value_counts().head(5)
        if not vc.empty:
            print(f"[info] ibes_act announcement time top values:\n{vc}")
