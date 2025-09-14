import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)


def parquet_to_df(artifacts: Dict[str, str], name: str) -> pd.DataFrame:
    """
    artifacts: mapping {parquet_name -> full_path} from wrds_extract_raw
    """
    path = artifacts.get(name)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Required artifact missing: {name}")
    return pd.read_parquet(path)


def coalesce_date_end(s: pd.Series) -> pd.Series:
    """CRSP uses NULL end as 'still valid'; convert to far-future date."""
    return s.fillna(pd.Timestamp("9999-12-31"))


def _ensure_datetime_cols(df: pd.DataFrame, cols: List[str], label: str) -> None:
    """
    Ensure the listed *columns* are datetime64 (convert in-place if needed).
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
    Ensure the listed *index levels* are datetime64 (return a DataFrame with fixed index).
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
    Make `cols` the (multi)index, optionally keep them as columns, and sort.
    Idempotent: if index already matches, just (re)sort.
    """
    if list(df.index.names or []) != cols:
        df = df.set_index(cols, drop=not keep_cols)
    if sort:
        df = df.sort_index()
    return df


def _get_key_frame(df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    """
    Build a temporary DataFrame with the requested keys as *columns*,
    regardless of whether they exist as columns or index levels.
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
    Ensure `key_cols` form a unique key (names can be columns OR index levels).
    Raises with top offenders if duplicates exist.
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
    Protect against accidental many-to-many fan-out.
    """
    if df_joined.shape[0] > df_left.shape[0]:
        raise AssertionError(
            f"{join_name}: produced {df_joined.shape[0]:,} rows > {left_name} {df_left.shape[0]:,}. "
            f"Likely interval overlap or non-unique keys."
        )


def _fill_prev_positive(series: pd.Series) -> pd.Series:
    """
    Replace non-positive values with NaN, then forward-fill/backfill to nearest positive value.
    """
    s = series.where(series > 0)  # keep > 0; null others
    return s.ffill().bfill()


def _to_columns(df: pd.DataFrame, names: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Ensure `names` exist as columns (copying them from index if needed). Returns (df, added_from_index).
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
    If the caller had an index, restore it by those names. Else, leave as columns.
    """
    if original_index_names and any(original_index_names):
        return df.set_index(original_index_names)
    return df


def pre_qa_dsf(dsf: pd.DataFrame) -> None:
    """
    Minimal checks:
      - 'date' dtype
      - warn on non-positive adjustment factors
      - warn on negative PRC share
      - check uniqueness on (permno, date)
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

    if (dsf["cfacpr"] <= 0).any():
        print("[warn] dsf: some cfacpr <= 0 (unexpected).")
    if (dsf["cfacshr"] <= 0).any():
        print("[warn] dsf: some cfacshr <= 0 (unexpected).")

    n_neg = int((dsf["prc"] < 0).sum())
    if n_neg:
        pct = round(n_neg / max(len(dsf), 1) * 100, 2)
        print(f"[info] dsf: {n_neg:,} rows have negative prices ({pct}%).")

    check_key_dupes(dsf, ["permno", "date"], "dsf")


def impute_negative_crsp_factors_and_price(dsf: pd.DataFrame) -> pd.DataFrame:
    """
    For each permno, replace non-positive values in {cfacpr, cfacshr, prc}
    with the nearest previous positive value (fallback to next positive via bfill).
    Then recompute adjusted fields (adj_prc, adj_shrout, adj_mktcap).
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

    def _fix_group(g: pd.DataFrame) -> pd.DataFrame:
        for col in ("cfacpr", "cfacshr", "prc"):
            g[col] = _fill_prev_positive(g[col])
        return g

    out = grouper.apply(_fix_group)

    # Recompute adjusted fields
    out["adj_prc"] = out["prc"] * out["cfacpr"]
    if "shrout" in out.columns:
        out["adj_shrout"] = out["shrout"] * out["cfacshr"]
        out["adj_mktcap"] = out["adj_prc"].abs() * out["adj_shrout"]

    return out


def pre_qa_stocknames(sn: pd.DataFrame) -> None:
    """
    Checks:
      - namedt/nameenddt datetime
      - warning on namedt > nameenddt
      - warning on overlapping windows (can duplicate rows on as-of join)
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


def post_join_qa_prices(df: pd.DataFrame) -> None:
    """
    Light checks after DSF ⟵ stocknames:
      - Uniqueness on (permno,date)
      - Ticker coverage
      - Adjusted price / market cap sanity
    """
    check_key_dupes(df, ["permno", "date"], "df_prices")

    if "ticker" in df.columns:
        null_ticker_rate = float(df["ticker"].isna().mean())
        if null_ticker_rate > 0:
            print(
                f"[info] df_prices: {null_ticker_rate:.2%} rows lack ticker mapping on that date."
            )

    if "adj_prc" in df.columns:
        near_zero = int((df["adj_prc"].abs() <= 1e-8).sum())
        if near_zero:
            print(f"[warn] df_prices: {near_zero:,} rows with ~0 adjusted price.")

    if "adj_mktcap" in df.columns:
        neg_cap = int((df["adj_mktcap"] < 0).sum())
        if neg_cap:
            print(f"[warn] df_prices: {neg_cap:,} rows with negative market cap.")


def pre_qa_ff(ff: pd.DataFrame) -> None:
    """
    Basic hygiene on Fama–French daily factors:
      - 'date' must be datetime64
      - unique by 'date'
      - finite values; warn on extreme magnitudes
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
    Left-join daily prices with Fama–French factors by 'date'.
    - Works whether df_prices is indexed or not; preserves original index.
    - Guarantees no row inflation.
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
    After adding FF factors:
      - Still unique on (permno, date)?
      - Missing rates for factors
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