import os
from typing import Dict

import numpy as np
import pandas as pd


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


def assert_no_new_rows(
    df_left: pd.DataFrame, df_joined: pd.DataFrame, *, left_name: str, join_name: str
) -> None:
    """Protect against accidental many-to-many fan-out."""
    if df_joined.shape[0] > df_left.shape[0]:
        raise AssertionError(
            f"{join_name}: produced {df_joined.shape[0]:,} rows > {left_name} {df_left.shape[0]:,}. "
            f"Likely interval overlap or non-unique keys."
        )


def check_key_dupes(df: pd.DataFrame, key_cols: list[str], label: str) -> None:
    """
    Ensure `key_cols` form a unique key. Each name may refer to a column OR an index level.
    """
    # Build a temporary DataFrame of the key values (columns only) for uniform checks
    idx_names = list(df.index.names) if df.index.names is not None else []
    data = {}
    for k in key_cols:
        if k in df.columns:
            data[k] = df[k].to_numpy()
        elif k in idx_names:
            data[k] = df.index.get_level_values(k).to_numpy()
        else:
            raise KeyError(f"{label}: key '{k}' not found as column or index level.")

    keys_df = pd.DataFrame(data)

    # Find duplicates across the composite key
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


def _ensure_datetime(df: pd.DataFrame, cols: list[str], label: str) -> None:
    """
    Ensure each name in `cols` is datetime64, whether it's a column or an index level.
    Converts to datetime.
    """
    for c in cols:
        # is column
        if c in df.columns:
            if not np.issubdtype(df[c].dtype, np.datetime64):
                try:
                    df[c] = pd.to_datetime(df[c], errors="raise")
                except Exception as e:
                    raise TypeError(
                        f"{label}.{c} (column) must be datetime64 and could not be converted. "
                        f"Got {df[c].dtype}. Error: {e}"
                    )
            continue

        # is index
        idx_names = list(df.index.names) if df.index.names is not None else []
        if c in idx_names:
            lvl_vals = df.index.get_level_values(c)
            if not np.issubdtype(lvl_vals.dtype, np.datetime64):
                try:
                    if df.index.nlevels == 1:
                        new_idx = pd.to_datetime(df.index, errors="raise")
                        new_idx.name = df.index.name
                        df.index = new_idx
                    else:
                        # Rebuild MultiIndex safely via a DataFrame
                        idx_df = df.index.to_frame(index=False)
                        idx_df[c] = pd.to_datetime(idx_df[c], errors="raise")
                        df.index = pd.MultiIndex.from_frame(idx_df)
                except Exception as e:
                    raise TypeError(
                        f"{label}.{c} (index level) must be datetime64 and could not be converted. "
                        f"Got {lvl_vals.dtype}. Error: {e}"
                    )
            continue

        # Not found
        raise KeyError(f"{label}: '{c}' not found as column or index.")


def _fill_prev_positive(series: pd.Series) -> pd.Series:
    """Replace non-positive values with NaN, then forward-fill, then backfill."""
    s = series.where(series > 0)  # keep strictly positive, null out others (<=0 or NaN)
    s = s.ffill().bfill()  # fill from previous valid; if starts invalid, backfill from next valid
    return s


def ensure_index(
    df: pd.DataFrame, cols: list[str], *, sort: bool = True, keep_cols: bool = True
) -> pd.DataFrame:
    """
    Make `cols` the (multi)index, optionally keep them as columns, and sort.
    - Faster groupby/lookup on repeated ops.
    - Idempotent: if index already matches, it just (re)sorts (if requested).

    Example:
      dsf = ensure_index(dsf, ["permno", "date"])
      sn  = ensure_index(sn,  ["permno", "namedt"])
    """
    need_set = list(df.index.names or []) != cols
    if need_set:
        df = df.set_index(cols, drop=not keep_cols)
    if sort:
        df = df.sort_index()
    return df


def pre_qa_dsf(dsf: pd.DataFrame) -> None:
    """
    Minimal checks that commonly bite modeling:
    - Required columns present
    - Date dtype
    - Adjustment factors positive
    - Returns outliers warning
    - Key uniqueness on (permno, date)
    """
    _ensure_datetime(dsf, ["date"], "dsf")

    if (dsf["cfacpr"] <= 0).any():
        print("[warn] dsf: some cfacpr <= 0 (unexpected); check those rows.")

    if (dsf["cfacshr"] <= 0).any():
        print("[warn] dsf: some cfacshr <= 0 (unexpected); check those rows.")

    # CRSP may store negative PRC for bid quotes - expected; adj price fixes sign.
    n_neg = int((dsf["prc"] < 0).sum())
    if n_neg:
        print(f"[info] dsf: {n_neg:,} rows have negative PRC (CRSP convention) out of {len(dsf)}.")

    # sanity on returns
    too_big = (dsf["ret"].abs() > 5.0).sum()
    if too_big:
        print(
            f"[warn] dsf: {too_big:,} rows with |ret| > 500% (possible stale prices/corp actions)."
        )

    # assert uniqueness of (permno,date)
    try:
        check_key_dupes(dsf, ["permno", "date"], "dsf")
    except AssertionError:
        # Let it bubble with context
        raise


def impute_negative_crsp_factors_and_price(dsf: pd.DataFrame) -> pd.DataFrame:
    """
    For each permno, replace non-positive values in cfacpr, cfacshr, prc
    with the most recent strictly positive value (falling back to the next one if needed).
    Then recompute adjusted fields.

    Works whether 'permno'/'date' are columns or index levels.
    """
    out = dsf.copy()

    # Sort by (permno, date) regardless of whether they are columns or index levels
    idx_names = list(out.index.names or [])
    has_permno_idx = "permno" in idx_names
    has_date_idx = "date" in idx_names

    if has_permno_idx and has_date_idx:
        out = out.sort_index()
    else:
        # If they’re columns (or mixed), sort by columns (no ambiguity)
        out = out.sort_values(by=["permno", "date"])

    # Group by permno (index level or column) and fill
    if has_permno_idx:
        grouper = out.groupby(level="permno", group_keys=False)
    else:
        grouper = out.groupby("permno", group_keys=False)

    def _fix_group(g: pd.DataFrame) -> pd.DataFrame:
        for col in ("cfacpr", "cfacshr", "prc"):
            g[col] = _fill_prev_positive(g[col])
        return g

    out = grouper.apply(_fix_group)

    # Recompute adjusted fields after imputation
    if "adj_prc" in out.columns:
        out["adj_prc"] = out["prc"] * out["cfacpr"]
    else:
        out.insert(len(out.columns), "adj_prc", out["prc"] * out["cfacpr"])

    if "adj_shrout" in out.columns:
        out["adj_shrout"] = out["shrout"] * out["cfacshr"]
    elif "shrout" in out.columns:
        out.insert(len(out.columns), "adj_shrout", out["shrout"] * out["cfacshr"])

    if "adj_mktcap" in out.columns:
        out["adj_mktcap"] = out["adj_prc"].abs() * out["adj_shrout"]

    return out


def pre_qa_stocknames(sn: pd.DataFrame) -> None:
    """
    - Required columns present
    - Date dtypes
    - Optional overlap warning for name windows
    """
    _ensure_datetime(sn, ["namedt", "nameenddt"], "stocknames")

    # warn if any namedt > nameenddt (data error)
    bad = (sn["nameenddt"].notna()) & (sn["namedt"] > sn["nameenddt"])
    if bad.any():
        n = int(bad.sum())
        print(f"[warn] stocknames: {n:,} rows have namedt > nameenddt.")

    # overlap warning (can cause multi-match on a date)
    sn = sn.assign(nameenddt_eff=coalesce_date_end(sn["nameenddt"]))
    overlaps = []
    for p, g in sn.groupby("permno", sort=False):
        g = g.sort_values("namedt")
        if len(g) > 1:
            # if any start <= previous effective end => overlap
            if (g["namedt"].values[1:] <= g["nameenddt_eff"].values[:-1]).any():
                overlaps.append(p)
    if overlaps:
        print(
            f"[warn] stocknames: {len(overlaps)} permno have overlapping name windows; "
            f"join logic will pick the record with the latest namedt per (permno,date)."
        )


def join_dsf_with_stocknames(dsf: pd.DataFrame, stock_names: pd.DataFrame) -> pd.DataFrame:
    """
    As-of join:
      1) Left-merge on permno.
      2) Keep rows where date ∈ [namedt, nameenddt_eff] (or keep nulls for pure left semantics).
      3) If multiple records match (overlapping windows), pick the one with the *latest* namedt.
      4) Ensure no row inflation vs. DSF (safety).
      5) Drop interval columns; prefer ncusip over dsf.cusip to avoid confusion.
    """
    # Ensure we work with columns (not index levels) to avoid ambiguity in merge/filtering
    dsf_c = (
        dsf.reset_index()
        if ("permno" in (dsf.index.names or []) or "date" in (dsf.index.names or []))
        else dsf.copy()
    )
    sn_c = (
        stock_names.reset_index()
        if (
            "permno" in (stock_names.index.names or [])
            or "namedt" in (stock_names.index.names or [])
        )
        else stock_names.copy()
    )

    sn_c["nameenddt_eff"] = coalesce_date_end(sn_c["nameenddt"])
    pre_rows = int(dsf_c.shape[0])

    merged = dsf_c.merge(
        sn_c[["permno", "ticker", "ncusip", "namedt", "nameenddt_eff"]],
        on="permno",
        how="left",
        suffixes=("", "_sn"),
    )

    # keep rows valid on the obs date
    valid = (merged["date"] >= merged["namedt"]) & (merged["date"] <= merged["nameenddt_eff"])
    merged = merged.loc[valid | merged["namedt"].isna()]

    # de-overlap: keep the record with the latest namedt per (permno, date)
    if {"namedt", "date", "permno"}.issubset(merged.columns):
        sentinel = pd.Timestamp("1900-01-01")  # put NaT last after sorting
        merged["_namedt_sort"] = merged["namedt"].fillna(sentinel)
        merged = merged.sort_values(["permno", "date", "_namedt_sort"])
        merged = merged.drop_duplicates(subset=["permno", "date"], keep="last")
        merged = merged.drop(columns=["_namedt_sort"])

    # safety: no row inflation
    assert_no_new_rows(dsf_c, merged, left_name="dsf", join_name="dsf <- stocknames")

    # keep a single key set
    merged = merged.drop(columns=["namedt", "nameenddt_eff"], errors="ignore")
    if "cusip" in merged.columns and "ncusip" in merged.columns:
        merged = merged.drop(columns=["cusip"])

    # (permno,date) uniqueness must hold
    check_key_dupes(merged, ["permno", "date"], "post-join dsf")

    post_rows = int(merged.shape[0])
    print(
        "[info] df_prices:",
        {"rows_in": pre_rows, "rows_out": post_rows, "lost_rows": pre_rows - post_rows},
    )
    return merged


def post_join_qa_prices(df: pd.DataFrame) -> None:
    """
    Light checks that help downstream modeling of returns:
      - Uniqueness on (permno,date)
      - Ticker coverage
      - Adj price / mktcap anomalies
      - Missing key fields
    """
    check_key_dupes(df, ["permno", "date"], "df_prices")

    # ticker coverage
    null_ticker_rate = float(df["ticker"].isna().mean()) if "ticker" in df.columns else 1.0
    if null_ticker_rate > 0:
        print(f"[info] df_prices: {null_ticker_rate:.2%} rows lack ticker mapping on that date.")

    # prices / caps
    if "adj_prc" in df.columns:
        near_zero = (df["adj_prc"].abs() <= 1e-8).sum()
        if near_zero:
            print(f"[warn] df_prices: {near_zero:,} rows with ~0 adjusted price.")
    if "adj_mktcap" in df.columns:
        neg_cap = (df["adj_mktcap"] < 0).sum()
        if neg_cap:
            print(
                f"[warn] df_prices: {neg_cap:,} rows with negative market cap (after abs(prc) fix?)."
            )

    # required for simple daily models
    needed = {"permno", "date", "ret", "adj_prc"}
    missing = needed - set(df.columns)
    if missing:
        print(f"[warn] df_prices: missing expected modeling columns: {missing}")
