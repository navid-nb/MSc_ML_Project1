import os
import numpy as np
import pandas as pd


def parquet_to_df(artifacts: dict, name: str) -> pd.DataFrame:
    path = artifacts.get(name)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Required artifact missing: {name}")
    return pd.read_parquet(path)


def coalesce_date_end(s: pd.Series) -> pd.Series:
    # CRSP uses NULL=ongoing; treat as far-future
    return s.fillna(pd.Timestamp("9999-12-31"))


def assert_no_new_rows(df_left: pd.DataFrame, df_joined: pd.DataFrame, left_name="left", join_name="join"):
    if df_joined.shape[0] > df_left.shape[0]:
        raise AssertionError(
            f"Join produced more rows ({df_joined.shape[0]:,}) than {left_name} ({df_left.shape[0]:,}). "
            f"Likely many-to-many; check key uniqueness and validity ranges."
        )


def check_key_dupes(df: pd.DataFrame, key_cols: list[str], label: str):
    dups = df.duplicated(key_cols, keep=False)
    if dups.any():
        ex = df.loc[dups, key_cols].value_counts().head(10)
        raise AssertionError(f"{label}: duplicate keys on {key_cols}. Examples:\n{ex}")


def basic_dsf_qa(dsf: pd.DataFrame):
    # required columns
    req = {"permno", "date", "prc", "ret", "cfacpr", "cfacshr"}
    missing = req - set(dsf.columns)
    if missing:
        raise AssertionError(f"dsf missing columns: {missing}")

    # types & monotonic sanity
    if not np.issubdtype(dsf["date"].dtype, np.datetime64):
        raise TypeError("dsf.date must be datetime64; ensure DuckDB/pyarrow preserved type.")
    if (dsf["cfacpr"] <= 0).any():
        print("[warn] dsf: some cfacpr <= 0 (unexpected); check splits/adjustments rows.")

    # price sanity (CRSP can store negative bid/ask-average in prc for bid quotes)
    n_neg = (dsf["prc"] < 0).sum()
    if n_neg:
        print(
            f"[info] dsf: {n_neg:,} rows have negative PRC (CRSP convention). adj_prc below handles sign via factor.")

    # returns sanity
    too_big = dsf["ret"].abs() > 5.0
    if too_big.any():
        print(
            f"[warn] dsf: {too_big.sum():,} rows with |ret|>500% — investigate potential stale prices/corporate actions.")


def basic_stocknames_qa(sn: pd.DataFrame):
    # required columns
    req = {"permno", "ticker", "ncusip", "namedt", "nameenddt"}
    missing = req - set(sn.columns)
    if missing:
        raise AssertionError(f"stocknames missing columns: {missing}")

    # ensure date types
    for c in ["namedt", "nameenddt"]:
        if not np.issubdtype(sn[c].dtype, np.datetime64):
            raise TypeError(f"stocknames.{c} must be datetime64.")

    # check overlapping name windows per permno (overlaps can explode rows on join)
    sn = sn.assign(nameenddt_eff=coalesce_date_end(sn["nameenddt"]))
    overlaps = []
    for p, g in sn.groupby("permno", sort=False):
        g = g.sort_values("namedt")
        # detect if any start < previous end
        ov = (g["namedt"].values[1:] <= g["nameenddt_eff"].values[:-1]).any()
        if ov:
            overlaps.append(p)
    if overlaps:
        print(f"[warn] stocknames: {len(overlaps)} permno have overlapping name windows; "
              f"join could duplicate rows. Consider deduping to most recent record per date.")


def dedupe_stocknames_for_asof(sn: pd.DataFrame) -> pd.DataFrame:
    """
    Optional: if overlapping intervals exist, collapse to the latest record per (permno, namedt)
    so that for any given date we pick the single record with the most recent 'namedt' not after the date.
    Strategy: keep all rows but we’ll pick the *latest namedt* per (permno,date) after merge.
    """
    # nothing to do structurally here; the pick happens after the merge using a rank
    return sn.copy()
