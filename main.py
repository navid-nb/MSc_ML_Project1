import pandas as pd

from src.helpers.data_cleanup import (
    assert_no_new_rows,
    basic_dsf_qa,
    basic_stocknames_qa,
    check_key_dupes,
    coalesce_date_end,
    parquet_to_df,
)
from src.helpers.data_extraction import wrds_extract_raw

if __name__ == "__main__":
    res = wrds_extract_raw(
        wrds_user="wboughattas",
        start="2020-01-01",
        end="2021-01-01",
        chunk_size=500_000,
        use_run="last",  # "last" or "new" or a specific folder like "run_20250101_120000"
        base_dir="wrds_extracts",
        artifacts=[
            ("src/migrations/001_base_extract.sql", "dsf.parquet"),
            ("src/migrations/002_crsp_names.sql", "stocknames.parquet"),
            ("src/migrations/003_comp_secm.sql", "secm.parquet"),
            ("src/migrations/004_comp_fundq.sql", "fundq.parquet"),
            ("src/migrations/005_ff_factors.sql", "ff.parquet"),
            ("src/migrations/006_ibes_statsumu.sql", "ibes_stats.parquet"),
            ("src/migrations/007_ibes_actu.sql", "ibes_act.parquet"),
            ("src/migrations/008_fisd_rating.sql", "fisd_rating.parquet"),
            ("src/migrations/009_cboe_cboe.sql", "cboe.parquet"),
        ],
    )
    print(res)

    dsf = parquet_to_df(res, "dsf.parquet")
    stock_names = parquet_to_df(res, "stocknames.parquet")

    basic_dsf_qa(dsf)
    basic_stocknames_qa(stock_names)

    # Ensure key uniqueness on DSF (permno, date) before any join
    check_key_dupes(dsf, ["permno", "date"], "dsf")

    # ---------- prepare/derive DSF features helpful for returns modeling ----------
    # Adjusted close and market cap (CRSP convention: PRC can be negative; use abs)
    dsf = dsf.copy()
    dsf["adj_prc"] = dsf["prc"] * dsf["cfacpr"]
    dsf["adj_shrout"] = dsf["shrout"] * dsf["cfacshr"]
    dsf["mktcap"] = dsf["adj_shrout"] * dsf["adj_prc"].abs()

    # ---------- join DSF with STOCKNAMES (as-of) ----------
    sn = stock_names.copy()
    sn["nameenddt_eff"] = coalesce_date_end(sn["nameenddt"])

    # 1) raw merge on permno (many-to-many), then filter validity, then collapse
    pre_rows = dsf.shape[0]
    merged = dsf.merge(
        sn[["permno", "ticker", "ncusip", "namedt", "nameenddt_eff"]],
        on="permno",
        how="left",
        suffixes=("", "_sn"),
    )

    # 2) keep only rows where date ∈ [namedt, nameenddt_eff]
    mask_valid = (merged["date"] >= merged["namedt"]) & (merged["date"] <= merged["nameenddt_eff"])
    merged = merged.loc[
        mask_valid | merged["namedt"].isna()
    ]  # keep unmatched (left join semantics)

    # 3) if overlaps created multiple rows for the same (permno,date), pick the record with the latest namedt
    if {"namedt", "date", "permno"}.issubset(merged.columns):
        merged["_rank_namedt"] = merged.groupby(["permno", "date"])["namedt"].transform(
            lambda s: s.fillna(pd.Timestamp("1900-01-01")).rank(method="first", ascending=False)
        )
        merged = merged.loc[(merged["_rank_namedt"] == 1) | merged["namedt"].isna()].drop(
            columns=["_rank_namedt"]
        )

    # 4) assert we didn't CREATE rows (left join guarantee)
    assert_no_new_rows(dsf, merged, left_name="dsf", join_name="dsf⟵stocknames")

    # 5) keep a single key set; drop stocknames’ window columns
    merged = merged.drop(columns=["namedt", "nameenddt_eff"], errors="ignore")
    # optional: prefer CRSP 6- or 8-CUSIP? keep ncusip (issue-level) and drop dsf.cusip to avoid confusion
    if "cusip" in merged.columns and "ncusip" in merged.columns:
        merged = merged.drop(columns=["cusip"])

    # ---------- post-join QA ----------
    # still unique by (permno,date)?
    check_key_dupes(merged, ["permno", "date"], "post-join dsf")

    # ticker coverage and nulls
    null_ticker = merged["ticker"].isna().mean()
    if null_ticker > 0:
        print(
            f"[info] {null_ticker:.1%} of dsf rows lack a ticker mapping on that date (OK if off-CRSP, mergers, stale names)."
        )

    # basic price/return issues that can poison models
    bad_adj = merged["adj_prc"].abs() <= 1e-8
    if bad_adj.any():
        print(
            f"[warn] {bad_adj.sum():,} rows with near-zero adjusted price; consider filtering before featurization."
        )

    # Optional: filter to U.S. common stocks later using share/exchange codes from CRSP monthly/headers if you add them.

    # ---------- result ----------
    df_prices = merged  # this is your time-series base: (permno, date, ticker, ncusip, adj_prc, ret, mktcap, …)
    print(df_prices.shape, df_prices.columns.tolist())
