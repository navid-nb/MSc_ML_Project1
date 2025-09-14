from src.helpers.data_cleanup import (
    join_dsf_with_stocknames,
    parquet_to_df,
    post_join_qa_prices,
    pre_qa_dsf,
    pre_qa_stocknames, impute_negative_crsp_factors_and_price, ensure_index,
)
from src.helpers.data_extraction import wrds_extract_raw

if __name__ == "__main__":
    res = wrds_extract_raw(
        wrds_user="wboughattas",
        start="2020-01-01",
        end="2021-01-01",
        chunk_size=500_000,
        use_run="last",  # "last" | "new" | specific folder like "run_20250101_120000"
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

    dsf = parquet_to_df(res["artifacts"], "dsf.parquet")
    dsf = ensure_index(dsf, ["permno", "date"], keep_cols=False)

    stock_names = parquet_to_df(res["artifacts"], "stocknames.parquet")

    pre_qa_dsf(dsf)
    impute_negative_crsp_factors_and_price(dsf)
    pre_qa_stocknames(stock_names)

    df_prices = join_dsf_with_stocknames(dsf, stock_names)
    df_prices = ensure_index(df_prices, ["permno", "date"], keep_cols=False)

    post_join_qa_prices(df_prices)

    print(f"[final] df_prices shape={df_prices.shape}")
    print(f"[final] index={list(df_prices.index.names)}")
    print(f"[final] columns={list(df_prices.columns)}")
