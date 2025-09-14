from src.helpers.data_cleanup import (
    ensure_index,
    impute_negative_crsp_factors_and_price,
    join_dsf_with_stocknames,
    join_prices_with_ff,
    parquet_to_df,
    post_join_qa_prices,
    post_join_qa_prices_with_ff,
    pre_qa_dsf,
    pre_qa_ff,
    pre_qa_stocknames,
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
            ("src/migrations/003_ff_factors.sql", "ff.parquet"),
            ("src/migrations/004_ibes_statsumu.sql", "ibes_stats.parquet"),
            ("src/migrations/005_ibes_actu.sql", "ibes_act.parquet"),
        ],
    )
    print(res)

    # load + index columns
    dsf = parquet_to_df(res["artifacts"], "dsf.parquet")
    dsf = ensure_index(dsf, ["permno", "date"], keep_cols=False)
    stock_names = parquet_to_df(res["artifacts"], "stocknames.parquet")
    ff = parquet_to_df(res["artifacts"], "ff.parquet")

    # QA source data + impute
    pre_qa_dsf(dsf)
    pre_qa_stocknames(stock_names)
    dsf = impute_negative_crsp_factors_and_price(dsf)
    pre_qa_ff(ff)

    # Joining tables, index columns and QA output
    df_prices = join_dsf_with_stocknames(dsf, stock_names)
    df_prices = ensure_index(df_prices, ["permno", "date"], keep_cols=False)
    post_join_qa_prices(df_prices)

    df_prices_with_ff = join_prices_with_ff(df_prices, ff)
    post_join_qa_prices_with_ff(df_prices_with_ff)

    print(f"[final] df_prices shape={df_prices_with_ff.shape}")
    print(f"[final] index={list(df_prices_with_ff.index.names)}")
    print(f"[final] columns={list(df_prices_with_ff.columns)}")
    print(f"[final] head=2\n{df_prices_with_ff.head(2)}")
