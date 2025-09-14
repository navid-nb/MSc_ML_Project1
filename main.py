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
from src.helpers.model_matrix import build_model_matrix, fillna_by_permno, null_report


def main():
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
    model_df = build_model_matrix(df_prices)

    print(f"[final] df_prices shape={df_prices.shape}")
    print(f"[final] index={list(df_prices.index.names)}")
    print(f"[final] columns={list(df_prices.columns)}")

    print(f"[model] shape={model_df.shape}")
    print(f"[model] index={list(model_df.index.names)}")
    print(f"[model] columns={list(model_df.columns)}")

    print(null_report(model_df))

    tickers = ["MSFT", "AAPL"]
    new_model_df = model_df[model_df["ticker"].isin(tickers)]
    print(null_report(new_model_df))


if __name__ == "__main__":
    main()
