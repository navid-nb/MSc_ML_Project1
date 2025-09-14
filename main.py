from src.helpers.extract_wrds_raw import wrds_extract_raw

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
