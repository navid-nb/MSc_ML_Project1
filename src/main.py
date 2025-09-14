import os
import datetime as dt
from typing import Dict, Any

from helpers import (
    ensure_dir,
    make_run_folder,
    compute_missing_artifacts,
    wrds_connect,
    extract_artifacts,
    setup_duckdb,
    register_views,
    render_final_select_sql,
    write_year_partition,
    year_iter,
)

BASE_DIR = "wrds_extracts"
ARTIFACTS = [
    ("migrations/001_base_extract.sql",  "dsf.parquet"),
    ("migrations/002_crsp_names.sql",    "stocknames.parquet"),
    ("migrations/003_comp_secm.sql",     "secm.parquet"),
    ("migrations/004_comp_fundq.sql",    "fundq.parquet"),
    ("migrations/005_ff_factors.sql",    "ff.parquet"),
    ("migrations/006_ibes_statsumu.sql", "ibes_stats.parquet"),
    ("migrations/007_ibes_actu.sql",     "ibes_act.parquet"),
]

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def wrds_extract_and_join(
    wrds_user: str,
    start: str,
    end: str,
    chunk_size: int,
    use_run: str,
    join_sql_select: str = "migrations/011_final_select.sql",
    duckdb_memory_limit: str = "8GB",
    duckdb_threads: int = 4,
    duckdb_temp_gib: str = "100GiB",
    *,
    tickers: list[str],
) -> Dict[str, Any]:
    # 0) Run folder & reuse
    ensure_dir(BASE_DIR)
    outdir, outdir_name, reuse = make_run_folder(BASE_DIR, use_run)
    print(f"[info] Using run folder: {outdir_name} (reuse={reuse})")

    # 1) Extract (skip existing, fill missing)
    missing = compute_missing_artifacts(outdir, ARTIFACTS)
    if reuse and missing:
        print(f"[warn] Missing files detected: {missing}. Will extract missing pieces.")
    need_extract = (not reuse) or bool(missing)

    if need_extract:
        print("[info] Connecting to WRDS ...")
        conn = wrds_connect(wrds_user)
        params = {"start": start, "end": end}
        extract_artifacts(conn, ARTIFACTS, outdir, params=params, chunk_size=chunk_size)
        print("[info] Extraction complete.")

    # 2) DuckDB session + register views
    tmp_dir = os.path.join(outdir, "duckdb_tmp")
    con = setup_duckdb(tmp_dir, duckdb_threads, duckdb_memory_limit, duckdb_temp_gib)

    register_views(con, {
        "dsf":        os.path.join(outdir, "dsf.parquet"),
        "stocknames": os.path.join(outdir, "stocknames.parquet"),
        "secm":       os.path.join(outdir, "secm.parquet"),
        "fundq":      os.path.join(outdir, "fundq.parquet"),
        "ff":         os.path.join(outdir, "ff.parquet"),
        "ibes_stats": os.path.join(outdir, "ibes_stats.parquet"),
        "ibes_act":   os.path.join(outdir, "ibes_act.parquet"),
    })

    # 3) Final join to yearly Parquet files
    if not os.path.isfile(join_sql_select):
        raise FileNotFoundError(f"Join SQL not found: {join_sql_select}")
    if not tickers:
        raise ValueError("tickers is mandatory: pass e.g. tickers=['AAPL','MSFT'].")

    select_sql_template = render_final_select_sql(join_sql_select, tickers)  # inject {{TICKER_FILTER}}

    final_dir = os.path.join(outdir, f"final_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ensure_dir(final_dir)

    total_rows = 0
    for yr, y_start, y_end in year_iter(start, end):
        date_filter = f"WHERE b.\"date\" >= DATE '{y_start}' AND b.\"date\" < DATE '{y_end}'"
        year_sql = select_sql_template.replace("--__DATE_FILTER__", date_filter)
        out_path = os.path.join(final_dir, f"year={yr}.parquet")
        print(f"[final] Writing year {yr} -> {out_path}")

        cnt = write_year_partition(con, year_sql, out_path)
        total_rows += cnt
        print(f"[final]   rows: {cnt:,}")

    print(f"[result] Total final rows across years: {total_rows:,}")
    print(f"[done] Final parquet folder: {final_dir}")

    return {
        "row_count": total_rows,
        "run_folder": outdir,
        "final_folder": final_dir,
        "reuse": reuse
    }


if __name__ == "__main__":
    res = wrds_extract_and_join(
        wrds_user="wboughattas",
        start="2020-01-01",
        end="2021-01-01",
        chunk_size=500_000,
        use_run="new",  # "last" or "new" or a specific run folder
        tickers=["AAPL", "MSFT"],
    )
    print(res)
