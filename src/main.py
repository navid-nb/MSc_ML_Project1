import os
import datetime as dt
from typing import Dict, Any, Optional, List

import pandas as pd
import duckdb
import wrds
import pyarrow as pa
import pyarrow.parquet as pq

BASE_DIR = "wrds_extracts"

# ----------------------------
# Helpers
# ----------------------------
def list_runs(base_dir: str = BASE_DIR) -> List[str]:
    if not os.path.isdir(base_dir):
        return []
    runs = [d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")]
    runs.sort()
    return runs

def latest_run(base_dir: str = BASE_DIR) -> Optional[str]:
    runs = list_runs(base_dir)
    return runs[-1] if runs else None

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def query_to_parquet(conn: wrds.Connection, sql_path: str, out_path: str,
                     params: Optional[Dict[str, Any]] = None, chunksize: int = 500_000) -> None:
    sql = open(sql_path).read()
    params = params or {}
    writer = None
    for chunk in pd.read_sql_query(sql, con=conn.connection, params=params, chunksize=chunksize):
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
    if writer is not None:
        writer.close()

def views_from_parquet(con: duckdb.DuckDBPyConnection, view_name: str, file_path: str) -> None:
    con.execute(f"CREATE VIEW {view_name} AS SELECT * FROM read_parquet('{file_path}');")

def year_iter(start: str, end: str):
    s = dt.date.fromisoformat(start)
    e = dt.date.fromisoformat(end)
    y = s.year
    while True:
        y_start = dt.date(y, 1, 1)
        y_end = dt.date(y + 1, 1, 1)
        # intersect with [s, e)
        a = max(y_start, s)
        b = min(y_end, e)
        if a < b:
            yield y, a.isoformat(), b.isoformat()
        y += 1
        if y_start >= e or y > e.year + 1:
            break

# ----------------------------
# Main function
# ----------------------------
def wrds_extract_and_join(
    wrds_user: str = "wboughattas",
    start: str = "2000-01-01",
    end: str = "2025-01-01",
    chunksize: int = 500_000,
    use_run: str = "last",            # "last", "new", or specific run folder
    migrations_dir: str = "migrations",
    join_sql_select: str = "migrations/011_final_select.sql",
    duckdb_memory_limit: str = "8GB", # tune as needed
    duckdb_threads: int = 4,
    duckdb_temp_gib: str = "100GiB",  # max temp spill size
) -> Dict[str, Any]:
    ensure_dir(BASE_DIR)

    # Decide run folder based on use_run
    reuse = False
    if use_run == "new":
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir_name = f"run_{stamp}"
    elif use_run == "last":
        last = latest_run(BASE_DIR)
        if last:
            outdir_name = last
            reuse = True
        else:
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            outdir_name = f"run_{stamp}"
    else:
        outdir_name = use_run
        reuse = os.path.isdir(os.path.join(BASE_DIR, outdir_name))

    outdir = os.path.join(BASE_DIR, outdir_name)
    ensure_dir(outdir)
    print(f"[info] Using run folder: {outdir_name} (reuse={reuse})")

    artifacts = [
        (os.path.join(migrations_dir, "001_base_extract.sql"),  "dsf.parquet"),
        (os.path.join(migrations_dir, "002_crsp_names.sql"),    "stocknames.parquet"),
        (os.path.join(migrations_dir, "003_comp_secm.sql"),     "secm.parquet"),
        (os.path.join(migrations_dir, "004_comp_fundq.sql"),    "fundq.parquet"),
        (os.path.join(migrations_dir, "005_ff_factors.sql"),    "ff.parquet"),
        (os.path.join(migrations_dir, "006_ibes_statsumu.sql"), "ibes_stats.parquet"),
        (os.path.join(migrations_dir, "007_ibes_actu.sql"),     "ibes_act.parquet"),
    ]

    need_extract = not reuse
    missing = []
    for _, fname in artifacts:
        if not os.path.isfile(os.path.join(outdir, fname)):
            missing.append(fname)
    if reuse and missing:
        print(f"[warn] Missing files detected: {missing}. Will extract missing pieces.")
        need_extract = True

    if need_extract:
        print("[info] Connecting to WRDS ...")
        conn = wrds.Connection(wrds_username=wrds_user, verbose=True)
        params = {"start": start, "end": end}

        for sqlfile, outfile in artifacts:
            outpath = os.path.join(outdir, outfile)
            if os.path.isfile(outpath):
                print(f"[skip] Already present: {outfile}")
                continue
            print(f"[extract] {sqlfile} -> {outfile}")
            query_to_parquet(conn, sqlfile, outpath, params=params, chunksize=chunksize)
            print(f"[ok] Saved {outfile}")

    # ----------------------------
    # DuckDB: tune + register views
    # ----------------------------
    tmp_dir = os.path.join(outdir, "duckdb_tmp")
    ensure_dir(tmp_dir)

    con = duckdb.connect(database=":memory:")
    # Tuning to avoid OOM
    con.execute(f"PRAGMA threads={duckdb_threads};")
    con.execute(f"PRAGMA memory_limit='{duckdb_memory_limit}';")
    con.execute(f"PRAGMA temp_directory='{tmp_dir}';")
    con.execute(f"PRAGMA max_temp_directory_size='{duckdb_temp_gib}';")
    con.execute("PRAGMA preserve_insertion_order=false;")

    # Register views
    views_from_parquet(con, "dsf",         os.path.join(outdir, "dsf.parquet"))
    views_from_parquet(con, "stocknames",  os.path.join(outdir, "stocknames.parquet"))
    views_from_parquet(con, "secm",        os.path.join(outdir, "secm.parquet"))
    views_from_parquet(con, "fundq",       os.path.join(outdir, "fundq.parquet"))
    views_from_parquet(con, "ff",          os.path.join(outdir, "ff.parquet"))
    views_from_parquet(con, "ibes_stats",  os.path.join(outdir, "ibes_stats.parquet"))
    views_from_parquet(con, "ibes_act",    os.path.join(outdir, "ibes_act.parquet"))

    # ----------------------------
    # Final join → partitioned parquet (per year) + total row count
    # ----------------------------
    if not os.path.isfile(join_sql_select):
        raise FileNotFoundError(f"Join SQL not found: {join_sql_select}")

    select_sql_template = open(join_sql_select).read()

    final_dir = os.path.join(outdir, f"final_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ensure_dir(final_dir)

    total_rows = 0
    for yr, y_start, y_end in year_iter(start, end):
        # inject date filter at the marker in the WITH…SELECT statement
        # the template must contain the token: --__DATE_FILTER__
        date_filter = f"WHERE b.\"date\" >= DATE '{y_start}' AND b.\"date\" < DATE '{y_end}'"
        year_sql = select_sql_template.replace("--__DATE_FILTER__", date_filter)

        out_path = os.path.join(final_dir, f"year={yr}.parquet")
        print(f"[final] Writing year {yr} -> {out_path}")

        # COPY will stream results to parquet (lower memory than fetching)
        con.execute(f"""
            COPY (
                {year_sql}
            )
            TO '{out_path}'
            (FORMAT PARQUET);
        """)

        # optional: count rows written
        cnt = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchone()[0]
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

res = wrds_extract_and_join(
    wrds_user="wboughattas",
    start="2010-01-01",
    end="2021-01-01",
    use_run="last",  # or "new" or a specific run folder
    duckdb_memory_limit="8GB",
    duckdb_threads=4,
    duckdb_temp_gib="200GiB",  # bump if you have disk space
)
print(res)