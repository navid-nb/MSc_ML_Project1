import os
import datetime as dt
from typing import Dict, Any, Optional, List

import pandas as pd
import duckdb
import wrds
import pyarrow as pa
import pyarrow.parquet as pq

def list_runs(base_dir: str) -> List[str]:
    if not os.path.isdir(base_dir):
        return []
    runs = [d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")]
    runs.sort()
    return runs

def latest_run(base_dir: str) -> Optional[str]:
    runs = list_runs(base_dir)
    return runs[-1] if runs else None

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def make_run_folder(base_dir: str, use_run: str) -> tuple[str, str, bool]:
    """Return (outdir_abs_path, outdir_name, reuse_flag)."""
    if use_run == "new":
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir_name = f"run_{stamp}"
        reuse = False
    elif use_run == "last":
        last = latest_run(base_dir)
        if last:
            outdir_name = last
            reuse = True
        else:
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            outdir_name = f"run_{stamp}"
            reuse = False
    else:
        outdir_name = use_run
        reuse = os.path.isdir(os.path.join(base_dir, outdir_name))

    outdir = os.path.join(base_dir, outdir_name)
    ensure_dir(outdir)
    return outdir, outdir_name, reuse

def compute_missing_artifacts(outdir: str, artifacts: List[tuple[str, str]]) -> List[str]:
    missing = []
    for _, fname in artifacts:
        if not os.path.isfile(os.path.join(outdir, fname)):
            missing.append(fname)
    return missing

def wrds_connect(wrds_user: str) -> wrds.Connection:
    return wrds.Connection(wrds_username=wrds_user, verbose=True)

def query_to_parquet(conn: wrds.Connection, sql_path: str, out_path: str,
                     params: Optional[Dict[str, Any]] = None, chunk_size: int = 500_000) -> None:
    sql = open(sql_path).read()
    params = params or {}
    writer = None
    for chunk in pd.read_sql_query(sql, con=conn.connection, params=params, chunksize=chunk_size):
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
    if writer is not None:
        writer.close()

def extract_artifacts(conn: wrds.Connection,
                      artifacts: List[tuple[str, str]],
                      outdir: str,
                      params: Optional[Dict[str, Any]] = None,
                      chunk_size: int = 500_000) -> None:
    """Run all SQL files to Parquet (skip existing)."""
    for sqlfile, outfile in artifacts:
        outpath = os.path.join(outdir, outfile)
        if os.path.isfile(outpath):
            print(f"[skip] Already present: {outfile}")
            continue
        print(f"[extract] {sqlfile} -> {outfile}")
        query_to_parquet(conn, sqlfile, outpath, params=params, chunk_size=chunk_size)
        print(f"[ok] Saved {outfile}")

def setup_duckdb(tmp_dir: str, threads: int, memory_limit: str, temp_gib: str) -> duckdb.DuckDBPyConnection:
    ensure_dir(tmp_dir)
    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={threads};")
    con.execute(f"PRAGMA memory_limit='{memory_limit}';")
    con.execute(f"PRAGMA temp_directory='{tmp_dir}';")
    con.execute(f"PRAGMA max_temp_directory_size='{temp_gib}';")
    con.execute("PRAGMA preserve_insertion_order=false;")
    return con

def views_from_parquet(con: duckdb.DuckDBPyConnection, view_name: str, file_path: str) -> None:
    con.execute(f"CREATE VIEW {view_name} AS SELECT * FROM read_parquet('{file_path}');")

def register_views(con: duckdb.DuckDBPyConnection, mapping: Dict[str, str]) -> None:
    """mapping: {view_name: parquet_path}"""
    for view_name, file_path in mapping.items():
        views_from_parquet(con, view_name, file_path)

def render_final_select_sql(join_sql_select: str, tickers: list[str]) -> str:
    """Load template, inject ticker filter for placeholder {{TICKER_FILTER}}."""
    template = open(join_sql_select).read()
    quoted = ", ".join("'" + t.replace("'", "''") + "'" for t in tickers)
    ticker_filter_sql = f"n.ticker IN ({quoted})"
    return template.replace("{{TICKER_FILTER}}", ticker_filter_sql)

def write_year_partition(con: duckdb.DuckDBPyConnection, year_sql: str, out_path: str) -> int:
    con.execute(f"""
        COPY (
            {year_sql}
        )
        TO '{out_path}'
        (FORMAT PARQUET);
    """)
    cnt = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchone()[0]
    return int(cnt)

def year_iter(start: str, end: str):
    s = dt.date.fromisoformat(start)
    e = dt.date.fromisoformat(end)
    y = s.year
    while True:
        y_start = dt.date(y, 1, 1)
        y_end = dt.date(y + 1, 1, 1)
        a = max(y_start, s)
        b = min(y_end, e)
        if a < b:
            yield y, a.isoformat(), b.isoformat()
        y += 1
        if y_start >= e or y > e.year + 1:
            break
