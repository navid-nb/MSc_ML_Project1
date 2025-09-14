import os
import datetime as dt
from typing import Dict, Any, Optional, List

import pandas as pd
import wrds
import pyarrow as pa
import pyarrow.parquet as pq

# -----------------------
# Run management
# -----------------------

def list_runs(base_dir: str) -> List[str]:
    if not os.path.isdir(base_dir):
        return []
    runs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")
    ]
    runs.sort()
    return runs

def latest_run(base_dir: str) -> Optional[str]:
    runs = list_runs(base_dir)
    return runs[-1] if runs else None

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def make_run_folder(base_dir: str, use_run: str) -> tuple[str, str, bool]:
    """
    Decide run folder name and create it if needed.
    Returns (abs_path, name, reuse_flag).

    - use_run == "new":     create a fresh timestamped folder (reuse=False)
    - use_run == "last":    reuse the latest run if exists, else create new
    - else:                 treat as explicit folder name; reuse if it exists
    """
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

# -----------------------
# Extraction
# -----------------------

def wrds_connect(wrds_user: str) -> wrds.Connection:
    return wrds.Connection(wrds_username=wrds_user, verbose=True)

def query_to_parquet(conn: wrds.Connection, sql_path: str, out_path: str,
                     params: Optional[Dict[str, Any]] = None, chunk_size: int = 500_000) -> None:
    sql = open(sql_path).read()
    params = params or {}
    writer = None
    for chunk in pd.read_sql_query(sql, con=conn.connection, params=params, chunksize=chunk_size):
        table = pa.Table.from_pandas(chunk, preserve_index=False)  # noqa
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
    if writer is not None:
        writer.close()

def extract_artifacts(conn: wrds.Connection,
                      artifacts: List[tuple[str, str]],
                      outdir: str,
                      params: Optional[Dict[str, Any]] = None,
                      chunk_size: int = 500_000,
                      force: bool = False) -> None:
    """
    Run all SQL files to Parquet.

    - If force=True: always (re)write Parquet files.
    - If force=False: skip files that already exist (not used in this flow).
    """
    for sqlfile, outfile in artifacts:
        outpath = os.path.join(outdir, outfile)
        if (not force) and os.path.isfile(outpath):
            print(f"[skip] Already present: {outfile}")
            continue
        if force and os.path.isfile(outpath):
            print(f"[overwrite] {outfile}")
            os.remove(outpath)
        print(f"[extract] {sqlfile} -> {outfile}")
        query_to_parquet(conn, sqlfile, outpath, params=params, chunk_size=chunk_size)
        print(f"[ok] Saved {outfile}")

def assert_artifacts_present(outdir: str, artifacts: List[tuple[str, str]]) -> None:
    """Raise AssertionError listing any missing Parquet outputs."""
    missing = []
    for _, fname in artifacts:
        if not os.path.isfile(os.path.join(outdir, fname)):
            missing.append(fname)
    if missing:
        raise AssertionError(
            f"Reuse mode requires all artifacts to exist. Missing: {', '.join(missing)}"
        )
