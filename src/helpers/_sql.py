import os
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import wrds


def wrds_connect(wrds_user: str) -> wrds.Connection:
    """
    Establish a connection to the WRDS database using the provided username.

    Args:
        wrdsuser (str): WRDS username for authentication.

    Returns:
        wrds.Connection: An active connection object to the WRDS database.
    """
    return wrds.Connection(wrds_username=wrds_user, verbose=True)


def query_to_parquet(
    conn: wrds.Connection,
    sql_path: str,
    out_path: str,
    params: Optional[Dict[str, Any]] = None,
    chunk_size: int = 500_000,
) -> None:
    """
    Execute a SQL query using WRDS connection and save the result to a Parquet file in chunks.

    Args:
        conn (wrds.Connection): Active WRDS connection object.
        sqlpath (str): Path to SQL file containing the query.
        outpath (str): Destination filepath for the Parquet output.
        params (Optional[Dict[str, Any]]): Dictionary of SQL parameters, if required.
        chunksize (int): Number of rows per chunk for processing large results.

    Returns:
        None
    """
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


def extract_artifacts(
    conn: wrds.Connection,
    artifacts: List[tuple[str, str]],
    outdir: str,
    params: Optional[Dict[str, Any]] = None,
    chunk_size: int = 500_000,
    force: bool = False,
) -> None:
    """
    Loop through all requested SQL/extraction artifacts, running queries and saving to Parquet.
    Skips outputs that already exist unless force is True.

    Args:
        conn (wrds.Connection): WRDS database connection.
        artifacts (List[tuple]): List of (SQL file path, output filename) pairs.
        outdir (str): Directory to save Parquet outputs.
        params (Optional[Dict[str, Any]]): Parameters to use in SQL queries.
        chunksize (int): Row count per chunk for extraction.
        force (bool): If True, overwrite existing files; If False, skip files that already exist (not used in this flow).

    Returns:
        None
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
    """
    Check that all expected Parquet files exist in the output directory.
    Raises AssertionError and lists any missing outputs.

    Args:
        outdir (str): Directory to verify for output files.
        artifacts (List[tuple]): List of (SQL file path, output filename) pairs.

    Raises:
        AssertionError: If any output files are missing.

    Returns:
        None
    """
    missing = []
    for _, fname in artifacts:
        if not os.path.isfile(os.path.join(outdir, fname)):
            missing.append(fname)
    if missing:
        raise AssertionError(
            f"Reuse mode requires all artifacts to exist. Missing: {', '.join(missing)}"
        )
