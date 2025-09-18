import os
from typing import Any, Dict, List, Tuple

from src.helpers._extract import ensure_dir, make_run_folder, safe_delete_dir
from src.helpers._sql import assert_artifacts_present, extract_artifacts, wrds_connect


def wrds_extract_raw(
    wrds_user: str,
    start: str,
    end: str,
    chunk_size: int,
    use_run: str,
    base_dir: str,
    artifacts: List[Tuple[str, str]],
) -> Dict[str, Any]:
    """
    Orchestrate raw data extraction from WRDS with directory and reuse management.

    This function sets up the run folder according to the user's choice (new run, reuse last, or specific run),
    connects to WRDS using given credentials, and runs SQL queries for all requested data artifacts.
    Data is extracted in chunks and saved as Parquet files in the run folder.
    If reuse mode is selected, it validates that the expected Parquet files already exist and skips extraction.

    Args:
        wrds_user (str): WRDS username for authentication.
        start (str): Start date for data extraction (e.g., "YYYY-MM-DD").
        end (str): End date for data extraction (e.g., "YYYY-MM-DD").
        chunk_size (int): Number of rows to fetch per chunk from WRDS.
        use_run (str): Specifies run mode ("new", "last", or explicit run folder name).
        base_dir (str): Base directory under which to organize run folders.
        artifacts (List[Tuple[str, str]]): List of (SQL file path, output Parquet filename) tuples to extract.

    Returns:
        Dict[str, Any]: Contains keys:
            - "run_folder": Path of the run folder used for extraction.
            - "reuse": Boolean flag whether the run folder was reused.
            - "artifacts": Dict mapping Parquet filenames to their absolute paths.
    """
    ensure_dir(base_dir)
    run_dir, run_name, reuse = make_run_folder(base_dir, use_run)
    print(f"[info] Using run folder: {run_name} (reuse={reuse})")

    if not reuse:
        conn = None
        try:
            conn = wrds_connect(wrds_user)
            params = {"start": start, "end": end}
            extract_artifacts(
                conn,
                artifacts,
                run_dir,
                params=params,
                chunk_size=chunk_size,
                force=True,
            )
            print("[info] Extraction complete (full refresh).")
        except Exception:
            safe_delete_dir(run_dir, base_dir)
            raise
        finally:
            if conn is not None:
                conn.close()
    else:
        assert_artifacts_present(run_dir, artifacts)
        print("[info] Reuse mode: all required Parquet files are present. No extraction performed.")

    produced = {
        parq: os.path.join(run_dir, parq)
        for _, parq in artifacts
        if os.path.isfile(os.path.join(run_dir, parq))
    }

    return {
        "run_folder": run_dir,
        "reuse": reuse,
        "artifacts": produced,
    }
