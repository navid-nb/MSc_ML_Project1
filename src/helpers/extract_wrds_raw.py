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
