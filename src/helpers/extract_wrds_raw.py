import os
from typing import Any, Dict

from src.helpers._extract import ensure_dir, make_run_folder, safe_delete_dir
from src.helpers._sql import assert_artifacts_present, extract_artifacts, wrds_connect


def wrds_extract_raw(
    wrds_user: str,
    start: str,
    end: str,
    chunk_size: int,
    use_run: str,
    out_dir: str,
    artifacts: list[tuple[str, str]],
) -> Dict[str, Any]:
    """
    If use_run == "new": extract ALL artifacts (overwrite if exist).
    Else (reuse an existing run): assert ALL artifacts are present; do not extract.
    """
    ensure_dir(out_dir)
    out_dir, out_dir_name, reuse = make_run_folder(out_dir, use_run)

    if not reuse:
        conn = None
        try:
            conn = wrds_connect(wrds_user)
            params = {"start": start, "end": end}
            extract_artifacts(
                conn, artifacts, out_dir, params=params, chunk_size=chunk_size, force=True
            )
        except Exception:
            safe_delete_dir(out_dir, out_dir)
            raise
        finally:
            if conn is not None:
                conn.close()
    else:
        assert_artifacts_present(out_dir, artifacts)

    produced = {
        parq: os.path.join(out_dir, parq)
        for _, parq in artifacts
        if os.path.isfile(os.path.join(out_dir, parq))
    }

    return {
        "run_folder": out_dir,
        "reuse": reuse,
        "artifacts": produced,
    }
