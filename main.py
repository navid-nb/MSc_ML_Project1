import os
from typing import Any, Dict

from src.helpers.extract import delete_dir, ensure_dir, make_run_folder
from src.helpers.sql import assert_artifacts_present, extract_artifacts, wrds_connect

BASE_DIR = "wrds_extracts"

ARTIFACTS = [
    ("src/migrations/001_base_extract.sql", "dsf.parquet"),
    ("src/migrations/002_crsp_names.sql", "stocknames.parquet"),
    ("src/migrations/003_comp_secm.sql", "secm.parquet"),
    ("src/migrations/004_comp_fundq.sql", "fundq.parquet"),
    ("src/migrations/005_ff_factors.sql", "ff.parquet"),
    ("src/migrations/006_ibes_statsumu.sql", "ibes_stats.parquet"),
    ("src/migrations/007_ibes_actu.sql", "ibes_act.parquet"),
    ("src/migrations/008_fisd_rating.sql", "fisd_rating.parquet"),
    ("src/migrations/009_cboe_cboe.sql", "cboe.parquet"),
]


def wrds_extract_raw(
    wrds_user: str,
    start: str,
    end: str,
    chunk_size: int,
    use_run: str,
) -> Dict[str, Any]:
    """
    If use_run == "new": extract ALL artifacts (overwrite if exist).
    Else (reuse an existing run): assert ALL artifacts are present; do not extract.
    """
    ensure_dir(BASE_DIR)
    out_dir, out_dir_name, reuse = make_run_folder(BASE_DIR, use_run)

    if not reuse:
        conn = None
        try:
            conn = wrds_connect(wrds_user)
            params = {"start": start, "end": end}
            extract_artifacts(
                conn, ARTIFACTS, out_dir, params=params, chunk_size=chunk_size, force=True
            )
        except Exception:
            if os.path.commonpath(
                [os.path.abspath(out_dir), os.path.abspath(BASE_DIR)]
            ) == os.path.abspath(BASE_DIR):
                delete_dir(out_dir)
            raise
        finally:
            if conn is not None:
                conn.close()
    else:
        assert_artifacts_present(out_dir, ARTIFACTS)

    produced = {
        parq: os.path.join(out_dir, parq)
        for _, parq in ARTIFACTS
        if os.path.isfile(os.path.join(out_dir, parq))
    }

    return {
        "run_folder": out_dir,
        "reuse": reuse,
        "artifacts": produced,
    }


if __name__ == "__main__":
    res = wrds_extract_raw(
        wrds_user="wboughattas",
        start="2020-01-01",
        end="2021-01-01",
        chunk_size=500_000,
        use_run="new",  # "last" or "new" or a specific folder like "run_20250101_120000"
    )
    print(res)
