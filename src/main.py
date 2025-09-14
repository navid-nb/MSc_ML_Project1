import os
from typing import Dict, Any

from helpers import (
    ensure_dir,
    make_run_folder,
    wrds_connect,
    extract_artifacts,
    assert_artifacts_present,
)

BASE_DIR = "wrds_extracts"

# Raw pulls only (SQL -> Parquet)
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

def wrds_extract_raw(
    wrds_user: str,
    start: str,
    end: str,
    chunk_size: int,
    use_run: str,  # "new", "last", or a specific run folder name
) -> Dict[str, Any]:
    """
    If use_run == "new": extract ALL artifacts (overwrite if exist).
    Else (reuse an existing run): assert ALL artifacts are present; do not extract.
    """
    ensure_dir(BASE_DIR)
    outdir, outdir_name, reuse = make_run_folder(BASE_DIR, use_run)
    print(f"[info] Using run folder: {outdir_name} (reuse={reuse})")

    if not reuse:
        # New run -> pull everything unconditionally
        print("[info] Connecting to WRDS ...")
        conn = wrds_connect(wrds_user)
        params = {"start": start, "end": end}
        extract_artifacts(conn, ARTIFACTS, outdir, params=params, chunk_size=chunk_size, force=True)
        print("[info] Extraction complete (full refresh).")
    else:
        # Reuse -> assert presence, no extraction
        assert_artifacts_present(outdir, ARTIFACTS)
        print("[info] Reuse mode: all required Parquet files are present. No extraction performed.")

    produced = {
        parq: os.path.join(outdir, parq)
        for _, parq in ARTIFACTS
        if os.path.isfile(os.path.join(outdir, parq))
    }
    print("[result] Parquet files:")
    for name, path in produced.items():
        print(f"         - {name}: {path}")

    return {
        "run_folder": outdir,
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
