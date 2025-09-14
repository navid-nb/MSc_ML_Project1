import datetime as dt
import os
import shutil
from typing import List, Optional


def list_runs(base_dir: str) -> List[str]:
    if not os.path.isdir(base_dir):
        return []
    runs = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")
    ]
    runs.sort()
    return runs


def latest_run(base_dir: str) -> Optional[str]:
    runs = list_runs(base_dir)
    return runs[-1] if runs else None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def delete_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)


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
        out_dir_name = f"run_{stamp}"
        reuse = False
    elif use_run == "last":
        last = latest_run(base_dir)
        if last:
            out_dir_name = last
            reuse = True
        else:
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir_name = f"run_{stamp}"
            reuse = False
    else:
        out_dir_name = use_run
        reuse = os.path.isdir(os.path.join(base_dir, out_dir_name))

    outdir = os.path.join(base_dir, out_dir_name)
    ensure_dir(outdir)
    return outdir, out_dir_name, reuse
