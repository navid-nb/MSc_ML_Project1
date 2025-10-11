import datetime as dt
import os
import shutil
from typing import List, Optional


def list_runs(base_dir: str) -> List[str]:
    """
    List all subdirectories in the base directory whose names start with 'run'.
    Args:
        base_dir (str): The directory path to search for run folders.

    Returns:
        List[str]: A sorted list of run folder names within the base directory.
    """
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
    """
    Return the latest (most recently sorted) run folder name in the base directory.
    Args:
        base_dir (str): The base directory containing run folders.

    Returns:
        Optional[str]: The name of the latest run folder, or None if none exist.
    """
    runs = list_runs(base_dir)
    return runs[-1] if runs else None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_delete_dir(path: str, base: str) -> None:
    """
    Safely delete the target directory only if it resides within the base directory.

    Args:
        path (str): The directory to delete.
        base (str): The allowed parent directory for deletion safety.

    Raises:
        ValueError: If the target path is not under the base directory.

    Returns:
        None
    """
    abs_path = os.path.abspath(path)
    abs_base = os.path.abspath(base)

    if os.path.commonpath([abs_path, abs_base]) != abs_base:
        raise ValueError(f"Refusing to delete {abs_path}: not inside {abs_base}")

    if os.path.isdir(abs_path):
        shutil.rmtree(abs_path)


def make_run_folder(base_dir: str, use_run: str) -> tuple[str, str, bool]:
    """
    Decide on a run folder name and create the folder if needed.

    Args:
        base_dir (str): Base directory to store run folders.
        use_run (str): Controls folder creation:
            - "new": create a fresh timestamped folder.
            - "last": reuse the most recent run folder if it exists.
            - explicit folder name: use/reuse that folder name.

    Returns:
        tuple: (absolute path to folder, folder name, reuse flag)
            - (str): absolute path to the run folder.
            - (str): run folder name.
            - (bool): True if reusing an existing folder, False if creating new.
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

    out_dir = os.path.join(base_dir, out_dir_name)
    ensure_dir(out_dir)
    return out_dir, out_dir_name, reuse
