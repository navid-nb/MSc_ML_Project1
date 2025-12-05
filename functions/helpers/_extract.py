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

    Note:
        This helper is intended for local filesystem use only.
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

    Note:
        This helper is intended for local filesystem use only.
    """
    runs = list_runs(base_dir)
    return runs[-1] if runs else None


def ensure_dir(path: str) -> None:
    """
    Ensure that a local directory exists. For S3 URIs (paths starting with 's3://'),
    this function is a no-op.

    Args:
        path (str): Local directory path or S3 URI.
    """
    if isinstance(path, str) and path.startswith("s3://"):
        # S3 has no real directories; caller should manage prefixes via S3 APIs.
        return
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

    Note:
        This helper is intended for local filesystem paths only. S3 deletion
        should be handled by S3-specific utilities.
    """
    abs_path = os.path.abspath(path)
    abs_base = os.path.abspath(base)

    if os.path.commonpath([abs_path, abs_base]) != abs_base:
        raise ValueError(f"Refusing to delete {abs_path}: not inside {abs_base}")

    if os.path.isdir(abs_path):
        shutil.rmtree(abs_path)


def make_run_folder(
    base_dir: str,
    use_run: str,
    *,
    s3_bucket: Optional[str] = None,
    s3_prefix: Optional[str] = None,
) -> tuple[str, str, bool]:
    """
    Decide on a run folder name and create the folder if needed.

    This helper supports both:
      - Local filesystem runs (default, when s3_bucket is None).
      - S3-based "run folders" (when s3_bucket is provided), in which case it
        returns an S3 URI and does not touch the local filesystem.

    Args:
        base_dir (str):
            Local base directory to store run folders (local mode) OR
            a logical prefix used when constructing S3 paths if s3_prefix is not given.
        use_run (str): Controls folder creation:
            - "new": create a fresh timestamped folder.
            - "last": reuse the most recent run folder if it exists (local mode only).
            - explicit folder name: use/reuse that folder name.
        s3_bucket (str | None):
            If provided, S3 mode is enabled and the function returns an S3 URI
            instead of a local path. No local directories are created.
        s3_prefix (str | None):
            Optional prefix within the S3 bucket under which run folders live.
            If not provided, base_dir is used as the prefix (if non-empty).

    Returns:
        tuple: (absolute path / URI to folder, folder name, reuse flag)
            - (str): absolute path to the run folder (local path or s3:// URI).
            - (str): run folder name (e.g. "run_20251204_153000").
            - (bool): True if reusing an existing local folder, False otherwise.

    Notes:
        - In S3 mode, `reuse` is always False (this helper does not inspect S3).
        - In S3 mode, the returned string is of the form:
              s3://{bucket}/{prefix}/{run_folder_name}
    """
    # ------------------------------------------------------------------
    # S3 MODE: construct an S3 URI and do not touch the local filesystem
    # ------------------------------------------------------------------
    if s3_bucket is not None:
        if use_run == "new":
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir_name = f"run_{stamp}"
            reuse = False
        elif use_run == "last":
            # We don't attempt to list S3 to find the latest run; just create a new one.
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir_name = f"run_{stamp}"
            reuse = False
            print(
                "[warn] make_run_folder: 'last' is not supported in S3 mode; "
                "creating a new run folder instead."
            )
        else:
            # Explicit run name; we cannot know if it already exists in S3.
            out_dir_name = use_run
            reuse = False

        prefix_parts: List[str] = []
        if s3_prefix:
            prefix_parts.append(s3_prefix.strip("/"))
        elif base_dir:
            prefix_parts.append(str(base_dir).strip("/"))

        prefix_str = "/".join(p for p in prefix_parts if p)
        if prefix_str:
            out_dir = f"s3://{s3_bucket}/{prefix_str}/{out_dir_name}"
        else:
            out_dir = f"s3://{s3_bucket}/{out_dir_name}"

        return out_dir, out_dir_name, reuse

    # ------------------------------------------------------------------
    # LOCAL FILESYSTEM MODE (original behavior)
    # ------------------------------------------------------------------
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
