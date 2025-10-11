import argparse
import os
import re
import subprocess as sp
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WHEELS = REPO / "wheels"
BASE_LOCK = REPO / "requirements.lock"
REQ_TXT = REPO / "requirements.txt"

DEF_PY_CANDIDATES = [
    # macOS/Linux first
    "python3.13",
    "python3.12",
    "python3.11",
    "python3.10",
    "python13",
    "python12",
    "python11",
    "python10",
    "python3",
    "python",
    # Windows launchers (if run on Windows)
    "py -3.13",
    "py -3.12",
    "py -3.11",
    "py -3.10",
]

DROP_PATTERNS = [r"^\-e\s", r"@ file://", r"@ git\+|^git\+"]


def sh(cmd: list[str], check: bool = True, env: dict | None = None) -> sp.CompletedProcess:
    print("$ " + " ".join(cmd))
    return sp.run(cmd, check=check, env=env)


def check_output(cmd: list[str]) -> str:
    return sp.check_output(cmd, text=True).strip()


def ensure_wheels_dir() -> None:
    WHEELS.mkdir(exist_ok=True)
    for p in WHEELS.glob("*.metadata"):
        p.unlink(missing_ok=True)


def py_info(py_cmd: str):
    """Return (version_tuple, real_executable_path, osname) or (None, None, None) if unavailable."""
    try:
        out = check_output(
            [
                *py_cmd.split(),
                "-c",
                "import sys,platform,os; "
                "print(f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}'); "
                "print(os.path.realpath(sys.executable)); "
                "print(platform.system().lower())",
            ]
        ).splitlines()
        ver = tuple(map(int, out[0].split(".")))
        exe = out[1]
        osname = out[2]
        return ver, exe, osname
    except Exception:
        return None, None, None


def freeze_filtered(py_exe: str) -> list[str]:
    """pip freeze, dropping editable/VCS/local-file entries."""
    out = check_output([py_exe, "-m", "pip", "freeze"])
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    keep: list[str] = []
    for ln in lines:
        if any(re.search(pat, ln) for pat in DROP_PATTERNS):
            print(f"[skip] {ln}")
            continue
        keep.append(ln)
    return keep


def write_lock(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote pinned lock: {path}")


def build_for_interpreter(py_exe: str, lock_path: Path) -> None:
    print(f"\n=== Building wheelhouse for {py_exe} using {lock_path.name} ===")
    # First: try to download only prebuilt binary wheels
    try:
        sh(
            [
                py_exe,
                "-m",
                "pip",
                "download",
                "--only-binary",
                ":all:",
                "--exists-action",
                "i",  # don't overwrite existing artifacts
                "-r",
                str(lock_path),
                "-d",
                str(WHEELS),
            ]
        )
    except sp.CalledProcessError:
        print("[info] Some packages had no prebuilt wheels; attempting local build …")
    # Second: build local wheels for anything missing
    sh(
        [
            py_exe,
            "-m",
            "pip",
            "wheel",
            "--exists-action",
            "i",  # keep existing wheels intact
            "-r",
            str(lock_path),
            "-w",
            str(WHEELS),
        ]
    )


def verify_coverage(lock_path: Path) -> None:
    """Light verification that each pinned dist base name appears among wheel basenames."""

    def norm_name(s: str) -> str:
        return re.split(r"[<>=!~]", s, 1)[0].strip().lower().replace("_", "-")

    pins = [
        norm_name(ln)
        for ln in lock_path.read_text().splitlines()
        if ln and not ln.startswith("#") and not ln.startswith("--")
    ]
    wheel_names = {wf.name.split("-")[0].lower().replace("_", "-") for wf in WHEELS.glob("*.whl")}
    missing = sorted({p for p in pins if p not in wheel_names})
    if missing:
        print("\n[warn] Some pinned distributions have no matching wheel basename in ./wheels:")
        for m in missing:
            print(" -", m)
        print(
            "They may still be satisfied by transitive deps or env markers, but review is recommended."
        )
    else:
        print("All pinned packages appear covered by wheels (basename check).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--py",
        nargs="*",
        default=None,
        help="Interpreter commands/paths to build for (default: try 3.13→3.10)",
    )
    ap.add_argument(
        "--lock", default=str(BASE_LOCK), help="Base lock file to use (default: requirements.lock)"
    )
    ap.add_argument(
        "--write-per-version-locks",
        action="store_true",
        help="Freeze and write requirements-<maj.min>.lock per interpreter",
        default=True,
    )
    args = ap.parse_args()

    base_lock = Path(args.lock)
    if not base_lock.exists():
        sys.exit(f"[!] Lock file not found: {base_lock}")

    ensure_wheels_dir()

    # Discover interpreters and de-duplicate by (real executable, major.minor)
    py_cmds = args.py or DEF_PY_CANDIDATES
    chosen: list[tuple[str, tuple[int, int, int]]] = []
    seen_versions: set[tuple[str, int, int]] = set()
    for cmd in py_cmds:
        ver, exe, _ = py_info(cmd)
        if not exe or not ver:
            continue
        key = (exe, ver[0], ver[1])
        if key in seen_versions:
            print(f"[dup ] Python {ver[0]}.{ver[1]}.{ver[2]} @ {exe} - skipping")
            continue
        seen_versions.add(key)
        if ver[0] == 3 and 10 <= ver[1] <= 13:
            print(f"[found] Python {ver[0]}.{ver[1]}.{ver[2]} @ {exe}")
            chosen.append((exe, ver))

    if not chosen:
        sys.exit("[!] No suitable Python 3.10–3.13 interpreters found.")

    # Optionally generate per-version locks by freezing each interpreter's resolution
    if args.write_per_version_locks:
        for exe, ver in chosen:
            # Create a temp venv to resolve what that interpreter can actually install from base_lock
            with tempfile.TemporaryDirectory() as td:
                venv = Path(td) / "v"
                sh([exe, "-m", "venv", str(venv)])
                # Use that interpreter's pip to install from the base lock
                pbin = venv / ("Scripts/pip.exe" if os.name == "nt" else "bin/pip")
                sh([str(pbin), "install", "-r", str(base_lock)])
                # Freeze using that interpreter
                pybin = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
                lines = freeze_filtered(str(pybin))
            lock_out = REPO / f"requirements-{ver[0]}.{ver[1]}.lock"
            write_lock(lock_out, lines)

    # Build wheelhouse for each interpreter (all into ./wheels)
    for exe, ver in chosen:
        lock_to_use = REPO / f"requirements-{ver[0]}.{ver[1]}.lock"
        if not lock_to_use.exists():
            lock_to_use = base_lock
        build_for_interpreter(exe, lock_to_use)

    # Light verification against the base lock
    verify_coverage(base_lock)

    # Write a requirements.txt that points to wheels and uses the base lock by default
    REQ_TXT.write_text(
        "--no-index\n--find-links ./wheels\n-r requirements.lock\n", encoding="utf-8"
    )
    print(f"\n[done] Unified wheelhouse in ./wheels for {len(chosen)} interpreters.")
    print("       requirements.txt written; per-version locks present where requested.")


if __name__ == "__main__":
    main()
