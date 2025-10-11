import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

SUPPORTED_MIN = (3, 10)  # inclusive
SUPPORTED_MAX = (3, 13)  # inclusive

ROOT = Path(__file__).resolve().parent


def in_supported_range(ver: tuple[int, int, int] | tuple[int, int]) -> bool:
    maj, min_ = ver[0], ver[1]
    return maj == 3 and SUPPORTED_MIN[1] <= min_ <= SUPPORTED_MAX[1]


def version_str(ver: tuple[int, int, int] | None) -> str:
    if not ver:
        return "unknown"
    return f"{ver[0]}.{ver[1]}.{ver[2]}"


def sh(cmd: list[str], env: dict | None = None) -> None:
    print("$ " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, env=env)


def install_offline(py: Path) -> None:
    wheels = ROOT / "wheels"
    req = ROOT / "requirements.txt"
    if not req.exists():
        raise FileNotFoundError("requirements.txt not found. Run tools/bundle_offline.py first.")
    if not wheels.exists():
        raise FileNotFoundError("wheels/ not found. Cannot install offline.")

    # Make pip offline and quiet about version checks
    offline_env = os.environ.copy()
    offline_env.update(
        {
            "PIP_NO_INDEX": "1",
            "PIP_FIND_LINKS": str(wheels.resolve()),
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        }
    )

    print("Installing (offline, from ./wheels) ...")

    # upgrade pip/setuptools/wheel from local wheels if present
    has_pip = any(wheels.glob("pip-*.whl"))
    has_setuptools = any(wheels.glob("setuptools-*.whl"))
    has_wheel = any(wheels.glob("wheel-*.whl"))
    if has_pip and has_setuptools and has_wheel:
        sh(
            [
                str(py),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--find-links",
                str(wheels),
                "--only-binary",
                ":all:",
                "--upgrade",
                "pip",
                "setuptools",
                "wheel",
            ],
            env=offline_env,
        )
    else:
        print("No local wheels for pip/setuptools/wheel; skipping their upgrade in offline mode.")

    # install deps from ./wheels
    sh(
        [
            str(py),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(wheels),
            "--only-binary",
            ":all:",
            "--no-build-isolation",
            "-r",
            str(req),
        ],
        env=offline_env,
    )


def get_version_and_exe(cmd: list[str]) -> tuple[tuple[int, int, int] | None, Path | None]:
    """
    Run `cmd -c` to get version + sys.executable. Return ((maj, min, mic), Path) or (None, None) on failure.
    """
    try:
        out = (
            subprocess.check_output(
                cmd
                + [
                    "-c",
                    "import sys; "
                    "print(sys.version_info[0],sys.version_info[1],sys.version_info[2]); "
                    "print(sys.executable)",
                ],
                text=True,
            )
            .strip()
            .splitlines()
        )
        if len(out) >= 2:
            maj, min_, mic = map(int, out[0].split())
            exe = Path(out[1]).resolve()
            return (maj, min_, mic), exe
    except (Exception,):
        return None, None
    return None, None


def find_supported_python(user_python: str | None) -> Path:
    """
    Resolve a Python 3.10–3.13 interpreter.
    Priority:
      1) --python (exact command/path)
      2) current interpreter if in range
      3) common launchers on this OS (tries 3.13, 3.12, 3.11, 3.10)
    """
    # 1) explicit
    if user_python:
        ver, exe = get_version_and_exe([user_python])
        if ver and in_supported_range(ver):
            return exe
        sys.exit(
            f"[!] --python must point to Python {SUPPORTED_MIN[0]}.{SUPPORTED_MIN[1]}–{SUPPORTED_MAX[1]}.*, "
            f"but got {version_str(ver)} from {user_python!r}"
        )

    # 2) current process
    cur = sys.version_info
    if in_supported_range((cur.major, cur.minor, cur.micro)):
        return Path(sys.executable).resolve()

    # 3) try common commands for 3.13 -> 3.10
    versions = [13, 12, 11, 10]
    candidates: list[list[str]] = []

    if platform.system() == "Windows":
        for v in versions:
            candidates.append(["py", f"-3.{v}"])
        for v in versions:
            candidates.append([f"python3.{v}"])
            candidates.append([f"python3{v}"])
            candidates.append([f"python{v}"])
        candidates += [["python3"], ["python"]]
    else:
        for v in versions:
            candidates.append([f"python3.{v}"])
            candidates.append([f"python{v}"])
        candidates += [["python3"], ["python"]]

    for cmd in candidates:
        ver, exe = get_version_and_exe(cmd)
        if ver and in_supported_range(ver):
            return exe

    # No suitable interpreter found
    msg = [
        f"[!] Could not find a supported Python in range "
        f"{SUPPORTED_MIN[0]}.{SUPPORTED_MIN[1]}–{SUPPORTED_MAX[1]}.* on this system.",
        "Install one and rerun, or pass --python PATH/COMMAND.",
        "",
        "Hints:",
    ]
    if platform.system() == "Windows":
        msg += [
            "  • Install Python 3.10–3.13 from python.org (ensure 'py' launcher installed).",
            "  • Then this usually works:  py -3 run_install_packages.py",
        ]
    elif platform.system() == "Darwin":
        msg += [
            "  • Homebrew:  brew install python@3.12  (or 3.10/3.11/3.13 as available)",
            "  • Then run:  python3 run_install_packages.py",
        ]
    else:
        raise EnvironmentError("Only Windows and Darwin(MacOS) are supported.")

    sys.exit("\n".join(msg))


def venv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")


def install_online(py: Path) -> None:
    print("Installing (online, from index) ...")
    sh([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    lock = ROOT / "requirements.lock"
    req = ROOT / "requirements.txt"
    spec = lock if lock.exists() else req
    if spec is None or not spec.exists():
        raise FileNotFoundError("No requirements.lock or requirements.txt found.")
    sh([str(py), "-m", "pip", "install", "-r", str(spec)])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--venv", default=".venv")
    p.add_argument("--force-recreate", action="store_true")
    p.add_argument(
        "--python",
        help=(
            f"Command or path to a Python {SUPPORTED_MIN[0]}.{SUPPORTED_MIN[1]}–{SUPPORTED_MAX[1]} "
            f"interpreter to create the venv"
        ),
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--offline-only", action="store_true")
    mode.add_argument("--online-only", action="store_true")
    args = p.parse_args()

    # Resolve a supported Python interpreter
    creator_py = find_supported_python(args.python)
    print(f"Using Python for venv creation: {creator_py}")

    venv_dir = (ROOT / args.venv).resolve()
    if args.force_recreate and venv_dir.exists():
        print(f"Removing existing venv at {venv_dir} ...")
        shutil.rmtree(venv_dir)

    if not venv_dir.exists():
        print(f"Creating virtual environment at {venv_dir} ...")
        sh([str(creator_py), "-m", "venv", str(venv_dir)])
    else:
        print(f"Using existing virtual environment at {venv_dir}")

    py = venv_python(venv_dir)
    if not py.exists():
        sys.exit(f"[!] venv Python not found at {py}")

    wheels_dir = ROOT / "wheels"

    try:
        if args.online_only:
            install_online(py)
        elif args.offline_only:
            if not wheels_dir.exists():
                raise FileNotFoundError("wheels/ not found for --offline-only")
            install_offline(py)
        else:
            # try offline first, then fall back to online
            if wheels_dir.exists():
                try:
                    install_offline(py)
                except subprocess.CalledProcessError as e:
                    print(
                        f"[warn] Offline install failed (exit {e.returncode}); falling back to online ..."
                    )
                    install_online(py)
            else:
                install_online(py)
    except Exception as e:
        sys.exit(f"[!] Installation failed: {e}")

    # Print the actual venv interpreter version
    ver, _ = get_version_and_exe([str(py)])
    print(f"\n Environment ready (Python {version_str(ver)} supported in [3.10.*, 3.13.*]).")
    print("Run WITHOUT activating the venv:")
    print(f"  {py} run_strategy.py")

    if platform.system() == "Windows":
        print("Or activate the environment first:")
        print("  PowerShell:  .\\.venv\\Scripts\\Activate.ps1")
        print("  CMD:         .\\.venv\\Scripts\\activate.bat")
    else:
        print("Or activate the environment first:")
        print("  source .venv/bin/activate")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
