import argparse
import platform
import subprocess
import sys
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent

def sh(cmd: list[str]) -> None:
    print("$ " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)

def venv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")

def install_offline(py: Path) -> None:
    req = ROOT / "requirements.txt"
    if not req.exists():
        raise FileNotFoundError("requirements.txt not found. Run tools/bundle_offline.py first.")
    print("Installing (offline, from ./wheels) …")
    sh([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    sh([str(py), "-m", "pip", "install", "-r", str(req)])  # requirements.txt already sets --no-index/--find-links

def install_online(py: Path) -> None:
    print("Installing (online, from index) …")
    sh([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    # Prefer fully pinned lock if present; otherwise fall back to requirements.txt
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
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--offline-only", action="store_true")
    mode.add_argument("--online-only", action="store_true")
    args = p.parse_args()

    venv_dir = (ROOT / args.venv).resolve()
    if args.force_recreate and venv_dir.exists():
        print(f"Removing existing venv at {venv_dir} …")
        shutil.rmtree(venv_dir)

    if not venv_dir.exists():
        print(f"Creating virtual environment at {venv_dir} …")
        sh([sys.executable, "-m", "venv", str(venv_dir)])
    else:
        print(f"Using existing virtual environment at {venv_dir}")

    py = venv_python(venv_dir)
    if not py.exists():
        sys.exit(f"[!] venv Python not found at {py}")

    wheels_dir = ROOT / "wheels"
    tried_offline = False

    try:
        if args.online_only:
            install_online(py)
        elif args.offline_only:
            if not wheels_dir.exists():
                raise FileNotFoundError("wheels/ not found for --offline-only")
            install_offline(py)
        else:
            # AUTO: try offline first, then fall back
            if wheels_dir.exists():
                tried_offline = True
                try:
                    install_offline(py)
                except subprocess.CalledProcessError as e:
                    print(f"[warn] Offline install failed (exit {e.returncode}); falling back to online …")
                    install_online(py)
            else:
                install_online(py)
    except Exception as e:
        sys.exit(f"[!] Installation failed: {e}")

    print("\n✅ Environment ready.")
    print("Run WITHOUT activating the venv:")
    print(f"  {py} run_strategy.py")
    print(f"  {py} run_data.py   # data-prep if applicable\n")

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