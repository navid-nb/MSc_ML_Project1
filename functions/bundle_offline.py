import re
import subprocess as sp
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WHEELS = REPO / "wheels"
LOCK = REPO / "requirements.lock"
REQ = REPO / "requirements.txt"

DROP_PATTERNS = [
    r"^\-e\s",  # editable installs
    r"@ file://",  # local file refs
    r"@ git\+|^git\+",  # VCS refs
]


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd))
    sp.run(cmd, check=True)


def pip_freeze_filtered() -> list[str]:
    print("Freezing current environment ...")
    out = sp.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    keep: list[str] = []
    for ln in lines:
        if any(re.search(pat, ln) for pat in DROP_PATTERNS):
            print(f"[skip] {ln}")
            continue
        keep.append(ln)
    return keep


def write_lock(lines: list[str]) -> None:
    LOCK.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote pinned lock: {LOCK}")


def ensure_wheels_dir() -> None:
    WHEELS.mkdir(exist_ok=True)
    # clean stray metadata (optional)
    for p in WHEELS.glob("*.metadata"):
        p.unlink(missing_ok=True)


def download_wheels() -> None:
    print("\nDownloading/bundling wheels (binary only preferred) ...")
    # Try pure download of prebuilt wheels first
    try:
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                "--only-binary",
                ":all:",
                "-r",
                str(LOCK),
                "-d",
                str(WHEELS),
            ]
        )
    except sp.CalledProcessError:
        print("\nSome packages lack prebuilt wheels for this platform; building wheels locally ...")
        # Build wheels for anything missing
        run([sys.executable, "-m", "pip", "wheel", "-r", str(LOCK), "-w", str(WHEELS)])


def verify_coverage() -> None:
    # Very light check: every pinned distribution name should have at least one wheel prefix match
    # (This is best-effort; platform tags vary.)
    def norm_name(s: str) -> str:
        return re.split(r"[<>=!~]", s, 1)[0].strip().lower().replace("_", "-")

    pins = [norm_name(ln) for ln in LOCK.read_text().splitlines() if ln and not ln.startswith("#")]
    wheel_names = []
    for wf in WHEELS.glob("*.whl"):
        name = wf.name.split("-")[0].lower().replace("_", "-")
        wheel_names.append(name)
    missing = sorted({p for p in pins if p not in wheel_names})
    if missing:
        print("\n[warn] Some pinned packages don’t have a matching wheel filename in ./wheels:")
        for m in missing:
            print(" -", m)
        print(
            "They may be satisfied via transitive wheels already present, or you may need to re-run pip wheel."
        )
    else:
        print("All pinned packages appear to be covered by wheels.")


def write_requirements_txt() -> None:
    content = """\
--no-index
--find-links ./wheels
-r requirements.lock
"""
    REQ.write_text(content, encoding="utf-8")
    print(f"Wrote {REQ} that points to ./wheels and uses the lock.")


def main() -> None:
    ensure_wheels_dir()
    lines = pip_freeze_filtered()
    if not lines:
        print("[!] Nothing to lock. Is your environment empty?")
        sys.exit(1)
    write_lock(lines)
    download_wheels()
    verify_coverage()
    write_requirements_txt()
    print(
        "\n✅ Offline bundle ready.\n- wheels/\n- requirements.lock\n- requirements.txt (points to wheels)\n"
    )


if __name__ == "__main__":
    main()
