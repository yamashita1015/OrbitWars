"""Build and optionally submit submission.tar.gz for orbit-wars.

Usage:
  python build_submission.py            # build only
  python build_submission.py --submit   # build + submit
  python build_submission.py --submit --message "my message"
"""
import argparse
import os
import py_compile
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
MAIN_PY = HERE / "main.py"
ORBIT_LITE = HERE / "orbit_lite"
OUTPUT = HERE / "submission.tar.gz"
COMPETITION = "orbit-wars"

EXPECTED_ORBIT_LITE_FILES = {
    "__init__.py", "adapter.py", "aiming.py", "constants.py",
    "distance_cache.py", "garrison_launch.py", "geometry.py",
    "intercept_aim.py", "movement.py", "movement_aiming.py",
    "movement_step.py", "obs.py", "planner_core.py",
}


def build():
    print("=== Building submission.tar.gz ===")

    # 1. Syntax check main.py
    py_compile.compile(str(MAIN_PY), doraise=True)
    print(f"  [OK] {MAIN_PY.name} compiles")

    # 2. Check orbit_lite completeness
    actual = {f.name for f in ORBIT_LITE.iterdir() if f.suffix == ".py"}
    missing = EXPECTED_ORBIT_LITE_FILES - actual
    if missing:
        print(f"  [WARN] orbit_lite missing files: {missing}")
    else:
        print(f"  [OK] orbit_lite has all {len(EXPECTED_ORBIT_LITE_FILES)} expected .py files")

    # 3. Build archive (exclude __pycache__ and .pyc)
    with tarfile.open(OUTPUT, "w:gz") as tar:
        tar.add(MAIN_PY, arcname="main.py")
        for f in sorted(ORBIT_LITE.iterdir()):
            if f.name == "__pycache__" or f.suffix == ".pyc":
                continue
            tar.add(f, arcname=f"orbit_lite/{f.name}")

    size_kb = OUTPUT.stat().st_size / 1024
    with tarfile.open(OUTPUT, "r:gz") as tar:
        names = tar.getnames()
    py_files = [n for n in names if n.endswith(".py")]
    print(f"  [OK] {OUTPUT.name}: {size_kb:.1f} KB, {len(py_files)} .py files")
    for n in sorted(names):
        print(f"       {n}")

    # 4. Smoke test: extract to temp dir and re-compile main.py
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with tarfile.open(OUTPUT, "r:gz") as tar:
            tar.extractall(tmp_path)
        py_compile.compile(str(tmp_path / "main.py"), doraise=True)
    print("  [OK] smoke compile after extraction passed")

    print(f"\nBuild complete: {OUTPUT}")
    return True


def submit(message: str):
    print(f"\n=== Submitting to {COMPETITION} ===")
    venv_kaggle = HERE / ".venv" / "bin" / "kaggle"
    kaggle_cmd = str(venv_kaggle) if venv_kaggle.exists() else "kaggle"

    cmd = [
        kaggle_cmd, "competitions", "submit",
        COMPETITION,
        "-f", str(OUTPUT),
        "-m", message,
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print(f"  [ERROR] submission failed (exit code {result.returncode})")
        return False
    print("  [OK] submitted")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--message", "-m", default="Producer Hybrid v2 + ring/adaptive/defense/terminal")
    args = parser.parse_args()

    ok = build()
    if ok and args.submit:
        submit(args.message)
