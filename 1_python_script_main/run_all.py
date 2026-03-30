from __future__ import annotations

import argparse
import datetime
from pathlib import Path
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS = [
    "1_making_master_file.py",
    "2_panel_design_clean.py",
    "3_population_assignment_and_pid.py",
]


def run_one(script_dir: Path, script_name: str, log_path: Path) -> None:
    script_path = script_dir / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    print("\n" + "=" * 60)
    print(f"RUN: {script_name}")
    print("=" * 60)

    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(script_dir),
        capture_output=True,
        text=True,
    )

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n" + "=" * 60 + "\n")
        handle.write(f"RUN: {script_name}\n")
        handle.write("=" * 60 + "\n")
        handle.write(proc.stdout)
        handle.write("\n--- STDERR ---\n")
        handle.write(proc.stderr)
        handle.write("\n")

    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"Script failed: {script_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the three core CacaoCipher scripts sequentially.")
    parser.add_argument(
        "--script-dir",
        type=str,
        default=str(SCRIPT_DIR),
        help="Directory containing the core scripts, data_raw/, and master/.",
    )
    args = parser.parse_args()

    script_dir = Path(args.script_dir).expanduser().resolve()
    log_dir = script_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_all_{ts}.log"

    print(f"[INFO] SCRIPT_DIR : {script_dir}")
    print(f"[INFO] LOGFILE    : {log_path}")

    for script_name in SCRIPTS:
        run_one(script_dir, script_name, log_path)

    print("\nALL DONE OK")
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
