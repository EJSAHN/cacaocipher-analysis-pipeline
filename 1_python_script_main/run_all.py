from pathlib import Path
import subprocess
import sys
import datetime

ROOT = Path(r"C:\projects\Bekele").resolve()
scripts = [
    "1_making_master_file.py",
    "2_panel_design_clean.py",
    "03_population_assignment_and_pid.py",
]

log_dir = ROOT / "logs"
log_dir.mkdir(exist_ok=True)

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = log_dir / f"run_all_{ts}.log"

def run_one(script_name: str) -> None:
    script_path = ROOT / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Missing: {script_path}")

    print("\n" + "=" * 60)
    print(f"RUN: {script_name}")
    print("=" * 60)

    # run with cwd fixed to ROOT (important for relative paths)
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )

    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"RUN: {script_name}\n")
        f.write("=" * 60 + "\n")
        f.write(proc.stdout)
        f.write("\n--- STDERR ---\n")
        f.write(proc.stderr)
        f.write("\n")

    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"Script failed: {script_name}")

def main():
    print(f"[INFO] ROOT    : {ROOT}")
    print(f"[INFO] LOGFILE : {log_path}")

    for s in scripts:
        run_one(s)

    print("\nALL DONE OK")
    print(f"Log saved to: {log_path}")

if __name__ == "__main__":
    main()
