from pathlib import Path
import subprocess, sys
ROOT = Path(__file__).resolve().parent
for name in ["core_workflow/01_build_master_tables.py","core_workflow/02_design_barcode_panels.py","core_workflow/03_population_assignment_pid.py"]:
    subprocess.check_call([sys.executable, name], cwd=ROOT)
