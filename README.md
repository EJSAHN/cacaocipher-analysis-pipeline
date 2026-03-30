# CacaoCipher Analysis Pipeline

Reproducible Python workflows for minimal SNP barcode design, benchmarking, and validation in cacao (*Theobroma cacao* L.).

## Repository layout

- `1_python_script_main/` - core manuscript workflow that assembles master tables and generates baseline outputs.
- `2_Add_on/` - add-on benchmarking and validation analyses used for the revision stage.
- `requirements.txt` - dependencies for the core workflow.
- `2_Add_on/requirements.txt` - dependencies for the add-on analyses.
- `2_Add_on/environment.yml` - optional conda environment for the add-on analyses.

## Expected local directories

The core workflow reads and writes within `1_python_script_main/`:

- `1_python_script_main/data_raw/` - local raw inputs (not redistributed in this repository).
- `1_python_script_main/master/` - generated master tables and core outputs.

The add-on workflow reads from the core output directory by default:

- `1_python_script_main/master/`

## Quick start

### 1) Run the core workflow

```bash
pip install -r requirements.txt
python 1_python_script_main/run_all.py
```

### 2) Run the add-on analyses

Using pip:

```bash
pip install -r 2_Add_on/requirements.txt
python 2_Add_on/run_analyses.py --project-root .
```

Using conda:

```bash
cd 2_Add_on
conda env create -f environment.yml
conda activate cacaocipher-addon
python run_analyses.py --project-root ..
```

## Outputs

Core outputs are written to `1_python_script_main/master/`.

Add-on outputs are written to `2_Add_on/addon_outputs/`, with tables saved as CSV and figures saved as both PDF and PNG.

## Notes

- Random seeds are fixed by default for reproducibility.
- Marker annotations included in Supplementary Data S1 are descriptive and are provided for interpretability only; they do not imply causal trait loci.
- Raw spreadsheets used during manuscript preparation are not redistributed here. See `1_python_script_main/DATA_SOURCES.md` for the expected local inputs.
