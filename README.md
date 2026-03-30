# CacaoCipher Analysis Pipeline

Reproducible Python workflows for minimal SNP barcode design, benchmarking, and validation in cacao (*Theobroma cacao* L.).

## Repository structure

This package contains two codebases:

1. `1_python_script_main/`  
   Core scripts for assembling cleaned master tables and baseline outputs.

2. `2_Add_on/`  
   Benchmarking and validation analyses, including:
   - population assignment benchmarking
   - trait–ancestry confounding control
   - LD redundancy and information-capacity summaries
   - genetic distance structure preservation

## Inputs

The workflows expect a project root containing:

- `master/`  
  Master CSV files used by the benchmarking/validation analyses  
  (for example: `icgt_geno_master.csv`, `icgt_pheno_master.csv`, and optional PR/TARS overlap files).

- `data_raw/` (optional)  
  Raw input files used by the core pipeline.

If raw inputs are not redistributed, obtain them from the original data sources described in the manuscript and place them under `data_raw/` using the expected filenames.

## Quick start

### A. Core pipeline (optional)

```bash
pip install -r Requirements.txt
python 1_python_script_main/run_all.py --project-root .
```

### B. Benchmarking and validation analyses

Using pip:

```bash
pip install -r 2_Add_on/requirements.txt
python 2_Add_on/run_analyses.py --project-root . --master-dir .\master
```

Using conda:

```bash
cd 2_Add_on
conda env create -f environment.yml
conda activate cacaocipher-addon
python run_analyses.py --project-root .. --master-dir ..\master
```

## Outputs

Add-on outputs are written to:

- `2_Add_on/addon_outputs/`

## Notes

- Figure and table numbering are not hard-coded.
- Random seeds are fixed by default for reproducibility.
- Marker annotations included in Supplementary Data S1 are descriptive and provided for interpretability only; they do not imply causal trait loci.
