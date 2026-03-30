# CacaoCipher Add-on Analyses

This directory contains benchmarking and validation analyses for the CacaoCipher manuscript.

## Expected inputs

A project root containing:

- `master/icgt_geno_master.csv`
- `master/icgt_pheno_master.csv`
- `master/barcode_panel_32_stats.csv`
- `master/accession_overlap_auto.csv` (optional; used for PR/TARS overlap checks)
- `master/pr_traits_master.csv` (optional; used for PR/TARS validation analyses)

## Quick start

Using pip:

```bash
pip install -r requirements.txt
python run_analyses.py --project-root .. --master-dir ..\master
```

Using conda:

```bash
conda env create -f environment.yml
conda activate cacaocipher-addon
python run_analyses.py --project-root .. --master-dir ..\master
```

If your files are nested differently, pass `--master-dir` explicitly.

## Outputs

Outputs are written to `addon_outputs/` by default.

## Notes

- Figure and table numbering are not hard-coded.
- Random seeds are fixed by default for reproducibility.
