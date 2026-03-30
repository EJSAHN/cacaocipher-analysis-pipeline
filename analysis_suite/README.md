# CacaoCipher Analyses Suite

This repository contains additional, reviewer-facing analyses for the CacaoCipher manuscript.
It is designed to be run from an Anaconda prompt and to produce publication-ready outputs.

## Expected inputs

A project root containing:

- `master/icgt_geno_master.csv`
- `master/icgt_pheno_master.csv`
- `master/barcode_panel_32_stats.csv`
- `master/accession_overlap_auto.csv` (optional; used for PR overlap checks)
- `master/pr_traits_master.csv` (optional; used for PR validation analyses)

## Quick start (conda)

```bash
conda env create -f environment.yml
conda activate cacaocipher
python run_analyses.py --project-root .

If your files are nested (e.g., `master/master/*.csv`), pass `--master-dir master/master`.
```

Outputs will be written to `analysis_outputs/` by default.

## Outputs

All figures are saved as both:
- Vector PDF (`.pdf`)
- 300 dpi PNG (`.png`)

Tables are saved as CSV.

## Notes

- The code avoids hard-coded figure/table numbering.
- Random seeds are fixed by default for reproducibility.
