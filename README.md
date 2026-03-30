# CacaoCipher Analysis Pipeline

Reproducible Python pipelines for minimal SNP barcode design, benchmarking, and validation in cacao (*Theobroma cacao* L.).

## Overview

This repository contains two workflow layers:

1. `core_workflow/`  
   Core pipeline for assembling cleaned master tables and generating baseline outputs used in the manuscript.

2. `analysis_suite/`  
   Extended benchmarking and validation analyses, including:
   - population assignment benchmarking (including repeated stratified cross-validation where applicable)
   - trait–ancestry confounding control (PC adjustment + stratified permutation)
   - LD redundancy and information-capacity summaries
   - genetic distance structure preservation (Mantel / Procrustes)
   - overlap-based validation analyses using matched ICGT and PR/TARS materials

## Repository layout

- `core_workflow/` — core scripts for data assembly and baseline outputs
- `analysis_suite/` — benchmarking and validation analyses (tables + figures)
- `requirements.txt` — Python dependencies for the core workflow
- `analysis_suite/requirements.txt` — Python dependencies for the extended analyses
- `analysis_suite/environment.yml` — optional Conda environment for the extended analyses
- `DATA_SOURCES.md` — expected local input files and source notes

## Expected local directories

This repository expects a project root containing the following directories:

- `core_workflow/data_raw/`  
  Local raw inputs used by the core workflow.

- `core_workflow/master/`  
  Generated master tables and baseline outputs.

The extended analysis suite reads from the core output directory by default:

- `core_workflow/master/`

## Inputs

Raw spreadsheets and local working tables are not redistributed in this repository.  
Users should obtain the relevant source files locally and place them under:

- `core_workflow/data_raw/`

See `DATA_SOURCES.md` for expected local input files, naming conventions, and source notes.

## Quick start

### A. Core workflow

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the core workflow:

```bash
python run_all.py
```

### B. Extended benchmarking and validation analyses

Using pip:

```bash
pip install -r analysis_suite/requirements.txt
python analysis_suite/run_analyses.py --project-root . --master-dir ./core_workflow/master
```

Or using conda:

```bash
cd analysis_suite
conda env create -f environment.yml
conda activate cacaocipher-addon
python run_analyses.py --project-root .. --master-dir ../core_workflow/master
```

## Outputs

Core outputs are written to:

- `core_workflow/master/`

Extended-analysis outputs are written to:

- `analysis_suite/analysis_outputs/`

Tables are written as CSV files. Figures are written as PDF and PNG files where applicable.

## Notes

- Random seeds are fixed by default where relevant for reproducibility.
- Marker annotations reported in manuscript-associated supplementary materials are descriptive and are provided for interpretability only; they do not imply causal trait loci.
- Supplementary Data S1 distributed with the manuscript contains curated reporting outputs and should be treated as the authoritative reference for final reported supplementary values.
- This repository documents the analysis workflow used for manuscript-associated outputs; it is not intended to redistribute all local working files used during manuscript preparation.

## Scope

This repository is intended to document the analytical workflow underlying the CacaoCipher study, including:

- assembly of integrated genotype/phenotype master tables,
- design of minimal SNP barcode panels,
- baseline population-assignment and probability-of-identity summaries,
- extended benchmarking and validation analyses.

It is not intended to function as a packaged general-purpose software product beyond the manuscript-associated workflow.

## Citation

If you use this repository or adapt its workflow, please cite the associated manuscript and acknowledge the original data sources described in `DATA_SOURCES.md`.
