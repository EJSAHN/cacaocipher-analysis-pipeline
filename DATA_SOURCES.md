# Data Sources

This repository does not redistribute raw or compiled spreadsheets placed under `core_workflow/data_raw/` or `core_workflow/master/`, because some files are locally curated working tables and may include third-party source material.

## Required local inputs

To reproduce the analyses, place the required input files in your local project directories as expected by the scripts. Preferred public-facing filenames are listed below.

### 1. ICGT genotype/phenotype master table
**Preferred local filename:** `icgt_genotype_phenotype_master.xlsx`  
**Expected role:** master input containing ICGT accession identifiers, SNP genotypes, phenotype fields, and Acc Group labels used for panel design, benchmarking, and downstream analyses.  
**Primary source:** Bekele et al. (2022), *PLOS ONE*  
**DOI:** https://doi.org/10.1371/journal.pone.0260907

### 2. PR/TARS phenotype validation table
**Preferred local filename:** `pr_tars_validation_traits.xlsx`  
**Expected role:** Puerto Rico (TARS) field-trial phenotypes used for overlap-based cross-environment validation analyses.  
**Primary source:** Baek et al. (2025), *BMC Plant Biology*  
**DOI:** https://doi.org/10.1186/s12870-025-07128-y

### 3. ICGT–PR/TARS accession overlap table
**Preferred local filename:** generated automatically or supplied locally as needed  
**Expected role:** accession-matching file linking ICGT materials to PR/TARS records for overlap-based validation.  
**Source:** locally curated linkage table derived from manuscript-associated working files.

## Optional contextual inputs

### 4. Climate/context table for the PR/TARS trial
**Preferred local filename:** `pr_tars_climate_summary.csv`  
**Expected role:** optional descriptive climate summary for the Puerto Rico trial.  
**Use in this repository:** contextual only; not required to reproduce barcode design, benchmarking, or validation analyses.  
**Primary source:** NOAA Global Historical Climatology Network-Daily (GHCN-Daily)  
**Source URL:** https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily

### 5. Optional local harmonization tables
**Expected role:** temporary alignment or harmonization spreadsheets used during manuscript preparation.  
**Use in this repository:** optional; not required for routine public use unless explicitly referenced by the scripts.

## Placement

Place the expected local inputs under:

- `core_workflow/data_raw/`

Generated master tables and downstream CSV outputs are written under:

- `core_workflow/master/`

## Notes

- If you use differently named local files, rename them to the expected filenames or update the file paths in your local configuration before running the scripts.
- Supplementary Data S1 distributed with the manuscript contains curated reporting outputs, not all raw upstream inputs.
- Public repository users should treat the deposited manuscript and supplementary materials as the authoritative reference for final reported values.
