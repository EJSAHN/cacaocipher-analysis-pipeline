# Data Sources

This repository does not redistribute the local spreadsheets placed under `1_python_script_main/data_raw/`. Some of those files are manuscript-associated working tables and may include third-party source material.

## Required local inputs

Place the following files under `1_python_script_main/data_raw/` using the preferred filenames below.

### 1. ICGT genotype/phenotype master table
**Preferred local filename:** `icgt_genotype_phenotype_master.xlsx`  
**Role:** ICGT accession identifiers, SNP genotypes, phenotype fields, and Acc Group labels used for panel design and downstream analyses.  
**Primary source:** Bekele et al. (2022), *PLOS ONE*  
**DOI:** https://doi.org/10.1371/journal.pone.0260907

### 2. PR/TARS phenotype validation table
**Preferred local filename:** `pr_tars_validation_traits.xlsx`  
**Role:** Puerto Rico (TARS) field-trial phenotypes used for overlap-based cross-environment validation analyses.  
**Primary source:** Baek et al. (2025), *BMC Plant Biology*  
**DOI:** https://doi.org/10.1186/s12870-025-07128-y

## Optional contextual input

### 3. Climate/context table for the PR/TARS trial
**Preferred local filename:** `pr_tars_climate_summary.csv`  
**Role:** optional descriptive climate summary for the Puerto Rico trial.  
**Use in this repository:** contextual only; not required to reproduce barcode design, benchmarking, or validation analyses.  
**Primary source:** NOAA Global Historical Climatology Network-Daily (GHCN-Daily)  
**Source URL:** https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily

## Notes

- Local alignment or harmonization spreadsheets used during manuscript preparation are not required for public use.
- Supplementary Data S1 distributed with the manuscript contains curated reporting outputs, not all raw upstream inputs.
