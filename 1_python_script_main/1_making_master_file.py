# 1_making_master_file.py

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Path setup
# ============================================================
BASE_DIR = Path(".").resolve()
RAW_DIR = BASE_DIR / "data_raw"
MASTER_DIR = BASE_DIR / "master"
MASTER_DIR.mkdir(parents=True, exist_ok=True)

print(f"[INFO] BASE_DIR   : {BASE_DIR}")
print(f"[INFO] RAW_DIR    : {RAW_DIR}")
print(f"[INFO] MASTER_DIR : {MASTER_DIR}")

ICGT_FILE = RAW_DIR / "Bekeley-structure analysis.xlsx"
PR_SUPP_FILE = RAW_DIR / "Supplementary Data 1_R1 ver.xlsx"
GRIN_FILE = RAW_DIR / "Methods GRIN-Global.xlsx"
TEMP_FILE = RAW_DIR / "Temperature data-Brain's PR work.csv"

ALIGN_FILES = [
    RAW_DIR / "Bekeley vs bekeley study structure.xlsx",
    RAW_DIR / "first paper align.xlsx",
    RAW_DIR / "Second paper alignment-criollo.xlsx",
    RAW_DIR / "PR-Osorio-Guarín et al Correlation.xlsx",
]

required = [ICGT_FILE, PR_SUPP_FILE, TEMP_FILE]
missing = [p for p in required if not p.exists()]
if missing:
    raise FileNotFoundError(
        "Missing required files (check under data_raw):\n"
        + "\n".join(f" - {m.name}" for m in missing)
    )


# ============================================================
# Utilities
# ============================================================
def add_clone_core_label(df: pd.DataFrame, acc_col: str = "Accession_ID") -> pd.DataFrame:
    def split_clone(v):
        if pd.isna(v):
            return pd.Series({"Clone_core": np.nan, "Clone_label": np.nan})
        s = str(v).strip()
        m = re.match(r"(.+?)\s+([A-Z])$", s)
        if m:
            return pd.Series({"Clone_core": m.group(1).strip(), "Clone_label": m.group(2)})
        else:
            return pd.Series({"Clone_core": s, "Clone_label": ""})

    extra = df[acc_col].apply(split_clone)
    return pd.concat([df, extra], axis=1)


def normalize_name(s):
    if pd.isna(s):
        return np.nan
    x = str(s).upper()
    x = re.sub(r"[^A-Z0-9]", "", x)
    return x


def find_column(df: pd.DataFrame, patterns, default=None):
    cols = list(df.columns)
    lower = [str(c).lower() for c in cols]
    for pat in patterns:
        p = pat.lower()
        for i, c in enumerate(lower):
            if p in c:
                return cols[i]
    return default


# ============================================================
# ICGT loader
# ============================================================
def load_icgt_bekele(path: Path):
    print(f"[LOAD] ICGT: {path.name}")
    df0 = pd.read_excel(path, sheet_name=0)
    print(f"[INFO] raw shape: {df0.shape}")

    first_col = df0.columns[0]
    print(f"[INFO] Using first column as Accession_ID: {first_col}")
    df0 = df0.rename(columns={first_col: "Accession_ID"})

    extra_acc_cols = [c for c in df0.columns[1:] if "accession" in str(c).lower()]
    if extra_acc_cols:
        print(f"[INFO] Dropping duplicate ACCESSION-like columns: {extra_acc_cols}")
        df0 = df0.drop(columns=extra_acc_cols)

    geno_cols = []
    for c in df0.columns:
        name = str(c)
        if re.match(r"^\d+_\d+(\.\d+)?$", name.strip()):
            geno_cols.append(c)
        elif name.upper().startswith("TCSNP"):
            geno_cols.append(c)
    geno_cols = list(dict.fromkeys(geno_cols))

    icgt_all = df0.copy()
    icgt_all["Accession_ID"] = icgt_all["Accession_ID"].astype(str)
    icgt_all = add_clone_core_label(icgt_all, "Accession_ID")
    icgt_all["Dataset"] = "ICGT"

    meta_cols = ["Accession_ID", "Clone_core", "Clone_label", "Dataset"]
    pheno_cols = [c for c in icgt_all.columns if c not in geno_cols and c not in meta_cols]

    icgt_pheno = icgt_all[meta_cols + pheno_cols].copy()
    icgt_geno = icgt_all[["Accession_ID", "Clone_core", "Clone_label"] + geno_cols].copy()

    print(f"[INFO] icgt_all   : {icgt_all.shape}")
    print(f"[INFO] icgt_pheno : {icgt_pheno.shape}")
    print(f"[INFO] icgt_geno  : {icgt_geno.shape}")
    print(f"[INFO] SNP cols   : {len(geno_cols)}")

    return icgt_all, icgt_pheno, icgt_geno


icgt_all, icgt_pheno, icgt_geno = load_icgt_bekele(ICGT_FILE)


# ============================================================
# PR phenotype loader (kept as-is; patched later exactly like your cell2)
# ============================================================
def load_pr_pheno(path: Path) -> pd.DataFrame:
    print(f"[LOAD] PR phenotypes: {path.name}")
    xls = pd.ExcelFile(path)

    target_sheet = None
    best_score = -1
    keys = ["yield", "total", "pod", "pod index", "infection", "disease", "seed", "bean"]

    for sheet in xls.sheet_names:
        df0 = xls.parse(sheet)
        if df0.empty:
            continue

        acc_col = find_column(df0, ["accession", "clone", "entry", "tree", "id", "name"], default=None)
        if acc_col is None:
            continue

        cols_lower = [str(c).lower() for c in df0.columns]
        score = 0
        for key in keys:
            if any(key in c for c in cols_lower):
                score += 1

        sname = sheet.lower()
        for k2 in ["trait", "field", "tars", "phenotype", "2007", "2008", "2009", "2010", "2011"]:
            if k2 in sname:
                score += 1

        if score > best_score:
            best_score = score
            target_sheet = sheet

    if target_sheet is None:
        raise ValueError("Could not find a PR sheet with both an ID-like column and trait keywords.")

    print(f"[INFO] Selected PR sheet: {target_sheet} (score={best_score})")
    df0 = xls.parse(target_sheet)

    acc_col = find_column(df0, ["accession", "clone", "entry", "tree", "id", "name"], default=None)
    if acc_col is None:
        raise ValueError("Selected PR sheet has no identifier-like column.")

    df = df0.rename(columns={acc_col: "Accession_ID"}).copy()
    df["Accession_ID"] = df["Accession_ID"].astype(str)
    df = add_clone_core_label(df, "Accession_ID")
    df["Dataset"] = "PR_TARS"
    return df


pr_pheno = load_pr_pheno(PR_SUPP_FILE)
print(f"[INFO] pr_pheno: {pr_pheno.shape}")


# ============================================================
# Climate yearly summary
# ============================================================
def load_climate_summary(path: Path) -> pd.DataFrame:
    print(f"[LOAD] Climate: {path.name}")
    df0 = pd.read_csv(path)
    if "DATE" not in df0.columns:
        raise ValueError("Climate file missing DATE column.")
    df0["DATE"] = pd.to_datetime(df0["DATE"])
    df0["year"] = df0["DATE"].dt.year

    agg_dict = {}
    for col in ["TAVG", "TMAX", "TMIN"]:
        if col in df0.columns:
            agg_dict[col] = "mean"
    if "PRCP" in df0.columns:
        agg_dict["PRCP"] = "sum"

    if not agg_dict:
        raise ValueError("Climate file missing TAVG/TMAX/TMIN/PRCP columns.")

    return df0.groupby("year").agg(agg_dict).reset_index()


clim_year = load_climate_summary(TEMP_FILE)
print(f"[INFO] climate yearly: {clim_year.shape}")


# ============================================================
# Save masters (cell1)
# ============================================================
icgt_all.to_csv(MASTER_DIR / "icgt_all_master.csv", index=False)
icgt_pheno.to_csv(MASTER_DIR / "icgt_pheno_master.csv", index=False)
icgt_geno.to_csv(MASTER_DIR / "icgt_geno_master.csv", index=False)
icgt_pheno.to_csv(MASTER_DIR / "icgt_traits_only.csv", index=False)
pr_pheno.to_csv(MASTER_DIR / "pr_traits_master.csv", index=False)
clim_year.to_csv(MASTER_DIR / "climate_PR_TARS_yearly.csv", index=False)

print("[SAVE] icgt_all_master.csv")
print("[SAVE] icgt_pheno_master.csv")
print("[SAVE] icgt_geno_master.csv")
print("[SAVE] icgt_traits_only.csv")
print("[SAVE] pr_traits_master.csv")
print("[SAVE] climate_PR_TARS_yearly.csv")


# ============================================================
# Initial combined/overlap (cell1; will be overwritten by patch)
# ============================================================
combined = pd.concat([icgt_pheno, pr_pheno], ignore_index=True, sort=False)
combined["Norm_ID"] = combined["Clone_core"].apply(normalize_name)
combined.to_csv(MASTER_DIR / "combined_traits_master.csv", index=False)
print("[SAVE] combined_traits_master.csv")

icgt_ids = icgt_pheno[["Accession_ID", "Clone_core"]].copy()
icgt_ids["Norm_ID"] = icgt_ids["Clone_core"].apply(normalize_name)

pr_ids = pr_pheno[["Accession_ID", "Clone_core"]].copy()
pr_ids["Norm_ID"] = pr_ids["Clone_core"].apply(normalize_name)

overlap = pd.merge(icgt_ids, pr_ids, on="Norm_ID", how="inner", suffixes=("_ICGT", "_PR"))
overlap = overlap.drop_duplicates(subset=["Norm_ID"])
overlap.to_csv(MASTER_DIR / "accession_overlap_auto.csv", index=False)
print(f"[SAVE] accession_overlap_auto.csv (auto overlap {len(overlap)})")


if GRIN_FILE.exists():
    try:
        grin = pd.read_excel(GRIN_FILE, sheet_name=0)
        grin.to_csv(MASTER_DIR / "grin_methods_dump.csv", index=False)
        print("[SAVE] grin_methods_dump.csv")
    except Exception as e:
        print(f"[WARN] GRIN methods dump error: {e}")
else:
    print("[INFO] Methods GRIN-Global.xlsx not found (optional)")

for f in ALIGN_FILES:
    if f.exists():
        try:
            xls_align = pd.ExcelFile(f)
            sheet0 = xls_align.sheet_names[0]
            df_align = xls_align.parse(sheet0)
            out_name = f.stem.replace(" ", "_") + "_dump.csv"
            df_align.to_csv(MASTER_DIR / out_name, index=False)
            print(f"[SAVE] {out_name}")
        except Exception as e:
            print(f"[WARN] Alignment dump error ({f.name}): {e}")


print("\n===== PATCH DONE =====")

pr_path = MASTER_DIR / "pr_traits_master.csv"
pr = pd.read_csv(pr_path)
print("[INFO] PR cols:", list(pr.columns))

if "Acession" not in pr.columns:
    raise ValueError("Column 'Acession' not found in pr_traits_master.csv.")

pr["Accession_ID"] = pr["Acession"].astype(str)

for col in ["Clone_core", "Clone_label"]:
    if col in pr.columns:
        pr = pr.drop(columns=[col])

pr = add_clone_core_label(pr, "Accession_ID")

if "Dataset" not in pr.columns:
    pr["Dataset"] = "PR_TARS"

pr.to_csv(pr_path, index=False)
print("[SAVE] pr_traits_master.csv (patched)")

icgt_traits_path = MASTER_DIR / "icgt_traits_only.csv"
if not icgt_traits_path.exists():
    raise FileNotFoundError(f"Missing file: {icgt_traits_path.name}")

icgt = pd.read_csv(icgt_traits_path)

combined2 = pd.concat([icgt, pr], ignore_index=True, sort=False)
combined2["Norm_ID"] = combined2["Clone_core"].apply(normalize_name)
combined2.to_csv(MASTER_DIR / "combined_traits_master.csv", index=False)
print("[SAVE] combined_traits_master.csv (patched)")

icgt_ids2 = icgt[["Accession_ID", "Clone_core"]].copy()
icgt_ids2["Norm_ID"] = icgt_ids2["Clone_core"].apply(normalize_name)

pr_ids2 = pr[["Accession_ID", "Clone_core"]].copy()
pr_ids2["Norm_ID"] = pr_ids2["Clone_core"].apply(normalize_name)

overlap2 = pd.merge(icgt_ids2, pr_ids2, on="Norm_ID", how="inner", suffixes=("_ICGT", "_PR"))
overlap2 = overlap2.drop_duplicates(subset=["Norm_ID"])
overlap2.to_csv(MASTER_DIR / "accession_overlap_auto.csv", index=False)
print(f"[SAVE] accession_overlap_auto.csv (patched overlap {len(overlap2)})")


# ============================================================
# 9) Summary
# ============================================================
print("\n===== MASTER COMPLETE =====")
print(f"ICGT accessions (icgt_pheno): {icgt_pheno['Accession_ID'].nunique()}")
print(f"ICGT accessions (icgt_geno) : {icgt_geno['Accession_ID'].nunique()}")
print(f"PR accessions (patched)     : {pr['Accession_ID'].nunique()}")
print(f"Final overlap (ICGT<->PR)     : {len(overlap2)}")
print(f"MASTER_DIR                  : {MASTER_DIR}")
