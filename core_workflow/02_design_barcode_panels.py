"""
Script 2: Barcode-panel design, QC, and Supplementary workbook regeneration.

This script reproduces the core table-based analysis outputs used for the manuscript.

It writes master-level CSV summaries and regenerates a manuscript-associated
supplementary workbook. Figure generation is handled separately in the
analysis suite when applicable.

Expected inputs (relative to BASE_DIR):
  master/icgt_geno_master.csv
  master/icgt_pheno_master.csv
  master/accession_overlap_auto.csv
  master/pr_traits_master.csv
  data_raw/pr_validation_traits.xlsx (preferred; auto-detected fallback supported)

Key outputs:
  master/barcode_panel_32_stats.csv
  master/panel_tradeoff_distance.csv
  master/panel_tradeoff_error.csv
  master/panel_pure_vs_trait_enriched.csv
  master/mislabel_within_core_candidates.csv
  master/synonym_candidates.csv
  master/synonym_clusters_by_core.csv
  master/overlap_MDS_PRtraits.csv
  master/overlap_MDS_PRtraits_MULTI.csv
  master/panel_tradeoff_distance_overlap.csv
  master/code_theoretic_summary.csv
  master/code_theoretic_summary_eff.csv
  master/panel_shannon_entropy.csv
  Supplementary Data 1.xlsx
"""
from __future__ import annotations

import math
import random
import re
from pathlib import Path
from sklearn.metrics import mutual_info_score

import numpy as np
import pandas as pd
from sklearn.manifold import MDS
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")

# ------------------------------
# Paths
# ------------------------------
BASE_DIR = Path(".").resolve()
MASTER_DIR = BASE_DIR / "master"
RAW_DIR = BASE_DIR / "data_raw"
MASTER_DIR.mkdir(parents=True, exist_ok=True)
RUN_BIT_BUDGET = False  # set True to compute MI/entropy bit budget (optional)

OUTPUT_FILE = BASE_DIR / "Supplementary Data 1.xlsx"
def resolve_optional_validation_workbook(raw_dir: Path) -> Path:
    preferred = raw_dir / "pr_validation_traits.xlsx"
    if preferred.exists():
        return preferred
    for p in sorted(raw_dir.glob("*.xlsx")):
        key = re.sub(r"[^a-z0-9]+", "", p.name.lower())
        if any(token in key for token in ["prvalidation", "tars", "supplementarydata", "validation"]):
            return p
    return preferred

PR_RAW_FILE = resolve_optional_validation_workbook(RAW_DIR)

print("[INFO] BASE_DIR   :", BASE_DIR)
print("[INFO] MASTER_DIR :", MASTER_DIR)
print("[INFO] RAW_DIR    :", RAW_DIR)

# ------------------------------
# Load master tables
# ------------------------------
icgt_geno = pd.read_csv(MASTER_DIR / "icgt_geno_master.csv")
icgt_pheno = pd.read_csv(MASTER_DIR / "icgt_pheno_master.csv")
overlap = pd.read_csv(MASTER_DIR / "accession_overlap_auto.csv")

print("[INFO] icgt_geno shape :", icgt_geno.shape)
print("[INFO] icgt_pheno shape:", icgt_pheno.shape)
print("[INFO] ICGT–PR overlap :", overlap.shape)

# Keep only accessions shared by genotype + phenotype tables (safety)
ids = sorted(set(icgt_geno["Accession_ID"]) & set(icgt_pheno["Accession_ID"]))
icgt_geno = icgt_geno[icgt_geno["Accession_ID"].isin(ids)].reset_index(drop=True)
icgt_pheno = icgt_pheno[icgt_pheno["Accession_ID"].isin(ids)].reset_index(drop=True)
print("[INFO] Common ICGT accessions:", len(ids))

# ------------------------------
# SNP column list and genotype matrix preprocessing
# ------------------------------
META_COLS = ["Accession_ID", "Clone_core", "Clone_label"]

snp_cols = [c for c in icgt_geno.columns if c not in META_COLS]

print("[INFO] Number of SNPs:", len(snp_cols))

geno_mat = icgt_geno[snp_cols].astype(str).copy()

geno_mat = geno_mat.replace({" ": np.nan, "nan": np.nan})
geno_mat = geno_mat.fillna("N")
geno_mat = geno_mat.replace({"M": "N"})

G = geno_mat.values
accessions = icgt_geno["Accession_ID"].tolist()
print("[INFO] G shape:", G.shape)

# ------------------------------
# Per-marker heterozygosity and missing rate
# ------------------------------
def snp_heterozygosity(col_values):
    
    s = pd.Series(col_values)
    s = s[s != "N"]
    if s.empty:
        return np.nan, 1.0

    missing_rate = 1.0 - len(s) / len(col_values)
    freqs = s.value_counts(normalize=True)
    H = 1.0 - np.sum(freqs.values ** 2)
    return H, missing_rate

hetero = []
for c in snp_cols:
    H, miss = snp_heterozygosity(G[:, snp_cols.index(c)])
    hetero.append((c, H, miss))

hetero_df = pd.DataFrame(hetero, columns=["marker", "H", "missing_rate"])
hetero_df = hetero_df.sort_values("H", ascending=False)
print(hetero_df.head())

# ------------------------------
# Binary encoding and pairwise SNP correlation helper
# ------------------------------
def encode_snp_binary(col_values):
    
    s = pd.Series(col_values)
    s = s.replace({"N": np.nan})
    counts = s.value_counts()
    if counts.empty:
        return None, None
    major = counts.index[0]
    b = (s != major).astype(float)
    return b, major

def snp_corr(b1, b2):
    mask = (~b1.isna()) & (~b2.isna())
    if mask.sum() < 10:
        return 0.0
    r = np.corrcoef(b1[mask], b2[mask])[0, 1]
    return r

# ------------------------------
# Barcode-panel selection (Pure-ID panel)
# ------------------------------
def select_barcode_markers(
    geno_df,
    snp_cols,
    max_markers=32,
    corr_thresh=0.8,
    max_missing=0.2
):
    
    stats = []
    G_local = geno_df[snp_cols].astype(str).replace({" ": np.nan, "nan": np.nan}).fillna("N").replace({"M":"N"}).values
    for j, c in enumerate(snp_cols):
        H, miss = snp_heterozygosity(G_local[:, j])
        stats.append((c, H, miss))
    df_stats = pd.DataFrame(stats, columns=["marker", "H", "missing_rate"])
    df_stats = df_stats[df_stats["missing_rate"] <= max_missing]
    df_stats = df_stats.sort_values("H", ascending=False).reset_index(drop=True)

    bin_enc = {}
    major_allele = {}
    for c in df_stats["marker"]:
        b, maj = encode_snp_binary(geno_df[c].values)
        if b is not None:
            bin_enc[c] = b
            major_allele[c] = maj

    selected = []
    for c in df_stats["marker"]:
        if c not in bin_enc:
            continue
        if not selected:
            selected.append(c)
        else:
            ok = True
            for s in selected:
                r = snp_corr(bin_enc[c], bin_enc[s])
                if abs(r) > corr_thresh:
                    ok = False
                    break
            if ok:
                selected.append(c)
        if len(selected) >= max_markers:
            break

    sel_df = df_stats[df_stats["marker"].isin(selected)].copy()
    return selected, sel_df

best_markers_32, best_stats_32 = select_barcode_markers(
    icgt_geno,
    snp_cols,
    max_markers=32,
    corr_thresh=0.8,
    max_missing=0.2
)

print("[INFO] Selected barcoding SNPs (n=32):")
print(best_markers_32)
best_stats_32.to_csv(MASTER_DIR / "barcode_panel_32_stats.csv", index=False)

# =============================================================================
# OPTIONAL: Bit budget analysis (Mutual Information + Panel entropy)
# - Default OFF (RUN_BIT_BUDGET=False) to keep public outputs identical
# - When ON, writes: master/bit_budget_MI_summary.csv
# =============================================================================
if RUN_BIT_BUDGET:
    print("[INFO] Running OPTIONAL bit budget analysis (MI + entropy)...")

    ID_COL = "Accession_ID"
    GROUP_COL = "Acc Group"
    YIELD_COL = "FR_WABW"     # Wet Bean Mass (yield proxy)
    POD_COL = "PodIndex"      # lower is better

    df_analysis = icgt_geno[[ID_COL] + best_markers_32].copy()
    df_pheno_sub = icgt_pheno[[ID_COL, GROUP_COL, POD_COL, YIELD_COL]].copy()
    df_pheno_sub = df_pheno_sub.dropna()

    df_mi = pd.merge(df_analysis, df_pheno_sub, on=ID_COL, how="inner")

    for m in best_markers_32:
        df_mi[m] = df_mi[m].astype(str).factorize()[0]

    df_mi["Group_Class"] = df_mi[GROUP_COL].astype(str)
    df_mi["Yield_Class"] = pd.qcut(df_mi[YIELD_COL], q=3, labels=["Low", "Med", "High"])
    df_mi["Pod_Class"] = pd.qcut(df_mi[POD_COL], q=3, labels=["HighEff", "Med", "LowEff"])

    mi_ancestry = 0.0
    mi_yield = 0.0
    mi_pod = 0.0

    for m in best_markers_32:
        mi_ancestry += mutual_info_score(df_mi[m], df_mi["Group_Class"]) / np.log(2)
        mi_yield    += mutual_info_score(df_mi[m], df_mi["Yield_Class"]) / np.log(2)
        mi_pod      += mutual_info_score(df_mi[m], df_mi["Pod_Class"]) / np.log(2)

    total_marker_entropy = 0.0
    for m in best_markers_32:
        counts = df_mi[m].value_counts(normalize=True)
        h = -np.sum(counts * np.log2(counts))
        total_marker_entropy += float(h)

    remaining = total_marker_entropy - (mi_ancestry + mi_yield + mi_pod)
    if remaining < 0:
        remaining = 0.0

    out_bit = MASTER_DIR / "bit_budget_MI_summary.csv"
    pd.DataFrame([{
        "n_samples_used": int(df_mi.shape[0]),
        "total_panel_entropy_bits": float(total_marker_entropy),
        "MI_ancestry_bits": float(mi_ancestry),
        "MI_yield_bits": float(mi_yield),
        "MI_podindex_bits": float(mi_pod),
        "residual_unique_id_bits": float(remaining),
        "markers_n": int(len(best_markers_32)),
        "yield_col": YIELD_COL,
        "podindex_col": POD_COL,
        "group_col": GROUP_COL,
    }]).to_csv(out_bit, index=False)

    print(f"[SAVE] {out_bit}")
    print(f"[RESULT] Total Panel Entropy: {total_marker_entropy:.2f} bits")
    print(f"[RESULT] Ancestry Structure: {mi_ancestry:.2f} bits")
    print(f"[RESULT] Yield Potential: {mi_yield:.2f} bits")
    print(f"[RESULT] Pod Efficiency: {mi_pod:.2f} bits")
    print(f"[RESULT] Unique ID Capacity (residual): {remaining:.2f} bits")

# ------------------------------
# Hamming distance utilities
# ------------------------------
def hamming_matrix(geno_df, marker_list):
    
    G_sub = geno_df[marker_list].astype(str).replace({" ": np.nan, "nan": np.nan}).fillna("N").replace({"M":"N"}).values
    n, m = G_sub.shape
    D = np.zeros((n, n), dtype=int)

    for i in range(n):
        gi = G_sub[i]
        for j in range(i+1, n):
            gj = G_sub[j]
            mask = (gi != "N") & (gj != "N")
            if mask.sum() == 0:
                d = 0
            else:
                d = np.sum(gi[mask] != gj[mask])
            D[i, j] = d
            D[j, i] = d
    return D

def pairwise_stats(D):
    n = D.shape[0]
    dists = []
    for i in range(n):
        for j in range(i+1, n):
            dists.append(D[i, j])
    dists = np.array(dists)
    return {
        "d_min": int(dists.min()),
        "d_mean": float(dists.mean()),
        "d_median": float(np.median(dists)),
        "d_max": int(dists.max()),
        "n_pairs": len(dists),
        "dists": dists,
    }

# ------------------------------
# Distance statistics: Full vs Best-32
# ------------------------------
D_full = hamming_matrix(icgt_geno, snp_cols)
stats_full = pairwise_stats(D_full)
print("[FULL] d_min, d_mean, d_max:", stats_full["d_min"], stats_full["d_mean"], stats_full["d_max"])

D_best32 = hamming_matrix(icgt_geno, best_markers_32)
stats_best32 = pairwise_stats(D_best32)
print("[BEST 32] d_min, d_mean, d_max:", stats_best32["d_min"], stats_best32["d_mean"], stats_best32["d_max"])

# ------------------------------
# MDS embedding in barcode space (computed once; used later for PR overlap merges)
# ------------------------------
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
coords = mds.fit_transform(D_best32)

# ------------------------------
# Mis-identification simulation helper
# ------------------------------
def simulate_misid(geno_df, marker_list, p_error=0.02, n_trials=1000, seed=123):
    random.seed(seed)
    np.random.seed(seed)

    G_sub = geno_df[marker_list].astype(str).replace({" ": np.nan, "nan": np.nan}).fillna("N").replace({"M":"N"}).values
    n, m = G_sub.shape

    def random_flip(geno_row):
        g = np.array(geno_row, dtype=object)
        mask = np.random.rand(m) < p_error
        g[mask] = "N"
        return g

    mis = 0
    for t in range(n_trials):
        i = random.randrange(n)
        true_row = G_sub[i]
        noisy = random_flip(true_row)

        best_j = None
        best_d = None
        for j in range(n):
            gj = G_sub[j]
            mask = (noisy != "N") & (gj != "N")
            if mask.sum() == 0:
                d = m
            else:
                d = np.sum(noisy[mask] != gj[mask])
            if (best_d is None) or (d < best_d):
                best_d = d
                best_j = j

        if best_j != i:
            mis += 1

    return mis / n_trials

# ------------------------------
# Tradeoff curves across panel sizes and error rates
# ------------------------------
k_list = [8, 16, 24, 32, 48, 64]
p_values = [0.0, 0.01, 0.02, 0.05]

panel_rows = []
error_rows = []

for k in k_list:
    print(f"\n[INFO] k={k} selecting panel...")
    markers_k, stats_k = select_barcode_markers(
        icgt_geno,
        snp_cols,
        max_markers=k,
        corr_thresh=0.8,
        max_missing=0.2
    )

    D_k = hamming_matrix(icgt_geno, markers_k)
    dist_stats = pairwise_stats(D_k)

    panel_rows.append({
        "k": k,
        "n_markers": len(markers_k),
        "d_min": dist_stats["d_min"],
        "d_mean": dist_stats["d_mean"],
        "d_median": dist_stats["d_median"],
        "d_max": dist_stats["d_max"],
        "n_pairs": dist_stats["n_pairs"]
    })

    for p in p_values:
        mis = simulate_misid(icgt_geno, markers_k, p_error=p, n_trials=300)
        error_rows.append({
            "k": k,
            "p_error": p,
            "mis_id": mis
        })
        print(f"  p={p:.2f}  mis-ID={mis:.4f}")

panel_df = pd.DataFrame(panel_rows)
error_df = pd.DataFrame(error_rows)

panel_df.to_csv(MASTER_DIR / "panel_tradeoff_distance.csv", index=False)
error_df.to_csv(MASTER_DIR / "panel_tradeoff_error.csv", index=False)

panel_df

# ------------------------------
# Trait correlation (single trait) and trait-enriched panel
# ------------------------------
TARGET_TRAIT = "PodIndex"

y = icgt_pheno[TARGET_TRAIT].values.astype(float)

trait_stats = []
for c in snp_cols:
    s = icgt_geno[c].astype(str).replace({" ": np.nan, "nan": np.nan, "M": np.nan, "N": np.nan})
    levels = s.dropna().unique()
    if len(levels) < 2:
        r = 0.0
    else:
        mapping = {allele: i for i, allele in enumerate(levels)}
        x = s.map(mapping).astype(float)
        mask = (~x.isna()) & (~pd.isna(y))
        if mask.sum() < 30:
            r = 0.0
        else:
            r = np.corrcoef(x[mask], y[mask])[0,1]
    trait_stats.append((c, abs(r)))

trait_df = pd.DataFrame(trait_stats, columns=["marker", "abs_r_trait"])
trait_df.head()
def select_trait_enriched_panel(
    geno_df,
    snp_cols,
    trait_df,
    max_markers=32,
    corr_thresh=0.8,
    max_missing=0.2,
    alpha=1.0,
    beta=1.0
):
    stats = []
    G_local = geno_df[snp_cols].astype(str).replace({" ": np.nan, "nan": np.nan}).fillna("N").replace({"M":"N"}).values
    for j, c in enumerate(snp_cols):
        H, miss = snp_heterozygosity(G_local[:, j])
        stats.append((c, H, miss))
    df_stats = pd.DataFrame(stats, columns=["marker", "H", "missing_rate"])

    df = df_stats.merge(trait_df, on="marker", how="left")
    df["abs_r_trait"] = df["abs_r_trait"].fillna(0.0)

    df = df[df["missing_rate"] <= max_missing].copy()
    df["score"] = alpha * df["H"] + beta * df["abs_r_trait"]

    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    bin_enc = {}
    for c in df["marker"]:
        b, _ = encode_snp_binary(geno_df[c].values)
        if b is not None:
            bin_enc[c] = b

    selected = []
    for c in df["marker"]:
        if c not in bin_enc:
            continue
        if not selected:
            selected.append(c)
        else:
            ok = True
            for s in selected:
                r = snp_corr(bin_enc[c], bin_enc[s])
                if abs(r) > corr_thresh:
                    ok = False
                    break
            if ok:
                selected.append(c)
        if len(selected) >= max_markers:
            break

    sel_df = df[df["marker"].isin(selected)].copy()
    return selected, sel_df
trait_markers_32, trait_panel_stats = select_trait_enriched_panel(
    icgt_geno,
    snp_cols,
    trait_df,
    max_markers=32,
    corr_thresh=0.8,
    max_missing=0.2,
    alpha=1.0,
    beta=1.0
)

D_pure = hamming_matrix(icgt_geno, best_markers_32)
stats_pure = pairwise_stats(D_pure)

D_trait = hamming_matrix(icgt_geno, trait_markers_32)
stats_trait = pairwise_stats(D_trait)

print("[PURE 32] d_min, d_mean:", stats_pure["d_min"], stats_pure["d_mean"])
print("[TRAIT 32] d_min, d_mean:", stats_trait["d_min"], stats_trait["d_mean"])

pure_corr = trait_df[trait_df["marker"].isin(best_markers_32)]["abs_r_trait"].mean()
trait_corr = trait_df[trait_df["marker"].isin(trait_markers_32)]["abs_r_trait"].mean()
print("[PURE 32] mean |r_trait|:", pure_corr)
print("[TRAIT 32] mean |r_trait|:", trait_corr)

comp = pd.DataFrame({
    "panel": ["pure_ID_32", "trait_enriched_32"],
    "n_markers": [32, 32],
    "d_min": [stats_pure["d_min"], stats_trait["d_min"]],
    "d_mean": [stats_pure["d_mean"], stats_trait["d_mean"]],
    "mean_abs_r_trait": [pure_corr, trait_corr]
})
comp.to_csv(MASTER_DIR / "panel_pure_vs_trait_enriched.csv", index=False)
comp

# ------------------------------
# QC: within-core distances and synonym candidates
# ------------------------------
def within_core_distances(geno_df, snp_cols):
    results = []
    G = geno_df[snp_cols].astype(str).replace({" ": np.nan, "nan": np.nan}).fillna("N").replace({"M":"N"}).values
    ids = geno_df["Accession_ID"].tolist()
    cores = geno_df["Clone_core"].tolist()

    core_to_indices = {}
    for i, core in enumerate(cores):
        core_to_indices.setdefault(core, []).append(i)

    for core, idxs in core_to_indices.items():
        if len(idxs) < 2:
            continue
        sub = G[idxs]
        n_sub = len(idxs)
        dists = []
        for i in range(n_sub):
            for j in range(i+1, n_sub):
                gi, gj = sub[i], sub[j]
                mask = (gi != "N") & (gj != "N")
                if mask.sum() == 0:
                    continue
                d = np.sum(gi[mask] != gj[mask])
                dists.append(d)
        if dists:
            results.append({
                "Clone_core": core,
                "n_labels": n_sub,
                "d_min_within": int(np.min(dists)),
                "d_max_within": int(np.max(dists)),
                "d_mean_within": float(np.mean(dists))
            })
    return pd.DataFrame(results)

within_df = within_core_distances(icgt_geno, snp_cols)
within_df.sort_values("d_max_within", ascending=False).head()

within_df.to_csv(MASTER_DIR / "mislabel_within_core_candidates.csv", index=False)
def synonym_candidates(geno_df, snp_cols, max_distance=1):
    G = geno_df[snp_cols].astype(str).replace({" ": np.nan, "nan": np.nan}).fillna("N").replace({"M":"N"}).values
    ids = geno_df["Accession_ID"].tolist()
    cores = geno_df["Clone_core"].tolist()
    n = len(ids)
    rows = []
    for i in range(n):
        for j in range(i+1, n):
            if cores[i] == cores[j]:
                continue
            gi, gj = G[i], G[j]
            mask = (gi != "N") & (gj != "N")
            if mask.sum() == 0:
                continue
            d = np.sum(gi[mask] != gj[mask])
            if d <= max_distance:
                rows.append({
                    "Accession_i": ids[i],
                    "Accession_j": ids[j],
                    "Clone_core_i": cores[i],
                    "Clone_core_j": cores[j],
                    "Hamming_distance": int(d)
                })
    return pd.DataFrame(rows)

syn_df = synonym_candidates(icgt_geno, snp_cols, max_distance=1)
syn_df.to_csv(MASTER_DIR / "synonym_candidates.csv", index=False)
syn_df.head()

# ------------------------------
# PR overlap merges using MDS coordinates
# ------------------------------
coords_df = pd.DataFrame(coords, columns=["MDS1", "MDS2"])
coords_df["Accession_ID"] = icgt_pheno["Accession_ID"].values

overlap = pd.read_csv(MASTER_DIR / "accession_overlap_auto.csv")
icgt_overlap_ids = overlap["Accession_ID_ICGT"].tolist()
pr_overlap_ids   = overlap["Accession_ID_PR"].tolist()

icgt_overlap_coords = coords_df[coords_df["Accession_ID"].isin(icgt_overlap_ids)].copy()

pr = pd.read_csv(MASTER_DIR / "pr_traits_master.csv")
pr_overlap = pr[pr["Acession"].isin(pr_overlap_ids)].copy()

print("[INFO] ICGT overlap coords:", icgt_overlap_coords.shape)
print("[INFO] PR overlap traits:", pr_overlap.shape)
map_icgt_to_pr = dict(zip(overlap["Accession_ID_ICGT"], overlap["Accession_ID_PR"]))

merged_rows = []
for _, row in icgt_overlap_coords.iterrows():
    acc_icgt = row["Accession_ID"]
    acc_pr = map_icgt_to_pr.get(acc_icgt, None)
    if acc_pr is None:
        continue
    pr_row = pr_overlap[pr_overlap["Acession"] == acc_pr]
    if pr_row.empty:
        continue
    merged_rows.append({
        "Accession_ICGT": acc_icgt,
        "Accession_PR": acc_pr,
        "MDS1": row["MDS1"],
        "MDS2": row["MDS2"],
        "YIELD": float(pr_row["YIELD"].values[0])
    })

merged = pd.DataFrame(merged_rows)
merged.to_csv(MASTER_DIR / "overlap_MDS_PRtraits.csv", index=False)
merged.head()

# ------------------------------
# Code-theoretic summaries
# ------------------------------
def code_summary_for_panel(markers, D=None):
    if D is None:
        D = hamming_matrix(icgt_geno, markers)
    st = pairwise_stats(D)
    n_clones = D.shape[0]
    k = len(markers)
    R = math.log2(n_clones) / k
    delta = st["d_min"] / k
    return {
        "n_clones": n_clones,
        "k": k,
        "d_min": st["d_min"],
        "d_mean": st["d_mean"],
        "R": R,
        "delta": delta
    }

summary_rows = []

full_summary = code_summary_for_panel(snp_cols, D_full)
full_summary["panel"] = "Full_536"
summary_rows.append(full_summary)

best32_summary = code_summary_for_panel(best_markers_32, D_best32)
best32_summary["panel"] = "Best_32"
summary_rows.append(best32_summary)

if 'trait_markers_32' in globals():
    D_trait32 = hamming_matrix(icgt_geno, trait_markers_32)
    trait32_summary = code_summary_for_panel(trait_markers_32, D_trait32)
    trait32_summary["panel"] = "TraitEnriched_32"
    summary_rows.append(trait32_summary)

for k in [8, 16, 48, 64]:
    pass

code_df = pd.DataFrame(summary_rows)
code_df.to_csv(MASTER_DIR / "code_theoretic_summary.csv", index=False)
code_df

# ------------------------------
# Multi-trait scan (trait-enriched 32 panel per phenotype column)
# ------------------------------
numeric_cols = [
    c for c in icgt_pheno.columns
    if np.issubdtype(icgt_pheno[c].dtype, np.number)
]

META_COLS = ["Accession_ID", "Clone_core", "Clone_label", "Dataset", "Acc Group"]

def looks_like_snp(name: str) -> bool:
    name = str(name)
    if re.match(r"^\d+[_ ]\d", name):
        return True
    return False

trait_cols = [
    c for c in numeric_cols
    if (c not in META_COLS) and (not looks_like_snp(c))
]

print("[INFO] Number of detected phenotype columns:", len(trait_cols))
print(trait_cols)
multi_rows = []

for trait in trait_cols:
    print(f"\n[INFO] Trait-enriched panel for {trait}")

    y = icgt_pheno[trait].values.astype(float)

    trait_stats = []
    for c in snp_cols:
        s = icgt_geno[c].astype(str).replace(
            {" ": np.nan, "nan": np.nan, "M": np.nan, "N": np.nan}
        )
        levels = s.dropna().unique()
        if len(levels) < 2:
            r = 0.0
        else:
            mapping = {allele: i for i, allele in enumerate(levels)}
            x = s.map(mapping).astype(float)
            mask = (~x.isna()) & (~pd.isna(y))
            if mask.sum() < 30:
                r = 0.0
            else:
                r = np.corrcoef(x[mask], y[mask])[0, 1]
        trait_stats.append((c, abs(r)))

    trait_df = pd.DataFrame(trait_stats, columns=["marker", "abs_r_trait"])

    D_pure = hamming_matrix(icgt_geno, best_markers_32)
    stats_pure = pairwise_stats(D_pure)
    pure_corr = trait_df[trait_df["marker"].isin(best_markers_32)]["abs_r_trait"].mean()

    trait_markers_32, trait_panel_stats = select_trait_enriched_panel(
        icgt_geno,
        snp_cols,
        trait_df,
        max_markers=32,
        corr_thresh=0.8,
        max_missing=0.2,
        alpha=1.0,
        beta=1.0,
    )
    D_trait = hamming_matrix(icgt_geno, trait_markers_32)
    stats_trait = pairwise_stats(D_trait)
    trait_corr = trait_df[trait_df["marker"].isin(trait_markers_32)]["abs_r_trait"].mean()

    multi_rows.append({
        "trait": trait,
        "panel": "pure_ID_32",
        "d_min": stats_pure["d_min"],
        "d_mean": stats_pure["d_mean"],
        "mean_abs_r_trait": pure_corr,
    })
    multi_rows.append({
        "trait": trait,
        "panel": "trait_enriched_32",
        "d_min": stats_trait["d_min"],
        "d_mean": stats_trait["d_mean"],
        "mean_abs_r_trait": trait_corr,
    })

multi_trait_panel = pd.DataFrame(multi_rows)
out_path = MASTER_DIR / "multi_trait_panel_summary.csv"
multi_trait_panel.to_csv(out_path, index=False)
print("[SAVE]", out_path)
multi_trait_panel

# ------------------------------
# Effective code rate summaries (unique genotype codewords)
# ------------------------------
G_full = icgt_geno[snp_cols].astype(str).replace(
    {" ": np.nan, "nan": np.nan, "M": "N"}
).fillna("N").values

keys_full = ["|".join(row) for row in G_full]
icgt_geno["geno_key_full"] = keys_full

unique_classes = icgt_geno.groupby("geno_key_full")["Accession_ID"].apply(list).reset_index()
unique_classes["class_size"] = unique_classes["Accession_ID"].apply(len)

print("[INFO] Number of unique genotype classes:", len(unique_classes))
print(unique_classes.sort_values("class_size", ascending=False).head(10))
n_unique = len(unique_classes)
R_full_eff = math.log2(n_unique) / 536
print("Effective R (full 536):", R_full_eff)
rows = []

rows.append({
    "panel": "Full_536_eff",
    "n_codewords": n_unique,
    "k": 536,
    "R": math.log2(n_unique)/536
})

G_32 = icgt_geno[best_markers_32].astype(str).replace(
    {" ": np.nan, "nan": np.nan, "M": "N"}
).fillna("N").values
keys_32 = ["|".join(row) for row in G_32]
icgt_geno["geno_key_32"] = keys_32
n_unique_32 = icgt_geno["geno_key_32"].nunique()

rows.append({
    "panel": "Best_32_eff",
    "n_codewords": n_unique_32,
    "k": 32,
    "R": math.log2(n_unique_32)/32
})

code_eff = pd.DataFrame(rows)
code_eff.to_csv(MASTER_DIR / "code_theoretic_summary_eff.csv", index=False)
code_eff

# ------------------------------
# Synonym clustering (union-find)
# ------------------------------
syn = pd.read_csv(MASTER_DIR / "synonym_candidates.csv")

parent = {}
def find(x):
    parent.setdefault(x, x)
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

for _, row in syn.iterrows():
    union(row["Clone_core_i"], row["Clone_core_j"])

clusters = {}
for node in parent.keys():
    r = find(node)
    clusters.setdefault(r, []).append(node)

cluster_rows = []
for root, members in clusters.items():
    cluster_rows.append({
        "root_core": root,
        "size": len(members),
        "members": ", ".join(sorted(members))
    })

cluster_df = pd.DataFrame(cluster_rows).sort_values("size", ascending=False)
out_cluster = MASTER_DIR / "synonym_clusters_by_core.csv"
cluster_df.to_csv(out_cluster, index=False)
print("[SAVE]", out_cluster)
cluster_df.head(10)

# ------------------------------
# PR overlap merge (multiple PR traits)
# ------------------------------
pr = pd.read_csv(MASTER_DIR / "pr_traits_master.csv")
overlap = pd.read_csv(MASTER_DIR / "accession_overlap_auto.csv")

coords_df = pd.DataFrame(coords, columns=["MDS1", "MDS2"])
coords_df["Accession_ID"] = icgt_pheno["Accession_ID"].values

map_icgt_to_pr = dict(zip(overlap["Accession_ID_ICGT"], overlap["Accession_ID_PR"]))

traits_PR = ["YIELD", "TOTAL_PODS", "POD_INDEX", "INFECTED_PODS",
             "DRY_WEIGHT", "FRESH_WEIGHT", "NUMBER_OF_SEEDS",
             "POD_LENGTH", "POD_WEIGHT"]

rows = []
for _, row in coords_df.iterrows():
    acc_icgt = row["Accession_ID"]
    acc_pr = map_icgt_to_pr.get(acc_icgt, None)
    if acc_pr is None:
        continue
    pr_sub = pr[pr["Acession"] == acc_pr]
    if pr_sub.empty:
        continue
    rec = {
        "Accession_ICGT": acc_icgt,
        "Accession_PR": acc_pr,
        "MDS1": row["MDS1"],
        "MDS2": row["MDS2"],
    }
    for t in traits_PR:
        if t in pr_sub.columns:
            rec[t] = float(pr_sub[t].values[0])
    rows.append(rec)

merged_multi = pd.DataFrame(rows)
out_path = MASTER_DIR / "overlap_MDS_PRtraits_MULTI.csv"
merged_multi.to_csv(out_path, index=False)
print("[SAVE]", out_path)
merged_multi.head()

# ------------------------------
# Panel tradeoff computed only on overlap subset
# ------------------------------
overlap = pd.read_csv(MASTER_DIR / "accession_overlap_auto.csv")
icgt_overlap_ids = overlap["Accession_ID_ICGT"].unique().tolist()

geno_sub = icgt_geno[icgt_geno["Accession_ID"].isin(icgt_overlap_ids)].reset_index(drop=True)

rows_overlap = []

for k in [8, 16, 24, 32, 48, 64]:
    markers_k, stats_k = select_barcode_markers(
        geno_sub,
        snp_cols,
        max_markers=k,
        corr_thresh=0.8,
        max_missing=0.2
    )
    D_k = hamming_matrix(geno_sub, markers_k)
    st = pairwise_stats(D_k)
    rows_overlap.append({
        "k": k,
        "d_min": st["d_min"],
        "d_mean": st["d_mean"],
        "d_median": st["d_median"],
        "d_max": st["d_max"],
        "n_pairs": st["n_pairs"]
    })

overlap_tradeoff = pd.DataFrame(rows_overlap)
out_path = MASTER_DIR / "panel_tradeoff_distance_overlap.csv"
overlap_tradeoff.to_csv(out_path, index=False)
print("[SAVE]", out_path)
overlap_tradeoff

# ------------------------------
# Shannon-entropy / information-capacity summary
# ------------------------------
def calculate_shannon_entropy(geno_df, markers):
    
    G = geno_df[markers].astype(str).replace({" ": np.nan, "nan": np.nan, "M": np.nan, "N": np.nan})
    
    total_entropy = 0
    marker_entropies = []
    
    for m in markers:
        counts = G[m].value_counts(normalize=True)
        H = -np.sum(counts * np.log2(counts))
        marker_entropies.append({"marker": m, "entropy_bits": H})
        total_entropy += H
        
    return pd.DataFrame(marker_entropies), total_entropy

try:
    if 'best_markers_32' not in globals():
        stats_path = MASTER_DIR / "barcode_panel_32_stats.csv"
        if stats_path.exists():
            stats_df = pd.read_csv(stats_path)
            best_markers_32 = stats_df["marker"].tolist()
            print(f"[INFO] Loaded {len(best_markers_32)} markers from file.")
        else:
            print("[WARN] best_markers_32 Could not find marker list. Please verify inputs.")
            best_markers_32 = []
except:
    pass

if best_markers_32:
    marker_ent_df, total_bits = calculate_shannon_entropy(icgt_geno, best_markers_32)
    
    print("\n[RESULT] 32-SNP Panel Information Capacity")
    print(f"Total Theoretical Capacity: {total_bits:.2f} bits")
    print(f"Requires bits to ID 419 clones: {math.log2(419):.2f} bits")
    print(f"Requires bits to ID 20,000 global clones: {math.log2(20000):.2f} bits")
    print(f"Redundancy Factor: {total_bits / math.log2(419):.2f}x")
    
    marker_ent_df.to_csv(MASTER_DIR / "panel_shannon_entropy.csv", index=False)

# ------------------------------
# Regenerate Supplementary Data 1.xlsx (no figures)
# ------------------------------
print(f"[INFO] Regenerating Supplementary Data 1 with FORCED S3 Injection...")

try:
    df_panel = pd.read_excel(OUTPUT_FILE, sheet_name='S1_CacaoCipher_32Panel')
    print("[LOAD] S1 Panel: OK (from existing file)")
except:
    df_panel = pd.read_csv(MASTER_DIR / "barcode_panel_32_stats.csv")
    print("[LOAD] S1 Panel: OK (from CSV backup)")

try:
    df_icgt = pd.read_excel(OUTPUT_FILE, sheet_name='S2_ICGT_Phenotypes')
    print("[LOAD] S2 ICGT: OK (from existing file)")
except:
    df_icgt = pd.read_csv(MASTER_DIR / "icgt_pheno_master.csv")
    print("[LOAD] S2 ICGT: OK (from CSV backup)")

print(f"   >>> Attempting to load PR Data from: {PR_RAW_FILE}")
try:
    try:
        df_pr = pd.read_excel(PR_RAW_FILE, sheet_name='TARS Genotypes')
    except:
        df_pr = pd.read_excel(PR_RAW_FILE, sheet_name=0)
        
    print(f"   >>> [SUCCESS] PR Data Loaded! Rows: {len(df_pr)}")
    
except Exception as e:
    print(f"   >>> [ERROR] Failed to load PR data. Please verify the raw file path.")
    print(f"   >>> Error message: {e}")
    df_pr = pd.DataFrame({"Error": ["PR Data file not found. Please check path."]})

try:
    df_metrics = pd.read_excel(OUTPUT_FILE, sheet_name='S4_Bit_Budget_Metrics')
    print("[LOAD] S4 Metrics: OK")
except:
    try:
        df_metrics = pd.read_csv(MASTER_DIR / "panel_tradeoff_distance.csv")
    except:
        df_metrics = pd.DataFrame()

with pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter') as writer:
    
    df_panel.to_excel(writer, sheet_name='S1_CacaoCipher_32Panel', index=False)
    writer.sheets['S1_CacaoCipher_32Panel'].set_column('A:A', 20)
    
    df_icgt.to_excel(writer, sheet_name='S2_ICGT_Phenotypes', index=False)
    
    df_pr.to_excel(writer, sheet_name='S3_PR_Validation_Data', index=False)
    writer.sheets['S3_PR_Validation_Data'].set_column('A:A', 15)
    
    df_metrics.to_excel(writer, sheet_name='S4_Bit_Budget_Metrics', index=False)

    readme_text = [
        ["Sheet Name", "Description"],
        ["S1_CacaoCipher_32Panel", "Details of the optimized 32-SNP barcode panel."],
        ["S2_ICGT_Phenotypes", "Phenotypic data for the ICGT (Trinidad) training population."],
        ["S3_PR_Validation_Data", "Phenotypic data for the TARS (Puerto Rico) validation population."],
        ["S4_Bit_Budget_Metrics", "Summary metrics of genetic distance and trait correlations."]
    ]
    pd.DataFrame(readme_text[1:], columns=readme_text[0]).to_excel(writer, sheet_name='README', index=False)
    writer.sheets['README'].set_column('A:A', 25)
    writer.sheets['README'].set_column('B:B', 80)

print("\n" + "="*60)
print("[FIX COMPLETE] Output file was generated (S3 sheet included).")
print(f"[OPEN] Verify output: {out_path}")
print("="*60)

print("[DONE] Script 2 completed.")
