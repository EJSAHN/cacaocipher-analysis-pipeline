# 03_population_assignment_and_pid.py
#
# Population assignment + Probability of Identity (P(ID)) analyses
# (cleaned for public/GitHub use; calculations preserved from the original notebook cells)
#
# Expected inputs (relative to project root):
#   Bekele/master/icgt_geno_master.csv
#   Bekele/master/icgt_pheno_master.csv
#   Bekele/master/barcode_panel_32_stats.csv
#
# Outputs:
#   Bekele/master/panel_population_assignment_accuracy.csv
#   Bekele/master/panel_PID_summary.csv
#
# Optional supplementary (CSV only by default):
#   Bekele/master/supplementary/S1_full_536_confusion_counts.csv
#   Bekele/master/supplementary/S1_full_536_confusion_norm_by_true.csv
#   Bekele/master/supplementary/S1_pureID_32_confusion_counts.csv
#   Bekele/master/supplementary/S1_pureID_32_confusion_norm_by_true.csv
#
# NOTE:
# - The “demo” cell that constructs an example genotype vector (for p≈0.5) is intentionally not executed here.
#   It was only a sanity-check example and is not part of manuscript outputs.

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder


# -----------------------------
# 0) Paths
# -----------------------------
BASE_DIR = Path(".").resolve()
MASTER_DIR = BASE_DIR / "master"
SUPP_DIR = MASTER_DIR / "supplementary"
SUPP_DIR.mkdir(parents=True, exist_ok=True)

print("[INFO] BASE_DIR   :", BASE_DIR)
print("[INFO] MASTER_DIR :", MASTER_DIR)
print("[INFO] SUPP_DIR   :", SUPP_DIR)

GENO_FP = MASTER_DIR / "icgt_geno_master.csv"
PHENO_FP = MASTER_DIR / "icgt_pheno_master.csv"
PANEL_FP = MASTER_DIR / "barcode_panel_32_stats.csv"

# Turn PNG plotting on/off (GitHub/public defaults to OFF).
MAKE_PLOTS = False


# -----------------------------
# 1) Load
# -----------------------------
icgt_geno = pd.read_csv(GENO_FP)
icgt_pheno = pd.read_csv(PHENO_FP)
panel_stats = pd.read_csv(PANEL_FP)

print("\n[INFO] icgt_geno shape :", icgt_geno.shape)
print("[INFO] icgt_pheno shape:", icgt_pheno.shape)
print("[INFO] panel_stats cols:", list(panel_stats.columns))


# -----------------------------
# 2) Panel definition (preserved)
# -----------------------------
meta_cols = ["Accession_ID", "Clone_core", "Clone_label", "Dataset", "Acc Group"]
snp_cols_full = [c for c in icgt_geno.columns if c not in meta_cols]
print("\n[INFO] Full SNP candidate columns:", len(snp_cols_full))

if "marker" not in panel_stats.columns:
    raise ValueError("barcode_panel_32_stats.csv must contain a 'marker' column.")

best_markers_32 = panel_stats["marker"].astype(str).tolist()
print("[INFO] PureID_32 markers:", len(best_markers_32))

panels = {
    "Full_536": snp_cols_full,
    "PureID_32": best_markers_32,
}

print("[INFO] Panels defined:", {k: len(v) for k, v in panels.items()})


# -----------------------------
# 3) Helper: encode SNP panel (preserved exactly)
# -----------------------------
def encode_panel(geno_df: pd.DataFrame, markers: list[str]) -> np.ndarray:
    """
    0/1 encoding with mean-imputation for missing:
      - most common allele -> 0
      - other allele(s)    -> 1
      - missing (N/M/blank) -> NaN, then replaced by column mean
    """
    G = geno_df[markers].astype(str).replace(
        {" ": np.nan, "nan": np.nan, "NaN": np.nan, "M": np.nan, "N": np.nan}
    )

    X_cols = []
    for m in markers:
        s = G[m]
        alleles = s.dropna().unique()
        if len(alleles) == 0:
            X_cols.append(np.full(len(G), np.nan))
        elif len(alleles) == 1:
            X_cols.append(np.zeros(len(G)))
        else:
            major = alleles[0]
            X_cols.append(np.where(s == major, 0, 1))

    X = np.vstack(X_cols).T  # (n_samples, n_markers)

    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

    return X


# -----------------------------
# 4) Population assignment (Table 2-style; preserved)
# -----------------------------
print("\n===== Population assignment (RandomForest, 5-fold CV) =====")

if "Acc Group" not in icgt_pheno.columns:
    raise ValueError("icgt_pheno_master.csv must contain an 'Acc Group' column.")

meta = icgt_pheno[["Accession_ID", "Acc Group"]].dropna()
meta["Acc Group"] = meta["Acc Group"].astype(str)

pop_results = []

for name, markers in panels.items():
    print(f"\n[POP] Panel: {name} (k={len(markers)})")

    cols_needed = ["Accession_ID"] + markers
    sub_geno = icgt_geno[cols_needed]

    sub = meta.merge(sub_geno, on="Accession_ID", how="inner")
    print(f"[POP]   merged rows: {sub.shape[0]}")

    y = sub["Acc Group"].values
    X = encode_panel(sub, markers)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, baccs = [], []

    for tr, te in skf.split(X, y):
        clf = RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        baccs.append(balanced_accuracy_score(y[te], pred))

    pop_results.append(
        {
            "panel": name,
            "k_markers": len(markers),
            "n_samples": len(y),
            "mean_accuracy": float(np.mean(accs)),
            "sd_accuracy": float(np.std(accs)),
            "mean_balanced_accuracy": float(np.mean(baccs)),
            "sd_balanced_accuracy": float(np.std(baccs)),
        }
    )

pop_df = pd.DataFrame(pop_results)
print("\n[POP] Summary:")
print(pop_df)

out_pop = MASTER_DIR / "panel_population_assignment_accuracy.csv"
pop_df.to_csv(out_pop, index=False)
print("[SAVE]", out_pop)


# -----------------------------
# 5) Probability of Identity (Waits 2001; preserved)
# -----------------------------
print("\n===== Probability of Identity (P(ID)) =====")

def locus_pid_unrelated(freqs) -> float:
    """
    P(ID) for unrelated individuals at one locus:
      PID = sum(p_i^4) + sum_{i<j} 2 p_i^2 p_j^2
    """
    freqs = np.asarray(freqs, dtype=float)
    p2 = freqs**2
    term1 = np.sum(freqs**4)
    term2 = 0.0
    for i in range(len(freqs)):
        for j in range(i + 1, len(freqs)):
            term2 += 2.0 * p2[i] * p2[j]
    return term1 + term2

def locus_pid_sibs(freqs) -> float:
    """
    Approximate P(ID) for siblings at one locus (Waits et al., 2001):
      PIDsibs = 0.25 + 0.5*sum(p_i^2) + 0.5*(sum(p_i^2))^2 - 0.25*sum(p_i^4)
    """
    freqs = np.asarray(freqs, dtype=float)
    p2 = freqs**2
    sum_p2 = np.sum(p2)
    sum_p4 = np.sum(freqs**4)
    return 0.25 + 0.5 * sum_p2 + 0.5 * (sum_p2**2) - 0.25 * sum_p4

def panel_pid(geno_df: pd.DataFrame, markers: list[str]):
    """
    Multi-locus PID (unrelated, sibs) on log10 scale:
      log10(PID_total) = sum log10(PID_locus)
    """
    G = geno_df[markers].astype(str).replace(
        {" ": np.nan, "nan": np.nan, "NaN": np.nan, "M": np.nan, "N": np.nan}
    )

    log10_pid_unrel = 0.0
    log10_pid_sibs = 0.0
    used_loci = 0

    for m in markers:
        s = G[m].dropna()
        if s.empty:
            continue

        vals, counts = np.unique(s.values, return_counts=True)
        freqs = counts / counts.sum()

        if len(freqs) < 2:
            continue

        pid_u = locus_pid_unrelated(freqs)
        pid_s = locus_pid_sibs(freqs)

        if pid_u <= 0 or pid_s <= 0:
            continue

        log10_pid_unrel += math.log10(pid_u)
        log10_pid_sibs += math.log10(pid_s)
        used_loci += 1

    return log10_pid_unrel, log10_pid_sibs, used_loci

pid_rows = []
for name, markers in panels.items():
    print(f"[PID] Panel: {name} (k={len(markers)})")
    log10_u, log10_s, used = panel_pid(icgt_geno, markers)
    pid_rows.append(
        {
            "panel": name,
            "k_input": len(markers),
            "k_used_for_PID": used,
            "log10_PID_unrelated": float(log10_u),
            "log10_PID_sibs": float(log10_s),
        }
    )

pid_df = pd.DataFrame(pid_rows)
print("\n[PID] Summary (log10 scale):")
print(pid_df)

out_pid = MASTER_DIR / "panel_PID_summary.csv"
pid_df.to_csv(out_pid, index=False)
print("[SAVE]", out_pid)


# -----------------------------
# 6) Supplementary confusion matrices (cell2 logic; CSV outputs preserved)
# -----------------------------
print("\n===== Supplementary confusion matrices (major groups, n>=5) =====")

ID_COL = "Accession_ID"
GROUP_COL = "Acc Group"

geno2 = icgt_geno.copy()
pheno2 = icgt_pheno.copy()

# preserve cell2 behavior: drop duplicates by ID to avoid merge inflation
geno2 = geno2.drop_duplicates(subset=[ID_COL], keep="first").copy()
pheno2 = pheno2.drop_duplicates(subset=[ID_COL], keep="first").copy()

snp_cols2 = [c for c in geno2.columns if re.match(r"^\d+_", str(c))]

markers_32 = panel_stats["marker"].astype(str).tolist()
missing_32 = [m for m in markers_32 if m not in snp_cols2]
if missing_32:
    raise ValueError(f"32-panel markers not found in geno columns: {missing_32[:10]} ...")

dfm = geno2[[ID_COL] + snp_cols2].merge(
    pheno2[[ID_COL, GROUP_COL]],
    on=ID_COL,
    how="inner",
)
dfm = dfm.dropna(subset=[GROUP_COL]).copy()

# preserve: cast 14.0 -> 14 -> "14"
dfm[GROUP_COL] = dfm[GROUP_COL].astype(int).astype(str)

counts = dfm[GROUP_COL].value_counts()
major_labels = counts[counts >= 5].index.tolist()
df_major = dfm[dfm[GROUP_COL].isin(major_labels)].copy()

print(f"[INFO] merged (non-missing groups): {dfm.shape}")
print(f"[INFO] major groups (n>=5): {len(major_labels)} groups, {df_major.shape[0]} samples")

def cv_confusion(df_in: pd.DataFrame, snps: list[str], label_col: str = GROUP_COL, n_splits: int = 5, random_state: int = 42):
    X = df_in[snps].copy()
    X = X.replace("N", np.nan)
    X = X.fillna("MISSING").astype(str)

    y = df_in[label_col].astype(str).values

    min_count = pd.Series(y).value_counts().min()
    n_splits_eff = min(n_splits, int(min_count))
    if n_splits_eff < 2:
        raise ValueError("Not enough samples per class for CV.")

    skf = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state)
    labels = sorted(np.unique(y), key=lambda s: int(s))

    y_true_all, y_pred_all = [], []

    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_tr_enc = enc.fit_transform(X_tr)
        X_te_enc = enc.transform(X_te)

        clf = RandomForestClassifier(
            n_estimators=500,
            random_state=random_state,
            n_jobs=-1,
        )
        clf.fit(X_tr_enc, y_tr)
        y_hat = clf.predict(X_te_enc)

        y_true_all.extend(y_te.tolist())
        y_pred_all.extend(y_hat.tolist())

    cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)
    acc = accuracy_score(y_true_all, y_pred_all)
    bal = balanced_accuracy_score(y_true_all, y_pred_all)
    return cm, labels, acc, bal, n_splits_eff

cm_full, labels, acc_full, bal_full, kfold = cv_confusion(df_major, snp_cols2)
cm_32, labels2, acc_32, bal_32, kfold2 = cv_confusion(df_major, markers_32)

assert labels == labels2

print(f"[RESULT] Full_536:  acc={acc_full:.3f}, bal_acc={bal_full:.3f}, folds={kfold}")
print(f"[RESULT] PureID_32: acc={acc_32:.3f}, bal_acc={bal_32:.3f}, folds={kfold2}")

def save_cm_csv(cm: np.ndarray, labels: list[str], prefix: str):
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    df_cm.to_csv(SUPP_DIR / f"{prefix}_confusion_counts.csv")

    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    df_norm = pd.DataFrame(cm_norm, index=labels, columns=labels)
    df_norm.to_csv(SUPP_DIR / f"{prefix}_confusion_norm_by_true.csv")

save_cm_csv(cm_full, labels, "S1_full_536")
save_cm_csv(cm_32, labels, "S1_pureID_32")

print("[SAVE] Supplementary confusion CSVs in:", SUPP_DIR)

# Optional: plotting (disabled by default for public repo)
if MAKE_PLOTS:
    import matplotlib.pyplot as plt

    def save_cm_png(cm: np.ndarray, labels: list[str], prefix: str):
        cm_norm = cm / cm.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm_norm, aspect="auto")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted Acc Group")
        ax.set_ylabel("True Acc Group")
        ax.set_title(f"{prefix} (row-normalized), major groups (n≥5)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Proportion")
        fig.tight_layout()
        fig.savefig(SUPP_DIR / f"{prefix}_confusion_heatmap.png", dpi=300)
        plt.close(fig)

    save_cm_png(cm_full, labels, "S1_full_536")
    save_cm_png(cm_32, labels, "S1_pureID_32")

    # combined A/B panel
    cmA = cm_full / cm_full.sum(axis=1, keepdims=True)
    cmB = cm_32 / cm_32.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
    for ax, cmn, title in zip(axes, [cmA, cmB], ["Full 536-SNP", "Pure-ID 32-SNP"]):
        im = ax.imshow(cmn, aspect="auto")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("Proportion (row-normalized)")
    fig.suptitle("Supplementary Fig. S1. Fold-aggregated confusion matrices (major groups, n≥5)")
    fig.savefig(SUPP_DIR / "S1_combined_confusion.png", dpi=300)
    plt.close(fig)

    print("[SAVE] Supplementary PNGs in:", SUPP_DIR)


# -----------------------------
# Appendix: PID helper for 0/1/2 dosages (not executed)
# -----------------------------
def pid_waits_snp_from_dosage(genotypes_012):
    """
    genotypes_012: array-like of {0,1,2} with NaN allowed
    returns: (pid_unrelated, pid_sibs)
    """
    g = np.asarray(genotypes_012, dtype=float)
    g = g[~np.isnan(g)]
    if g.size == 0:
        return None, None

    p = g.mean() / 2.0
    p = min(max(p, 0.0), 1.0)
    q = 1.0 - p

    pid_u = (p**4) + (2 * p * q) ** 2 + (q**4)

    sum_p2 = (p**2) + (q**2)
    sum_p4 = (p**4) + (q**4)
    pid_s = 0.25 + 0.5 * sum_p2 + 0.5 * (sum_p2**2) - 0.25 * sum_p4

    return pid_u, pid_s


def multilocus_log10_pid(geno_matrix_012):
    """
    geno_matrix_012: 2D array (n_samples, n_loci) of 0/1/2 with NaN
    returns: (log10_pid_unrelated, log10_pid_sibs, used_loci)
    """
    G = np.asarray(geno_matrix_012, dtype=float)

    log10_u = 0.0
    log10_s = 0.0
    used = 0

    for j in range(G.shape[1]):
        pid_u, pid_s = pid_waits_snp_from_dosage(G[:, j])
        if pid_u is None:
            continue
        if pid_u <= 0 or pid_s <= 0:
            continue
        log10_u += np.log10(pid_u)
        log10_s += np.log10(pid_s)
        used += 1

    return log10_u, log10_s, used


print("\n===== DONE =====")
