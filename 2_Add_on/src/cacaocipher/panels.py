from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .encoding import binary_encode_major_minor


def snp_heterozygosity(values: np.ndarray) -> tuple[float, float]:
    """
    Compute heterozygosity-like diversity (1 - sum p^2) and missing rate,
    treating 'N' as missing.

    This matches the manuscript's use as a fast polymorphism proxy.
    """
    s = pd.Series(values.astype(str))
    s = s.replace({"nan": "N", "NaN": "N", " ": "N"})
    non = s[s != "N"]
    if len(non) == 0:
        return 0.0, 1.0
    missing_rate = 1.0 - len(non) / len(s)
    freqs = non.value_counts(normalize=True).values
    H = float(1.0 - np.sum(freqs ** 2))
    return H, float(missing_rate)


def marker_abs_pearson_corr_with_trait(
    geno_df: pd.DataFrame,
    pheno_df: pd.DataFrame,
    markers: list[str],
    trait_col: str,
    *,
    treat_m_as_missing: bool = True,
    min_n: int = 30,
) -> pd.DataFrame:
    """
    Compute |r| between each marker and a numeric trait.

    Genotypes are converted to integer categories based on observed levels.
    'M' and 'N' can be treated as missing for conservatism.
    """
    df = pheno_df[["Accession_ID", trait_col]].dropna().merge(
        geno_df[["Accession_ID"] + markers],
        on="Accession_ID",
        how="inner",
    )
    y = df[trait_col].values.astype(float)

    rows = []
    for m in markers:
        s = df[m].astype(str).replace({"nan": np.nan, "NaN": np.nan, " ": np.nan})
        if treat_m_as_missing:
            s = s.replace({"M": np.nan, "N": np.nan})
        else:
            s = s.replace({"N": np.nan})
        levels = s.dropna().unique()
        if len(levels) < 2:
            rows.append({"marker": m, "abs_r_trait": 0.0, "n_used": 0})
            continue
        mapping = {a: i for i, a in enumerate(levels)}
        x = s.map(mapping).astype(float).values
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() < min_n:
            rows.append({"marker": m, "abs_r_trait": 0.0, "n_used": int(mask.sum())})
            continue
        r = np.corrcoef(x[mask], y[mask])[0, 1]
        if np.isnan(r):
            r = 0.0
        rows.append({"marker": m, "abs_r_trait": float(abs(r)), "n_used": int(mask.sum())})

    out = pd.DataFrame(rows).sort_values("abs_r_trait", ascending=False).reset_index(drop=True)
    return out


def select_pure_id_panel(
    geno_df: pd.DataFrame,
    snp_cols: list[str],
    *,
    k: int = 32,
    max_missing: float = 0.2,
    corr_thresh: float = 0.8,
) -> list[str]:
    """
    Unsupervised selection: prioritize high heterozygosity with missingness + correlation pruning.
    """
    stats = []
    G = geno_df[snp_cols].values
    for j, m in enumerate(snp_cols):
        H, miss = snp_heterozygosity(G[:, j])
        stats.append((m, H, miss))
    df = pd.DataFrame(stats, columns=["marker", "H", "missing_rate"])
    df = df[df["missing_rate"] <= max_missing].sort_values("H", ascending=False)

    ranked = df["marker"].astype(str).tolist()
    # Use a stable binary encoding for correlation pruning
    markers = _corr_prune_binary(geno_df, ranked, k=k, corr_thresh=corr_thresh)
    return markers


def _corr_prune_binary(
    geno_df: pd.DataFrame,
    ranked_markers: list[str],
    *,
    k: int,
    corr_thresh: float,
) -> list[str]:
    enc = binary_encode_major_minor(geno_df, ranked_markers, treat_m_as_missing=True)
    X = enc.X
    selected: list[int] = []
    for j in range(X.shape[1]):
        if len(selected) == 0:
            selected.append(j)
        else:
            ok = True
            for s in selected:
                r = np.corrcoef(X[:, j], X[:, s])[0, 1]
                if np.isnan(r):
                    r = 0.0
                if abs(r) > corr_thresh:
                    ok = False
                    break
            if ok:
                selected.append(j)
        if len(selected) >= k:
            break
    return [ranked_markers[j] for j in selected]


def select_trait_aware_panel(
    geno_df: pd.DataFrame,
    pheno_df: pd.DataFrame,
    snp_cols: list[str],
    *,
    trait_col: str = "PodIndex",
    k: int = 32,
    alpha: float = 1.0,
    beta: float = 1.0,
    max_missing: float = 0.2,
    corr_thresh: float = 0.8,
    treat_m_as_missing_for_trait: bool = True,
) -> list[str]:
    """
    Trait-aware selection: score = alpha * H + beta * |r(marker, trait)|, then correlation pruning.

    This mirrors the manuscript's trait-aware heuristic while keeping deterministic behavior.
    """
    trait_df = marker_abs_pearson_corr_with_trait(
        geno_df,
        pheno_df,
        snp_cols,
        trait_col,
        treat_m_as_missing=treat_m_as_missing_for_trait,
    )

    stats = []
    G = geno_df[snp_cols].values
    for j, m in enumerate(snp_cols):
        H, miss = snp_heterozygosity(G[:, j])
        stats.append((m, H, miss))
    df_stats = pd.DataFrame(stats, columns=["marker", "H", "missing_rate"])
    df = df_stats.merge(trait_df[["marker", "abs_r_trait"]], on="marker", how="left").fillna({"abs_r_trait": 0.0})
    df = df[df["missing_rate"] <= max_missing].copy()
    df["score"] = alpha * df["H"] + beta * df["abs_r_trait"]
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    ranked = df["marker"].astype(str).tolist()
    markers = _corr_prune_binary(geno_df, ranked, k=k, corr_thresh=corr_thresh)
    return markers
