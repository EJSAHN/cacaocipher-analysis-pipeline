from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LdSummary:
    n_markers: int
    n_pairs: int
    r2_mean: float
    r2_median: float
    r2_max: float
    frac_r2_gt_0p2: float
    frac_r2_gt_0p5: float
    effective_rank_entropy: float
    effective_rank_ipr: float


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = (~np.isnan(x)) & (~np.isnan(y))
    if mask.sum() < 10:
        return 0.0
    r = np.corrcoef(x[mask], y[mask])[0, 1]
    if np.isnan(r):
        return 0.0
    return float(r)


def pairwise_r2_from_binary(X: np.ndarray) -> np.ndarray:
    """
    Compute pairwise r^2 for columns of X (binary/imputed values allowed).
    Returns a vector of length m*(m-1)/2.
    """
    m = X.shape[1]
    r2 = []
    for i in range(m):
        for j in range(i + 1, m):
            r = _safe_corr(X[:, i], X[:, j])
            r2.append(r * r)
    return np.asarray(r2, dtype=float)


def effective_rank(corr: np.ndarray) -> tuple[float, float]:
    """
    Two effective-rank summaries of a correlation matrix:
      - exp(Shannon entropy of normalized eigenvalues)
      - 1 / sum(p_i^2)  (inverse participation ratio)
    """
    evals = np.linalg.eigvalsh(corr.astype(float))
    evals = np.clip(evals, 0.0, None)
    s = float(evals.sum())
    if s <= 0:
        return 0.0, 0.0
    p = evals / s
    p = np.clip(p, 1e-12, 1.0)
    er_entropy = float(np.exp(-np.sum(p * np.log(p))))
    er_ipr = float(1.0 / np.sum(p * p))
    return er_entropy, er_ipr


def summarize_ld(X: np.ndarray) -> LdSummary:
    m = X.shape[1]
    r2 = pairwise_r2_from_binary(X)
    n_pairs = int(len(r2))
    r2_mean = float(np.mean(r2)) if n_pairs else 0.0
    r2_median = float(np.median(r2)) if n_pairs else 0.0
    r2_max = float(np.max(r2)) if n_pairs else 0.0
    frac_02 = float(np.mean(r2 > 0.2)) if n_pairs else 0.0
    frac_05 = float(np.mean(r2 > 0.5)) if n_pairs else 0.0

    corr = np.corrcoef(X, rowvar=False)
    er_entropy, er_ipr = effective_rank(corr)

    return LdSummary(
        n_markers=m,
        n_pairs=n_pairs,
        r2_mean=r2_mean,
        r2_median=r2_median,
        r2_max=r2_max,
        frac_r2_gt_0p2=frac_02,
        frac_r2_gt_0p5=frac_05,
        effective_rank_entropy=er_entropy,
        effective_rank_ipr=er_ipr,
    )


def per_locus_entropy_bits(geno_df: pd.DataFrame, markers: list[str]) -> pd.Series:
    """
    Shannon entropy per marker (bits), using observed non-missing allele frequencies.
    Missing is assumed to be 'N'.
    """
    H = {}
    for m in markers:
        s = geno_df[m].astype(str)
        s = s.replace({"nan": "N", "NaN": "N", " ": "N"})
        s = s[s != "N"]
        if len(s) == 0:
            H[m] = 0.0
            continue
        p = s.value_counts(normalize=True).values
        h = -np.sum(p * np.log2(p))
        H[m] = float(h)
    return pd.Series(H, name="entropy_bits")


@dataclass(frozen=True)
class CodewordEntropy:
    n_samples: int
    n_markers: int
    n_unique_codewords: int
    log2_unique: float
    shannon_entropy_bits: float
    total_locus_entropy_bits: float
    total_correlation_bits: float


def codeword_entropy_bits(
    geno_df: pd.DataFrame,
    markers: list[str],
    *,
    missing_token: str = "N",
) -> CodewordEntropy:
    """
    Compute codeword entropy for a panel:

    - unique codeword count and log2(unique)
    - Shannon entropy of codeword distribution
    - sum of per-locus entropies (upper bound)
    - total correlation = sum(H_i) - H_joint (redundancy/dependence)
    """
    G = geno_df[markers].astype(str).copy()
    G = G.replace({"nan": missing_token, "NaN": missing_token, " ": missing_token})
    codewords = G.apply(lambda r: "|".join(r.values.tolist()), axis=1)
    counts = codewords.value_counts()
    p = (counts / counts.sum()).values
    H_joint = float(-np.sum(p * np.log2(p))) if len(p) else 0.0
    n_unique = int(counts.shape[0])
    log2_unique = float(np.log2(n_unique)) if n_unique > 0 else 0.0

    locus_H = per_locus_entropy_bits(geno_df, markers)
    sum_H = float(locus_H.sum())
    total_corr = float(sum_H - H_joint)

    return CodewordEntropy(
        n_samples=int(G.shape[0]),
        n_markers=int(len(markers)),
        n_unique_codewords=n_unique,
        log2_unique=log2_unique,
        shannon_entropy_bits=H_joint,
        total_locus_entropy_bits=sum_H,
        total_correlation_bits=total_corr,
    )
