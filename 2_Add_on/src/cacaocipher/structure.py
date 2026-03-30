from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
from scipy.spatial import procrustes
from scipy.stats import spearmanr


@dataclass(frozen=True)
class MantelResult:
    method: str
    r: float
    p_value: float
    n: int
    n_permutations: int


def _upper_tri_values(D: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(D.shape[0], k=1)
    return D[iu].astype(float)


def mantel_test(
    D1: np.ndarray,
    D2: np.ndarray,
    *,
    method: Literal["pearson", "spearman"] = "pearson",
    n_permutations: int = 999,
    seed: int = 0,
) -> MantelResult:
    """
    Mantel test for correlation between two distance matrices.

    Permutation is done by shuffling the labels of D2 (rows/cols) together.
    """
    if D1.shape != D2.shape:
        raise ValueError("D1 and D2 must have the same shape.")
    if D1.shape[0] < 3:
        raise ValueError("Distance matrices too small.")

    v1 = _upper_tri_values(D1)
    v2 = _upper_tri_values(D2)

    if method == "spearman":
        r_obs = spearmanr(v1, v2).correlation
    else:
        r_obs = float(np.corrcoef(v1, v2)[0, 1])

    rng = np.random.default_rng(seed)
    count = 0
    n = D1.shape[0]

    for _ in range(n_permutations):
        perm = rng.permutation(n)
        D2p = D2[np.ix_(perm, perm)]
        v2p = _upper_tri_values(D2p)
        if method == "spearman":
            r_p = spearmanr(v1, v2p).correlation
        else:
            r_p = float(np.corrcoef(v1, v2p)[0, 1])
        if np.isnan(r_p):
            continue
        if abs(r_p) >= abs(r_obs):
            count += 1

    p = (count + 1) / (n_permutations + 1)
    return MantelResult(method=method, r=float(r_obs), p_value=float(p), n=n, n_permutations=n_permutations)


def hamming_distance_matrix_excluding_missing(
    G: np.ndarray,
    *,
    missing_value: int = -1,
) -> np.ndarray:
    """
    Compute Hamming distance counts between rows of G, excluding missing positions.

    G must be an integer matrix with missing coded as `missing_value`.
    Distance is the number of mismatching loci among loci observed in both samples.

    If a pair has zero jointly observed loci, distance is defined as 0.
    """
    if G.ndim != 2:
        raise ValueError("G must be 2D.")
    n, m = G.shape
    D = np.zeros((n, n), dtype=np.int32)

    for i in range(n):
        gi = G[i]
        valid = (gi != missing_value) & (G != missing_value)
        mism = (G != gi) & valid
        d = mism.sum(axis=1)
        D[i, :] = d

    # enforce symmetry and zeros on diagonal
    D = np.triu(D, 1)
    D = D + D.T
    return D


def classical_mds(
    D: np.ndarray,
    n_components: int = 2,
) -> np.ndarray:
    """
    Classical MDS (Torgerson) on a distance matrix.

    Returns coordinates with shape (n_samples, n_components).
    """
    D = D.astype(float)
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    evals, evecs = np.linalg.eigh(B)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    evals_pos = np.maximum(evals[:n_components], 0.0)
    coords = evecs[:, :n_components] * np.sqrt(evals_pos)
    return coords


def procrustes_similarity(
    X: np.ndarray,
    Y: np.ndarray,
) -> float:
    """
    Procrustes similarity in [0,1], where 1 is identical up to translation/scale/rotation.
    """
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape for Procrustes.")
    _, _, disparity = procrustes(X, Y)
    sim = 1.0 - float(disparity)
    if sim < 0:
        sim = 0.0
    return sim


def mean_neighbor_overlap(
    D_ref: np.ndarray,
    D_alt: np.ndarray,
    *,
    k: int = 10,
) -> float:
    """
    Average Jaccard overlap between k-NN sets derived from two distance matrices.
    """
    if D_ref.shape != D_alt.shape:
        raise ValueError("Distance matrices must have the same shape.")
    n = D_ref.shape[0]
    if k >= n:
        raise ValueError("k must be < n.")

    overlaps = []
    for i in range(n):
        nn_ref = np.argsort(D_ref[i])[1 : k + 1]
        nn_alt = np.argsort(D_alt[i])[1 : k + 1]
        a = set(nn_ref.tolist())
        b = set(nn_alt.tolist())
        overlaps.append(len(a & b) / len(a | b))
    return float(np.mean(overlaps))
