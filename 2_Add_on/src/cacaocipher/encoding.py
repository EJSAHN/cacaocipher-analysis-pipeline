from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


MISSING_TOKENS_DEFAULT = {"", " ", "nan", "NaN", "None", "N"}


@dataclass(frozen=True)
class BinaryEncoding:
    X: np.ndarray
    majors: list[str]
    markers: list[str]


def _clean_series(
    s: pd.Series,
    treat_m_as_missing: bool,
    missing_tokens: set[str],
) -> pd.Series:
    s2 = s.astype(str).replace({t: np.nan for t in missing_tokens})
    if treat_m_as_missing:
        s2 = s2.replace({"M": np.nan})
    return s2


def binary_encode_major_minor(
    geno_df: pd.DataFrame,
    markers: list[str],
    *,
    treat_m_as_missing: bool = True,
    missing_tokens: set[str] = MISSING_TOKENS_DEFAULT,
    min_non_missing: int = 10,
    impute_strategy: Literal["mean", "most_frequent"] = "mean",
) -> BinaryEncoding:
    """
    Encode each SNP into 0/1 using the most frequent (major) observed allele as 0,
    and any other observed allele as 1. Missing values are NaN and later imputed to 0.0 to keep feature count fixed across splits.

    This encoding is intended for ML baselines and is consistent with a conservative
    treatment of ambiguous calls (e.g., 'M') as missing when treat_m_as_missing=True.
    """
    n = len(geno_df)
    majors: list[str] = []
    X = np.full((n, len(markers)), np.nan, dtype=float)

    for j, m in enumerate(markers):
        s = _clean_series(geno_df[m], treat_m_as_missing=treat_m_as_missing, missing_tokens=missing_tokens)
        s_non = s.dropna()
        if len(s_non) < min_non_missing:
            majors.append("NA")
            continue
        major = s_non.value_counts().idxmax()
        majors.append(str(major))
        X[:, j] = np.where(s == major, 0.0, np.where(s.isna(), np.nan, 1.0))

    X_imp = np.nan_to_num(X, nan=0.0)
    return BinaryEncoding(X=X_imp, majors=majors, markers=markers)


def categorical_encode(
    geno_df: pd.DataFrame,
    markers: list[str],
    *,
    categories: Optional[list[str]] = None,
    missing_tokens: set[str] = MISSING_TOKENS_DEFAULT,
    treat_m_as_missing: bool = False,
) -> np.ndarray:
    """
    Ordinal encoding for distance/structure work.

    By default, uses categories ['A','C','M'] (missing as -1) if categories is None.
    Missing tokens and optionally 'M' are treated as missing.

    Returns an integer matrix with missing coded as -1.
    """
    if categories is None:
        categories = ["A", "C", "M"]

    cat_to_int = {c: i for i, c in enumerate(categories)}
    G = np.full((len(geno_df), len(markers)), -1, dtype=np.int16)

    for j, m in enumerate(markers):
        s = geno_df[m].astype(str)
        s = s.replace({t: np.nan for t in missing_tokens})
        if treat_m_as_missing:
            s = s.replace({"M": np.nan})
        for i, v in enumerate(s.values):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            vv = str(v)
            if vv in cat_to_int:
                G[i, j] = cat_to_int[vv]
            else:
                # unseen token -> missing
                G[i, j] = -1
    return G


def compute_genotype_pca(
    geno_df: pd.DataFrame,
    markers: list[str],
    *,
    n_components: int = 10,
    treat_m_as_missing: bool = True,
    missing_tokens: set[str] = MISSING_TOKENS_DEFAULT,
    random_state: int = 0,
) -> np.ndarray:
    """
    Compute PCA components from major/minor binary encoding of genotypes.
    """
    enc = binary_encode_major_minor(
        geno_df,
        markers,
        treat_m_as_missing=treat_m_as_missing,
        missing_tokens=missing_tokens,
    )
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(enc.X)