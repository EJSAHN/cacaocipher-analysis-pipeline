from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr


@dataclass(frozen=True)
class RegressionResult:
    n: int
    coef: float
    se: float
    t: float
    p: float
    r2: float


def _design_matrix(
    x: np.ndarray,
    covariates: Optional[np.ndarray] = None,
) -> np.ndarray:
    if covariates is None:
        X = x.reshape(-1, 1)
    else:
        X = np.column_stack([x.reshape(-1, 1), covariates])
    X = sm.add_constant(X, has_constant="add")
    return X


def robust_ols(
    y: np.ndarray,
    x: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    *,
    cov_type: str = "HC3",
) -> RegressionResult:
    """
    Robust OLS of y ~ x + covariates with heteroskedasticity-consistent SE.
    Returns coefficient and robust p-value for x.
    """
    mask = ~np.isnan(y) & ~np.isnan(x)
    if covariates is not None:
        mask = mask & ~np.any(np.isnan(covariates), axis=1)

    y2 = y[mask]
    x2 = x[mask]
    cov2 = covariates[mask] if covariates is not None else None

    if len(y2) < 20:
        return RegressionResult(n=int(len(y2)), coef=float("nan"), se=float("nan"), t=float("nan"), p=float("nan"), r2=float("nan"))

    X = _design_matrix(x2, cov2)
    model = sm.OLS(y2, X).fit(cov_type=cov_type)
    coef = float(model.params[1])
    se = float(model.bse[1])
    t = float(model.tvalues[1])
    p = float(model.pvalues[1])
    r2 = float(model.rsquared)
    return RegressionResult(n=int(len(y2)), coef=coef, se=se, t=t, p=p, r2=r2)


def stratified_permutation_pvalue(
    y: np.ndarray,
    x: np.ndarray,
    strata: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    *,
    n_permutations: int = 5000,
    seed: int = 0,
) -> float:
    """
    Permutation p-value for the absolute x coefficient in y ~ x + covariates,
    permuting x within strata to preserve ancestry/group structure.

    Returns a two-sided p-value based on |coef|.
    """
    rng = np.random.default_rng(seed)
    obs = robust_ols(y, x, covariates=covariates)
    if np.isnan(obs.coef):
        return float("nan")
    coef_obs = abs(obs.coef)

    # Precompute indices per stratum
    strata = np.asarray(strata).astype(str)
    idx_by = {}
    for i, g in enumerate(strata):
        idx_by.setdefault(g, []).append(i)
    idx_by = {g: np.asarray(idxs, dtype=int) for g, idxs in idx_by.items()}

    count = 0
    valid = 0
    for _ in range(n_permutations):
        x_perm = x.copy()
        for g, idxs in idx_by.items():
            if len(idxs) < 2:
                continue
            x_perm[idxs] = rng.permutation(x_perm[idxs])
        res = robust_ols(y, x_perm, covariates=covariates)
        if np.isnan(res.coef):
            continue
        valid += 1
        if abs(res.coef) >= coef_obs:
            count += 1

    if valid == 0:
        return float("nan")
    return float((count + 1) / (valid + 1))


def within_group_correlations(
    df: pd.DataFrame,
    *,
    group_col: str,
    x_col: str,
    y_col: str,
    min_n: int = 10,
) -> pd.DataFrame:
    rows = []
    for g, sub in df.groupby(group_col):
        sub = sub[[x_col, y_col]].dropna()
        if sub.shape[0] < min_n:
            continue
        r_p = np.corrcoef(sub[x_col].values, sub[y_col].values)[0, 1]
        r_s = spearmanr(sub[x_col].values, sub[y_col].values).correlation
        rows.append(
            {
                "group": str(g),
                "n": int(sub.shape[0]),
                "pearson_r": float(r_p) if not np.isnan(r_p) else 0.0,
                "spearman_r": float(r_s) if not np.isnan(r_s) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False)
