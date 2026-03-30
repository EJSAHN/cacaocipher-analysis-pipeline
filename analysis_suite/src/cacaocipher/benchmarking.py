from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Literal, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    mean_squared_error,
)
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.feature_selection import mutual_info_classif


from .encoding import binary_encode_major_minor


@dataclass(frozen=True)
class PanelSpec:
    name: str
    k: int
    markers: Optional[list[str]] = None  # fixed panels
    selection: Optional[Callable[[pd.DataFrame, np.ndarray, list[str], int, int], list[str]]] = None  # fold-wise selection

def _collapse_rare_classes(
    y: np.ndarray,
    *,
    min_count: int,
    strategy: Literal["collapse", "drop"] = "collapse",
    rare_label: str = "Other",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Handle rare classes for stratified CV.

    Returns:
      - y_new: modified labels
      - keep_mask: boolean mask (all True for 'collapse')
      - collapsed_labels: list of labels that were collapsed/dropped
    """
    y = np.asarray(y).astype(str)
    counts = pd.Series(y).value_counts()
    collapsed = counts[counts < min_count].index.astype(str).tolist()
    if len(collapsed) == 0:
        return y, np.ones(len(y), dtype=bool), []

    if strategy == "drop":
        keep = ~np.isin(y, collapsed)
        return y[keep], keep, collapsed

    y2 = y.copy()
    y2[np.isin(y2, collapsed)] = rare_label
    return y2, np.ones(len(y), dtype=bool), collapsed



def _corr_prune(
    geno_df: pd.DataFrame,
    ranked_markers: list[str],
    *,
    k: int,
    corr_thresh: float = 0.8,
) -> list[str]:
    """
    Greedy correlation pruning on binary major/minor encoding.
    """
    enc = binary_encode_major_minor(geno_df, ranked_markers, treat_m_as_missing=True)
    X = enc.X
    selected: list[int] = []
    for j in range(X.shape[1]):
        if len(selected) == 0:
            selected.append(j)
            if len(selected) >= k:
                break
            continue
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


def select_by_mutual_information(
    geno_train: pd.DataFrame,
    y_train: np.ndarray,
    snp_cols: list[str],
    k: int,
    seed: int,
) -> list[str]:
    enc = binary_encode_major_minor(geno_train, snp_cols, treat_m_as_missing=True, impute_strategy="most_frequent")
    X = enc.X
    mi = mutual_info_classif(X, y_train, discrete_features=True, random_state=seed)
    order = np.argsort(mi)[::-1]
    ranked = [snp_cols[i] for i in order]
    return _corr_prune(geno_train, ranked, k=k, corr_thresh=0.8)


def select_by_l1_logistic(
    geno_train: pd.DataFrame,
    y_train: np.ndarray,
    snp_cols: list[str],
    k: int,
    seed: int,
) -> list[str]:
    enc = binary_encode_major_minor(geno_train, snp_cols, treat_m_as_missing=True, impute_strategy="most_frequent")
    X = enc.X

    clf = LogisticRegression(
        penalty="l1",
        solver="saga",
        max_iter=5000,
        C=0.5,
        random_state=seed,
    )
    clf.fit(X, y_train)
    coef = np.abs(clf.coef_)
    scores = coef.mean(axis=0)
    order = np.argsort(scores)[::-1]
    ranked = [snp_cols[i] for i in order]
    return _corr_prune(geno_train, ranked, k=k, corr_thresh=0.8)


def select_by_rf_importance(
    geno_train: pd.DataFrame,
    y_train: np.ndarray,
    snp_cols: list[str],
    k: int,
    seed: int,
) -> list[str]:
    enc = binary_encode_major_minor(geno_train, snp_cols, treat_m_as_missing=True, impute_strategy="most_frequent")
    X = enc.X
    clf = RandomForestClassifier(
        n_estimators=800,
        max_features="sqrt",
        random_state=seed,
    )
    clf.fit(X, y_train)
    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1]
    ranked = [snp_cols[i] for i in order]
    return _corr_prune(geno_train, ranked, k=k, corr_thresh=0.8)


def select_random_panel(
    geno_train: pd.DataFrame,
    y_train: np.ndarray,
    snp_cols: list[str],
    k: int,
    seed: int,
) -> list[str]:
    rng = np.random.default_rng(seed)
    return rng.choice(snp_cols, size=k, replace=False).tolist()


@dataclass(frozen=True)
class PopAssignFoldResult:
    panel: str
    k: int
    fold: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    n_test: int


def evaluate_population_assignment(
    geno_df: pd.DataFrame,
    pheno_df: pd.DataFrame,
    snp_cols: list[str],
    *,
    target_col: str = "Acc Group",
    panels: Sequence[PanelSpec],
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
    rare_class_strategy: Literal["collapse", "drop"] = "collapse",
    min_class_count: Optional[int] = None,
    rare_label: str = "Other",
) -> pd.DataFrame:
    """
    Repeated stratified CV population assignment benchmark.

    Supervised panels are selected on each training fold only (no leakage).
    """
    meta = pheno_df[["Accession_ID", target_col]].dropna().copy()
    meta[target_col] = meta[target_col].astype(str)

    df = meta.merge(geno_df[["Accession_ID"] + snp_cols], on="Accession_ID", how="inner")
    y = df[target_col].values
    min_count = int(min_class_count) if min_class_count is not None else int(n_splits)
    y2, keep, collapsed = _collapse_rare_classes(
        y,
        min_count=min_count,
        strategy=rare_class_strategy,
        rare_label=rare_label,
    )
    if not np.all(keep):
        df = df.loc[keep].reset_index(drop=True)
    y = y2
    collapsed_str = ";".join(collapsed)
    all_labels = np.unique(y)
    fold_results: list[dict] = []

    rng = np.random.default_rng(seed)
    # generate deterministic seeds per repeat
    repeat_seeds = rng.integers(0, 1_000_000, size=n_repeats)

    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(repeat_seeds[rep]))
        for fold, (tr, te) in enumerate(skf.split(df, y)):
            df_tr = df.iloc[tr].reset_index(drop=True)
            df_te = df.iloc[te].reset_index(drop=True)
            y_tr = df_tr[target_col].values
            y_te = df_te[target_col].values

            for p in panels:
                if p.selection is not None:
                    markers = p.selection(df_tr, y_tr, snp_cols, p.k, int(repeat_seeds[rep]) + fold)
                else:
                    if p.markers is None:
                        raise ValueError(f"Panel '{p.name}' requires markers or selection.")
                    markers = p.markers

                X_tr = binary_encode_major_minor(df_tr, markers, treat_m_as_missing=True).X
                X_te = binary_encode_major_minor(df_te, markers, treat_m_as_missing=True).X

                clf = RandomForestClassifier(
                    n_estimators=600,
                    max_features="sqrt",
                    random_state=int(repeat_seeds[rep]) + fold,
                )
                clf.fit(X_tr, y_tr)
                pred = clf.predict(X_te)

                fold_results.append(
                    {
                        "panel": p.name,
                        "k": int(p.k),
                        "repeat": int(rep),
                        "fold": int(fold),
                        "n_classes": int(len(np.unique(y))),
                        "collapsed_labels": collapsed_str,
                        "accuracy": float(accuracy_score(y_te, pred)),
                        "balanced_accuracy": float(recall_score(y_te, pred, average="macro", labels=all_labels, zero_division=0)),
                        "macro_f1": float(f1_score(y_te, pred, average="macro", labels=all_labels, zero_division=0)),
                        "n_test": int(len(y_te)),
                    }
                )

    return pd.DataFrame(fold_results)


@dataclass(frozen=True)
class TraitPredFoldResult:
    panel: str
    k: int
    fold: int
    rmse: float
    r_pearson: float
    n_test: int


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    r = np.corrcoef(x, y)[0, 1]
    if np.isnan(r):
        return 0.0
    return float(r)


def evaluate_trait_prediction(
    geno_df: pd.DataFrame,
    pheno_df: pd.DataFrame,
    snp_cols: list[str],
    *,
    trait_col: str = "PodIndex",
    panels: Sequence[PanelSpec],
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Trait prediction benchmark with strict CV (panel markers are fixed or chosen without using the test fold).
    Uses Ridge regression for a conservative, low-variance baseline.
    """
    df = pheno_df[["Accession_ID", trait_col, "Acc Group"]].dropna().copy()
    df = df.merge(geno_df[["Accession_ID"] + snp_cols], on="Accession_ID", how="inner").reset_index(drop=True)
    y = df[trait_col].values.astype(float)

    # Use Group-stratified KFold to preserve ancestry composition (approx).
    # If Acc Group is very imbalanced, this acts as a conservative split strategy.
    groups = df["Acc Group"].astype(str).values
    groups2, keep, collapsed = _collapse_rare_classes(
        groups,
        min_count=int(n_splits),
        strategy="collapse",
        rare_label="Other",
    )
    if not np.all(keep):
        df = df.loc[keep].reset_index(drop=True)
        y = y[keep]
    groups = groups2

    min_ct = int(pd.Series(groups).value_counts().min())
    if min_ct < n_splits:
        n_splits = min_ct
        if n_splits < 2:
            raise ValueError("Not enough samples per group for stratified CV.")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    rows: list[dict] = []
    for fold, (tr, te) in enumerate(skf.split(df, groups)):
        df_tr = df.iloc[tr].reset_index(drop=True)
        df_te = df.iloc[te].reset_index(drop=True)
        y_tr = df_tr[trait_col].values.astype(float)
        y_te = df_te[trait_col].values.astype(float)

        for p in panels:
            if p.selection is not None:
                # trait prediction benchmarks use only fixed panels by default
                markers = p.selection(df_tr, y_tr, snp_cols, p.k, seed + fold)
            else:
                if p.markers is None:
                    raise ValueError(f"Panel '{p.name}' requires markers or selection.")
                markers = p.markers

            X_tr = binary_encode_major_minor(df_tr, markers, treat_m_as_missing=True).X
            X_te = binary_encode_major_minor(df_te, markers, treat_m_as_missing=True).X

            model = Ridge(alpha=1.0, random_state=seed)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)

            rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
            r = _safe_corr(pred, y_te)

            rows.append(
                {
                    "panel": p.name,
                    "k": int(p.k),
                    "fold": int(fold),
                    "rmse": rmse,
                    "r_pearson": r,
                    "n_test": int(len(y_te)),
                }
            )
    return pd.DataFrame(rows)