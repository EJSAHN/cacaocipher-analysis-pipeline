from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Allow running without installation
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cacaocipher.io_utils import ensure_output_dir, load_master_tables, resolve_project_root, get_snp_columns
from cacaocipher.encoding import categorical_encode, compute_genotype_pca
from cacaocipher.structure import (
    hamming_distance_matrix_excluding_missing,
    mantel_test,
    classical_mds,
    procrustes_similarity,
    mean_neighbor_overlap,
)
from cacaocipher.plotting import save_figure
from cacaocipher.ld_entropy import summarize_ld, codeword_entropy_bits
from cacaocipher.panels import select_trait_aware_panel, select_pure_id_panel
from cacaocipher.confounding import robust_ols, stratified_permutation_pvalue, within_group_correlations
from cacaocipher.benchmarking import (
    PanelSpec,
    evaluate_population_assignment,
    evaluate_trait_prediction,
    select_by_mutual_information,
    select_by_l1_logistic,
    select_by_rf_importance,
    select_random_panel,
)


def _summary_ci(x: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
    ci_lo = mean - 1.96 * se
    ci_hi = mean + 1.96 * se
    return mean, ci_lo, ci_hi


def run_structure_preservation(
    geno: pd.DataFrame,
    pheno: pd.DataFrame,
    snp_cols: list[str],
    panels: dict[str, list[str]],
    out_dir: Path,
    *,
    seed: int,
    n_perm: int,
) -> None:
    fig_dir = out_dir / "figures" / "structure"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    # Full distance matrix
    G_full = categorical_encode(geno, snp_cols, treat_m_as_missing=False)
    D_full = hamming_distance_matrix_excluding_missing(G_full, missing_value=-1)

    coords_full = classical_mds(D_full, n_components=2)

    group_map_df = pheno[["Accession_ID", "Acc Group"]].dropna(subset=["Accession_ID"]).copy()
    group_map_df["Acc Group"] = group_map_df["Acc Group"].astype("Int64").astype(str)
    group_map = dict(zip(group_map_df["Accession_ID"].astype(str).values, group_map_df["Acc Group"].values))
    groups = np.array([group_map.get(str(a), "NA") for a in geno["Accession_ID"].astype(str).values])
    group_codes, _ = pd.factorize(groups)

    results = []

    for name, markers in panels.items():
        G_panel = categorical_encode(geno, markers, treat_m_as_missing=False)
        D_panel = hamming_distance_matrix_excluding_missing(G_panel, missing_value=-1)
        coords_panel = classical_mds(D_panel, n_components=2)

        m_pear = mantel_test(D_full, D_panel, method="pearson", n_permutations=n_perm, seed=seed)
        m_spear = mantel_test(D_full, D_panel, method="spearman", n_permutations=n_perm, seed=seed)
        proc = procrustes_similarity(coords_full, coords_panel)
        nn10 = mean_neighbor_overlap(D_full, D_panel, k=10)

        results.append(
            {
                "panel": name,
                "k": int(len(markers)),
                "mantel_r_pearson": m_pear.r,
                "mantel_p_pearson": m_pear.p_value,
                "mantel_r_spearman": m_spear.r,
                "mantel_p_spearman": m_spear.p_value,
                "procrustes_similarity": proc,
                "mean_neighbor_overlap_k10": nn10,
            }
        )

        # MDS scatter
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(coords_panel[:, 0], coords_panel[:, 1], c=group_codes, s=12, alpha=0.8)
        ax.set_xlabel("MDS1")
        ax.set_ylabel("MDS2")
        ax.set_title(f"MDS (panel: {name})")
        save_figure(fig, fig_dir, f"mds_{name}")
        plt.close(fig)

    pd.DataFrame(results).to_csv(tab_dir / "structure_preservation_metrics.csv", index=False)

    # Pairwise distance correlation plot (full vs pure32 if available)
    if "PureID_32" in panels:
        G_p = categorical_encode(geno, panels["PureID_32"], treat_m_as_missing=False)
        D_p = hamming_distance_matrix_excluding_missing(G_p, missing_value=-1)
        v_full = D_full[np.triu_indices(D_full.shape[0], 1)]
        v_p = D_p[np.triu_indices(D_p.shape[0], 1)]

        rng = np.random.default_rng(seed)
        idx = rng.choice(len(v_full), size=min(50000, len(v_full)), replace=False)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(v_full[idx], v_p[idx], s=5, alpha=0.25)
        ax.set_xlabel("Full panel pairwise distance")
        ax.set_ylabel("32-marker pairwise distance")
        ax.set_title("Pairwise distances: full vs 32-marker panel")
        save_figure(fig, fig_dir, "pairwise_distance_scatter_full_vs_32")
        plt.close(fig)


def run_ld_and_entropy(
    geno: pd.DataFrame,
    panels: dict[str, list[str]],
    out_dir: Path,
) -> None:
    fig_dir = out_dir / "figures" / "ld_entropy"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    r2_frames = []

    for name, markers in panels.items():
        # binary encoding for LD
        from cacaocipher.encoding import binary_encode_major_minor
        enc = binary_encode_major_minor(geno, markers, treat_m_as_missing=True)
        ld = summarize_ld(enc.X)
        cw = codeword_entropy_bits(geno, markers)

        rows.append(
            {
                "panel": name,
                "k": int(len(markers)),
                "ld_r2_mean": ld.r2_mean,
                "ld_r2_median": ld.r2_median,
                "ld_r2_max": ld.r2_max,
                "ld_frac_r2_gt_0p2": ld.frac_r2_gt_0p2,
                "ld_frac_r2_gt_0p5": ld.frac_r2_gt_0p5,
                "ld_effective_rank_entropy": ld.effective_rank_entropy,
                "ld_effective_rank_ipr": ld.effective_rank_ipr,
                "codeword_n_unique": cw.n_unique_codewords,
                "codeword_log2_unique": cw.log2_unique,
                "codeword_entropy_bits": cw.shannon_entropy_bits,
                "sum_locus_entropy_bits": cw.total_locus_entropy_bits,
                "total_correlation_bits": cw.total_correlation_bits,
            }
        )

        # r^2 distribution
        from cacaocipher.ld_entropy import pairwise_r2_from_binary
        r2 = pairwise_r2_from_binary(enc.X)
        r2_frames.append(pd.DataFrame({"panel": name, "r2": r2}))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(r2, bins=30)
        ax.set_xlabel("Pairwise r^2 (binary major/minor)")
        ax.set_ylabel("Count")
        ax.set_title(f"LD distribution ({name})")
        save_figure(fig, fig_dir, f"ld_r2_hist_{name}")
        plt.close(fig)

    pd.DataFrame(rows).to_csv(tab_dir / "ld_and_entropy_summary.csv", index=False)

    if r2_frames:
        r2_all = pd.concat(r2_frames, ignore_index=True)
        r2_all.to_csv(tab_dir / "ld_pairwise_r2_values.csv", index=False)


def run_confounding_tests(
    geno: pd.DataFrame,
    pheno: pd.DataFrame,
    snp_cols: list[str],
    panels: dict[str, list[str]],
    out_dir: Path,
    *,
    seed: int,
    n_perm: int,
) -> None:
    fig_dir = out_dir / "figures" / "confounding"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    # Ancestry PCs from full panel
    pcs = compute_genotype_pca(geno, snp_cols, n_components=10, treat_m_as_missing=True, random_state=seed)
    pcs_df = pd.DataFrame(pcs, columns=[f"PC{i+1}" for i in range(pcs.shape[1])])
    pcs_df["Accession_ID"] = geno["Accession_ID"].values

    # Prepare phenotype subset
    meta = pheno[["Accession_ID", "Acc Group", "PodIndex", "FR_WABW"]].copy()
    meta = meta.merge(pcs_df, on="Accession_ID", how="inner")
    meta["Acc Group"] = meta["Acc Group"].astype("Int64").astype(str)

    results = []

    for panel_name, markers in panels.items():
        # MDS coords for panel
        G_panel = categorical_encode(geno, markers, treat_m_as_missing=False)
        D_panel = hamming_distance_matrix_excluding_missing(G_panel, missing_value=-1)
        coords = classical_mds(D_panel, n_components=2)
        coords_df = pd.DataFrame(coords, columns=["MDS1", "MDS2"])
        coords_df["Accession_ID"] = geno["Accession_ID"].values

        df = meta.merge(coords_df, on="Accession_ID", how="inner").dropna(subset=["MDS1"])

        strata = df["Acc Group"].values
        cov_pcs = df[[f"PC{i+1}" for i in range(10)]].values

        for trait in ["PodIndex", "FR_WABW"]:
            y = df[trait].values.astype(float)
            x = df["MDS1"].values.astype(float)

            # Unadjusted
            res0 = robust_ols(y, x, covariates=None)
            p0 = stratified_permutation_pvalue(y, x, strata=strata, covariates=None, n_permutations=n_perm, seed=seed)

            # Adjusted by PCs
            res1 = robust_ols(y, x, covariates=cov_pcs)
            p1 = stratified_permutation_pvalue(y, x, strata=strata, covariates=cov_pcs, n_permutations=n_perm, seed=seed + 1)

            results.append(
                {
                    "panel": panel_name,
                    "trait": trait,
                    "model": "unadjusted",
                    "n": res0.n,
                    "coef_mds1": res0.coef,
                    "se": res0.se,
                    "p_robust": res0.p,
                    "p_perm_stratified": p0,
                    "r2": res0.r2,
                }
            )
            results.append(
                {
                    "panel": panel_name,
                    "trait": trait,
                    "model": "adjusted_PCs10",
                    "n": res1.n,
                    "coef_mds1": res1.coef,
                    "se": res1.se,
                    "p_robust": res1.p,
                    "p_perm_stratified": p1,
                    "r2": res1.r2,
                }
            )

            # Within-group correlations
            wgc = within_group_correlations(df, group_col="Acc Group", x_col="MDS1", y_col=trait, min_n=10)
            wgc.to_csv(tab_dir / f"within_group_correlations_{panel_name}_{trait}.csv", index=False)

            # Scatter plot
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(df["MDS1"].values, y, s=14, alpha=0.7)
            ax.set_xlabel("MDS1")
            ax.set_ylabel(trait)
            ax.set_title(f"{trait} vs MDS1 ({panel_name})")
            save_figure(fig, fig_dir, f"scatter_{panel_name}_{trait}_vs_mds1")
            plt.close(fig)

    pd.DataFrame(results).to_csv(tab_dir / "trait_confounding_tests.csv", index=False)


def run_pr_validation(
    geno: pd.DataFrame,
    pheno: pd.DataFrame,
    pr_traits: pd.DataFrame | None,
    overlap: pd.DataFrame | None,
    panels: dict[str, list[str]],
    out_dir: Path,
    *,
    seed: int,
    n_perm: int,
) -> None:
    """
    Independent validation on PR_TARS traits for overlapping accessions, if available.
    """
    if pr_traits is None or overlap is None:
        return

    fig_dir = out_dir / "figures" / "pr_validation"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    # Map ICGT accession -> Acc Group
    gmap = (
        pheno[["Accession_ID", "Acc Group"]]
        .dropna(subset=["Accession_ID"])
        .assign(Accession_ID=lambda d: d["Accession_ID"].astype(str))
        .assign(Acc_Group=lambda d: d["Acc Group"].astype("Int64").astype(str))
        .set_index("Accession_ID")["Acc_Group"]
        .to_dict()
    )

    ov = overlap.copy()
    ov["Accession_ID_ICGT"] = ov["Accession_ID_ICGT"].astype(str)
    ov["Accession_ID_PR"] = ov["Accession_ID_PR"].astype(str)
    ov["Acc Group"] = ov["Accession_ID_ICGT"].map(lambda x: gmap.get(x, "NA"))

    pr = pr_traits.copy()
    pr["Accession_ID"] = pr["Accession_ID"].astype(str)

    results = []

    for panel_name, markers in panels.items():
        G_panel = categorical_encode(geno, markers, treat_m_as_missing=False)
        D_panel = hamming_distance_matrix_excluding_missing(G_panel, missing_value=-1)
        coords = classical_mds(D_panel, n_components=2)
        coords_df = pd.DataFrame(coords, columns=["MDS1", "MDS2"])
        coords_df["Accession_ID_ICGT"] = geno["Accession_ID"].astype(str).values

        df = ov.merge(coords_df, on="Accession_ID_ICGT", how="inner")
        df = df.merge(pr, left_on="Accession_ID_PR", right_on="Accession_ID", how="inner")

        if df.shape[0] < 10:
            continue

        strata = df["Acc Group"].astype(str).values
        x = df["MDS1"].values.astype(float)

        for trait in ["YIELD", "POD_INDEX", "TOTAL_PODS"]:
            if trait not in df.columns:
                continue
            y = df[trait].values.astype(float)
            res = robust_ols(y, x, covariates=None)
            p_perm = stratified_permutation_pvalue(y, x, strata=strata, covariates=None, n_permutations=n_perm, seed=seed)

            results.append(
                {
                    "panel": panel_name,
                    "trait": trait,
                    "n": res.n,
                    "coef_mds1": res.coef,
                    "se": res.se,
                    "p_robust": res.p,
                    "p_perm_stratified": p_perm,
                    "r2": res.r2,
                }
            )

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(x, y, s=22, alpha=0.8)
            ax.set_xlabel("MDS1")
            ax.set_ylabel(trait)
            ax.set_title(f"PR validation: {trait} vs MDS1 ({panel_name})")
            save_figure(fig, fig_dir, f"pr_scatter_{panel_name}_{trait}_vs_mds1")
            plt.close(fig)

    if results:
        pd.DataFrame(results).to_csv(tab_dir / "pr_validation_tests.csv", index=False)

def run_benchmarks(
    geno: pd.DataFrame,
    pheno: pd.DataFrame,
    snp_cols: list[str],
    fixed_panels: dict[str, list[str]],
    out_dir: Path,
    *,
    seed: int,
    pop_splits: int,
    pop_repeats: int,
    trait_splits: int,
) -> None:
    fig_dir = out_dir / "figures" / "benchmarks"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    # Population assignment benchmarks (32 markers, plus optional 96 baseline if provided)
    panels = []
    if "PureID_32" in fixed_panels:
        panels.append(PanelSpec(name="CacaoCipher_PureID", k=32, markers=fixed_panels["PureID_32"]))
    if "TraitAware_32" in fixed_panels:
        panels.append(PanelSpec(name="TraitAware_heuristic", k=32, markers=fixed_panels["TraitAware_32"]))

    panels += [
        PanelSpec(name="MI_pruned", k=32, selection=select_by_mutual_information),
        PanelSpec(name="L1_logistic_pruned", k=32, selection=select_by_l1_logistic),
        PanelSpec(name="RF_importance_pruned", k=32, selection=select_by_rf_importance),
        PanelSpec(name="Random_32", k=32, selection=select_random_panel),
    ]

    if "PureID_96" in fixed_panels:
        panels.append(PanelSpec(name="CacaoCipher_PureID_96", k=96, markers=fixed_panels["PureID_96"]))
        panels.append(PanelSpec(name="MI_pruned_96", k=96, selection=select_by_mutual_information))
        panels.append(PanelSpec(name="L1_logistic_pruned_96", k=96, selection=select_by_l1_logistic))
        panels.append(PanelSpec(name="RF_importance_pruned_96", k=96, selection=select_by_rf_importance))

    pop = evaluate_population_assignment(
        geno,
        pheno,
        snp_cols,
        panels=panels,
        n_splits=pop_splits,
        n_repeats=pop_repeats,
        seed=seed,
    )
    pop.to_csv(tab_dir / "benchmark_population_assignment_folds.csv", index=False)

    # Summary
    summary_rows = []
    for (panel, k), sub in pop.groupby(["panel", "k"]):
        mean_bacc, lo_bacc, hi_bacc = _summary_ci(sub["balanced_accuracy"].values)
        mean_acc, lo_acc, hi_acc = _summary_ci(sub["accuracy"].values)
        mean_f1, lo_f1, hi_f1 = _summary_ci(sub["macro_f1"].values)
        summary_rows.append(
            {
                "panel": panel,
                "k": int(k),
                "mean_balanced_accuracy": mean_bacc,
                "ci95_balanced_accuracy_lo": lo_bacc,
                "ci95_balanced_accuracy_hi": hi_bacc,
                "mean_accuracy": mean_acc,
                "ci95_accuracy_lo": lo_acc,
                "ci95_accuracy_hi": hi_acc,
                "mean_macro_f1": mean_f1,
                "ci95_macro_f1_lo": lo_f1,
                "ci95_macro_f1_hi": hi_f1,
                "n_folds": int(sub.shape[0]),
            }
        )
    pop_summary = pd.DataFrame(summary_rows).sort_values(["k", "mean_balanced_accuracy"], ascending=[True, False])
    pop_summary.to_csv(tab_dir / "benchmark_population_assignment_summary.csv", index=False)

    # Plot balanced accuracy distributions
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = []
    data = []
    for (panel, k), sub in pop.groupby(["panel", "k"]):
        labels.append(f"{panel} (k={k})")
        data.append(sub["balanced_accuracy"].values)
    ax.boxplot(data, labels=labels, vert=True, showfliers=False)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Population assignment benchmark (repeated CV)")
    ax.tick_params(axis="x", labelrotation=45)
    save_figure(fig, fig_dir, "benchmark_population_assignment_balanced_accuracy")
    plt.close(fig)

    # Trait prediction benchmarks (PodIndex by default)
    trait_panels = []
    if "PureID_32" in fixed_panels:
        trait_panels.append(PanelSpec(name="CacaoCipher_PureID", k=32, markers=fixed_panels["PureID_32"]))
    if "TraitAware_32" in fixed_panels:
        trait_panels.append(PanelSpec(name="TraitAware_heuristic", k=32, markers=fixed_panels["TraitAware_32"]))
    if "PureID_96" in fixed_panels:
        trait_panels.append(PanelSpec(name="CacaoCipher_PureID_96", k=96, markers=fixed_panels["PureID_96"]))
    trait_panels.append(PanelSpec(name="Random_32", k=32, selection=select_random_panel))

    pred = evaluate_trait_prediction(
        geno,
        pheno,
        snp_cols,
        panels=trait_panels,
        trait_col="PodIndex",
        n_splits=trait_splits,
        seed=seed,
    )
    pred.to_csv(tab_dir / "benchmark_trait_prediction_podindex_folds.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = []
    data = []
    for (panel, k), sub in pred.groupby(["panel", "k"]):
        labels.append(f"{panel} (k={k})")
        data.append(sub["r_pearson"].values)
    ax.boxplot(data, labels=labels, vert=True, showfliers=False)
    ax.set_ylabel("Pearson r (predicted vs observed)")
    ax.set_title("Trait prediction benchmark (PodIndex, CV)")
    ax.tick_params(axis="x", labelrotation=45)
    save_figure(fig, fig_dir, "benchmark_trait_prediction_podindex_r")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=str, default=".", help="Project root containing master/ directory.")
    ap.add_argument("--out-dir", type=str, default="addon_outputs", help="Output directory.")
    ap.add_argument("--master-dir", type=str, default="master", help="Directory containing master CSV files.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--mantel-permutations", type=int, default=999, help="Number of permutations for Mantel tests.")
    ap.add_argument("--confounding-permutations", type=int, default=5000, help="Number of permutations for stratified confounding tests.")
    ap.add_argument("--skip-structure", action="store_true", help="Skip structure preservation analyses.")
    ap.add_argument("--skip-ld-entropy", action="store_true", help="Skip LD/entropy analyses.")
    ap.add_argument("--skip-confounding", action="store_true", help="Skip ancestry-confounding tests.")
    ap.add_argument("--skip-benchmarks", action="store_true", help="Skip benchmarking analyses.")

    ap.add_argument("--benchmark-pop-splits", type=int, default=5, help="CV splits for population assignment benchmarks.")
    ap.add_argument("--benchmark-pop-repeats", type=int, default=5, help="CV repeats for population assignment benchmarks.")
    ap.add_argument("--benchmark-trait-splits", type=int, default=5, help="CV splits for trait prediction benchmarks.")
    args = ap.parse_args()

    project_root = resolve_project_root(args.project_root)
    master_dir = Path(args.master_dir)
    if not master_dir.is_absolute():
        master_dir = project_root / master_dir
    out_dir = ensure_output_dir(project_root / args.out_dir)

    print("[INFO] project_root:", project_root)
    print("[INFO] master_dir   :", master_dir)
    print("[INFO] out_dir      :", out_dir)

    tables = load_master_tables(master_dir)
    geno = tables.geno.copy()
    pheno = tables.pheno.copy()

    # Harmonize IDs
    ids = sorted(set(geno["Accession_ID"]) & set(pheno["Accession_ID"]))
    geno = geno[geno["Accession_ID"].isin(ids)].reset_index(drop=True)
    pheno = pheno[pheno["Accession_ID"].isin(ids)].reset_index(drop=True)

    snp_cols = get_snp_columns(geno)
    print("[INFO] n_samples:", len(ids))
    print("[INFO] n_snps_full:", len(snp_cols))

    # Fixed panels
    pure32 = tables.panel_32["marker"].astype(str).tolist()

    # Trait-aware panel regenerated (PodIndex)
    trait32 = select_trait_aware_panel(
        geno,
        pheno,
        snp_cols,
        trait_col="PodIndex",
        k=32,
        alpha=1.0,
        beta=1.0,
        max_missing=0.2,
        corr_thresh=0.8,
    )

    # 96-marker baseline (unsupervised)
    pure96 = select_pure_id_panel(geno, snp_cols, k=96, max_missing=0.2, corr_thresh=0.8)

    fixed_panels = {
        "PureID_32": pure32,
        "TraitAware_32": trait32,
        "PureID_96": pure96,
    }
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"marker": pure32}).to_csv(out_dir / "tables" / "panel_markers_pure32.csv", index=False)
    pd.DataFrame({"marker": trait32}).to_csv(out_dir / "tables" / "panel_markers_trait32.csv", index=False)
    pd.DataFrame({"marker": pure96}).to_csv(out_dir / "tables" / "panel_markers_pure96.csv", index=False)

    # Structure preservation: compare each panel to full
    if not args.skip_structure:
        run_structure_preservation(
            geno,
            pheno,
            snp_cols,
            panels={
                "PureID_32": pure32,
                "TraitAware_32": trait32,
                "PureID_96": pure96,
            },
            out_dir=out_dir,
            seed=args.seed,
            n_perm=args.mantel_permutations,
        )

    # LD and entropy summaries
    if not args.skip_ld_entropy:
        run_ld_and_entropy(
            geno,
            panels={
                "PureID_32": pure32,
                "TraitAware_32": trait32,
                "PureID_96": pure96,
            },
            out_dir=out_dir,
        )

    # Confounding tests for ICGT traits
    if not args.skip_confounding:
        run_confounding_tests(
            geno,
            pheno,
            snp_cols,
            panels={
                "PureID_32": pure32,
                "TraitAware_32": trait32,
            },
            out_dir=out_dir,
            seed=args.seed,
            n_perm=args.confounding_permutations,
        )
        run_pr_validation(
            geno,
            pheno,
            tables.pr_traits,
            tables.overlap,
            panels={
                "PureID_32": pure32,
                "TraitAware_32": trait32,
            },
            out_dir=out_dir,
            seed=args.seed,
            n_perm=args.confounding_permutations,
        )


    # Benchmarks vs alternative panel selection methods
    if not args.skip_benchmarks:
        run_benchmarks(
            geno,
            pheno,
            snp_cols,
            fixed_panels=fixed_panels,
            out_dir=out_dir,
            seed=args.seed,
            pop_splits=args.benchmark_pop_splits,
            pop_repeats=args.benchmark_pop_repeats,
            trait_splits=args.benchmark_trait_splits,
        )

    print("[DONE] All add-on analyses completed.")


if __name__ == "__main__":
    main()