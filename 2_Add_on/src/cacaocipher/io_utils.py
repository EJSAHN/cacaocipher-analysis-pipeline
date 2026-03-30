from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class MasterTables:
    geno: pd.DataFrame
    pheno: pd.DataFrame
    panel_32: pd.DataFrame
    pr_traits: Optional[pd.DataFrame]
    overlap: Optional[pd.DataFrame]


def resolve_project_root(project_root: str | Path) -> Path:
    p = Path(project_root).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Project root does not exist: {p}")
    return p


def load_master_tables(master_dir: str | Path) -> MasterTables:
    master_dir = Path(master_dir).expanduser().resolve()
    if not master_dir.exists():
        raise FileNotFoundError(f"master_dir does not exist: {master_dir}")

    geno_fp = master_dir / "icgt_geno_master.csv"
    pheno_fp = master_dir / "icgt_pheno_master.csv"
    panel_fp = master_dir / "barcode_panel_32_stats.csv"
    pr_fp = master_dir / "pr_traits_master.csv"
    overlap_fp = master_dir / "accession_overlap_auto.csv"

    geno = pd.read_csv(geno_fp)
    pheno = pd.read_csv(pheno_fp)
    panel_32 = pd.read_csv(panel_fp)

    pr_traits = pd.read_csv(pr_fp) if pr_fp.exists() else None
    overlap = pd.read_csv(overlap_fp) if overlap_fp.exists() else None

    return MasterTables(geno=geno, pheno=pheno, panel_32=panel_32, pr_traits=pr_traits, overlap=overlap)


def get_snp_columns(geno_df: pd.DataFrame) -> list[str]:
    meta_cols = {"Accession_ID", "Clone_core", "Clone_label", "Dataset", "Acc Group"}
    return [c for c in geno_df.columns if c not in meta_cols]


def ensure_output_dir(out_dir: str | Path) -> Path:
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
