from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> tuple[Path, Path]:
    """
    Save a figure as vector PDF and 300 dpi PNG.

    Returns (pdf_path, png_path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    return pdf_path, png_path
