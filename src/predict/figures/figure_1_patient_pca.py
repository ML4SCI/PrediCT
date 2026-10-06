"""F1: Patient PCA scatter, coloured by Agatston tertile.

Descriptive: shows where the burden tertiles lie on the first two
principal components. It does not establish a continuum or discrete
groups (D021.1, D036.1, D038; D044).

Two panels:
  (a) PC1 vs PC2 scatter, coloured by qcut burden tertile
  (b) Histogram of PC1 stacked by tertile
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from predict.figures.style import PALETTE


def build(reduce_dir: Path) -> "plt.Figure":
    pca_scores = np.load(reduce_dir / "pca_scores.npy")
    pid_order = pd.read_csv(
        reduce_dir / "pca_scores_pid_order.csv", dtype={"pid": str}
    )["pid"].tolist()
    meta = pd.read_csv(
        reduce_dir / "cohort_metadata.csv", dtype={"pid": str}
    ).set_index("pid")
    agatston = meta.loc[pid_order, "agatston_total"].astype(float)
    tertile = pd.qcut(agatston, q=3,
                       labels=["low", "mid", "high"],
                       duplicates="drop")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5),
                              gridspec_kw={"width_ratios": [1.5, 1]})
    ax_scatter, ax_hist = axes

    # (a) PC1 vs PC2 scatter
    for t in ("low", "mid", "high"):
        m = (tertile == t).to_numpy()
        ax_scatter.scatter(
            pca_scores[m, 0], pca_scores[m, 1],
            c=PALETTE[t], s=22, alpha=0.65, edgecolors="white", linewidths=0.4,
            label=f"{t} burden (n={int(m.sum())})",
        )
    ax_scatter.set_xlabel("PC1")
    ax_scatter.set_ylabel("PC2")
    ax_scatter.set_title("(a) Patient PCA on representative features\n"
                          "coloured by Agatston tertile")
    ax_scatter.legend(loc="upper right", title="Burden")
    ax_scatter.axhline(0, c="black", lw=0.4, alpha=0.4)
    ax_scatter.axvline(0, c="black", lw=0.4, alpha=0.4)

    # (b) PC1 marginal stacked histogram by tertile
    bins = np.linspace(pca_scores[:, 0].min() - 0.5,
                        pca_scores[:, 0].max() + 0.5, 30)
    pc1_by_tertile = [pca_scores[(tertile == t).to_numpy(), 0]
                       for t in ("low", "mid", "high")]
    ax_hist.hist(
        pc1_by_tertile, bins=bins, stacked=True,
        color=[PALETTE["low"], PALETTE["mid"], PALETTE["high"]],
        label=["low", "mid", "high"], alpha=0.85, edgecolor="white",
        linewidth=0.3,
    )
    ax_hist.set_xlabel("PC1 score")
    ax_hist.set_ylabel("patients")
    ax_hist.set_title("(b) PC1 distribution\n"
                       "stacked by burden tertile")
    ax_hist.legend(loc="upper right", title="Burden")

    fig.suptitle(
        f"Figure 1. Patient principal components (N = {len(pid_order)}) "
        "coloured by Agatston tertile",
        y=1.03, fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    return fig
