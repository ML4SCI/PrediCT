"""SF1: Robust cross-cohort discriminator heatmap.

132 features that pass the D027 three-rule criterion, shown as a heatmap
of signed Cliff's delta across (partition, cluster) combinations. Rows
sorted by overall effect-size magnitude.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build(analyse_dir: Path) -> "plt.Figure":
    df = pd.read_csv(analyse_dir / "cross_cohort_feature_consistency.csv")
    robust = df[df["robust_discriminator"]].copy()
    if len(robust) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no robust discriminators", ha="center", va="center")
        return fig

    # Build pivot: rows = feature, cols = (partition, cluster); values =
    # signed delta_full (use full cohort as the visualised effect)
    robust["signed_delta_full"] = robust["sign_full"] * robust["delta_full"]
    pivot = robust.pivot_table(
        index="feature",
        columns=["partition", "cluster"],
        values="signed_delta_full",
    )
    # Sort rows by mean |delta|
    abs_mean = pivot.abs().mean(axis=1).sort_values(ascending=False)
    pivot = pivot.loc[abs_mean.index]

    fig, ax = plt.subplots(figsize=(11, max(6, 0.16 * len(pivot))))
    im = ax.imshow(
        pivot.to_numpy(), cmap="RdBu_r", aspect="auto",
        vmin=-0.6, vmax=0.6, interpolation="nearest",
    )
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=6)
    col_labels = [f"{p}\n{c}" for p, c in pivot.columns]
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("(partition, cluster)")
    cb = fig.colorbar(im, ax=ax, shrink=0.7, label="signed Cliff's $\\delta$\n"
                                                     "(direction-of-effect)")
    ax.set_title(f"Supplementary Figure 1. {len(pivot)} cross-cohort robust "
                  f"discriminators\n(D027 3-rule criterion: direction "
                  f"consistent + significance in $\\geq$2 of 3 + "
                  f"$|\\delta| \\geq 0.20$ in all 3)",
                  fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig
