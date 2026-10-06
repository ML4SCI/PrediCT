"""F2: Gap-statistic curves grid (3 algorithms x 3 feature spaces).

Descriptive only (D021.1): the curves, their bootstrap SE and the selected
k are shown as outputs of the specified procedure. Selected k, including
k = 1 or selection near the upper boundary, is not evidence establishing
or excluding discrete subtypes; the gap arms failed the planted-group
positive control on this geometry (D036.1). A monotone curve is not read
as the signature of a continuum either (D044).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from predict.figures.style import PALETTE


ALGORITHM_COLOURS = {
    "kmeans": PALETTE["neutral_blue"],
    "ward":   PALETTE["neutral_orange"],
    "gmm":    PALETTE["neutral_green"],
}

SPACES = ("full", "burden_residualised", "spatial_only")
ALGORITHMS = ("kmeans", "ward", "gmm")


def build(reduce_dir: Path) -> "plt.Figure":
    gap_records = json.loads((reduce_dir / "gap_statistic.json").read_text())
    # Index by (space, algo) -> record
    by_key = {(r["feature_space"], r["algorithm"]): r for r in gap_records}

    fig, axes = plt.subplots(
        len(SPACES), len(ALGORITHMS),
        figsize=(11, 9), sharex=True, sharey=False,
    )

    for i, space in enumerate(SPACES):
        for j, algo in enumerate(ALGORITHMS):
            ax = axes[i, j]
            rec = by_key.get((space, algo))
            if rec is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes)
                continue
            k = np.array(rec["k_range"])
            gap = np.array(rec["gap_values"])
            sk = np.array(rec["sk_values"])
            sel_k = rec["selected_k"]
            colour = ALGORITHM_COLOURS[algo]
            ax.errorbar(k, gap, yerr=sk, fmt="o-",
                        color=colour, markersize=4, capsize=2, linewidth=1.2,
                        ecolor=colour, alpha=0.95)
            # Mark selected k
            sel_idx = list(k).index(sel_k)
            ax.scatter([sel_k], [gap[sel_idx]], s=80, marker="*",
                       facecolor="white", edgecolor=colour, linewidth=1.5,
                       zorder=10,
                       label=f"selected k={sel_k}")
            # Annotate "boundary hit" if selected_k == k_max
            if sel_k == int(k.max()):
                ax.annotate("boundary",
                            xy=(sel_k, gap[sel_idx]),
                            xytext=(sel_k - 1, gap[sel_idx] - 0.15),
                            fontsize=8, color=PALETTE["refuted"],
                            ha="right")
            ax.legend(loc="lower right")
            if i == 0:
                ax.set_title(algo, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{space}\ngap value")
            if i == len(SPACES) - 1:
                ax.set_xlabel("k")
            ax.set_xticks(k[::2])

    fig.suptitle(
        "Figure 2. Gap statistic per algorithm and feature space (full cohort; "
        "error bars = SE, star = selected k)\n"
        "Descriptive output of the procedure: the selected k is not evidence "
        "for or against discrete subtypes (D021.1).",
        y=1.02, fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    return fig
