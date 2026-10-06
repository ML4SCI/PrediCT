"""SF2: Monotonicity classification scatter.

Each cross-cohort robust feature positioned by its Spearman
rho against agatston_total, coloured by D026 classification
(burden / structure / spatial / mixed).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from predict.figures.style import PALETTE


CLASS_COLOURS = {
    "burden_tracking":    PALETTE["high"],
    "structure_tracking": PALETTE["neutral_blue"],
    "spatial_tracking":   PALETTE["focal"],
    "mixed":              "#999999",
}


def build(analyse_dir: Path) -> "plt.Figure":
    df = pd.read_csv(analyse_dir / "monotonicity_classification.csv")
    # Only the full cohort for the visual; the strata are a sensitivity probe
    full = df[df["cohort"] == "full"].copy()
    # Features without a correlation (D026.1) cannot be placed on the axis.
    n_not_eval = int((full["classification"] == "not_evaluable").sum())
    full = full[full["classification"] != "not_evaluable"]
    full = full.sort_values("spearman_rho")
    full["abs_rho"] = full["spearman_rho"].abs()

    fig, ax = plt.subplots(figsize=(8, max(6, 0.22 * len(full))))
    y_pos = np.arange(len(full))
    for cls in sorted(set(full["classification"])):
        m = (full["classification"] == cls).to_numpy()
        ax.scatter(
            full.loc[m, "spearman_rho"], y_pos[m],
            c=CLASS_COLOURS.get(cls, "black"), s=80, alpha=0.85,
            edgecolors="white", linewidths=0.5,
            label=cls.replace("_", " "),
        )
    ax.axvline(0.0, c="black", lw=0.5, alpha=0.4)
    ax.axvline(0.5, c=PALETTE["high"], lw=0.5, ls="--", alpha=0.5,
                label="|$\\rho$| = 0.5 (burden cutoff)")
    ax.axvline(-0.5, c=PALETTE["high"], lw=0.5, ls="--", alpha=0.5)
    ax.axvline(0.3, c=PALETTE["neutral_blue"], lw=0.5, ls=":", alpha=0.5,
                label="|$\\rho$| = 0.3 (structure / spatial cutoff)")
    ax.axvline(-0.3, c=PALETTE["neutral_blue"], lw=0.5, ls=":", alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(full["feature"], fontsize=7)
    ax.set_xlabel("Spearman $\\rho$ vs agatston_total")
    ax.set_title(
        f"Supplementary Figure 2. Monotonicity classification of the "
        f"{len(full)} robust features (full cohort)"
        + (f"; {n_not_eval} not evaluable" if n_not_eval else ""),
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(-1.0, 1.0)
    fig.tight_layout()
    return fig
