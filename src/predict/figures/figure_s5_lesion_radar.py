"""SF5: Lesion-cluster radar (spider) chart.

Each of the 12 lesion morphology clusters as a radar over 6 normalised
morphology axes (volume, area, mean_hu, max_hu, n_rois, log10
(volume/area)). One axis per spoke; one polygon per cluster, coloured
by broad GMM-k=3 class.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from predict.figures._lesion_classes import discover_broad_classes
from predict.figures.style import PALETTE


def build(lesion_morph_dir: Path) -> "plt.Figure":
    prof = pd.read_csv(lesion_morph_dir / "cluster_profiles.csv")
    # Dynamically classify on THIS rerun's medians; k-means labels are
    # arbitrary across reruns so a hard-coded label->class dict is unsafe.
    primary_to_broad = discover_broad_classes(prof)
    axes_features = [
        ("volume_mm3_median",       "log volume"),
        ("total_area_mm2_median",   "log area"),
        ("mean_hu_weighted_median", "mean HU"),
        ("max_hu_median",           "max HU"),
        ("n_rois_median",           "log n_rois"),
    ]
    # Apply log10 to size axes for better visual spread
    for col, _ in axes_features:
        if col.startswith(("volume", "total_area", "n_rois")):
            prof[col + "_log"] = np.log10(prof[col].clip(lower=0.1))

    # Build axis values (use _log for size/count)
    axis_cols = []
    for col, label in axes_features:
        if col.startswith(("volume", "total_area", "n_rois")):
            axis_cols.append((col + "_log", "log10 " + label.replace("log ", "")))
        else:
            axis_cols.append((col, label))

    # Min-max normalise each axis across clusters to [0, 1]
    normed = pd.DataFrame()
    for col, _ in axis_cols:
        v = prof[col].astype(float)
        normed[col] = (v - v.min()) / max(v.max() - v.min(), 1e-9)

    n_axes = len(axis_cols)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(
        3, 4, figsize=(13, 10),
        subplot_kw=dict(polar=True),
    )

    for i, cluster_id in enumerate(sorted(prof["cluster"].astype(int).tolist())):
        ax = axes.flat[i]
        vals = normed.loc[prof["cluster"] == cluster_id].iloc[0].tolist()
        vals += vals[:1]
        broad = primary_to_broad.get(cluster_id, "moderate_nodules")
        col = PALETTE[broad]
        ax.plot(angles, vals, color=col, linewidth=1.6, alpha=0.95)
        ax.fill(angles, vals, color=col, alpha=0.35)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([label for _, label in axis_cols], fontsize=7)
        ax.set_yticklabels([])
        ax.set_ylim(0, 1)
        n = int(prof.loc[prof["cluster"] == cluster_id, "n_lesions"].iloc[0])
        ax.set_title(f"C{cluster_id}  (n={n})\n{broad}",
                      fontsize=9, fontweight="bold", pad=12)

    fig.suptitle(
        "Supplementary Figure 5. Per-cluster morphology profile\n"
        "(min-max normalised across clusters; broad class shaded by colour)",
        y=1.02, fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    return fig
