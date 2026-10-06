"""SF4: Hennig clusterboot stability across cohorts.

Median + mean bootstrap Jaccard per cluster for the spatial-only k=2
partition, shown across all three cohorts. The 0.75 stable threshold
is drawn for reference.

(The pipeline records median + mean only, not raw bootstrap samples,
so we show those as paired bars rather than a true boxplot.)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from predict.figures.style import PALETTE


COHORT_DIRS = {
    "full":    "",
    "Qr36d/2": "stratified_Qr36d_2",
    "I30f/3":  "stratified_I30f_3",
}


def build(reduce_dir: Path) -> "plt.Figure":
    rows = []
    for coh_name, sub in COHORT_DIRS.items():
        cdir = reduce_dir if sub == "" else reduce_dir / sub
        v = pd.read_csv(cdir / "validity_checks.csv")
        spat = v[(v["test"] == "hennig_clusterboot")
                 & (v["feature_space"] == "spatial_only")]
        for _, r in spat.iterrows():
            rows.append({
                "cohort": coh_name,
                "cluster_id": int(r["cluster_id"]),
                "median": float(r["jaccard_median"]),
                "mean":   float(r["jaccard_mean"]),
            })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    cohort_order = ["full", "Qr36d/2", "I30f/3"]
    width = 0.32
    x = np.arange(len(cohort_order))
    for cid, marker in zip(sorted(df["cluster_id"].unique()), ("o", "s")):
        med = [df.loc[(df["cohort"] == c) & (df["cluster_id"] == cid),
                       "median"].iloc[0]
               for c in cohort_order]
        mean = [df.loc[(df["cohort"] == c) & (df["cluster_id"] == cid),
                        "mean"].iloc[0]
                for c in cohort_order]
        offset = -width / 2 if cid == 0 else width / 2
        c = PALETTE["focal"] if cid == 0 else PALETTE["diffuse"]
        ax.bar(x + offset, med, width=width, color=c, alpha=0.85,
                edgecolor="white", linewidth=1.2,
                label=f"cluster {cid} (median)")
        ax.scatter(x + offset, mean, marker=marker, s=70,
                    c="black", zorder=10, label=f"cluster {cid} (mean)")

    ax.axhline(0.75, c=PALETTE["high"], lw=1, ls="--",
                label="Hennig stable threshold (0.75)")
    ax.set_xticks(x)
    ax.set_xticklabels(cohort_order)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Hennig bootstrap Jaccard")
    ax.set_title(
        "Supplementary Figure 4. Hennig clusterboot stability of the\n"
        "spatial-only x GMM x k=2 partition (Finding 3 reproducibility)",
        fontsize=11, fontweight="bold",
    )
    handles, labels = ax.get_legend_handles_labels()
    # De-duplicate
    seen = set()
    handles_uniq, labels_uniq = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            handles_uniq.append(h); labels_uniq.append(l); seen.add(l)
    ax.legend(handles_uniq, labels_uniq, loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig
