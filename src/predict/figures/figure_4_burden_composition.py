"""F4: Lesion-cluster composition across burden tertiles (cross-sectional).

Shows the mean per-patient lesion fractions per Agatston tertile. The
k-means lesion clusters are grouped by the fixed median-threshold rule of
``predict.figures._lesion_classes`` (dense: max HU >= 500; soft: max HU
<= 230 and volume <= 15 mm3; moderate: the rest), NOT by the GMM k=3
partition (ARI 0.21 with these groups). The tertiles are different
patients, so the figure shows composition across burden levels, not a
time course (D044).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from predict.figures._lesion_classes import discover_broad_classes
from predict.figures.style import PALETTE


# Legend labels state the grouping rule itself.
CLASS_LABELS = {
    "soft_microspots": "soft (max HU <= 230, vol <= 15 mm$^3$)",
    "moderate_nodules": "moderate (other)",
    "dense_plaques": "dense (max HU >= 500)",
}


def build(lesion_morph_dir: Path) -> "plt.Figure":
    # Dynamically classify each cluster on THIS rerun's cluster_profiles
    # rather than relying on a hard-coded label -> class dict (k-means
    # labels are arbitrary across reruns).
    profiles = pd.read_csv(lesion_morph_dir / "cluster_profiles.csv")
    primary_to_broad = discover_broad_classes(profiles)

    by_tert = pd.read_csv(
        lesion_morph_dir / "mixture_by_burden_tertile.csv"
    )
    # Index on the tertile column (first col)
    tert_col = by_tert.columns[0]
    by_tert = by_tert.set_index(tert_col)
    # Each column is frac_C0, frac_C1, ..., frac_C11
    frac_cols = sorted(
        [c for c in by_tert.columns if c.startswith("frac_C")],
        key=lambda x: int(x.replace("frac_C", "")),
    )
    # Aggregate to broad class
    broad_sum = pd.DataFrame(index=by_tert.index)
    for cls in ("soft_microspots", "moderate_nodules", "dense_plaques"):
        member_cols = [f"frac_C{c}" for c, b in primary_to_broad.items()
                       if b == cls and f"frac_C{c}" in by_tert.columns]
        if member_cols:
            broad_sum[cls] = by_tert[member_cols].sum(axis=1)

    # JT p-values
    jt = pd.read_csv(lesion_morph_dir / "finalise"
                     / "jonckheere_terpstra_trends.csv")
    jt = jt.set_index("cluster")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_stack, ax_indiv = axes

    # (a) Stacked bar by broad class
    tertiles = list(broad_sum.index)
    bottom = np.zeros(len(tertiles))
    for cls in ("soft_microspots", "moderate_nodules", "dense_plaques"):
        if cls not in broad_sum.columns:
            continue
        vals = broad_sum[cls].to_numpy()
        ax_stack.bar(
            tertiles, vals, bottom=bottom,
            color=PALETTE[cls], edgecolor="white", linewidth=1,
            label=CLASS_LABELS[cls],
        )
        for i, (t, v, b) in enumerate(zip(tertiles, vals, bottom)):
            if v > 0.05:
                ax_stack.text(
                    i, b + v / 2, f"{v:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="black", weight="bold",
                )
        bottom += vals
    ax_stack.set_xlabel("Burden tertile (Agatston-total qcut)")
    ax_stack.set_ylabel("Mean per-patient lesion fraction")
    ax_stack.set_title("(a) Lesion composition by burden tertile\n"
                        "(k-means clusters grouped by median thresholds)")
    ax_stack.legend(loc="upper right")
    ax_stack.set_ylim(0, 1.0)

    # (b) Per-cluster fraction by tertile with JT trend p-values
    colours_for_class = {
        "soft_microspots": PALETTE["soft_microspots"],
        "moderate_nodules": PALETTE["moderate_nodules"],
        "dense_plaques": PALETTE["dense_plaques"],
    }
    for cid in sorted(primary_to_broad.keys()):
        col = f"frac_C{cid}"
        if col not in by_tert.columns:
            continue
        vals = by_tert[col].to_numpy()
        broad = primary_to_broad.get(cid, "moderate_nodules")
        colour = colours_for_class.get(broad, "grey")
        jt_p = jt.loc[cid, "JT_p_two_sided"] if cid in jt.index else float("nan")
        sig = "***" if (jt_p < 0.001 and not np.isnan(jt_p)) else ""
        lw = 2.0 if sig else 0.9
        alpha = 0.95 if sig else 0.45
        ax_indiv.plot(tertiles, vals, "o-", color=colour, linewidth=lw,
                       alpha=alpha, markersize=5,
                       label=f"C{cid}{sig}")
    ax_indiv.set_xlabel("Burden tertile")
    ax_indiv.set_ylabel("Mean fraction")
    ax_indiv.set_title("(b) Per-cluster mean fraction by burden tertile\n"
                        "(*** = Jonckheere-Terpstra trend p < 0.001)")
    ax_indiv.legend(loc="upper right", fontsize=7, ncol=2)

    fig.suptitle(
        "Figure 4. Lesion composition across Agatston tertiles "
        "(cross-sectional; different patients per tertile)",
        y=1.04, fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    return fig
