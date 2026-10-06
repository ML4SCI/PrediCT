"""F3: Spatial-only k=2 partition against patient burden (Finding 3).

Every number and verdict on the figure is read from the stage 6 and 7
files (Hennig stability, D024 interpretation); nothing is fixed text
(D044).

Three panels:
  (a) PCA on spatial-only feature subspace, points coloured by GMM-k=2
  (b) Box / violin of agatston_total per cluster, with Cliff's delta
  (c) Lesion-count / n_calcified_arteries per cluster (the structural
      biology that explains the burden split)
"""
from __future__ import annotations

from pathlib import Path

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from predict.analyse.profiles import determine_focal_diffuse_mapping
from predict.figures.style import PALETTE


SPATIAL_FEATURES = (
    "lesion_count_lad", "lesion_count_rca", "lesion_count_lcx", "lesion_count_lm",
    "lesion_count_total", "n_calcified_arteries", "gini_lesion_volume",
    "dist_from_top_max", "dist_from_top_mean", "center_of_mass_z",
    "inter_lesion_dist_mean_lad", "inter_lesion_dist_max_lad",
    "first_to_last_dist_lad",
)


# NOTE: the D023 focal/diffuse rule is NOT reimplemented here. A local copy
# can get the tie case wrong: on equal medians it would label whichever
# cluster iterated first as focal and the rest diffuse, silently and
# arbitrarily, producing a mislabelled figure with no error. `n_calcified_arteries` is an integer in {1,2,3,4}, so ties
# are ordinary. Use the canonical helper, which raises instead.


def build(reduce_dir: Path, analyse_dir: Path, features_csv: Path) -> "plt.Figure":
    prep = pd.read_csv(reduce_dir / "prepared_matrix.csv", dtype={"pid": str})
    spatial_cols = [c for c in SPATIAL_FEATURES if c in prep.columns]
    pid_order = prep["pid"].tolist()
    X = prep[spatial_cols].to_numpy(dtype=float)
    pca = PCA(n_components=2, random_state=42).fit(X)
    Z = pca.transform(X)

    labels_df = pd.read_csv(
        reduce_dir / "cluster_labels_spatial_k2.csv", dtype={"pid": str}
    ).set_index("pid")
    raw_labels = labels_df["spatial_only_gmm_k2"].astype(int).loc[pid_order]

    features = pd.read_csv(features_csv, dtype={"pid": str}).set_index("pid")
    n_calc = features.loc[pid_order, "n_calcified_arteries"]   # panel (c)
    focal_map = determine_focal_diffuse_mapping(
        features.loc[pid_order], raw_labels,
    )
    cluster_name = raw_labels.map(focal_map)

    meta = pd.read_csv(
        reduce_dir / "cohort_metadata.csv", dtype={"pid": str}
    ).set_index("pid")
    agatston = meta.loc[pid_order, "agatston_total"].astype(float)

    # Cliff's delta from burden_orthogonality.csv (full cohort row)
    bo = pd.read_csv(analyse_dir / "burden_orthogonality.csv")
    full_row = bo[bo["cohort"] == "full"].iloc[0]
    delta = float(full_row["cliffs_delta_agatston"])
    mw_p = float(full_row["mannwhitney_pval"])
    interp = str(full_row["interpretation"])

    # Stage 6 Hennig median Jaccard of the spatial k=2 clusters, by name.
    val = pd.read_csv(reduce_dir / "validity_checks.csv")
    hen = val[(val["test"] == "hennig_clusterboot") & (val["feature_space"] == "spatial_only")]
    jac = {focal_map[int(r.cluster_id)]: float(r.jaccard_median) for r in hen.itertuples()}
    if set(jac) != {"focal", "diffuse"}:
        raise ValueError(f"spatial-only Hennig rows do not cover both clusters: {jac}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ax_pca, ax_box, ax_struct = axes

    # (a) Spatial PCA scatter
    for name in ("focal", "diffuse"):
        m = (cluster_name == name).to_numpy()
        ax_pca.scatter(Z[m, 0], Z[m, 1],
                        c=PALETTE[name], s=22, alpha=0.7,
                        edgecolors="white", linewidths=0.4,
                        label=f"{name} (n={int(m.sum())})")
    ax_pca.set_xlabel(f"Spatial PC1 ({pca.explained_variance_ratio_[0]:.0%})")
    ax_pca.set_ylabel(f"Spatial PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    ax_pca.set_title("(a) Spatial-only PCA + GMM k=2\n"
                      f"(Hennig median Jaccard focal {jac['focal']:.2f} / "
                      f"diffuse {jac['diffuse']:.2f})")
    ax_pca.legend(loc="best")

    # (b) Box of agatston by cluster
    focal_burden = agatston[cluster_name == "focal"].dropna()
    diffuse_burden = agatston[cluster_name == "diffuse"].dropna()
    box = ax_box.boxplot(
        [focal_burden, diffuse_burden],
        labels=["focal", "diffuse"], widths=0.55, showfliers=False,
        patch_artist=True, medianprops={"color": "black", "linewidth": 1.4},
    )
    for patch, name in zip(box["boxes"], ("focal", "diffuse")):
        patch.set_facecolor(PALETTE[name]); patch.set_alpha(0.7)
    ax_box.set_ylabel("Agatston total (log scale)")
    ax_box.set_yscale("log")
    ax_box.set_title("(b) Burden distribution per cluster")
    ax_box.text(
        0.05, 0.95,
        f"Cliff's $\\delta$ = {delta:.3f}\nMann-Whitney p = {mw_p:.1e}\n"
        f"D024 interpretation: {interp}",
        transform=ax_box.transAxes, va="top", ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="lightgrey",
                  pad=5),
    )

    # (c) Structural biology: lesion_count_total + n_calcified_arteries
    lc_focal  = features.loc[pid_order, "lesion_count_total"][cluster_name == "focal"].dropna()
    lc_diff   = features.loc[pid_order, "lesion_count_total"][cluster_name == "diffuse"].dropna()
    nc_focal  = n_calc[cluster_name == "focal"].dropna()
    nc_diff   = n_calc[cluster_name == "diffuse"].dropna()

    positions = [1, 2, 4, 5]
    data = [lc_focal, lc_diff, nc_focal, nc_diff]
    cols = [PALETTE["focal"], PALETTE["diffuse"]] * 2
    box2 = ax_struct.boxplot(
        data, positions=positions, widths=0.6,
        showfliers=False, patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.4},
    )
    for p, c in zip(box2["boxes"], cols):
        p.set_facecolor(c); p.set_alpha(0.7)
    ax_struct.set_xticks([1.5, 4.5])
    ax_struct.set_xticklabels(["lesion count\n(total)", "vessels\ncalcified"])
    ax_struct.set_title("(c) Lesion count and calcified vessels\n"
                         "per cluster")
    # Custom legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=PALETTE["focal"], alpha=0.7),
        plt.Rectangle((0, 0), 1, 1, fc=PALETTE["diffuse"], alpha=0.7),
    ]
    ax_struct.legend(handles, ["focal", "diffuse"], loc="upper right")

    fig.suptitle(
        "Figure 3. Spatial-only k=2 partition and patient burden (full cohort)\n"
        f"D024 burden orthogonality: {interp} (Cliff's $\\delta$ = {delta:+.2f})",
        y=1.05, fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    return fig
