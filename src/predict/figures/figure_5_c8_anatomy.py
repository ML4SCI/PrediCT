"""F5: The C8-like lesion cluster: vessel distribution, position within the
RCA, and patient burden.

Every panel reads the stage 9 results for this run's C8-like cluster
(``finalise/c8_deep_dive.json`` and ``finalise/c8_rca_relative_z.csv``), so
the figure and the stage 9 analysis use one comparison: within the C8
carriers, their C8 RCA lesions against their other RCA lesions (D044).

Position is reported as the superior-inferior coordinate, relative within
each carrier's RCA lesions: (z - min z) / (max z - min z), 0 = the lowest
(most inferior) RCA lesion, 1 = the highest, with +z superior (DICOM).
Height alone does not establish proximal or distal position along the
curving RCA, so no proximal / distal claim is made. If no
cluster matches the C8 signature, the figure fails instead of picking the
most RCA-biased cluster.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from predict.figures._lesion_classes import discover_c8_like_cluster
from predict.figures.style import PALETTE


def build(lesion_morph_dir: Path) -> "plt.Figure":
    vc = pd.read_csv(lesion_morph_dir / "cluster_vessel_chi_square.csv")
    profiles = pd.read_csv(lesion_morph_dir / "cluster_profiles.csv")
    c8_id = discover_c8_like_cluster(profiles, vc)
    if c8_id is None:
        raise ValueError("no cluster matches the C8-like signature in this run")
    deep = json.loads((lesion_morph_dir / "finalise" / "c8_deep_dive.json").read_text())
    if int(deep["c8_like_cluster_id"]) != c8_id:
        raise ValueError(f"stage 9 C8 cluster {deep['c8_like_cluster_id']} != discovered {c8_id}")
    c8_row = vc[vc["cluster"] == c8_id].iloc[0]

    rel = pd.read_csv(lesion_morph_dir / "finalise" / "c8_rca_relative_z.csv", dtype={"pid": str})
    c8_rel_z = rel.loc[rel["in_c8"].astype(bool), "relative_z_within_RCA"].to_numpy()
    other_rel_z = rel.loc[~rel["in_c8"].astype(bool), "relative_z_within_RCA"].to_numpy()
    n_carriers_rca = int(rel["pid"].nunique())

    vessels = ("LAD", "RCA", "LCx", "LM")
    obs_over_exp = [float(c8_row[f"{v.lower()}_obs_over_exp"]) for v in vessels]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ax_vessel, ax_z, ax_burden = axes

    # (a) Vessel observed/expected ratio
    colours = [PALETTE[v] for v in vessels]
    bars = ax_vessel.bar(vessels, obs_over_exp,
                         color=colours, edgecolor="white", linewidth=1)
    ax_vessel.axhline(1.0, c="black", lw=0.7, linestyle="--",
                      alpha=0.7, label="cohort baseline")
    for v, r, bar in zip(vessels, obs_over_exp, bars):
        label = "0 lesions" if r == 0 else f"{r:.2f}"
        ax_vessel.text(bar.get_x() + bar.get_width() / 2, r + 0.05, label,
                       ha="center", va="bottom", fontsize=9)
    ax_vessel.set_ylabel("observed / expected ratio")
    ax_vessel.set_title(
        f"(a) Vessel distribution\n"
        f"cluster id (this run) = {c8_id}, n = {int(c8_row['n_lesions'])}\n"
        f"$\\chi^2$ p = {c8_row['chi2_p']:.1e};  "
        f"Cramer's V = {c8_row['cramers_v']:.2f}"
    )
    ax_vessel.set_ylim(0, max(2.5, max(obs_over_exp) + 0.4))
    ax_vessel.legend(loc="upper left")

    # (b) Relative superior-inferior position within the carrier's RCA lesions
    bins = np.linspace(0, 1, 21)
    if c8_rel_z.size > 0 and other_rel_z.size > 0:
        ax_z.hist(other_rel_z, bins=bins, alpha=0.55, color="#888888",
                  label="other RCA lesions (same carriers)",
                  edgecolor="white", linewidth=0.4, density=True)
        ax_z.hist(c8_rel_z, bins=bins, alpha=0.85,
                  color=PALETTE["refuted"], label="C8-like RCA lesions",
                  edgecolor="white", linewidth=0.4, density=True)
        p = deep.get("c8_vs_other_RCA_z_mw_p_two_sided", float("nan"))
        ax_z.set_title(
            f"(b) Position within the carrier's RCA lesions ({n_carriers_rca} carriers)\n"
            f"median {np.median(c8_rel_z):.2f} (C8-like) vs {np.median(other_rel_z):.2f}; "
            f"MW p = {p:.1e}"
        )
        ax_z.legend(loc="upper center")
    else:
        ax_z.text(0.5, 0.5, "too few RCA lesions in C8 carriers",
                  ha="center", va="center", transform=ax_z.transAxes)
    ax_z.set_xlabel("relative z within the patient's RCA lesions\n"
                    "(0 = lowest / most inferior, 1 = highest; +z superior)")
    ax_z.set_ylabel("density")

    # (c) Burden of C8-like carriers vs other patients (stage 9 values)
    meds = [float(deep["non_c8_patient_burden_median"]), float(deep["c8_patient_burden_median"])]
    ax_burden.barh(["other patients", "C8-like carriers"], meds,
                   color=[PALETTE["neutral_blue"], PALETTE["refuted"]],
                   edgecolor="white", linewidth=1)
    for i, val in enumerate(meds):
        ax_burden.text(val + max(meds) * 0.03, i, f"{val:.0f}", va="center", ha="left", fontsize=9)
    ax_burden.set_xlabel("median Agatston total")
    ax_burden.set_title(
        f"(c) Patient burden\n"
        f"Cliff's $\\delta$ = {float(deep['cliffs_delta_burden_c8_vs_non']):.2f}; "
        f"MW p = {float(deep['mannwhitney_p_one_sided_c8_higher']):.1e}\n"
        f"(N carriers = {int(deep['n_c8_patients'])})"
    )

    fig.suptitle(
        f"Figure 5. The C8-like lesion cluster (C{c8_id}): vessel distribution, "
        "superior-inferior position within the RCA,\n"
        "and patient burden (cluster identified by its morphology signature "
        "in this run, not by label)",
        y=1.05, fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    return fig
