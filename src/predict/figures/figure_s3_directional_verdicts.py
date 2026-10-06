"""SF3: Directional hypothesis verdicts.

The 6 pre-registered D025 hypotheses (rows) x 3 cohorts (columns).
Cell colour indicates confirmed vs refuted vs missing. Cell text shows
the focal vs diffuse median pair.
"""
from __future__ import annotations

from pathlib import Path

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from predict.figures.style import PALETTE


def build(analyse_dir: Path) -> "plt.Figure":
    df = pd.read_csv(analyse_dir / "directional_hypotheses.csv")
    verdict = json.loads((analyse_dir / "directional_verdict.json").read_text())

    cohorts = ["full", "Qr36d/2", "I30f/3"]
    features = list(df[df["cohort"] == "full"]["feature"])

    fig, ax = plt.subplots(figsize=(11, 4.5))
    n_rows = len(features)
    n_cols = len(cohorts)
    cell_colours = np.empty((n_rows, n_cols), dtype=object)
    cell_text = np.empty((n_rows, n_cols), dtype=object)

    for i, feat in enumerate(features):
        for j, coh in enumerate(cohorts):
            row = df[(df["cohort"] == coh) & (df["feature"] == feat)]
            if len(row) == 0 or not bool(row.iloc[0]["feature_present"]):
                cell_colours[i, j] = PALETTE["ns"]
                cell_text[i, j] = "n/a"
                continue
            r = row.iloc[0]
            if coh == "full":
                # Use confirmed (which is direction_match AND FDR<0.05)
                if bool(r["confirmed"]):
                    cell_colours[i, j] = PALETTE["confirmed"]
                    cell_text[i, j] = "✓ confirmed"
                elif bool(r["direction_match"]):
                    cell_colours[i, j] = "#F4D03F"
                    cell_text[i, j] = "sign OK\nn.s."
                else:
                    cell_colours[i, j] = PALETTE["refuted"]
                    cell_text[i, j] = "✗ refuted"
            else:
                # Stratum: direction-only criterion
                if bool(r["direction_match"]):
                    cell_colours[i, j] = PALETTE["confirmed"]
                    cell_text[i, j] = "✓ dir"
                else:
                    cell_colours[i, j] = PALETTE["refuted"]
                    cell_text[i, j] = "✗ dir"

    for i in range(n_rows):
        for j in range(n_cols):
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                  facecolor=cell_colours[i, j],
                                  edgecolor="white", linewidth=2)
            ax.add_patch(rect)
            ax.text(j, i, cell_text[i, j], ha="center", va="center",
                    fontsize=9, color="white", weight="bold")

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(cohorts)
    ax.set_yticks(range(n_rows))
    pred_dirs = [df[df["feature"] == f].iloc[0]["predicted_direction"]
                  for f in features]
    ax.set_yticklabels([f"{f}\n({d})" for f, d in zip(features, pred_dirs)],
                        fontsize=8)
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)
    ax.set_title(
        "Supplementary Figure 3. D025 pre-registered directional hypotheses\n"
        f"Primary verdict: {verdict['overall_verdict'].upper()}: "
        f"{verdict['primary']['n_confirmed']} of 6 confirmed in the full cohort "
        f"(FDR p < 0.05); secondary {verdict['secondary']['qr36d_2_match_count']} "
        f"and {verdict['secondary']['i30f_3_match_count']} direction-matches "
        "in strata.",
        fontsize=11, fontweight="bold", loc="left",
    )
    fig.tight_layout()
    return fig
