"""F6: Phenotype paper table as a publication-grade table figure.

Renders the 15-row phenotype_paper_table.csv as a typeset matplotlib
table, with column-aware formatting (large numbers rounded, percentages
displayed as such, signature text truncated).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from predict.figures.style import PALETTE


COHORT_PALETTE = {
    "full":     PALETTE["neutral_blue"],
    "Qr36d/2":  PALETTE["neutral_orange"],
    "I30f/3":   PALETTE["neutral_green"],
}

CLUSTER_PALETTE = {
    "focal":   PALETTE["focal"],
    "diffuse": PALETTE["diffuse"],
    "low":     PALETTE["low"],
    "mid":     PALETTE["mid"],
    "high":    PALETTE["high"],
}


def _fmt(row, col):
    v = row.get(col)
    if pd.isna(v):
        return ""
    if col == "N":
        return f"{int(v)}"
    if col == "agatston_median":
        return f"{float(v):.0f}"
    if col == "agatston_iqr_lower":
        return f"[{float(v):.0f}-{float(row['agatston_iqr_upper']):.0f}]"
    if col in ("pct_qr36d_2", "pct_i30f_3", "pct_low_burden",
               "pct_cat_1_99", "pct_cat_100_399", "pct_cat_ge_400"):
        return f"{float(v):.0f}%"
    if col == "hennig_jaccard_median":
        if v is None or pd.isna(v):
            return "n/a"
        return f"{float(v):.2f}"
    if col == "top_signature_features":
        s = str(v)
        if not s:
            return "n/a"
        # Truncate at ~60 chars
        return s[:60] + ("..." if len(s) > 60 else "")
    return str(v)


def build(analyse_dir: Path) -> "plt.Figure":
    df = pd.read_csv(analyse_dir / "phenotype_paper_table.csv")

    display_cols = [
        ("cohort",                "Cohort"),
        ("partition",             "Partition"),
        ("cluster",               "Cluster"),
        ("N",                     "N"),
        ("agatston_median",       "Agatston med"),
        ("agatston_iqr_lower",    "[IQR]"),
        ("pct_qr36d_2",           "%Qr36d/2"),
        ("pct_low_burden",        "%low burden"),
        ("hennig_jaccard_median", "Hennig"),
    ]

    n_rows = len(df)
    fig_height = 0.35 * n_rows + 1.5
    fig, ax = plt.subplots(figsize=(13, fig_height))
    ax.axis("off")

    cell_text = []
    for _, row in df.iterrows():
        cell_text.append([_fmt(row, col_name) for col_name, _ in display_cols])
    col_labels = [label for _, label in display_cols]

    table = ax.table(
        cellText=cell_text, colLabels=col_labels,
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    # Header styling
    for col_idx in range(len(col_labels)):
        cell = table[0, col_idx]
        cell.set_facecolor("#404040")
        cell.set_text_props(color="white", weight="bold")
        cell.set_edgecolor("white")

    # Row styling: alternate background by partition; left bar by cluster
    cluster_idx_in_disp = [c for c, _ in display_cols].index("cluster")
    partition_idx = [c for c, _ in display_cols].index("partition")
    for row_idx in range(1, n_rows + 1):
        partition = df.iloc[row_idx - 1]["partition"]
        bg = "#F5F5F5" if partition == "spatial_k2" else "#FFFFFF"
        for col_idx in range(len(col_labels)):
            cell = table[row_idx, col_idx]
            cell.set_facecolor(bg)
            cell.set_edgecolor("#DDDDDD")
        # Cluster cell bg
        cluster_name = df.iloc[row_idx - 1]["cluster"]
        cluster_cell = table[row_idx, cluster_idx_in_disp]
        c = CLUSTER_PALETTE.get(cluster_name)
        if c is not None:
            cluster_cell.set_facecolor(c)
            cluster_cell.set_text_props(color="white", weight="bold")

    fig.suptitle(
        "Figure 6. Phenotype paper table (15 rows = 3 cohorts x 5 phenotype "
        "clusters)\n"
        "spatial_k2: focal vs diffuse;  burden_k3: low/mid/high Agatston tertiles",
        y=1.0, fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    return fig
