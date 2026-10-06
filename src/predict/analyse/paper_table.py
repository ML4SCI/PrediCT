"""Stage 7 publication-grade phenotype paper table (D028).

Two output tables:

1. **Main paper table** (15 rows): 3 cohorts x 5 phenotype clusters
   (2 spatial-k=2 + 3 burden-k=3) with N, agatston median + IQR, kernel
   composition, low_burden rate, category composition, top-3 signature
   features, and optional Hennig median Jaccard from stage 6.

2. **Robust sensitivity table** (5 rows): 1 cohort (full cohort restricted
   to low_burden_flag = False) x 5 phenotype clusters. Same schema as
   the main table.

The burden-k=3 partition is the agatston_total tertile partition computed
via ``pd.qcut(agatston_total, q=3, labels=['low','mid','high'])``, NOT
the stage-6 forced k=3 PC-space cluster labels. The tertile partition is
the conservative, distribution-defined choice that does not impose
structure beyond burden rank.

Decisions referencing this module:
    D028 - low-burden sensitivity output schema
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from predict.analyse.signatures import signature_paragraph_for_paper


# Standard Agatston categories (0, 1-99, 100-399, >= 400) as exhaustive,
# non-overlapping intervals for fractional scores: zero is exactly 0, the
# others are [lo, hi) with "1-99" covering (0, 100). Integer-style inclusive
# ranges would drop scores below 1 and between 99 and 100 (8 of 436
# patients).
AGATSTON_CATEGORIES: tuple[tuple[str, float, float], ...] = (
    ("zero", 0.0, 0.0),
    ("cat_1_99", 0.0, 100.0),
    ("cat_100_399", 100.0, 400.0),
    ("cat_ge_400", 400.0, float("inf")),
)


# ─────────────────────── helpers ───────────────────────


def _agatston_tertile_labels(agatston: pd.Series) -> pd.Series:
    """Assign burden tertile labels {low, mid, high} per patient.

    NaN patients propagate NaN labels.
    """
    return pd.qcut(
        agatston, q=3,
        labels=["low", "mid", "high"],
        duplicates="drop",
    )


def _category_percentages(agatston: pd.Series) -> dict:
    """Percentage of patients in each Agatston category. Every non-missing
    score falls in exactly one category; a negative score raises."""
    vals = agatston.dropna()
    n = int(vals.size)
    out: dict = {}
    if n == 0:
        for name, _, _ in AGATSTON_CATEGORIES:
            out[f"pct_{name}"] = float("nan")
        return out
    if (vals < 0).any():
        raise ValueError(f"negative Agatston score(s): {vals[vals < 0].tolist()[:5]}")
    assigned = pd.Series(0, index=vals.index)
    for name, lo, hi in AGATSTON_CATEGORIES:
        if name == "zero":
            mask = vals == 0
        elif name == "cat_1_99":
            mask = (vals > lo) & (vals < hi)
        else:
            mask = (vals >= lo) & (vals < hi)
        assigned += mask.astype(int)
        out[f"pct_{name}"] = float(100.0 * mask.sum() / n)
    if not (assigned == 1).all():
        raise AssertionError("Agatston categories are not exhaustive and disjoint")
    return out


def _build_row(
    cohort: str,
    partition: str,
    cluster: str,
    pids_in_cluster: list[str],
    cohort_meta: pd.DataFrame,
    signatures_df: pd.DataFrame,
    hennig_lookup: dict | None,
    n_signature_features: int = 3,
) -> dict:
    """One row of the paper table for a single (cohort, partition, cluster)
    tuple.
    """
    meta_in_cluster = cohort_meta.set_index("pid").loc[pids_in_cluster]
    agatston_vals = meta_in_cluster["agatston_total"].dropna()

    if len(agatston_vals) == 0:
        agat_median = float("nan")
        iqr_lo = iqr_hi = float("nan")
    else:
        agat_median = float(np.median(agatston_vals))
        iqr_lo = float(np.percentile(agatston_vals, 25))
        iqr_hi = float(np.percentile(agatston_vals, 75))

    # Kernel composition
    kernel_counts = meta_in_cluster["kernel"].value_counts(dropna=False)
    n_total = int(len(meta_in_cluster))
    if n_total == 0:
        pct_qr = pct_i30 = float("nan")
    else:
        pct_qr = 100.0 * float(kernel_counts.get("Qr36d/2", 0)) / n_total
        pct_i30 = 100.0 * float(kernel_counts.get("I30f/3", 0)) / n_total

    # Low burden flag rate
    if "low_burden_flag" in meta_in_cluster.columns:
        lb = meta_in_cluster["low_burden_flag"]
        # Handle bool / string forms
        lb_bool = lb.astype(str).str.lower().isin({"true", "1"})
        pct_low_burden = (
            100.0 * float(lb_bool.sum()) / n_total
            if n_total > 0 else float("nan")
        )
    else:
        pct_low_burden = float("nan")

    cat_pcts = _category_percentages(agatston_vals)

    signature_text = signature_paragraph_for_paper(
        signatures_df, cohort=cohort, partition=partition, cluster=cluster,
        n_features=n_signature_features,
    )

    hennig_value = (
        hennig_lookup.get((cohort, partition, cluster))
        if hennig_lookup is not None else None
    )

    row = {
        "cohort": cohort,
        "partition": partition,
        "cluster": cluster,
        "N": n_total,
        "agatston_median": agat_median,
        "agatston_iqr_lower": iqr_lo,
        "agatston_iqr_upper": iqr_hi,
        "pct_qr36d_2": pct_qr,
        "pct_i30f_3": pct_i30,
        "pct_low_burden": pct_low_burden,
        "top_signature_features": signature_text,
        "hennig_jaccard_median": (
            float(hennig_value) if hennig_value is not None
            else float("nan")
        ),
    }
    row.update(cat_pcts)
    return row


def _assemble_cohort_rows(
    cohort: str,
    cohort_meta: pd.DataFrame,
    spatial_labels: pd.Series,
    burden_tertile_labels: pd.Series,
    signatures_df: pd.DataFrame,
    hennig_lookup: dict | None,
) -> list[dict]:
    """Assemble the 5 phenotype rows for a single cohort."""
    rows: list[dict] = []
    # Spatial k=2 (focal, diffuse)
    for cluster_name in ("focal", "diffuse"):
        pids = spatial_labels.index[spatial_labels == cluster_name].tolist()
        rows.append(_build_row(
            cohort, "spatial_k2", cluster_name, pids,
            cohort_meta, signatures_df, hennig_lookup,
        ))
    # Burden k=3 (low, mid, high)
    for tertile in ("low", "mid", "high"):
        pids = burden_tertile_labels.index[
            burden_tertile_labels == tertile
        ].tolist()
        rows.append(_build_row(
            cohort, "burden_k3", tertile, pids,
            cohort_meta, signatures_df, hennig_lookup,
        ))
    return rows


# ─────────────────────── public API ───────────────────────


def build_paper_table(
    per_cohort_inputs: dict,
    hennig_lookup: dict | None = None,
) -> pd.DataFrame:
    """Build the 15-row main paper table.

    Parameters
    ----------
    per_cohort_inputs : dict mapping cohort label -> dict with keys
        ``cohort_metadata`` (DataFrame with pid + agatston_total + kernel +
        low_burden_flag + category), ``spatial_labels`` (Series of
        focal/diffuse indexed by pid), ``signatures`` (DataFrame from
        top_n_signatures).
    hennig_lookup : optional dict {(cohort, partition, cluster): jaccard}
        from the stage 6 validity_checks.csv.

    Returns
    -------
    15-row DataFrame.
    """
    all_rows: list[dict] = []
    for cohort, inputs in per_cohort_inputs.items():
        cohort_meta = inputs["cohort_metadata"]
        spatial_labels = inputs["spatial_labels"]
        burden_tertile_labels = _agatston_tertile_labels(
            cohort_meta.set_index("pid")["agatston_total"],
        )
        all_rows.extend(_assemble_cohort_rows(
            cohort, cohort_meta, spatial_labels,
            burden_tertile_labels, inputs["signatures"],
            hennig_lookup,
        ))
    return pd.DataFrame(all_rows)


def build_robust_sensitivity_table(
    full_cohort_inputs: dict,
    hennig_lookup: dict | None = None,
    robust_cohort_label: str = "robust",
    robust_signatures: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the 5-row robust-sensitivity table.

    Restricts the full cohort to patients with low_burden_flag == False
    and re-applies the same 5 phenotype rows. Spatial labels are restricted
    to the robust pid set; burden tertiles are recomputed within the
    restricted cohort to keep the tertile sizes balanced.

    Parameters
    ----------
    full_cohort_inputs : dict matching the structure of one entry in
        ``per_cohort_inputs`` from ``build_paper_table``.
    hennig_lookup : optional dict; if provided, the lookup keys for the
        robust cohort are (robust_cohort_label, partition, cluster).
    robust_cohort_label : the label string written into the ``cohort``
        column of the returned DataFrame.
    robust_signatures : signatures recomputed on the restricted patients,
        labelled ``robust_cohort_label`` (what stage 7 passes). Without it
        the full cohort's signatures are used, whose cohort label does not
        match, so the signature text is empty.

    Returns
    -------
    5-row DataFrame.
    """
    cohort_meta = full_cohort_inputs["cohort_metadata"].copy()
    spatial_labels = full_cohort_inputs["spatial_labels"].copy()
    signatures_df = (robust_signatures if robust_signatures is not None
                     else full_cohort_inputs["signatures"]).copy()

    # Patients with low_burden_flag == False
    lb_mask = cohort_meta["low_burden_flag"].astype(str).str.lower().isin(
        {"false", "0"}
    )
    robust_pids = cohort_meta.loc[lb_mask, "pid"].tolist()

    robust_meta = cohort_meta[cohort_meta["pid"].isin(robust_pids)].copy()
    robust_spatial = spatial_labels.loc[
        spatial_labels.index.intersection(robust_pids)
    ]
    # Recompute burden tertiles within the restricted set
    robust_burden_tertiles = _agatston_tertile_labels(
        robust_meta.set_index("pid")["agatston_total"],
    )

    rows = _assemble_cohort_rows(
        robust_cohort_label, robust_meta, robust_spatial,
        robust_burden_tertiles, signatures_df, hennig_lookup,
    )
    return pd.DataFrame(rows)
