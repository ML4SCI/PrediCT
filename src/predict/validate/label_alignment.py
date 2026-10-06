"""Stage 8 shared helper: focal/diffuse mapping (D023) for raw GMM labels.

The spatial-only k=2 GMM produces labels in ``{0, 1}`` whose identity is
arbitrary (depends on init seed + per-fold refit data). Per D023, the
canonical mapping is::

    focal   = cluster with LOWER median n_calcified_arteries
    diffuse = cluster with HIGHER median n_calcified_arteries

This module wraps the stage-7 helper
``predict.analyse.profiles.determine_focal_diffuse_mapping`` and adds
convenience functions used by ``predict.validate.{external_holdout,
leave_k_out}``:

* :func:`canonical_numeric_labels`: relabel raw GMM ``{0, 1}`` so that
  ``0 = focal`` and ``1 = diffuse``. Used for cross-fold comparison
  where a stable numeric encoding is needed.
* :func:`canonical_string_labels`: same mapping but emits string
  labels (``"focal" / "diffuse"``) for human-readable / paper outputs.
* :func:`apply_mapping`: apply an arbitrary ``{old_id: new_label}`` dict
  to a label series; idempotent if already in canonical form.

D030.5 explicitly states ARI is computed on RAW GMM labels (permutation
invariant), so this module is for INTERPRETIVE columns only.

Tie handling: D023's underlying helper raises on identical medians.
This module adds an optional tie-breaker (lower MEAN n_calcified_arteries)
to keep stage-8 LOO folds robust to rare median-ties in a per-fold refit,
while preserving the strict behaviour at the cohort level.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from predict.analyse.profiles import determine_focal_diffuse_mapping

FOCAL_LABEL_STR = "focal"
DIFFUSE_LABEL_STR = "diffuse"
FOCAL_LABEL_INT = 0
DIFFUSE_LABEL_INT = 1


def determine_mapping(
    raw_features: pd.DataFrame,
    spatial_labels: pd.Series,
    n_calcified_arteries_col: str = "n_calcified_arteries",
    *,
    tie_break: str = "mean",
) -> dict[int, str]:
    """Return ``{cluster_id: "focal" | "diffuse"}`` for the 2-cluster partition.

    Wraps :func:`predict.analyse.profiles.determine_focal_diffuse_mapping`.
    If the strict-median helper raises on identical medians and
    ``tie_break != "raise"``, falls back to MEAN as a deterministic
    tie-breaker (``tie_break = "mean"``, the default for stage 8).
    """
    try:
        return determine_focal_diffuse_mapping(
            raw_features, spatial_labels,
            n_calcified_arteries_col=n_calcified_arteries_col,
        )
    except ValueError as exc:
        msg = str(exc)
        is_tie = "identical median" in msg
        if not is_tie or tie_break == "raise":
            raise
    # Tie-break by mean
    cluster_ids = sorted(spatial_labels.unique())
    means = {}
    for cid in cluster_ids:
        vals = raw_features.loc[
            spatial_labels == cid, n_calcified_arteries_col,
        ].dropna()
        means[cid] = float(np.mean(vals)) if len(vals) else float("nan")
    if means[cluster_ids[0]] == means[cluster_ids[1]]:
        raise ValueError(
            "determine_mapping: identical median AND mean "
            f"{n_calcified_arteries_col} in both clusters; cannot break tie."
        )
    focal_id = min(means, key=means.get)
    diffuse_id = max(means, key=means.get)
    return {focal_id: FOCAL_LABEL_STR, diffuse_id: DIFFUSE_LABEL_STR}


def apply_mapping(
    labels: pd.Series,
    mapping: Mapping,
) -> pd.Series:
    """Apply ``{old_id: new_label}`` to ``labels``. Idempotent.

    If ``labels`` already contains values from ``mapping.values()``, those
    pass through (identity entries inferred). This means calling
    apply_mapping(apply_mapping(x, m), m) == apply_mapping(x, m).
    """
    extended = dict(mapping)
    for v in mapping.values():
        extended.setdefault(v, v)
    return labels.map(extended)


def canonical_string_labels(
    raw_features: pd.DataFrame,
    spatial_labels: pd.Series,
    n_calcified_arteries_col: str = "n_calcified_arteries",
    *,
    tie_break: str = "mean",
) -> pd.Series:
    """Relabel ``spatial_labels`` (in ``{0, 1}``) to ``{"focal", "diffuse"}``."""
    mapping = determine_mapping(
        raw_features, spatial_labels,
        n_calcified_arteries_col=n_calcified_arteries_col,
        tie_break=tie_break,
    )
    return apply_mapping(spatial_labels, mapping).astype("string")


def canonical_numeric_labels(
    raw_features: pd.DataFrame,
    spatial_labels: pd.Series,
    n_calcified_arteries_col: str = "n_calcified_arteries",
    *,
    tie_break: str = "mean",
) -> pd.Series:
    """Relabel ``spatial_labels`` so ``0 = focal`` and ``1 = diffuse``.

    Convenience for cross-fold numeric outputs (e.g. confusion matrix vs
    a frozen reference). Use ``canonical_string_labels`` for any
    human-readable / paper column.
    """
    str_to_int = {FOCAL_LABEL_STR: FOCAL_LABEL_INT,
                  DIFFUSE_LABEL_STR: DIFFUSE_LABEL_INT}
    return canonical_string_labels(
        raw_features, spatial_labels,
        n_calcified_arteries_col=n_calcified_arteries_col,
        tie_break=tie_break,
    ).map(str_to_int).astype(int)
