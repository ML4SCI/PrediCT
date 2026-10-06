"""Stage 7 monotonicity test and burden-axis classification (D026).

For each robust kernel-independent feature (or any other
feature list passed in), compute Spearman rho and Kendall tau against
``agatston_total`` and classify into one of four mechanistic groups:

    burden_tracking     |spearman_rho| >= 0.5
    structure_tracking  |spearman_rho| < 0.3 AND block in
                          {hu_statistics, texture, shape, burden,
                           density_tier}* (any non-spatial block)
    spatial_tracking    |spearman_rho| < 0.3 AND block == spatial
    mixed               0.3 <= |spearman_rho| < 0.5 (intermediate band)

    * D026 specifies "structure_tracking" specifically for HU-statistics,
      texture, and shape. Burden and density_tier features in this band
      would be mixed by definition. We follow D026 strictly: only
      hu_statistics / texture / shape get the structure_tracking label;
      burden / density_tier features at |rho| < 0.3 are classified as
      mixed (this is rare in practice because burden features by
      construction track agatston).

Spearman is the primary classifier (radiomics literature standard).
Kendall tau-b is reported as a sensitivity column.

Decisions referencing this module:
    D026 - monotonicity test and burden-axis classification
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


# Classification thresholds (D026).
BURDEN_TRACKING_RHO = 0.5
WEAK_RHO_THRESHOLD = 0.3

# Block names that qualify for the structure_tracking classification.
STRUCTURE_BLOCKS: frozenset[str] = frozenset({
    "hu_statistics", "texture", "shape",
})

# Block name that qualifies for the spatial_tracking classification.
SPATIAL_BLOCK = "spatial"

# Missing feature, too few values, or a constant feature / burden: no
# correlation exists, so no class is assigned (D026.1).
NOT_EVALUABLE = "not_evaluable"


def classify_feature(spearman_rho: float, block: str) -> str:
    """Apply the D026 4-class classification rule.

    Parameters
    ----------
    spearman_rho : Spearman correlation against agatston_total
                   (signed; the magnitude is what matters)
    block        : the D022 multi-block partition this feature belongs to

    Returns
    -------
    One of "burden_tracking", "structure_tracking", "spatial_tracking",
    "mixed", or "not_evaluable" when rho is NaN (D026.1).
    """
    if np.isnan(spearman_rho):
        return NOT_EVALUABLE
    abs_rho = abs(spearman_rho)
    if abs_rho >= BURDEN_TRACKING_RHO:
        return "burden_tracking"
    if abs_rho < WEAK_RHO_THRESHOLD:
        if block == SPATIAL_BLOCK:
            return "spatial_tracking"
        if block in STRUCTURE_BLOCKS:
            return "structure_tracking"
        # Block is burden / density_tier / something else: mixed.
        return "mixed"
    return "mixed"


def compute_monotonicity(
    raw_features: pd.DataFrame,
    feature_names: Iterable[str],
    agatston: pd.Series,
    cohort: str,
    block_lookup: dict[str, str],
) -> pd.DataFrame:
    """Compute Spearman + Kendall vs agatston_total and classify.

    Parameters
    ----------
    raw_features : RAW (not z-scored) feature dataframe indexed by pid
    feature_names: list of feature columns to evaluate
    agatston     : RAW agatston_total Series indexed by pid
    cohort       : human-readable cohort label
    block_lookup : dict {feature_name: block_name} from the multi-block
                   partition (read from multi_block_assignments.csv)

    Returns
    -------
    DataFrame with one row per feature. Columns:
      cohort, feature, block, n_used, spearman_rho, spearman_p,
      kendall_tau, kendall_p, classification.
    """
    common = raw_features.index.intersection(agatston.index)
    if len(common) == 0:
        raise ValueError(
            f"compute_monotonicity [{cohort}]: no shared pids between "
            f"features and agatston"
        )
    raw_features = raw_features.loc[common]
    agatston_aligned = agatston.loc[common]

    rows: list[dict] = []
    for feature in feature_names:
        block = block_lookup.get(feature, "unknown")
        if feature not in raw_features.columns:
            rows.append({
                "cohort": cohort,
                "feature": feature,
                "block": block,
                "n_used": 0,
                "spearman_rho": float("nan"),
                "spearman_p": float("nan"),
                "kendall_tau": float("nan"),
                "kendall_p": float("nan"),
                "classification": NOT_EVALUABLE,
            })
            continue
        feature_vals = raw_features[feature].to_numpy(dtype=float)
        burden_vals = agatston_aligned.to_numpy(dtype=float)
        valid = ~(np.isnan(feature_vals) | np.isnan(burden_vals))
        f_v = feature_vals[valid]
        b_v = burden_vals[valid]
        if f_v.size < 5:
            # Too few values for meaningful correlation
            rows.append({
                "cohort": cohort,
                "feature": feature,
                "block": block,
                "n_used": int(f_v.size),
                "spearman_rho": float("nan"),
                "spearman_p": float("nan"),
                "kendall_tau": float("nan"),
                "kendall_p": float("nan"),
                "classification": NOT_EVALUABLE,
            })
            continue
        # If either array is constant, correlation is undefined; report NaN.
        if np.all(f_v == f_v[0]) or np.all(b_v == b_v[0]):
            s_rho, s_p = float("nan"), float("nan")
            k_tau, k_p = float("nan"), float("nan")
        else:
            s_rho, s_p = stats.spearmanr(f_v, b_v)
            k_tau, k_p = stats.kendalltau(f_v, b_v, variant="b")
        classification = classify_feature(s_rho, block)
        rows.append({
            "cohort": cohort,
            "feature": feature,
            "block": block,
            "n_used": int(f_v.size),
            "spearman_rho": float(s_rho),
            "spearman_p": float(s_p),
            "kendall_tau": float(k_tau),
            "kendall_p": float(k_p),
            "classification": classification,
        })
    return pd.DataFrame(rows)


def classification_summary(
    monotonicity_df: pd.DataFrame,
) -> pd.DataFrame:
    """Count features per classification per cohort.

    Returns a wide-format table: rows = cohorts, columns = classification
    labels (burden_tracking, structure_tracking, spatial_tracking, mixed,
    not_evaluable).
    """
    summary = (
        monotonicity_df
        .groupby(["cohort", "classification"])
        .size()
        .unstack(fill_value=0)
    )
    # Ensure every classification column is present even if empty
    cols = ["burden_tracking", "structure_tracking", "spatial_tracking",
            "mixed", NOT_EVALUABLE]
    for col in cols:
        if col not in summary.columns:
            summary[col] = 0
    summary["total"] = summary[cols].sum(axis=1)
    return summary.reset_index()
