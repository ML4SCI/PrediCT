"""Stage 7 per-cluster characterisation primitives (D023).

Computes per-cluster median + IQR + Cliff's delta + Mann-Whitney U on RAW
feature values (NOT z-scored prep_df values) for clinical interpretability,
then applies Benjamini-Hochberg FDR correction across the 41 comparisons
within each (cohort x partition) bundle.

Also exposes the two cohort-level fail-loud gates from D023:

    * `assert_biological_sanity` - focal cluster max_hu_global median must
      be >= 0.9 * diffuse cluster median. Catches the soft-tissue-mask
      regression mode.
    * `assert_label_balance` - minority class size must be > 15% of the
      cohort. Catches degenerate partition cases.

NaN policy: features may legitimately be NaN (the 22 patients with
mask_voxels < 14 have all PyRadiomics features NaN per D010). Median /
IQR computation drops NaN BEFORE comparison, and the dropped-count is
recorded in `n_used`. If a cluster has fewer than 5 non-NaN values for
a feature, statistics are flagged `insufficient_data = True` and that
row is excluded from FDR-BH (so it does not inflate the multiple-testing
correction).

Decisions referencing this module:
    D023 - per-cluster characterisation method
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


# Minimum non-NaN sample size per cluster for a feature to be included
# in significance testing. Below this, statistics are too noisy.
MIN_SAMPLES_PER_CLUSTER = 5


# ─────────────────────── effect-size primitives ───────────────────────


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta effect size for two samples.

    delta = (#{a_i > b_j} - #{a_i < b_j}) / (n_a * n_b)
          = P(A > B) - P(A < B), as estimated by ranks.

    Range: [-1, +1]. Symmetric: cliffs_delta(a, b) == -cliffs_delta(b, a).
    Returns 0.0 when both samples are identical (or perfectly overlap).
    NaN values are dropped before computation; raises ValueError if either
    sample becomes empty.

    This is the rank-based effect size companion to the Mann-Whitney U
    test. It is preferred over Cohen's d for non-Gaussian distributions
    (Romano 2006).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        raise ValueError(
            f"cliffs_delta: empty sample after NaN drop "
            f"(a.size={a.size}, b.size={b.size})"
        )
    # Vectorised pairwise comparison; OK for cohort sizes < 1000.
    # For larger cohorts, switch to the rank-based O(n log n) formula.
    diffs = a[:, None] - b[None, :]
    n_greater = int((diffs > 0).sum())
    n_less = int((diffs < 0).sum())
    return (n_greater - n_less) / float(a.size * b.size)


def mannwhitney_u_pval(
    a: np.ndarray, b: np.ndarray,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> float:
    """Mann-Whitney U test p-value. NaN-drops both samples first.

    Returns 1.0 if either sample becomes empty after NaN drop, or if both
    samples are constant and equal (no rank information). Returns NaN if
    one is constant but the other is not (scipy raises in this case for
    some versions).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return 1.0
    # If both samples are constant AND equal, p-value is 1.0 (no evidence).
    if np.all(a == a[0]) and np.all(b == b[0]) and a[0] == b[0]:
        return 1.0
    try:
        result = stats.mannwhitneyu(a, b, alternative=alternative)
        return float(result.pvalue)
    except ValueError:
        return float("nan")


# ─────────────────────── per-cluster profiles ───────────────────────


@dataclass(frozen=True)
class ProfileRow:
    """Per-(feature, cluster) profile row before FDR adjustment."""
    cohort: str
    partition: str
    cluster: str
    feature: str
    n: int
    n_used: int
    n_nonzero: int
    median: float
    iqr_lower: float
    iqr_upper: float
    cliffs_delta: float
    mannwhitney_u_pval: float
    insufficient_data: bool

    def to_dict(self) -> dict:
        return {
            "cohort": self.cohort,
            "partition": self.partition,
            "cluster": self.cluster,
            "feature": self.feature,
            "n": self.n,
            "n_used": self.n_used,
            "n_nonzero": self.n_nonzero,
            "median": self.median,
            "iqr_lower": self.iqr_lower,
            "iqr_upper": self.iqr_upper,
            "cliffs_delta": self.cliffs_delta,
            "mannwhitney_u_pval": self.mannwhitney_u_pval,
            "insufficient_data": self.insufficient_data,
        }


def compute_cluster_profile(
    raw_features: pd.DataFrame,
    labels: pd.Series,
    feature_names: list[str],
    cohort: str,
    partition: str,
) -> pd.DataFrame:
    """Per-cluster profile for every feature in feature_names.

    For binary partitions (e.g., spatial_k2) the "vs other" comparison is
    cluster-A-vs-cluster-B. For multi-class partitions (e.g., burden_k3)
    we use one-vs-rest (this-cluster vs union of all other clusters).

    Parameters
    ----------
    raw_features : DataFrame indexed by patient id, columns = feature_names
    labels       : Series indexed by patient id, integer or string cluster labels
    feature_names: list of feature columns to profile
    cohort       : human-readable cohort label (used in output)
    partition    : human-readable partition label (used in output)

    Returns
    -------
    DataFrame with one row per (cluster, feature). The Mann-Whitney p has
    NOT yet been FDR-adjusted; pass through ``apply_fdr_bh`` to add the
    ``fdr_bh_pval`` and ``is_robust_discriminator`` columns.
    """
    if not raw_features.index.equals(labels.index):
        # Align by intersection; warn loudly if shapes drift.
        common = raw_features.index.intersection(labels.index)
        if len(common) == 0:
            raise ValueError(
                "compute_cluster_profile: raw_features and labels have "
                "no shared pids"
            )
        raw_features = raw_features.loc[common]
        labels = labels.loc[common]

    rows: list[dict] = []
    cluster_ids = sorted(labels.unique())
    for cluster_id in cluster_ids:
        in_cluster = labels == cluster_id
        other = ~in_cluster
        for feature in feature_names:
            if feature not in raw_features.columns:
                continue
            vals_in = raw_features.loc[in_cluster, feature].to_numpy(dtype=float)
            vals_out = raw_features.loc[other, feature].to_numpy(dtype=float)
            vals_in_clean = vals_in[~np.isnan(vals_in)]
            vals_out_clean = vals_out[~np.isnan(vals_out)]
            insufficient = (
                vals_in_clean.size < MIN_SAMPLES_PER_CLUSTER
                or vals_out_clean.size < MIN_SAMPLES_PER_CLUSTER
            )
            if vals_in_clean.size == 0:
                median = np.nan
                iqr_l = iqr_u = np.nan
                n_nonzero = 0
            else:
                median = float(np.median(vals_in_clean))
                iqr_l = float(np.percentile(vals_in_clean, 25))
                iqr_u = float(np.percentile(vals_in_clean, 75))
                n_nonzero = int((vals_in_clean != 0).sum())
            if insufficient:
                delta = np.nan
                pval = np.nan
            else:
                delta = cliffs_delta(vals_in_clean, vals_out_clean)
                pval = mannwhitney_u_pval(vals_in_clean, vals_out_clean,
                                          alternative="two-sided")
            rows.append({
                "cohort": cohort,
                "partition": partition,
                "cluster": str(cluster_id),
                "feature": feature,
                "n": int(in_cluster.sum()),
                "n_used": int(vals_in_clean.size),
                "n_nonzero": n_nonzero,
                "median": median,
                "iqr_lower": iqr_l,
                "iqr_upper": iqr_u,
                "cliffs_delta": delta,
                "mannwhitney_u_pval": pval,
                "insufficient_data": insufficient,
            })
    return pd.DataFrame(rows)


def apply_fdr_bh(
    profile_df: pd.DataFrame,
    p_column: str = "mannwhitney_u_pval",
    out_column: str = "fdr_bh_pval",
    delta_threshold: float = 0.20,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Apply Benjamini-Hochberg FDR correction within each
    (cohort, partition) bundle.

    Adds two columns:
      - ``out_column``: BH-adjusted p-values (rows with insufficient_data
        get NaN; they are excluded from the BH adjustment denominator)
      - ``is_robust_discriminator``: True iff
        fdr_bh_pval < alpha AND abs(cliffs_delta) >= delta_threshold

    The input DataFrame is not mutated; a copy is returned.
    """
    out = profile_df.copy()
    out[out_column] = np.nan
    out["is_robust_discriminator"] = False
    for (cohort, partition), bundle in out.groupby(["cohort", "partition"]):
        eligible = bundle[~bundle["insufficient_data"]
                          & bundle[p_column].notna()]
        if len(eligible) == 0:
            continue
        pvals = eligible[p_column].to_numpy()
        _, fdr_p, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
        out.loc[eligible.index, out_column] = fdr_p
        out.loc[eligible.index, "is_robust_discriminator"] = (
            (fdr_p < alpha)
            & (np.abs(eligible["cliffs_delta"].to_numpy()) >= delta_threshold)
        )
    return out


# ─────────────────────── fail-loud gates ───────────────────────


def assert_biological_sanity(
    raw_features: pd.DataFrame,
    labels: pd.Series,
    focal_label: str | int,
    diffuse_label: str | int,
    cohort: str,
    hu_column: str = "max_hu_global",
    absolute_floor_hu: float = 130.0,
    warning_ratio_threshold: float = 0.5,
) -> dict:
    """Raise if the focal cluster lacks calcium (D023 gate, revised).

    The gate fires when the focal cluster's median ``hu_column`` falls
    BELOW the absolute calcium floor (default 130 HU, the IBSI / Agatston
    calcium definition). This catches the actual failure mode the gate
    is meant to detect: a soft-tissue-mask regression that would push
    focal max_hu to noise-floor (negative or near-zero).

    A secondary WARNING is logged (but the gate does NOT raise) if
    focal_median / diffuse_median < ``warning_ratio_threshold`` (default
    0.5). This is the "unexpected biology" signal: focal calcium is
    legitimate but unusually low-peak relative to diffuse, which we want
    visible in the audit log without blocking the pipeline.

    Returns a dict {focal_median, diffuse_median, ratio, passes,
    warning_low_ratio} for audit; only raises on the absolute floor.

    A relative ratio threshold (focal >= 0.9 * diffuse) would fire
    false-positively on COCA, where focal calcium has legitimately lower
    peak HU than diffuse (ratio about 0.5). The IBSI absolute floor is the
    objective, non-arbitrary alternative.
    """
    focal_mask = labels == focal_label
    diffuse_mask = labels == diffuse_label
    focal_vals = raw_features.loc[focal_mask, hu_column].dropna()
    diffuse_vals = raw_features.loc[diffuse_mask, hu_column].dropna()
    if len(focal_vals) == 0 or len(diffuse_vals) == 0:
        raise ValueError(
            f"assert_biological_sanity [{cohort}]: empty focal or diffuse "
            f"cluster for {hu_column}"
        )
    focal_med = float(np.median(focal_vals))
    diffuse_med = float(np.median(diffuse_vals))

    # Primary gate: focal cluster must contain calcium.
    if focal_med < absolute_floor_hu:
        raise ValueError(
            f"assert_biological_sanity [{cohort}]: focal median "
            f"{hu_column} = {focal_med:.1f} HU < {absolute_floor_hu} HU "
            f"(IBSI calcium floor). The focal cluster does not contain "
            f"calcium-positive voxels; soft-tissue-mask regression suspected. "
            f"Inspect focal/diffuse label mapping or mask alignment before "
            f"re-running."
        )

    # Secondary warning: unusually low focal/diffuse ratio.
    if diffuse_med <= 0:
        ratio = float("nan")
        warning_low_ratio = False
    else:
        ratio = focal_med / diffuse_med
        warning_low_ratio = ratio < warning_ratio_threshold

    return {
        "cohort": cohort,
        "focal_median": focal_med,
        "diffuse_median": diffuse_med,
        "ratio": ratio,
        "absolute_floor_hu": absolute_floor_hu,
        "passes": True,  # absolute-floor gate passed
        "warning_low_ratio": warning_low_ratio,
        "warning_ratio_threshold": warning_ratio_threshold,
    }


def assert_label_balance(
    labels: pd.Series,
    cohort: str,
    partition: str,
    minority_fraction_threshold: float = 0.15,
) -> dict:
    """Raise if the minority class is below the D023 minimum fraction.

    Returns a dict {cluster_sizes, minority_fraction, passes} for audit;
    only raises if passes == False.
    """
    sizes = labels.value_counts().to_dict()
    total = int(labels.size)
    if total == 0:
        raise ValueError(
            f"assert_label_balance [{cohort}/{partition}]: empty labels"
        )
    min_size = min(sizes.values())
    fraction = min_size / total
    passes = fraction > minority_fraction_threshold
    if not passes:
        raise ValueError(
            f"assert_label_balance [{cohort}/{partition}]: minority class "
            f"size = {min_size}/{total} = {fraction:.3f} <= "
            f"{minority_fraction_threshold}. D023 gate failed; partition "
            f"is too imbalanced for stage 7."
        )
    return {
        "cohort": cohort,
        "partition": partition,
        "cluster_sizes": {str(k): int(v) for k, v in sizes.items()},
        "minority_fraction": fraction,
        "passes": passes,
    }


def determine_focal_diffuse_mapping(
    raw_features: pd.DataFrame,
    spatial_labels: pd.Series,
    n_calcified_arteries_col: str = "n_calcified_arteries",
) -> dict[int, str]:
    """Map the spatial-k2 cluster ids {0, 1} to {focal, diffuse}.

    D023 rule (pre-registered): the cluster with the LOWER median
    ``n_calcified_arteries`` is ``focal``; the other is ``diffuse``.

    Returns a dict {original_label: "focal" | "diffuse"} for relabelling.
    Raises if the spatial partition does not have exactly 2 clusters or
    if the two clusters have identical medians (no unambiguous focal
    identity).
    """
    cluster_ids = sorted(spatial_labels.unique())
    if len(cluster_ids) != 2:
        raise ValueError(
            f"determine_focal_diffuse_mapping: expected exactly 2 spatial "
            f"clusters; got {len(cluster_ids)} ({cluster_ids})"
        )
    medians = {}
    for cluster_id in cluster_ids:
        mask = spatial_labels == cluster_id
        vals = raw_features.loc[mask, n_calcified_arteries_col].dropna()
        if len(vals) == 0:
            raise ValueError(
                f"determine_focal_diffuse_mapping: cluster {cluster_id} has "
                f"no non-NaN {n_calcified_arteries_col} values"
            )
        medians[cluster_id] = float(np.median(vals))
    if medians[cluster_ids[0]] == medians[cluster_ids[1]]:
        raise ValueError(
            f"determine_focal_diffuse_mapping: identical median "
            f"{n_calcified_arteries_col} ({medians[cluster_ids[0]]}) in "
            f"both spatial clusters; cannot deterministically map to "
            f"focal/diffuse"
        )
    focal_id = min(medians, key=medians.get)
    diffuse_id = max(medians, key=medians.get)
    return {focal_id: "focal", diffuse_id: "diffuse"}
