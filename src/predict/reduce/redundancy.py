"""Stage 5 redundancy clustering and representative selection (D020 part 1).

Takes the prepared analysis matrix from ``prepare_matrix.run_matrix_prep``
plus the ICC report from stage 4 and produces:

  1. A Spearman r^2 distance matrix on the feature columns.
  2. Hierarchical clustering on (1 - r^2) using average linkage (primary).
     Sensitivity reruns on ward and complete linkage produce alternative
     cluster counts for the audit table.
  3. A representative feature per cluster, selected by the layered rule:
        a. highest ICC value (D016/D013)
        b. canonical features beat PyRadiomics on ties
        c. alphabetical for final tiebreak
     This rule is deterministic and easy to audit.

Mathematical conventions:

* Spearman correlation captures monotone redundancy regardless of distribution
  shape. Pearson underestimates redundancy on non-Gaussian features. We assert NO NaN at entry; pairwise-complete fallback is exposed
  as a parameter but defaults off.
* Distance is 1 - r^2, in [0, 1]. Diagonal is forced to exactly 0.
* Average linkage (UPGMA) is the primary; ward and complete are reported as
  sensitivity. Note: ward's mathematical "minimise within-cluster variance"
  interpretation strictly requires Euclidean inputs; on (1 - r^2) distances
  it remains a valid agglomeration but should be interpreted as a sensitivity
  probe, not a variance argument.

Decisions referencing this module:
    D013, D016 - ICC values consumed for representative selection
    D020       - this module

NaN policy: every function asserts no NaN on active feature columns at
entry. Upstream prepare_matrix is contracted to deliver finite values; NaN
here means a bug somewhere earlier in the pipeline and we want to fail loud.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy import stats
from scipy.spatial.distance import squareform


IccSource = Literal["empirical", "invariant_by_construction", "missing"]
LinkageMethod = Literal["average", "ward", "complete", "single"]


# ───────────────────────── containers ─────────────────────────


@dataclass(frozen=True)
class IccInfo:
    """ICC value plus its source label for one feature."""
    icc: float
    icc_source: IccSource


@dataclass
class RedundancyResult:
    """Output of run_redundancy_clustering.

    `linkage` is the scipy linkage matrix from the primary method. `labels`
    is the per-feature cluster assignment (1-indexed, scipy convention).
    `feature_order` is the order the input features appeared in; the i-th
    entry of `labels` corresponds to `feature_order[i]`.

    `cluster_assignments` is the audit dataframe with one row per input
    feature: feature, cluster_id, is_representative, icc, icc_source,
    is_canonical, selection_key_value.

    `sensitivity_per_method` maps each non-primary linkage method to a dict
    with cluster_count, cut_threshold, representatives, jaccard_vs_primary.
    """
    linkage: np.ndarray
    labels: np.ndarray
    cut_threshold: float
    feature_order: list[str]
    representatives: list[str]
    cluster_assignments: pd.DataFrame
    sensitivity_per_method: dict[str, dict]
    primary_method: str = "average"

    def n_clusters(self) -> int:
        return int(len(set(int(x) for x in self.labels)))


# ───────────────────────── helpers ─────────────────────────


def _assert_no_nan(df: pd.DataFrame, columns: Iterable[str], step: str) -> None:
    cols = list(columns)
    if not cols:
        return
    if df.loc[:, cols].isna().any().any():
        bad = df.loc[:, cols].columns[df.loc[:, cols].isna().any()].tolist()
        raise ValueError(
            f"[{step}] NaN in feature columns: {bad[:5]} "
            f"(and {max(0, len(bad) - 5)} more)."
        )


# ───────────────────────── Spearman distance ─────────────────────────


def spearman_r2_distance(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    nan_policy: Literal["propagate", "omit", "raise"] = "raise",
) -> np.ndarray:
    """Pairwise (1 - r^2) distance using Spearman correlation.

    Returns a (p x p) symmetric matrix in [0, 1] with zeros on the diagonal.

    NaN policy:
    - "raise" (default): asserts no NaN at entry. Recommended for production.
    - "omit": uses pairwise-complete observations for each pair.
    - "propagate": NaN in any column propagates to its row and column.

    Numerical notes:
    - Constant columns (zero variance) produce undefined Spearman correlation.
      We force their pairwise distance to 1.0 (maximally distant from
      everything; effectively they will form a singleton cluster).
    - r is clipped to [-1, 1] before squaring to absorb tiny floating-point
      drift.
    """
    if nan_policy == "raise":
        _assert_no_nan(df, feature_cols, "spearman_r2_distance")

    p = len(feature_cols)
    if p == 0:
        return np.zeros((0, 0), dtype=float)
    if p == 1:
        return np.zeros((1, 1), dtype=float)

    X = df[feature_cols].to_numpy(dtype=float)
    distance = np.zeros((p, p), dtype=float)

    # Constant-column detection (sd == 0 means undefined rank correlation).
    sds = X.std(axis=0, ddof=1)
    constant_mask = ~(sds > 0)

    for i in range(p):
        if constant_mask[i]:
            distance[i, :] = 1.0
            distance[:, i] = 1.0
            distance[i, i] = 0.0
            continue
        for j in range(i + 1, p):
            if constant_mask[j]:
                distance[i, j] = 1.0
                distance[j, i] = 1.0
                continue
            r, _ = stats.spearmanr(
                X[:, i], X[:, j], nan_policy=nan_policy,
            )
            if np.isnan(r):
                d = 1.0  # treat as maximally distant
            else:
                r = float(np.clip(r, -1.0, 1.0))
                d = 1.0 - r * r
            distance[i, j] = d
            distance[j, i] = d

    np.fill_diagonal(distance, 0.0)
    return np.clip(distance, 0.0, 1.0)


# ───────────────────────── hierarchical clustering ─────────────────────────


def hierarchical_cluster(
    distance_matrix: np.ndarray,
    feature_names: list[str],
    *,
    method: LinkageMethod = "average",
    min_gap: float = 0.05,
    fallback_distance: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Cluster features by hierarchical agglomeration on a precomputed
    distance matrix.

    Returns (linkage_matrix, cluster_labels, cut_threshold).

    Cut strategy:
    - Detect the largest gap between consecutive merge distances.
    - If the largest gap >= min_gap, cut in the middle of that gap.
    - Otherwise, cut at fallback_distance (default 0.25, equivalent to
      Spearman r^2 > 0.75 per D020).

    Note on ward: on (1 - r^2) distances ward remains a valid agglomeration
    but loses its "minimise within-cluster variance" mathematical
    interpretation. Used here only as a sensitivity probe, not as a primary
    statistical claim.
    """
    if distance_matrix.shape != (len(feature_names), len(feature_names)):
        raise ValueError(
            f"distance_matrix shape {distance_matrix.shape} does not match "
            f"feature_names length {len(feature_names)}"
        )
    if not np.allclose(distance_matrix, distance_matrix.T, atol=1e-8):
        raise ValueError("distance_matrix must be symmetric")
    if np.any(np.diag(distance_matrix) > 1e-8):
        raise ValueError("distance_matrix diagonal must be zero")

    n = len(feature_names)
    if n == 0:
        return np.empty((0, 4)), np.array([], dtype=int), 0.0
    if n == 1:
        return (
            np.array([[0.0, 0.0, 0.0, 1.0]]),
            np.array([1], dtype=int),
            0.0,
        )

    condensed = squareform(distance_matrix, checks=False)
    linkage = sch.linkage(condensed, method=method)

    merge_distances = np.sort(linkage[:, 2])
    gaps = np.diff(merge_distances)
    if len(gaps) > 0 and float(gaps.max()) >= min_gap:
        idx = int(np.argmax(gaps))
        cut_threshold = float((merge_distances[idx] + merge_distances[idx + 1]) / 2.0)
    else:
        cut_threshold = float(fallback_distance)

    labels = sch.fcluster(linkage, t=cut_threshold, criterion="distance")
    return linkage, labels, cut_threshold


# ───────────────────────── ICC lookup ─────────────────────────


def build_icc_lookup(
    icc_report_df: pd.DataFrame,
    *,
    derived_report_df: pd.DataFrame | None = None,
    invariant_names: Iterable[str] = (),
) -> dict[str, IccInfo]:
    """Build {feature_name: IccInfo} from the gate's ICC report, plus the
    measured reliability of the stage-5 transformed features
    (``derived_report_df``, stage 5b ``derived_icc.csv``; D040).

    ``invariant_names`` assigns ICC 1.0 / invariant_by_construction only to
    the names given, and only when no report row exists for them. It is for
    the D014 classic gate, where canonical inputs are invariant by
    construction; under D033 the transformed features vary across draws and
    must come from ``derived_report_df``.
    """
    required = {"feature", "icc", "icc_source"}
    lookup: dict[str, IccInfo] = {}
    for df in (icc_report_df, derived_report_df):
        if df is None:
            continue
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"ICC report missing columns: {sorted(missing)}")
        for _, row in df.iterrows():
            try:
                icc = float(row["icc"])
            except (TypeError, ValueError):
                icc = float("nan")
            source = str(row["icc_source"])
            lookup[str(row["feature"])] = IccInfo(icc=icc, icc_source=source)  # type: ignore[arg-type]

    for name in invariant_names:
        if name not in lookup:
            lookup[name] = IccInfo(icc=1.0, icc_source="invariant_by_construction")

    return lookup


def is_canonical_feature(name: str, canonical_set: set[str]) -> bool:
    """True if the feature is a member of the canonical bypass set (D016).

    Derived features added by prepare_matrix are not in feature_schema's
    canonical set, but they are *treated* as canonical for representative
    selection because they are XML-derived. The caller should include them
    explicitly in ``canonical_set`` when building the lookup.
    """
    return name in canonical_set


# ───────────────────────── representative selection ─────────────────────────


def selection_sort_key(
    name: str,
    icc_lookup: dict[str, IccInfo],
    canonical_set: set[str],
) -> tuple[int, float, int, str]:
    """Build the sort key used to pick a representative within a cluster.

    Sort order (smaller wins):
      1. Has known ICC (not NaN, not missing source) -> 0 ; else -> 1
      2. Negative ICC value (higher ICC sorts first) ; 0.0 if NaN
      3. Non-canonical flag (canonical sorts first) ; True converts to 1
      4. Name (alphabetical asc)

    This produces a fully deterministic ordering.
    """
    info = icc_lookup.get(name)
    if info is None or info.icc_source == "missing":
        nan_flag = 1
        neg_icc = 0.0
    elif np.isnan(info.icc):
        nan_flag = 1
        neg_icc = 0.0
    else:
        nan_flag = 0
        neg_icc = -float(info.icc)
    non_canonical = 0 if (name in canonical_set) else 1
    return (nan_flag, neg_icc, non_canonical, name)


def select_representative_per_cluster(
    cluster_members: list[str],
    icc_lookup: dict[str, IccInfo],
    canonical_set: set[str],
) -> tuple[str, dict]:
    """Pick the representative of a cluster using the layered rule.

    Returns (representative_name, record). Record fields:
      n_members, max_icc, n_canonical, decided_by ∈ {'singleton',
      'icc', 'canonical_tiebreak', 'alphabetical_tiebreak'}.
    """
    if not cluster_members:
        raise ValueError("cluster_members must be non-empty")
    if len(cluster_members) == 1:
        rep = cluster_members[0]
        return rep, {
            "n_members": 1,
            "max_icc": icc_lookup.get(rep, IccInfo(float("nan"), "missing")).icc,
            "n_canonical": int(rep in canonical_set),
            "decided_by": "singleton",
        }

    ranked = sorted(
        cluster_members,
        key=lambda n: selection_sort_key(n, icc_lookup, canonical_set),
    )
    rep = ranked[0]

    iccs = [icc_lookup.get(n, IccInfo(float("nan"), "missing")).icc
            for n in cluster_members]
    iccs_valid = [v for v in iccs if not (v is None or np.isnan(v))]
    max_icc = max(iccs_valid) if iccs_valid else float("nan")
    n_canonical = sum(1 for n in cluster_members if n in canonical_set)

    # Decide which tier of the layered rule actually picked the winner.
    iccs_eq_max = [n for n, v in zip(cluster_members, iccs)
                   if (v is not None) and (not np.isnan(v))
                   and (max_icc - v < 1e-9)]
    if len(iccs_eq_max) == 1:
        decided_by = "icc"
    else:
        canonical_in_tie = [n for n in iccs_eq_max if n in canonical_set]
        if len(canonical_in_tie) == 1:
            decided_by = "canonical_tiebreak"
        elif len(canonical_in_tie) > 1:
            decided_by = "alphabetical_tiebreak"
        else:
            decided_by = "alphabetical_tiebreak"

    return rep, {
        "n_members": len(cluster_members),
        "max_icc": max_icc,
        "n_canonical": n_canonical,
        "decided_by": decided_by,
    }


# ───────────────────────── orchestrator ─────────────────────────


def run_redundancy_clustering(
    df: pd.DataFrame,
    feature_cols: list[str],
    icc_lookup: dict[str, IccInfo],
    canonical_set: set[str],
    *,
    primary_method: LinkageMethod = "average",
    sensitivity_methods: Iterable[LinkageMethod] = ("ward", "complete"),
    min_gap: float = 0.05,
    fallback_distance: float = 0.25,
) -> RedundancyResult:
    """Build the Spearman r^2 distance, cluster with the primary linkage,
    pick representatives, and rerun on each sensitivity linkage.

    Returns ``RedundancyResult`` (see dataclass).
    """
    distance = spearman_r2_distance(df, feature_cols)
    linkage_primary, labels, cut = hierarchical_cluster(
        distance, feature_cols,
        method=primary_method,
        min_gap=min_gap, fallback_distance=fallback_distance,
    )

    # Group features by cluster label.
    by_cluster: dict[int, list[str]] = {}
    for name, lab in zip(feature_cols, labels):
        by_cluster.setdefault(int(lab), []).append(name)

    representatives: list[str] = []
    selection_records: dict[str, dict] = {}
    for cluster_id in sorted(by_cluster):
        members = by_cluster[cluster_id]
        rep, rec = select_representative_per_cluster(
            members, icc_lookup, canonical_set,
        )
        representatives.append(rep)
        for m in members:
            selection_records[m] = {
                "cluster_id": int(cluster_id),
                "is_representative": (m == rep),
                "n_members": rec["n_members"],
                "max_icc_in_cluster": rec["max_icc"],
                "decided_by": rec["decided_by"],
            }

    # Per-feature audit dataframe.
    rows: list[dict] = []
    for name in feature_cols:
        info = icc_lookup.get(name, IccInfo(float("nan"), "missing"))
        rec = selection_records[name]
        rows.append({
            "feature": name,
            "cluster_id": rec["cluster_id"],
            "is_representative": rec["is_representative"],
            "icc": float(info.icc),
            "icc_source": info.icc_source,
            "is_canonical": (name in canonical_set),
            "cluster_size": rec["n_members"],
            "cluster_max_icc": rec["max_icc_in_cluster"],
            "decided_by": rec["decided_by"],
        })
    cluster_assignments = pd.DataFrame(rows)

    # Sensitivity reruns: report cluster count, cut threshold, and the
    # Jaccard similarity of the representative set vs the primary's.
    sensitivity: dict[str, dict] = {}
    primary_reps = set(representatives)
    for method in sensitivity_methods:
        _, alt_labels, alt_cut = hierarchical_cluster(
            distance, feature_cols,
            method=method,
            min_gap=min_gap, fallback_distance=fallback_distance,
        )
        alt_by_cluster: dict[int, list[str]] = {}
        for name, lab in zip(feature_cols, alt_labels):
            alt_by_cluster.setdefault(int(lab), []).append(name)
        alt_reps: list[str] = []
        for cid in sorted(alt_by_cluster):
            rep, _ = select_representative_per_cluster(
                alt_by_cluster[cid], icc_lookup, canonical_set,
            )
            alt_reps.append(rep)
        jacc = (len(primary_reps & set(alt_reps)) /
                len(primary_reps | set(alt_reps))) if primary_reps else 0.0
        sensitivity[str(method)] = {
            "n_clusters": int(len(set(int(x) for x in alt_labels))),
            "cut_threshold": float(alt_cut),
            "representatives": list(alt_reps),
            "jaccard_vs_primary": float(jacc),
        }

    return RedundancyResult(
        linkage=linkage_primary,
        labels=labels,
        cut_threshold=cut,
        feature_order=list(feature_cols),
        representatives=representatives,
        cluster_assignments=cluster_assignments,
        sensitivity_per_method=sensitivity,
        primary_method=str(primary_method),
    )


# ───────────────────────── multi-block clustering (D022) ─────────────────────────


@dataclass(frozen=True)
class FeatureBlock:
    """Prospective block definition for multi-block redundancy clustering.

    The membership rule is a list of names + a list of name-prefix patterns.
    A feature belongs to the block if its full name is in ``exact_names`` OR
    if any prefix in ``prefixes`` is a prefix of the feature name.
    ``exclude_exact_names`` lets a more specific block override a broader
    prefix-based block.
    """
    name: str
    exact_names: tuple[str, ...] = ()
    prefixes: tuple[str, ...] = ()
    exclude_exact_names: tuple[str, ...] = ()

    def contains(self, feature: str) -> bool:
        if feature in self.exclude_exact_names:
            return False
        if feature in self.exact_names:
            return True
        return any(feature.startswith(p) for p in self.prefixes)


# D022 locked block partition. Each feature is assigned to exactly one block.
# The texture block excludes ``original_firstorder_Range`` because it is an
# HU-statistic (max - min) more naturally grouped with mean_hu_* / max_hu_*.
DEFAULT_BLOCKS: tuple[FeatureBlock, ...] = (
    FeatureBlock(
        name="burden",
        prefixes=("agatston_", "mass_", "volume_"),
    ),
    FeatureBlock(
        name="hu_statistics",
        exact_names=(
            "max_hu_global", "mean_hu_weighted_global",
            "original_firstorder_Range",
        ),
        prefixes=("max_hu_", "mean_hu_"),
    ),
    FeatureBlock(
        name="density_tier",
        exact_names=("has_dense_calcium", "high_density_fraction",
                     "dense_calcium_count"),
        prefixes=("n_rois_d",),
    ),
    FeatureBlock(
        name="spatial",
        exact_names=("n_calcified_arteries", "gini_lesion_volume",
                     "center_of_mass_z", "vessel_burden_gini"),
        prefixes=("lesion_count_", "dist_from_top_",
                  "inter_lesion_dist_", "first_to_last_dist_"),
    ),
    FeatureBlock(
        name="texture",
        prefixes=("original_glcm_", "original_glszm_", "original_glrlm_",
                  "original_ngtdm_", "original_gldm_", "original_firstorder_"),
        exclude_exact_names=("original_firstorder_Range",),
    ),
    FeatureBlock(
        name="shape",
        prefixes=("original_shape_",),
    ),
)


def assign_features_to_blocks(
    feature_cols: list[str],
    blocks: tuple[FeatureBlock, ...] = DEFAULT_BLOCKS,
) -> tuple[dict[str, list[str]], list[str]]:
    """Partition feature_cols across blocks; return (block_to_features, unassigned).

    Asserts mutual exclusivity: each feature is assigned to at most one
    block. Features matching no block end up in the ``unassigned`` list
    (typically per-vessel diffusivity or similar; should be empty for our
    post-D017 feature set).
    """
    block_to_features: dict[str, list[str]] = {b.name: [] for b in blocks}
    assigned: dict[str, str] = {}
    unassigned: list[str] = []
    for feat in feature_cols:
        matches = [b.name for b in blocks if b.contains(feat)]
        if not matches:
            unassigned.append(feat)
            continue
        if len(matches) > 1:
            raise ValueError(
                f"feature {feat!r} matches multiple blocks: {matches}. "
                "Block partition must be mutually exclusive."
            )
        block_to_features[matches[0]].append(feat)
        assigned[feat] = matches[0]
    return block_to_features, unassigned


@dataclass
class MultiBlockResult:
    """Output of run_multi_block_redundancy_clustering.

    Aggregates a RedundancyResult per block plus the combined representative
    list across all blocks.
    """
    per_block: dict[str, RedundancyResult]
    block_assignment: dict[str, str]   # feature -> block name
    representatives: list[str]          # union across all blocks
    unassigned: list[str]               # features matched no block
    block_definitions: tuple[FeatureBlock, ...]

    def n_representatives(self) -> int:
        return len(self.representatives)

    def assignments_dataframe(self) -> pd.DataFrame:
        rows: list[dict] = []
        for block_name, result in self.per_block.items():
            for _, row in result.cluster_assignments.iterrows():
                rows.append({
                    "feature": row["feature"],
                    "block": block_name,
                    **{k: row[k] for k in row.index if k != "feature"},
                })
        return pd.DataFrame(rows)


def run_multi_block_redundancy_clustering(
    df: pd.DataFrame,
    feature_cols: list[str],
    icc_lookup: dict[str, IccInfo],
    canonical_set: set[str],
    *,
    blocks: tuple[FeatureBlock, ...] = DEFAULT_BLOCKS,
    primary_method: LinkageMethod = "average",
    sensitivity_methods: Iterable[LinkageMethod] = (),
    min_gap: float = 0.05,
    fallback_distance: float = 0.25,
) -> MultiBlockResult:
    """Run Spearman r^2 clustering independently within each block (D022).

    Each block uses the existing ``run_redundancy_clustering`` machinery, so
    the per-block selection rule (ICC-first, canonical-vs-PyRadiomics
    tiebreaker, alphabetical final tiebreaker) is identical to the
    single-matrix variant. The final representative list is the union of
    per-block representatives.
    """
    block_to_features, unassigned = assign_features_to_blocks(
        feature_cols, blocks=blocks,
    )

    per_block: dict[str, RedundancyResult] = {}
    block_assignment: dict[str, str] = {}
    representatives: list[str] = []

    for block in blocks:
        members = block_to_features[block.name]
        if not members:
            continue
        result = run_redundancy_clustering(
            df, members, icc_lookup, canonical_set,
            primary_method=primary_method,
            sensitivity_methods=sensitivity_methods,
            min_gap=min_gap,
            fallback_distance=fallback_distance,
        )
        per_block[block.name] = result
        for f in members:
            block_assignment[f] = block.name
        representatives.extend(result.representatives)

    return MultiBlockResult(
        per_block=per_block,
        block_assignment=block_assignment,
        representatives=representatives,
        unassigned=unassigned,
        block_definitions=blocks,
    )


__all__ = [
    "DEFAULT_BLOCKS",
    "FeatureBlock",
    "IccInfo",
    "IccSource",
    "LinkageMethod",
    "MultiBlockResult",
    "RedundancyResult",
    "assign_features_to_blocks",
    "build_icc_lookup",
    "hierarchical_cluster",
    "is_canonical_feature",
    "run_multi_block_redundancy_clustering",
    "run_redundancy_clustering",
    "select_representative_per_cluster",
    "selection_sort_key",
    "spearman_r2_distance",
]
