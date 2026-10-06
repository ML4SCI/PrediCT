"""Tests for predict.reduce.redundancy (D020).

Coverage: Spearman r^2 distance properties; hierarchical clustering elbow
detection + fallback; ICC lookup builder; canonical-membership check; the
layered representative-selection rule; the orchestrator on synthetic data.

The critical correctness invariant we test heavily: representative selection
under the layered rule (ICC -> canonical -> alphabetical) is fully
deterministic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predict.reduce.redundancy import (
    DEFAULT_BLOCKS,
    FeatureBlock,
    IccInfo,
    MultiBlockResult,
    RedundancyResult,
    assign_features_to_blocks,
    build_icc_lookup,
    hierarchical_cluster,
    is_canonical_feature,
    run_multi_block_redundancy_clustering,
    run_redundancy_clustering,
    select_representative_per_cluster,
    selection_sort_key,
    spearman_r2_distance,
)


# ───────────────────── helpers ─────────────────────


def _df(cols: dict[str, list[float]]) -> pd.DataFrame:
    n = len(next(iter(cols.values())))
    return pd.DataFrame({"pid": [str(i) for i in range(n)], **cols})


# ───────────────────── spearman_r2_distance ─────────────────────


def test_spearman_distance_identity_zero():
    df = _df({"a": [1.0, 2, 3, 4, 5], "b": [1.0, 2, 3, 4, 5]})
    D = spearman_r2_distance(df, ["a", "b"])
    assert D.shape == (2, 2)
    assert D[0, 1] == pytest.approx(0.0, abs=1e-9)


def test_spearman_distance_perfect_monotone_zero():
    """Spearman is rank-based; any strictly monotone relation gives r=1."""
    df = _df({"a": [1.0, 2, 3, 4, 5], "b": [1.0, 4, 9, 16, 25]})  # b = a^2 (monotone)
    D = spearman_r2_distance(df, ["a", "b"])
    assert D[0, 1] == pytest.approx(0.0, abs=1e-9)


def test_spearman_distance_anti_correlation_zero():
    df = _df({"a": [1.0, 2, 3, 4, 5], "b": [5.0, 4, 3, 2, 1]})
    D = spearman_r2_distance(df, ["a", "b"])
    # r^2 = 1 -> distance = 0
    assert D[0, 1] == pytest.approx(0.0, abs=1e-9)


def test_spearman_distance_independent_near_one():
    rng = np.random.default_rng(0)
    n = 500
    df = _df({"a": rng.normal(size=n).tolist(), "b": rng.normal(size=n).tolist()})
    D = spearman_r2_distance(df, ["a", "b"])
    assert D[0, 1] > 0.95   # nearly maximal distance


def test_spearman_distance_symmetric_and_diagonal_zero():
    rng = np.random.default_rng(1)
    df = _df({c: rng.normal(size=30).tolist() for c in "abcd"})
    D = spearman_r2_distance(df, list("abcd"))
    np.testing.assert_allclose(D, D.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)


def test_spearman_distance_in_unit_interval():
    rng = np.random.default_rng(2)
    df = _df({c: rng.normal(size=50).tolist() for c in "abcd"})
    D = spearman_r2_distance(df, list("abcd"))
    assert D.min() >= 0.0
    assert D.max() <= 1.0


def test_spearman_distance_raises_on_nan_by_default():
    df = _df({"a": [1.0, 2, 3, 4, 5], "b": [1.0, np.nan, 3, 4, 5]})
    with pytest.raises(ValueError, match="NaN"):
        spearman_r2_distance(df, ["a", "b"])


def test_spearman_distance_constant_column_distance_one():
    df = _df({"a": [1.0, 2, 3, 4, 5], "b": [7.0, 7, 7, 7, 7]})  # b constant
    D = spearman_r2_distance(df, ["a", "b"])
    assert D[0, 1] == pytest.approx(1.0, abs=1e-9)
    assert D[1, 1] == 0.0


def test_spearman_distance_empty_input_handled():
    df = _df({"a": [1.0, 2, 3]})
    D = spearman_r2_distance(df, [])
    assert D.shape == (0, 0)


def test_spearman_distance_single_column():
    df = _df({"a": [1.0, 2, 3]})
    D = spearman_r2_distance(df, ["a"])
    assert D.shape == (1, 1)
    assert D[0, 0] == 0.0


# ───────────────────── hierarchical_cluster ─────────────────────


def test_hierarchical_cluster_two_correlated_one_cluster():
    D = np.array([[0.0, 0.02, 0.95],
                  [0.02, 0.0, 0.93],
                  [0.95, 0.93, 0.0]])
    _, labels, _ = hierarchical_cluster(D, ["a", "b", "c"],
                                         method="average",
                                         min_gap=0.1, fallback_distance=0.25)
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]


def test_hierarchical_cluster_three_independent_three_clusters():
    D = np.array([[0.0, 0.8, 0.9],
                  [0.8, 0.0, 0.85],
                  [0.9, 0.85, 0.0]])
    _, labels, cut = hierarchical_cluster(D, ["a", "b", "c"],
                                           method="average",
                                           min_gap=0.5, fallback_distance=0.25)
    assert len(set(labels)) == 3
    assert cut == pytest.approx(0.25)


def test_hierarchical_cluster_elbow_finds_clear_gap():
    """Two tight pairs separated by a large gap; the cut should land in
    the gap (between ~0.02 and ~0.80)."""
    D = np.array([
        [0.0, 0.02, 0.80, 0.82],
        [0.02, 0.0, 0.79, 0.81],
        [0.80, 0.79, 0.0, 0.03],
        [0.82, 0.81, 0.03, 0.0],
    ])
    _, labels, cut = hierarchical_cluster(D, list("abcd"),
                                           method="average",
                                           min_gap=0.1, fallback_distance=0.25)
    assert 0.03 < cut < 0.79
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_hierarchical_cluster_fallback_when_no_elbow():
    D = np.array([[0.0, 0.10, 0.12],
                  [0.10, 0.0, 0.14],
                  [0.12, 0.14, 0.0]])
    _, _, cut = hierarchical_cluster(D, list("abc"),
                                       method="average",
                                       min_gap=0.5, fallback_distance=0.25)
    assert cut == pytest.approx(0.25)


def test_hierarchical_cluster_single_feature_returns_single_cluster():
    D = np.array([[0.0]])
    _, labels, _ = hierarchical_cluster(D, ["solo"], method="average")
    assert list(labels) == [1]


def test_hierarchical_cluster_empty_input():
    D = np.zeros((0, 0))
    linkage, labels, cut = hierarchical_cluster(D, [], method="average")
    assert linkage.shape == (0, 4)
    assert len(labels) == 0


def test_hierarchical_cluster_raises_on_non_symmetric():
    D = np.array([[0.0, 0.1], [0.2, 0.0]])
    with pytest.raises(ValueError, match="symmetric"):
        hierarchical_cluster(D, ["a", "b"], method="average")


def test_hierarchical_cluster_raises_on_nonzero_diagonal():
    D = np.array([[0.5, 0.1], [0.1, 0.0]])
    with pytest.raises(ValueError, match="diagonal"):
        hierarchical_cluster(D, ["a", "b"], method="average")


def test_hierarchical_cluster_shape_mismatch_raises():
    D = np.array([[0.0, 0.1], [0.1, 0.0]])
    with pytest.raises(ValueError, match="shape"):
        hierarchical_cluster(D, ["a", "b", "c"], method="average")


@pytest.mark.parametrize("method", ["average", "ward", "complete", "single"])
def test_hierarchical_cluster_methods_all_run(method):
    rng = np.random.default_rng(0)
    # Build a valid symmetric distance matrix.
    p = 6
    raw = rng.uniform(0, 1, size=(p, p))
    D = (raw + raw.T) / 2
    np.fill_diagonal(D, 0.0)
    _, labels, _ = hierarchical_cluster(D, list("abcdef"),
                                          method=method,
                                          min_gap=0.05, fallback_distance=0.25)
    assert len(labels) == p


# ───────────────────── build_icc_lookup ─────────────────────


def test_build_icc_lookup_basic():
    report = pd.DataFrame({
        "feature": ["agatston_lad", "original_glcm_Contrast"],
        "icc": [1.0, 0.83],
        "icc_source": ["invariant_by_construction", "empirical"],
    })
    lookup = build_icc_lookup(report)
    assert lookup["agatston_lad"].icc == 1.0
    assert lookup["agatston_lad"].icc_source == "invariant_by_construction"
    assert lookup["original_glcm_Contrast"].icc == pytest.approx(0.83)


def test_build_icc_lookup_uses_measured_derived_reliability():
    """D040: under D033 the transformed features carry their measured ICC
    from 5b's derived_icc.csv, never an assumed 1.0."""
    report = pd.DataFrame({"feature": ["agatston_lad"], "icc": [1.0],
                           "icc_source": ["invariant_by_construction"]})
    derived = pd.DataFrame({"feature": ["vessel_burden_gini"], "icc": [0.998],
                            "icc_source": ["empirical"]})
    lookup = build_icc_lookup(report, derived_report_df=derived)
    assert lookup["vessel_burden_gini"].icc == pytest.approx(0.998)
    assert lookup["vessel_burden_gini"].icc_source == "empirical"
    assert "high_density_fraction" not in lookup     # no silent default


def test_build_icc_lookup_invariant_names_only_for_classic_gate():
    report = pd.DataFrame({"feature": ["agatston_lad"], "icc": [1.0],
                           "icc_source": ["invariant_by_construction"]})
    lookup = build_icc_lookup(report, invariant_names=["high_density_fraction"])
    assert lookup["high_density_fraction"].icc == 1.0
    assert lookup["high_density_fraction"].icc_source == "invariant_by_construction"


def test_build_icc_lookup_nan_string_becomes_nan():
    report = pd.DataFrame({
        "feature": ["f"],
        "icc": ["NaN"],
        "icc_source": ["empirical"],
    })
    lookup = build_icc_lookup(report)
    assert np.isnan(lookup["f"].icc)


def test_build_icc_lookup_missing_columns_raises():
    report = pd.DataFrame({"feature": ["f"], "icc": [0.5]})
    with pytest.raises(KeyError, match="icc_source"):
        build_icc_lookup(report)


# ───────────────────── is_canonical_feature ─────────────────────


def test_is_canonical_basic():
    canon = {"agatston_lad", "high_density_fraction"}
    assert is_canonical_feature("agatston_lad", canon)
    assert is_canonical_feature("high_density_fraction", canon)
    assert not is_canonical_feature("original_glcm_Contrast", canon)


# ───────────────────── selection_sort_key ─────────────────────


def test_selection_key_higher_icc_wins():
    lookup = {
        "a": IccInfo(0.9, "empirical"),
        "b": IccInfo(0.7, "empirical"),
    }
    canon = set()
    ka = selection_sort_key("a", lookup, canon)
    kb = selection_sort_key("b", lookup, canon)
    assert ka < kb   # smaller key wins; higher ICC has smaller (-icc)


def test_selection_key_canonical_wins_at_equal_icc():
    lookup = {
        "a": IccInfo(1.0, "invariant_by_construction"),
        "b": IccInfo(1.0, "empirical"),
    }
    canon = {"a"}
    ka = selection_sort_key("a", lookup, canon)
    kb = selection_sort_key("b", lookup, canon)
    assert ka < kb


def test_selection_key_alphabetical_at_full_tie():
    lookup = {
        "alpha": IccInfo(1.0, "invariant_by_construction"),
        "bravo": IccInfo(1.0, "invariant_by_construction"),
    }
    canon = {"alpha", "bravo"}
    ka = selection_sort_key("alpha", lookup, canon)
    kb = selection_sort_key("bravo", lookup, canon)
    assert ka < kb


def test_selection_key_nan_loses_to_valid_icc():
    lookup = {
        "valid": IccInfo(0.5, "empirical"),
        "nan_one": IccInfo(float("nan"), "empirical"),
    }
    canon = set()
    kv = selection_sort_key("valid", lookup, canon)
    kn = selection_sort_key("nan_one", lookup, canon)
    assert kv < kn


def test_selection_key_missing_lookup_treated_as_nan():
    lookup: dict[str, IccInfo] = {}
    canon = set()
    k_missing = selection_sort_key("absent", lookup, canon)
    # First element of tuple should be nan_flag = 1.
    assert k_missing[0] == 1


# ───────────────────── select_representative_per_cluster ─────────────────────


def test_representative_singleton_cluster():
    lookup = {"solo": IccInfo(0.8, "empirical")}
    rep, rec = select_representative_per_cluster(["solo"], lookup, set())
    assert rep == "solo"
    assert rec["decided_by"] == "singleton"
    assert rec["n_members"] == 1


def test_representative_picked_by_icc():
    lookup = {
        "a": IccInfo(0.8, "empirical"),
        "b": IccInfo(0.95, "empirical"),
        "c": IccInfo(0.7, "empirical"),
    }
    rep, rec = select_representative_per_cluster(["a", "b", "c"], lookup, set())
    assert rep == "b"
    assert rec["decided_by"] == "icc"


def test_representative_canonical_tiebreaks_equal_icc():
    lookup = {
        "agatston_lad": IccInfo(1.0, "invariant_by_construction"),
        "original_glcm_X": IccInfo(1.0, "empirical"),
    }
    canon = {"agatston_lad"}
    rep, rec = select_representative_per_cluster(
        ["agatston_lad", "original_glcm_X"], lookup, canon,
    )
    assert rep == "agatston_lad"
    assert rec["decided_by"] == "canonical_tiebreak"


def test_representative_alphabetical_at_full_tie():
    lookup = {
        "agatston_lad": IccInfo(1.0, "invariant_by_construction"),
        "agatston_rca": IccInfo(1.0, "invariant_by_construction"),
    }
    canon = {"agatston_lad", "agatston_rca"}
    rep, rec = select_representative_per_cluster(
        ["agatston_lad", "agatston_rca"], lookup, canon,
    )
    assert rep == "agatston_lad"
    assert rec["decided_by"] == "alphabetical_tiebreak"


def test_representative_empty_cluster_raises():
    with pytest.raises(ValueError):
        select_representative_per_cluster([], {}, set())


def test_representative_deterministic_under_input_reordering():
    """Order of cluster_members must not change the choice."""
    lookup = {
        "a": IccInfo(0.9, "empirical"),
        "b": IccInfo(0.9, "empirical"),
        "c": IccInfo(0.9, "empirical"),
    }
    canon = {"a", "b", "c"}   # all canonical
    rep1, _ = select_representative_per_cluster(["a", "b", "c"], lookup, canon)
    rep2, _ = select_representative_per_cluster(["c", "b", "a"], lookup, canon)
    rep3, _ = select_representative_per_cluster(["b", "a", "c"], lookup, canon)
    assert rep1 == rep2 == rep3 == "a"


# ───────────────────── orchestrator ─────────────────────


def _orchestrator_synthetic() -> tuple[pd.DataFrame, list[str], dict, set]:
    """Build a small synthetic case: 3 redundancy clusters,
    one of which has a clear ICC tiebreak."""
    rng = np.random.default_rng(0)
    n = 100
    base1 = rng.normal(size=n)
    base2 = rng.normal(size=n)
    base3 = rng.normal(size=n)
    df = pd.DataFrame({
        "pid": [str(i) for i in range(n)],
        "burden_a": base1,
        "burden_b": base1 + 0.01 * rng.normal(size=n),     # tightly correlated to burden_a
        "burden_c": base1 + 0.02 * rng.normal(size=n),
        "texture_a": base2,
        "texture_b": base2 + 0.005 * rng.normal(size=n),   # tight with texture_a
        "shape_a": base3,
    })
    feat = ["burden_a", "burden_b", "burden_c",
            "texture_a", "texture_b",
            "shape_a"]
    icc_lookup = {
        "burden_a": IccInfo(1.0, "invariant_by_construction"),
        "burden_b": IccInfo(1.0, "invariant_by_construction"),
        "burden_c": IccInfo(1.0, "invariant_by_construction"),
        "texture_a": IccInfo(0.85, "empirical"),
        "texture_b": IccInfo(0.92, "empirical"),
        "shape_a": IccInfo(1.0, "empirical"),
    }
    canonical_set = {"burden_a", "burden_b", "burden_c"}
    return df, feat, icc_lookup, canonical_set


def test_orchestrator_groups_correlated_into_clusters():
    df, feat, icc_lookup, canon = _orchestrator_synthetic()
    result = run_redundancy_clustering(df, feat, icc_lookup, canon,
                                         primary_method="average",
                                         sensitivity_methods=("complete",),
                                         min_gap=0.05, fallback_distance=0.25)
    # Three independent base signals => three clusters.
    assert result.n_clusters() == 3
    # Burden cluster should pick one of burden_a/b/c.
    rep_set = set(result.representatives)
    burden_picks = rep_set & {"burden_a", "burden_b", "burden_c"}
    assert len(burden_picks) == 1
    # Texture cluster picks texture_b (higher ICC).
    assert "texture_b" in rep_set
    # Shape singleton.
    assert "shape_a" in rep_set


def test_orchestrator_cluster_assignments_columns_and_count():
    df, feat, icc_lookup, canon = _orchestrator_synthetic()
    result = run_redundancy_clustering(df, feat, icc_lookup, canon,
                                         sensitivity_methods=("complete",))
    ca = result.cluster_assignments
    expected_cols = {"feature", "cluster_id", "is_representative", "icc",
                     "icc_source", "is_canonical", "cluster_size",
                     "cluster_max_icc", "decided_by"}
    assert expected_cols.issubset(set(ca.columns))
    assert len(ca) == len(feat)
    # Exactly one representative per cluster.
    n_reps_by_cluster = ca.groupby("cluster_id")["is_representative"].sum()
    assert (n_reps_by_cluster == 1).all()


def test_orchestrator_sensitivity_per_method_includes_jaccard():
    df, feat, icc_lookup, canon = _orchestrator_synthetic()
    result = run_redundancy_clustering(df, feat, icc_lookup, canon,
                                         sensitivity_methods=("ward", "complete"))
    assert set(result.sensitivity_per_method.keys()) == {"ward", "complete"}
    for method, info in result.sensitivity_per_method.items():
        assert "n_clusters" in info
        assert "cut_threshold" in info
        assert "representatives" in info
        assert "jaccard_vs_primary" in info
        assert 0.0 <= info["jaccard_vs_primary"] <= 1.0


def test_orchestrator_deterministic_across_runs():
    df, feat, icc_lookup, canon = _orchestrator_synthetic()
    r1 = run_redundancy_clustering(df, feat, icc_lookup, canon,
                                     sensitivity_methods=())
    r2 = run_redundancy_clustering(df, feat, icc_lookup, canon,
                                     sensitivity_methods=())
    assert r1.representatives == r2.representatives
    np.testing.assert_array_equal(r1.labels, r2.labels)


def test_orchestrator_preserves_feature_order():
    df, feat, icc_lookup, canon = _orchestrator_synthetic()
    result = run_redundancy_clustering(df, feat, icc_lookup, canon,
                                         sensitivity_methods=())
    assert result.feature_order == feat


# ───────────────────── multi-block clustering (D022) ─────────────────────


def test_default_blocks_have_six():
    assert len(DEFAULT_BLOCKS) == 6
    names = {b.name for b in DEFAULT_BLOCKS}
    assert names == {"burden", "hu_statistics", "density_tier",
                     "spatial", "texture", "shape"}


def test_default_blocks_are_mutually_exclusive_on_schema():
    """Every name in feature_schema must match at most one block."""
    from predict.features.feature_schema import feature_names

    schema = list(feature_names())
    # Also include the post-D018/D019 derived names.
    schema += ["has_dense_calcium", "high_density_fraction", "vessel_burden_gini"]
    # PyRadiomics features (sample of names from each family).
    schema += [
        "original_shape_Sphericity", "original_shape_Elongation",
        "original_firstorder_Range", "original_firstorder_Mean",
        "original_glcm_Contrast", "original_glszm_ZoneEntropy",
        "original_glrlm_RunLengthNonUniformity",
        "original_ngtdm_Coarseness", "original_gldm_DependenceEntropy",
    ]
    block_to_features, _ = assign_features_to_blocks(schema)
    seen = set()
    for feats in block_to_features.values():
        for f in feats:
            assert f not in seen, f"{f} assigned to more than one block"
            seen.add(f)


def test_burden_block_contains_agatston_and_mass_and_volume():
    bf, _ = assign_features_to_blocks(
        ["agatston_lad", "mass_rca", "volume_lcx_mm3", "agatston_total"],
    )
    assert set(bf["burden"]) == {
        "agatston_lad", "mass_rca", "volume_lcx_mm3", "agatston_total",
    }


def test_hu_statistics_block_contains_firstorder_range_but_not_other_firstorders():
    bf, _ = assign_features_to_blocks([
        "max_hu_global", "mean_hu_lad",
        "original_firstorder_Range",
        "original_firstorder_Mean",
        "original_firstorder_Energy",
    ])
    assert "original_firstorder_Range" in bf["hu_statistics"]
    assert "max_hu_global" in bf["hu_statistics"]
    assert "mean_hu_lad" in bf["hu_statistics"]
    assert "original_firstorder_Mean" in bf["texture"]
    assert "original_firstorder_Energy" in bf["texture"]


def test_density_tier_block_includes_derived():
    bf, _ = assign_features_to_blocks([
        "n_rois_d1_lad", "has_dense_calcium", "high_density_fraction",
    ])
    assert set(bf["density_tier"]) == {
        "n_rois_d1_lad", "has_dense_calcium", "high_density_fraction",
    }


def test_spatial_block_includes_vessel_burden_gini():
    bf, _ = assign_features_to_blocks([
        "lesion_count_total", "n_calcified_arteries",
        "vessel_burden_gini", "center_of_mass_z",
        "inter_lesion_dist_mean_lad",
    ])
    assert set(bf["spatial"]) == {
        "lesion_count_total", "n_calcified_arteries",
        "vessel_burden_gini", "center_of_mass_z",
        "inter_lesion_dist_mean_lad",
    }


def test_shape_block_only_contains_shape_features():
    bf, _ = assign_features_to_blocks([
        "original_shape_Sphericity", "original_shape_Elongation",
    ])
    assert set(bf["shape"]) == {
        "original_shape_Sphericity", "original_shape_Elongation",
    }


def test_unassigned_features_are_returned():
    bf, unassigned = assign_features_to_blocks([
        "agatston_lad", "made_up_feature_xyz",
    ])
    assert unassigned == ["made_up_feature_xyz"]
    assert "agatston_lad" in bf["burden"]


def test_multi_block_clustering_runs_each_block_independently():
    """Construct a synthetic case where a burden feature and a shape feature
    are perfectly correlated. Single-matrix clustering would merge them
    and absorb one. Multi-block clustering keeps both because they live in
    different blocks."""
    rng = np.random.default_rng(0)
    n = 100
    base = rng.normal(size=n)
    df = pd.DataFrame({
        "pid": [str(i) for i in range(n)],
        "agatston_lad": base,
        "agatston_rca": base + 0.005 * rng.normal(size=n),   # tight with agatston_lad
        "original_shape_Sphericity": base,                    # PERFECTLY correlated with burden
    })
    feat = ["agatston_lad", "agatston_rca", "original_shape_Sphericity"]
    icc_lookup = {
        "agatston_lad": IccInfo(1.0, "invariant_by_construction"),
        "agatston_rca": IccInfo(1.0, "invariant_by_construction"),
        "original_shape_Sphericity": IccInfo(0.9, "empirical"),
    }
    canon = {"agatston_lad", "agatston_rca"}

    # Multi-block: should keep at least one rep per non-empty block.
    result = run_multi_block_redundancy_clustering(
        df, feat, icc_lookup, canon,
    )
    assert isinstance(result, MultiBlockResult)
    burden_reps = result.per_block["burden"].representatives
    shape_reps = result.per_block["shape"].representatives
    assert len(burden_reps) >= 1
    assert len(shape_reps) == 1
    assert "original_shape_Sphericity" in result.representatives


def test_multi_block_assignments_dataframe_columns():
    df, feat, icc_lookup, canon = _orchestrator_synthetic()
    # Reuse the orchestrator synthetic. Add canonical-friendly column names.
    df_aliased = df.rename(columns={
        "burden_a": "agatston_lad", "burden_b": "agatston_rca",
        "burden_c": "agatston_lcx",
        "texture_a": "original_glcm_Contrast",
        "texture_b": "original_glcm_Correlation",
        "shape_a": "original_shape_Sphericity",
    })
    feat = ["agatston_lad", "agatston_rca", "agatston_lcx",
            "original_glcm_Contrast", "original_glcm_Correlation",
            "original_shape_Sphericity"]
    icc_lookup = {
        f: IccInfo(1.0, "invariant_by_construction"
                   if f.startswith("agatston_") else "empirical")
        for f in feat
    }
    canon = {"agatston_lad", "agatston_rca", "agatston_lcx"}
    result = run_multi_block_redundancy_clustering(
        df_aliased, feat, icc_lookup, canon,
    )
    asdf = result.assignments_dataframe()
    assert "block" in asdf.columns
    assert "feature" in asdf.columns
    assert set(asdf["block"].unique()).issubset(
        {"burden", "texture", "shape"}
    )


def test_multi_block_deterministic_across_runs():
    df, feat, icc_lookup, canon = _orchestrator_synthetic()
    df_aliased = df.rename(columns={
        "burden_a": "agatston_lad", "burden_b": "agatston_rca",
        "burden_c": "agatston_lcx",
        "texture_a": "original_glcm_Contrast",
        "texture_b": "original_glcm_Correlation",
        "shape_a": "original_shape_Sphericity",
    })
    feat = ["agatston_lad", "agatston_rca", "agatston_lcx",
            "original_glcm_Contrast", "original_glcm_Correlation",
            "original_shape_Sphericity"]
    icc_lookup = {
        f: IccInfo(1.0, "invariant_by_construction"
                   if f.startswith("agatston_") else "empirical")
        for f in feat
    }
    canon = {"agatston_lad", "agatston_rca", "agatston_lcx"}
    r1 = run_multi_block_redundancy_clustering(df_aliased, feat, icc_lookup, canon)
    r2 = run_multi_block_redundancy_clustering(df_aliased, feat, icc_lookup, canon)
    assert r1.representatives == r2.representatives


def test_orchestrator_no_nan_in_audit():
    df, feat, icc_lookup, canon = _orchestrator_synthetic()
    result = run_redundancy_clustering(df, feat, icc_lookup, canon,
                                         sensitivity_methods=())
    ca = result.cluster_assignments
    # cluster_id, is_representative, cluster_size, decided_by should never be NaN.
    for col in ("cluster_id", "is_representative", "cluster_size", "decided_by"):
        assert not ca[col].isna().any()
