"""Tests for predict.stability.icc.

Covers:

- ICC(3,1) core formula on toy matrices with known answers.
- Listwise NaN handling and degeneracy.
- `build_reliability_matrix` patient-intersection logic.
- `gate_features` threshold application and NaN handling.
- The 31-feature invariant-by-construction registry against feature_schema.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predict.features.feature_schema import feature_names
from predict.stability.icc import (
    IccRecord,
    build_reliability_matrix,
    gate_features,
    icc_3_1_absolute,
    invariant_by_construction_features,
)


# ─────────────────────── icc_3_1_absolute ───────────────────────


def test_icc_identical_columns_is_one():
    """Identical raters, varying subjects => perfect agreement."""
    matrix = np.column_stack([
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    ])
    assert icc_3_1_absolute(matrix) == pytest.approx(1.0, abs=1e-9)


def test_icc_systematic_shift_penalised_under_absolute_agreement():
    """rater 2 = rater 1 + constant => consistency ICC=1, but absolute ICC<1."""
    r1 = np.array([1.0, 2.0, 3.0, 4.0])
    r2 = r1 + 10.0
    matrix = np.column_stack([r1, r2])
    icc = icc_3_1_absolute(matrix)
    assert icc < 1.0
    # The shift dominates the subject variance => ICC should be small.
    assert icc < 0.2


def test_icc_anti_correlated_columns_near_zero_or_negative():
    """Reversed ranking => no real agreement."""
    matrix = np.column_stack([
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([4.0, 3.0, 2.0, 1.0]),
    ])
    assert icc_3_1_absolute(matrix) < 0.1


def test_icc_random_noise_near_zero():
    rng = np.random.default_rng(0)
    n = 200
    base = rng.normal(0.0, 1.0, size=n)
    perturbed = rng.normal(0.0, 1.0, size=n)  # independent
    matrix = np.column_stack([base, perturbed])
    assert abs(icc_3_1_absolute(matrix)) < 0.15


def test_icc_handles_three_raters():
    """ICC(3,1) with k=3 raters that all agree exactly => 1.0."""
    col = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    matrix = np.column_stack([col, col, col])
    assert icc_3_1_absolute(matrix) == pytest.approx(1.0, abs=1e-9)


def test_icc_listwise_deletion_for_nan_rows():
    """Rows with any NaN are dropped before computation."""
    matrix = np.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [np.nan, 4.0],
        [5.0, np.nan],
    ])
    # After deletion: 3 perfectly-agreeing rows.
    assert icc_3_1_absolute(matrix) == pytest.approx(1.0, abs=1e-9)


def test_icc_returns_nan_when_fewer_than_two_complete_rows():
    matrix = np.array([
        [1.0, np.nan],
        [np.nan, 2.0],
        [3.0, 3.0],
    ])
    assert np.isnan(icc_3_1_absolute(matrix))


def test_icc_returns_nan_when_all_values_identical():
    """Zero total variance => degenerate."""
    matrix = np.full((5, 3), 7.0)
    assert np.isnan(icc_3_1_absolute(matrix))


def test_icc_raises_on_1d_input():
    with pytest.raises(ValueError, match="must be 2D"):
        icc_3_1_absolute(np.array([1.0, 2.0, 3.0]))


def test_icc_matches_manual_calculation():
    """Spot-check against a hand-derived ICC value."""
    matrix = np.array([
        [0.5, 0.5],
        [1.5, 1.7],
        [2.5, 2.2],
        [3.5, 3.7],
    ])
    n, k = matrix.shape
    grand = matrix.mean()
    subj = matrix.mean(axis=1)
    rater = matrix.mean(axis=0)
    ss_s = k * ((subj - grand) ** 2).sum()
    ss_r = n * ((rater - grand) ** 2).sum()
    ss_t = ((matrix - grand) ** 2).sum()
    ss_e = ss_t - ss_s - ss_r
    ms_s = ss_s / (n - 1)
    ms_r = ss_r / (k - 1)
    ms_e = ss_e / ((n - 1) * (k - 1))
    expected = (ms_s - ms_e) / (
        ms_s + (k - 1) * ms_e + (k * (ms_r - ms_e) / n)
    )
    assert icc_3_1_absolute(matrix) == pytest.approx(expected, abs=1e-12)


# ─────────────────────── build_reliability_matrix ───────────────────────


def _baseline(pids: list[str], values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"pid": pids, "f1": values})


def _pert(pids: list[str], values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"pid": pids, "f1": values})


def test_build_matrix_baseline_first_then_perturbations():
    baseline = _baseline(["1", "2", "3"], [10.0, 20.0, 30.0])
    perts = {
        "rotate_+5": _pert(["1", "2", "3"], [10.1, 20.2, 30.3]),
        "noise_5":   _pert(["1", "2", "3"], [10.5, 19.5, 30.1]),
    }
    matrix, pids, raters = build_reliability_matrix("f1", baseline, perts)
    assert raters == ["baseline", "rotate_+5", "noise_5"]
    assert pids == ["1", "2", "3"]
    assert matrix.shape == (3, 3)
    np.testing.assert_allclose(matrix[:, 0], [10.0, 20.0, 30.0])
    np.testing.assert_allclose(matrix[:, 1], [10.1, 20.2, 30.3])
    np.testing.assert_allclose(matrix[:, 2], [10.5, 19.5, 30.1])


def test_build_matrix_intersects_patient_ids():
    """A patient missing from one perturbation is excluded from the matrix."""
    baseline = _baseline(["1", "2", "3"], [10.0, 20.0, 30.0])
    perts = {
        "rotate_+5": _pert(["1", "2"],      [10.1, 20.2]),       # missing 3
        "noise_5":   _pert(["1", "2", "3"], [10.5, 19.5, 30.1]),
    }
    matrix, pids, _ = build_reliability_matrix("f1", baseline, perts)
    assert pids == ["1", "2"]
    assert matrix.shape == (2, 3)


def test_build_matrix_raises_on_missing_feature_column():
    baseline = pd.DataFrame({"pid": ["1"], "f1": [1.0]})
    perts = {"rotate_+5": pd.DataFrame({"pid": ["1"], "f2": [1.0]})}
    with pytest.raises(KeyError, match="f1"):
        build_reliability_matrix("f1", baseline, perts)


def test_build_matrix_raises_on_missing_pid_column():
    baseline = pd.DataFrame({"id": ["1"], "f1": [1.0]})
    perts = {"rotate_+5": pd.DataFrame({"pid": ["1"], "f1": [1.0]})}
    with pytest.raises(KeyError, match="pid"):
        build_reliability_matrix("f1", baseline, perts)


def test_build_matrix_empty_intersection_returns_empty():
    baseline = _baseline(["1"], [1.0])
    perts = {"rotate_+5": _pert(["2"], [2.0])}
    matrix, pids, raters = build_reliability_matrix("f1", baseline, perts)
    assert matrix.size == 0
    assert pids == []
    assert raters == []


# ─────────────────────── gate_features ───────────────────────


def _rec(feature: str, icc: float, source: str = "empirical") -> IccRecord:
    return IccRecord(
        feature=feature,
        icc=icc,
        icc_source=source,  # type: ignore[arg-type]
        n_subjects=100,
        n_raters=15,
        passes_gate=False,
    )


def test_gate_features_above_threshold_passes():
    records = [_rec("f_high", 0.90), _rec("f_low", 0.50)]
    updated, passing = gate_features(records, threshold=0.75)
    assert passing == ["f_high"]
    assert updated[0].passes_gate is True
    assert updated[1].passes_gate is False


def test_gate_features_exactly_at_threshold_passes():
    records = [_rec("f_edge", 0.75)]
    _, passing = gate_features(records, threshold=0.75)
    assert passing == ["f_edge"]


def test_gate_features_nan_icc_fails():
    records = [_rec("f_nan", float("nan"))]
    _, passing = gate_features(records, threshold=0.75)
    assert passing == []


def test_gate_features_recomputes_passes_gate_field():
    """The input passes_gate value is ignored; the threshold rule is canonical."""
    bad = IccRecord(
        feature="f", icc=0.50, icc_source="empirical",
        n_subjects=10, n_raters=15, passes_gate=True,  # wrong
    )
    updated, passing = gate_features([bad], threshold=0.75)
    assert updated[0].passes_gate is False
    assert passing == []


# ─────────────────────── invariant_by_construction registry ───────────────────────


def test_bypass_registry_has_68_features():
    """D016: all 68 canonical features bypass the empirical D014 gate
    because the pipeline reads HU values from the XML's
    frozen Max / Mean fields, never from the CT array."""
    assert len(invariant_by_construction_features()) == 68


def test_bypass_registry_no_duplicates():
    feats = invariant_by_construction_features()
    assert len(set(feats)) == len(feats)


def test_bypass_registry_is_exactly_the_schema():
    """The bypass list is the canonical schema (one source of truth)."""
    assert set(invariant_by_construction_features()) == set(feature_names())


def test_bypass_registry_includes_all_per_vessel_stems():
    """Every per-vessel canonical feature (geometric AND HU-touching) is in
    the bypass because none of them read the CT array."""
    bypass = set(invariant_by_construction_features())
    expected_stems = (
        "lesion_count",
        "max_hu", "mean_hu",
        "mass", "agatston",
        "inter_lesion_dist_mean",
        "inter_lesion_dist_max",
        "first_to_last_dist",
        "diffusivity",
        "n_rois_d1", "n_rois_d2", "n_rois_d3", "n_rois_d4",
    )
    for vessel in ("lad", "rca", "lcx", "lm"):
        for stem in expected_stems:
            assert f"{stem}_{vessel}" in bypass, f"missing {stem}_{vessel}"
        # volume has a different naming convention.
        assert f"volume_{vessel}_mm3" in bypass


def test_bypass_registry_includes_all_globals():
    bypass = set(invariant_by_construction_features())
    for g in (
        "lesion_count_total",
        "n_calcified_arteries",
        "gini_lesion_volume",
        "dist_from_top_max",
        "dist_from_top_mean",
        "dense_calcium_count",
        "agatston_total",
        "volume_total_mm3",
        "mass_total",
        "mean_hu_weighted_global",
        "max_hu_global",
        "center_of_mass_z",
    ):
        assert g in bypass


def test_bypass_registry_excludes_pyradiomics():
    """Anti-regression: PyRadiomics features must NEVER be in the bypass list."""
    bypass = set(invariant_by_construction_features())
    for prefix in ("original_shape_", "original_firstorder_", "original_glcm_",
                   "original_glszm_", "original_glrlm_", "original_ngtdm_",
                   "original_gldm_"):
        assert not any(f.startswith(prefix) for f in bypass), \
            f"PyRadiomics ({prefix}*) leaked into the bypass list"


def test_bypass_registry_excludes_metadata():
    """Anti-regression: metadata columns are not features and must not appear."""
    bypass = set(invariant_by_construction_features())
    for m in ("pid", "kernel", "scanner_model", "mask_voxels", "low_burden_flag",
              "roundtrip_quality", "category", "radiomics_status", "radiomics_reason"):
        assert m not in bypass
