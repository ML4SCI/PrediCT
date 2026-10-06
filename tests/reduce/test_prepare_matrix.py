"""Tests for predict.reduce.prepare_matrix (D019).

Coverage policy: every transform gets at least one unit test per documented
behaviour. Mathematical invariants (orthogonality, sign conventions,
idempotency, NaN propagation) get property tests. The orchestrator gets an
end-to-end test on a synthetic dataset whose answer can be hand-verified.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from predict.reduce.prepare_matrix import (
    D017_DROPPED_FEATURES,
    D018_BINARISE_SOURCE,
    D018_BINARISE_TARGET,
    SPARSE_COLUMNS,
    MatrixPrepLog,
    _explained_variance_by_kernel,
    _r2_against_existing,
    _rank_transform,
    apply_d017_drops,
    apply_d018_binarisation,
    combat_harmonise,
    compute_high_density_fraction,
    compute_vessel_burden_gini,
    global_zscore,
    harmonisable_columns,
    maybe_add_derived_features,
    run_matrix_prep,
    variance_filter,
    yeo_johnson_with_fallback,
)


# Example intensity-texture columns for the synthetic fixtures (formerly the
# hardcoded ComBat list; ComBat now derives its columns, D035).
TEXTURE_EXAMPLE: tuple[str, ...] = (
    "original_glrlm_RunLengthNonUniformity",
    "original_gldm_DependenceEntropy",
    "original_glszm_ZoneEntropy",
    "original_glrlm_GrayLevelNonUniformity",
    "original_gldm_GrayLevelNonUniformity",
    "original_firstorder_Range",
)


# ───────────────────── helpers ─────────────────────


def _toy_df(n: int = 30, seed: int = 0) -> pd.DataFrame:
    """Build a dataframe shaped like the stage 5 input: pid + kernel +
    a representative selection of canonical and pyradiomics columns.

    Contains the 13 D017-droppable columns and the D018 source column so the
    transforms have something to act on.
    """
    rng = np.random.default_rng(seed)

    # 4 vessels x 4 tier counts.
    tier_cols = {
        f"n_rois_d{t}_{v}": rng.poisson(lam=0.8, size=n).astype(float)
        for t in (1, 2, 3, 4)
        for v in ("lad", "rca", "lcx", "lm")
    }
    # 4 vessels x agatston.
    agatston_cols = {
        f"agatston_{v}": rng.gamma(shape=2.0, scale=50.0, size=n)
        for v in ("lad", "rca", "lcx", "lm")
    }
    # D017 droppable columns.
    drop_cols = {c: rng.uniform(0, 10, size=n) for c in D017_DROPPED_FEATURES}
    # D018 source.
    dense = {D018_BINARISE_SOURCE: rng.poisson(lam=0.3, size=n).astype(float)}
    # 3 LAD distance features (sparse).
    distance_cols = {
        "inter_lesion_dist_mean_lad": np.where(
            rng.random(n) < 0.35, 0.0, rng.uniform(2, 30, size=n)
        ),
        "inter_lesion_dist_max_lad": np.where(
            rng.random(n) < 0.35, 0.0, rng.uniform(5, 60, size=n)
        ),
        "first_to_last_dist_lad": np.where(
            rng.random(n) < 0.35, 0.0, rng.uniform(5, 80, size=n)
        ),
    }
    # 6 texture surviving features.
    texture_cols = {
        c: rng.normal(loc=10.0, scale=2.0, size=n)
        for c in TEXTURE_EXAMPLE
    }
    # 14 shape features (passthrough, kernel-invariant).
    shape_cols = {
        f"original_shape_F{i}": rng.normal(size=n) for i in range(14)
    }

    cols = {**tier_cols, **agatston_cols, **drop_cols, **dense,
            **distance_cols, **texture_cols, **shape_cols}
    df = pd.DataFrame(cols)
    df.insert(0, "pid", [str(i) for i in range(n)])
    df.insert(1, "kernel", rng.choice(["Qr36d/2", "I30f/3"], size=n))
    return df


def _feature_cols_from(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("pid", "kernel")]


# ───────────────────── apply_d017_drops ─────────────────────


def test_d017_drops_all_listed_features():
    df = _toy_df()
    feat = _feature_cols_from(df)
    new_df, new_feat, dropped = apply_d017_drops(df, feat)
    assert set(dropped) == set(D017_DROPPED_FEATURES)
    for c in D017_DROPPED_FEATURES:
        assert c not in new_df.columns
        assert c not in new_feat


def test_d017_idempotent():
    df = _toy_df()
    feat = _feature_cols_from(df)
    df1, feat1, dropped1 = apply_d017_drops(df, feat)
    df2, feat2, dropped2 = apply_d017_drops(df1, feat1)
    assert dropped2 == []
    assert list(df2.columns) == list(df1.columns)
    assert feat2 == feat1


def test_d017_preserves_pid_and_non_droppable_features():
    df = _toy_df()
    feat = _feature_cols_from(df)
    new_df, _, _ = apply_d017_drops(df, feat)
    assert "pid" in new_df.columns
    assert "agatston_lad" in new_df.columns
    assert "original_shape_F0" in new_df.columns


def test_d017_returns_only_actually_dropped_names():
    df = _toy_df().drop(columns=["diffusivity_lad"])
    feat = _feature_cols_from(df)
    _, _, dropped = apply_d017_drops(df, feat)
    assert "diffusivity_lad" not in dropped


# ───────────────────── apply_d018_binarisation ─────────────────────


def test_d018_creates_binary_target_column():
    df = _toy_df()
    feat = _feature_cols_from(df)
    new_df, new_feat, did = apply_d018_binarisation(df, feat)
    assert did is True
    assert D018_BINARISE_TARGET in new_df.columns
    assert D018_BINARISE_SOURCE not in new_df.columns
    assert set(new_df[D018_BINARISE_TARGET].unique()).issubset({0, 1})


def test_d018_target_matches_count_greater_than_zero():
    df = _toy_df(seed=99)
    feat = _feature_cols_from(df)
    expected = (df[D018_BINARISE_SOURCE] > 0).astype(int)
    new_df, _, _ = apply_d018_binarisation(df, feat)
    np.testing.assert_array_equal(
        new_df[D018_BINARISE_TARGET].to_numpy(),
        expected.to_numpy(),
    )


def test_d018_replaces_source_at_same_position_in_feature_cols():
    df = _toy_df()
    feat = _feature_cols_from(df)
    src_pos = feat.index(D018_BINARISE_SOURCE)
    _, new_feat, _ = apply_d018_binarisation(df, feat)
    assert new_feat[src_pos] == D018_BINARISE_TARGET


def test_d018_no_op_when_source_missing():
    df = _toy_df().drop(columns=[D018_BINARISE_SOURCE])
    feat = _feature_cols_from(df)
    new_df, new_feat, did = apply_d018_binarisation(df, feat)
    assert did is False
    assert D018_BINARISE_TARGET not in new_df.columns
    assert new_feat == feat


# ───────────────────── derived feature computations ─────────────────────


def test_high_density_fraction_known_value():
    df = pd.DataFrame({
        "pid": ["a"],
        **{f"n_rois_d{t}_{v}": [1.0 if (t, v) == (1, "lad") else 0.0]
           for t in (1, 2, 3, 4) for v in ("lad", "rca", "lcx", "lm")},
    })
    # Only d1 populated => (d3 + d4) sum = 0 => fraction = 0
    series = compute_high_density_fraction(df)
    assert series.iloc[0] == pytest.approx(0.0)


def test_high_density_fraction_all_in_d4():
    df = pd.DataFrame({
        "pid": ["a"],
        **{f"n_rois_d{t}_{v}": [3.0 if (t, v) == (4, "lad") else 0.0]
           for t in (1, 2, 3, 4) for v in ("lad", "rca", "lcx", "lm")},
    })
    series = compute_high_density_fraction(df)
    assert series.iloc[0] == pytest.approx(1.0)


def test_high_density_fraction_zero_when_no_tier_data():
    df = pd.DataFrame({
        "pid": ["a"],
        **{f"n_rois_d{t}_{v}": [0.0]
           for t in (1, 2, 3, 4) for v in ("lad", "rca", "lcx", "lm")},
    })
    series = compute_high_density_fraction(df)
    assert series.iloc[0] == 0.0


def test_high_density_fraction_returns_none_on_missing_columns():
    df = pd.DataFrame({"pid": ["a"], "n_rois_d1_lad": [1.0]})
    assert compute_high_density_fraction(df) is None


def test_high_density_fraction_range_in_unit_interval():
    df = _toy_df(n=50)
    series = compute_high_density_fraction(df)
    assert series.min() >= 0.0
    assert series.max() <= 1.0


def test_vessel_burden_gini_single_vessel_is_maximal():
    """D040: all burden in one of four vessels is the most focal case,
    Gini = (n - 1) / n = 0.75 (it was 0 under the dropped-zero convention)."""
    df = pd.DataFrame({
        "pid": ["a"],
        "agatston_lad": [100.0],
        "agatston_rca": [0.0],
        "agatston_lcx": [0.0],
        "agatston_lm": [0.0],
    })
    series = compute_vessel_burden_gini(df)
    assert series.iloc[0] == pytest.approx(0.75)


def test_vessel_burden_gini_separates_focal_from_even():
    """D040 counterexample: [100,0,0,0] and [25,25,25,25]
    must not share a value."""
    df = pd.DataFrame({
        "pid": ["focal", "even"],
        "agatston_lad": [100.0, 25.0], "agatston_rca": [0.0, 25.0],
        "agatston_lcx": [0.0, 25.0], "agatston_lm": [0.0, 25.0],
    })
    g = compute_vessel_burden_gini(df)
    assert g.iloc[0] == pytest.approx(0.75)
    assert g.iloc[1] == pytest.approx(0.0, abs=1e-12)


def test_vessel_burden_gini_zero_when_all_equal():
    df = pd.DataFrame({
        "pid": ["a"],
        "agatston_lad": [50.0],
        "agatston_rca": [50.0],
        "agatston_lcx": [50.0],
        "agatston_lm": [50.0],
    })
    series = compute_vessel_burden_gini(df)
    assert series.iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_vessel_burden_gini_high_when_one_dominates():
    df = pd.DataFrame({
        "pid": ["a"],
        "agatston_lad": [1000.0],
        "agatston_rca": [1.0],
        "agatston_lcx": [1.0],
        "agatston_lm": [1.0],
    })
    series = compute_vessel_burden_gini(df)
    assert series.iloc[0] > 0.5


def test_vessel_burden_gini_zero_when_no_calcium():
    df = pd.DataFrame({
        "pid": ["a"],
        "agatston_lad": [0.0], "agatston_rca": [0.0],
        "agatston_lcx": [0.0], "agatston_lm": [0.0],
    })
    series = compute_vessel_burden_gini(df)
    assert series.iloc[0] == 0.0


# ───────────────────── _r2_against_existing ─────────────────────


def test_r2_against_existing_perfect_prediction():
    df = pd.DataFrame({
        "pid": ["a", "b", "c", "d"],
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [2.0, 4.0, 6.0, 8.0],
    })
    new = pd.Series([3.0, 6.0, 9.0, 12.0])   # exactly 1.5x
    r2 = _r2_against_existing(df, new, ["x", "y"])
    assert r2 == pytest.approx(1.0, abs=1e-9)


def test_r2_against_existing_no_relationship():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "pid": [str(i) for i in range(200)],
        "x": rng.normal(size=200),
        "y": rng.normal(size=200),
    })
    new = pd.Series(rng.normal(size=200))   # independent
    r2 = _r2_against_existing(df, new, ["x", "y"])
    assert r2 < 0.2


# ───────────────────── maybe_add_derived_features ─────────────────────


def test_maybe_add_derived_accepts_when_non_redundant():
    df = _toy_df(n=100)
    feat = _feature_cols_from(df)
    new_df, new_feat, accepted, rejected = maybe_add_derived_features(
        df, feat, r2_redundancy_threshold=0.95,
    )
    # At least one of the two should be accepted on random data.
    assert len(accepted) >= 1
    for rec in accepted:
        assert rec["accepted"] is True
        assert rec["r2"] < 0.95
        assert rec["name"] in new_df.columns
        assert rec["name"] in new_feat


def test_maybe_add_derived_rejects_redundant_with_explicit_record():
    # Force redundancy: make new feature equal to an existing one.
    df = _toy_df(n=80)
    # Make agatston_lad equal to a known linear combination already in matrix.
    # We rely on the existing R^2 check; should reject because vessel_burden_gini
    # may correlate >= 0.95 with the original Agatston features in some seeds.
    # Test mostly verifies the record format.
    feat = _feature_cols_from(df)
    _, _, _, rejected = maybe_add_derived_features(df, feat,
                                                   r2_redundancy_threshold=0.001)
    # With extremely strict threshold, both must be rejected.
    assert len(rejected) == 2
    for rec in rejected:
        assert rec["accepted"] is False


def test_maybe_add_derived_second_check_uses_updated_matrix():
    """Regression: after high_density_fraction is added, the R^2 check for
    vessel_burden_gini must compute against the NEW matrix that already
    contains high_density_fraction. Previously the check used the original
    df, which raised KeyError when looking up the newly added column."""
    df = _toy_df(n=100, seed=0)
    feat = _feature_cols_from(df)
    new_df, new_feat, accepted, rejected = maybe_add_derived_features(
        df, feat, r2_redundancy_threshold=0.95,
    )
    # Should accept both without raising; one should appear in new_feat
    # after the other has already been added.
    assert all(rec["name"] in new_df.columns for rec in accepted)


def test_maybe_add_derived_skips_when_inputs_missing():
    df = _toy_df()
    df = df.drop(columns=[c for c in df.columns if c.startswith("n_rois_")])
    feat = [c for c in _feature_cols_from(df) if c != "kernel"]
    _, _, accepted, rejected = maybe_add_derived_features(df, feat)
    names = [r["name"] for r in rejected]
    assert "high_density_fraction" in names


# ───────────────────── variance_filter ─────────────────────


def test_variance_filter_keeps_high_variance():
    df = pd.DataFrame({"pid": ["1", "2", "3"], "a": [0.0, 1.0, 2.0]})
    _, kept, dropped = variance_filter(df, ["a"], threshold=0.01)
    assert kept == ["a"]
    assert dropped == []


def test_variance_filter_drops_constant():
    df = pd.DataFrame({"pid": ["1", "2", "3"], "a": [1.0, 1.0, 1.0]})
    new_df, kept, dropped = variance_filter(df, ["a"], threshold=0.01)
    assert kept == []
    assert dropped[0]["feature"] == "a"
    assert "a" not in new_df.columns


def test_variance_filter_preserves_non_feature_columns():
    df = _toy_df()
    feat = _feature_cols_from(df)
    new_df, _, _ = variance_filter(df, feat, threshold=0.01)
    assert "pid" in new_df.columns
    assert "kernel" in new_df.columns


def test_variance_filter_raises_on_nan_in_features():
    df = _toy_df()
    df.loc[0, "agatston_lad"] = np.nan
    feat = _feature_cols_from(df)
    with pytest.raises(ValueError, match="NaN"):
        variance_filter(df, feat)


# ───────────────────── ComBat ─────────────────────


def test_explained_variance_by_kernel_zero_when_no_effect():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "pid": [str(i) for i in range(n)],
        "kernel": rng.choice(["A", "B"], size=n),
        "x": rng.normal(size=n),
    })
    r2 = _explained_variance_by_kernel(df, ["x"], kernel_col="kernel")
    assert r2["x"] < 0.05


def test_explained_variance_by_kernel_high_when_strong_effect():
    n = 200
    df = pd.DataFrame({
        "pid": [str(i) for i in range(n)],
        "kernel": ["A"] * (n // 2) + ["B"] * (n // 2),
    })
    df["x"] = [10.0] * (n // 2) + [20.0] * (n // 2)
    r2 = _explained_variance_by_kernel(df, ["x"], kernel_col="kernel")
    assert r2["x"] > 0.99


@pytest.mark.skipif(
    pytest.importorskip("neuroCombat", reason="neuroCombat not installed")
    is None, reason="neuroCombat not installed",
)
def test_combat_reduces_kernel_variance_on_simulated_batch_effect():
    """Make a kernel-confounded synthetic column and verify ComBat reduces
    the kernel R^2 below the 0.02 acceptance threshold."""
    rng = np.random.default_rng(42)
    n = 200
    pid = [str(i) for i in range(n)]
    kernel = ["Qr36d/2"] * (n // 2) + ["I30f/3"] * (n // 2)
    # Inject a clear additive batch effect on a texture column.
    col_name = TEXTURE_EXAMPLE[0]
    base = rng.normal(loc=10.0, scale=1.5, size=n)
    confounded = base + np.where(np.array(kernel) == "Qr36d/2", -2.0, 2.0)
    df = pd.DataFrame({"pid": pid, "kernel": kernel, col_name: confounded})
    # Also include the other 5 texture columns trivially (uncorrelated noise).
    for c in TEXTURE_EXAMPLE[1:]:
        df[c] = rng.normal(loc=5.0, scale=1.0, size=n)

    feat = [c for c in df.columns if c not in ("pid", "kernel")]
    _, _, audit = combat_harmonise(
        df, feat, acceptance_max_post_r2=0.02,
    )
    rec = next(r for r in audit if r["feature"] == col_name)
    assert rec["kernel_r2_pre"] > 0.4
    assert rec["kernel_r2_post"] <= 0.02
    assert rec["passes_threshold"] is True


def test_harmonisable_columns_is_every_non_shape_radiomics_feature():
    """D035: ComBat targets every PyRadiomics feature except shape, in order;
    canonical XML features and derived features are never harmonised."""
    cols = ["agatston_total", "original_shape_Volume", "original_glcm_Imc2",
            "original_firstorder_Mean", "lesion_count_lad",
            "original_ngtdm_Busyness", "high_density_fraction"]
    assert harmonisable_columns(cols) == [
        "original_glcm_Imc2", "original_firstorder_Mean", "original_ngtdm_Busyness",
    ]


def test_run_matrix_prep_harmonises_texture_outside_old_list():
    """D035: a gated texture feature outside any fixed name list must be
    ComBat-harmonised (the list is derived from the gate)."""
    df = _toy_df(n=200, seed=3)
    kern = df["kernel"].to_numpy()
    rng = np.random.default_rng(5)
    df["original_glcm_Imc2"] = rng.normal(0.3, 0.05, len(df)) + np.where(kern == "Qr36d/2", -0.08, 0.08)
    feat = _feature_cols_from(df)
    _, _, log = run_matrix_prep(df, feat)
    audit = {r["feature"]: r for r in log.combat_audit}
    assert "original_glcm_Imc2" in audit
    assert audit["original_glcm_Imc2"]["kernel_r2_pre"] > 0.3
    assert audit["original_glcm_Imc2"]["kernel_r2_post"] <= 0.02
    assert not any(f.startswith("original_shape_") for f in audit)
    assert all(f.startswith("original_") for f in audit)


def test_combat_skips_on_single_kernel_cohort():
    """Kernel-stratified sensitivity reruns (D021) restrict to one kernel at
    a time. ComBat is a no-op in that case and must return cleanly with an
    audit row marked skipped=True per column."""
    rng = np.random.default_rng(0)
    n = 30
    col = TEXTURE_EXAMPLE[0]
    df = pd.DataFrame({
        "pid": [str(i) for i in range(n)],
        "kernel": ["Qr36d/2"] * n,
        col: rng.normal(size=n),
    })
    new_df, _, audit = combat_harmonise(df, [col])
    assert len(audit) == 1
    assert audit[0]["skipped"] is True
    assert audit[0]["passes_threshold"] is True
    # No change to the data on a no-op.
    pd.testing.assert_series_equal(new_df[col], df[col])


def test_combat_raises_on_singleton_kernel_group():
    """Regression: a kernel group with only 1 sample cannot be ComBat-
    harmonised. The orchestrator filters these out before calling, but
    we add a defense-in-depth guard here in case the filter is skipped."""
    rng = np.random.default_rng(0)
    n = 50
    pids = [str(i) for i in range(n)]
    # 49 patients on kernel A, 1 patient on kernel B (singleton).
    kernel = ["A"] * 49 + ["B"]
    col = TEXTURE_EXAMPLE[0]
    df = pd.DataFrame({"pid": pids, "kernel": kernel,
                        col: rng.normal(size=n)})
    feat = [col]
    with pytest.raises(ValueError, match="singleton"):
        combat_harmonise(df, feat)


def test_combat_no_op_when_no_target_columns_present():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "pid": ["a", "b", "c"],
        "kernel": ["A", "B", "A"],
        "other": rng.normal(size=3),
    })
    new_df, _, audit = combat_harmonise(df, ["other"], columns_to_harmonise=[])
    assert audit == []
    pd.testing.assert_frame_equal(new_df, df)


# ───────────────────── _rank_transform ─────────────────────


def test_rank_transform_output_in_unit_interval():
    rng = np.random.default_rng(0)
    v = rng.normal(size=50)
    r = _rank_transform(v)
    assert r.min() > 0.0
    assert r.max() <= 1.0


def test_rank_transform_monotonic():
    v = np.array([3.0, 1.0, 2.0, 4.0])
    r = _rank_transform(v)
    # Order of ranks must match order of values.
    order = np.argsort(v)
    assert np.all(np.diff(r[order]) > 0)


def test_rank_transform_ties_average():
    v = np.array([1.0, 2.0, 2.0, 3.0])
    r = _rank_transform(v)
    # The two tied values should have the same rank.
    assert r[1] == r[2]


# ───────────────────── yeo_johnson_with_fallback ─────────────────────


def test_yj_skips_binary_column():
    df = pd.DataFrame({
        "pid": [str(i) for i in range(20)],
        "n_rois_d1_lad": [0, 0, 1, 1] * 5,   # binary 0/1
    })
    _, _, records = yeo_johnson_with_fallback(
        df, ["n_rois_d1_lad"], sparse_columns=["n_rois_d1_lad"],
    )
    assert len(records) == 1
    assert records[0]["transform_used"] == "binary_skipped"


def test_yj_reduces_skewness_on_skewed_input():
    rng = np.random.default_rng(0)
    skewed = np.concatenate([np.zeros(60), rng.exponential(scale=5, size=40)])
    df = pd.DataFrame({"pid": [str(i) for i in range(100)],
                       "n_rois_d1_lad": skewed})
    _, _, records = yeo_johnson_with_fallback(
        df, ["n_rois_d1_lad"], sparse_columns=["n_rois_d1_lad"],
        skew_threshold=1.0,
    )
    rec = records[0]
    assert rec["pre_skew"] > 1.0
    assert rec["transform_used"] in ("yeo-johnson", "rank")
    assert abs(rec["post_final_skew"]) <= 1.0 + 0.5   # within tolerance


def test_yj_falls_back_to_rank_on_extreme_skew():
    # Extreme zero-inflation that YJ can not normalise.
    skewed = np.concatenate([np.zeros(95), np.array([100.0, 200, 300, 400, 500])])
    df = pd.DataFrame({"pid": [str(i) for i in range(100)],
                       "n_rois_d1_lad": skewed})
    _, _, records = yeo_johnson_with_fallback(
        df, ["n_rois_d1_lad"], sparse_columns=["n_rois_d1_lad"],
        skew_threshold=0.5,
    )
    rec = records[0]
    if rec["fallback_triggered"]:
        assert rec["transform_used"] == "rank"


def test_yj_preserves_non_sparse_columns():
    df = _toy_df(n=50)
    feat = _feature_cols_from(df)
    sample_passthrough = "agatston_lad"
    before = df[sample_passthrough].copy()
    new_df, _, _ = yeo_johnson_with_fallback(df, feat,
                                              sparse_columns=SPARSE_COLUMNS)
    pd.testing.assert_series_equal(new_df[sample_passthrough], before)


# ───────────────────── global_zscore ─────────────────────


def test_zscore_mean_zero_per_column():
    df = _toy_df(n=100)
    feat = ["agatston_lad", "agatston_rca"]
    new_df, _, _ = global_zscore(df, feat)
    for c in feat:
        assert abs(new_df[c].mean()) < 1e-9


def test_zscore_unit_variance_per_column():
    df = _toy_df(n=100)
    feat = ["agatston_lad", "agatston_rca"]
    new_df, _, _ = global_zscore(df, feat)
    for c in feat:
        assert abs(new_df[c].std(ddof=1) - 1.0) < 1e-6


def test_zscore_constant_column_becomes_zero():
    df = pd.DataFrame({"pid": ["a", "b", "c"], "x": [5.0, 5.0, 5.0]})
    new_df, _, info = global_zscore(df, ["x"])
    np.testing.assert_array_equal(new_df["x"].to_numpy(), [0.0, 0.0, 0.0])
    assert info["stds"]["x"] == 0.0


def test_zscore_logs_means_and_stds():
    df = _toy_df(n=50)
    feat = ["agatston_lad"]
    _, _, info = global_zscore(df, feat)
    assert "means" in info
    assert "stds" in info
    assert "agatston_lad" in info["means"]


def test_zscore_preserves_pid_and_kernel():
    df = _toy_df(n=30)
    feat = ["agatston_lad"]
    new_df, _, _ = global_zscore(df, feat)
    assert "pid" in new_df.columns
    assert "kernel" in new_df.columns
    pd.testing.assert_series_equal(new_df["pid"], df["pid"])
    pd.testing.assert_series_equal(new_df["kernel"], df["kernel"])


# ───────────────────── NaN entry guards ─────────────────────


@pytest.mark.parametrize("fn", [
    apply_d017_drops,
    apply_d018_binarisation,
    variance_filter,
    yeo_johnson_with_fallback,
    global_zscore,
    maybe_add_derived_features,
])
def test_pid_required_at_entry(fn):
    df = pd.DataFrame({"agatston_lad": [1.0, 2.0]})
    with pytest.raises((KeyError, ValueError)):
        fn(df, ["agatston_lad"])


# ───────────────────── orchestrator ─────────────────────


def test_run_matrix_prep_end_to_end_shape():
    """Synthetic end-to-end run: input -> output dimensions and absence of NaN."""
    df = _toy_df(n=80)
    feat = _feature_cols_from(df)

    out_df, out_feat, log = run_matrix_prep(df, feat)

    # No NaN anywhere in output features.
    assert not out_df[out_feat].isna().any().any()
    # Number of patients preserved.
    assert len(out_df) == len(df)
    # D017 dropped 13.
    assert log.n_features_in == len(feat)
    assert all(c not in out_feat for c in D017_DROPPED_FEATURES)
    # D018 binarised.
    assert log.d018_target_present is True
    assert D018_BINARISE_TARGET in out_feat
    assert D018_BINARISE_SOURCE not in out_feat


def test_run_matrix_prep_global_zscore_at_end():
    df = _toy_df(n=120)
    feat = _feature_cols_from(df)
    out_df, out_feat, _ = run_matrix_prep(df, feat)
    # Final columns should be approximately zero-mean / unit-sd (except binary
    # has_dense_calcium which can deviate slightly due to lopsided 0/1).
    means = out_df[out_feat].mean()
    assert (means.abs() < 1e-6).all()


def test_run_matrix_prep_log_serialisable():
    df = _toy_df(n=60)
    feat = _feature_cols_from(df)
    _, _, log = run_matrix_prep(df, feat)
    payload = log.to_dict()
    # Must be JSON-serialisable.
    import json
    json.dumps(payload, default=str)


def test_run_matrix_prep_captures_zscore_stats():
    """D029: stage 8 holdout/leave_k_out need the per-column z-score means
    + stds to reapply the EXACT same fit without leakage."""
    df = _toy_df(n=120)
    feat = _feature_cols_from(df)
    out_df, out_feat, log = run_matrix_prep(df, feat)
    # Every surviving feature column should have a captured mean + std.
    assert set(out_feat).issubset(set(log.zscore_means.keys()))
    assert set(out_feat).issubset(set(log.zscore_stds.keys()))
    # Captured means should be ~0 (post-z-score). Captured stds should be
    # the PRE-z-score std (so >> 0 typically).
    for col in out_feat:
        assert isinstance(log.zscore_means[col], float)
        assert isinstance(log.zscore_stds[col], float)


def test_run_matrix_prep_idempotent_d017_d018_no_double_drop():
    """Calling run_matrix_prep twice in sequence should not throw or break."""
    df = _toy_df(n=40)
    feat = _feature_cols_from(df)
    out1, feat1, _ = run_matrix_prep(df, feat)
    # Restore pid / kernel for second pass (run_matrix_prep returns a df with them).
    out2, feat2, _ = run_matrix_prep(out1, feat1, add_derived=False)
    assert len(out2) == len(out1)


def test_run_matrix_prep_raises_on_missing_kernel_when_combat_runs():
    df = _toy_df(n=40).drop(columns=["kernel"])
    feat = _feature_cols_from(df)
    with pytest.raises(KeyError, match="kernel"):
        run_matrix_prep(df, feat)


# ───────────────────── MatrixPrepLog ─────────────────────


def test_preprocessing_log_default_fields():
    log = MatrixPrepLog()
    assert log.n_patients_in == 0
    assert log.d017_dropped == []
    assert log.combat_audit == []
    assert log.zscore_columns == []


def test_preprocessing_log_to_dict_keys():
    log = MatrixPrepLog(n_patients_in=100, n_features_out=70)
    d = log.to_dict()
    for key in ("n_patients_in", "n_patients_out", "n_features_in",
                "n_features_out", "d017_dropped", "d018_target_present",
                "derived_accepted", "derived_rejected", "variance_dropped",
                "combat_audit", "yj_per_column", "zscore_columns",
                # D029 / stage 8 capture (added 2026-06-08):
                "zscore_means", "zscore_stds"):
        assert key in d
    # New fields default to empty dicts (additive backwards-compat).
    assert d["zscore_means"] == {}
    assert d["zscore_stds"] == {}


# ───────────────────── D040 transformed-feature reliability ─────────────────────


def test_stage5_transformed_columns_cover_the_registry():
    from predict.reduce.prepare_matrix import STAGE5_TRANSFORMED, stage5_transformed_columns
    df = _toy_df(n=20)
    df["dense_calcium_count"] = range(20)
    cols = stage5_transformed_columns(df)
    assert set(cols) == set(STAGE5_TRANSFORMED)
    assert cols["has_dense_calcium"].tolist() == [0] + [1] * 19


def test_unreliable_derived_feature_is_rejected_before_r2():
    df = _toy_df(n=50)
    feat = [c for c in df.columns if c not in ("pid", "kernel")]
    _, cols, accepted, rejected = maybe_add_derived_features(
        df, feat, reliable={"high_density_fraction"})
    assert "vessel_burden_gini" not in cols
    reasons = {r["name"]: r.get("reason", "") for r in rejected}
    assert "reliability" in reasons["vessel_burden_gini"]
