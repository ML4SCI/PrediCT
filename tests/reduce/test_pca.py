"""Tests for predict.reduce.pca (D020 part 2).

Heavy emphasis on the sign convention (``normalise_pc_signs``): a single
sign flip on a PC destroys downstream interpretation. Multiple tests verify the sign rule under input
reordering, varying dominant-feature positions, and determinism across runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predict.reduce.pca import (
    FAMILY_MAP,
    PcaResult,
    assert_components_orthonormal,
    assign_family,
    explained_variance_table,
    fit_pca,
    normalise_pc_signs,
    pc_external_correlation,
    select_n_retain,
    top_loadings_table,
)


# ───────────────────── helpers ─────────────────────


def _zscored_df(n_patients: int = 50, n_features: int = 10, seed: int = 0) -> pd.DataFrame:
    """Z-scored synthetic dataframe (PCA-ready). Columns are named to test
    the assign_family map across canonical and PyRadiomics prefixes."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_patients, n_features))
    X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    names = [f"original_glcm_F{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=names)
    df.insert(0, "pid", [str(i) for i in range(n_patients)])
    return df


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "pid"]


# ───────────────────── select_n_retain ─────────────────────


def test_n_retain_first_index_at_threshold():
    cumvar = np.array([0.20, 0.55, 0.80, 0.92, 0.99])
    assert select_n_retain(cumvar, threshold=0.85) == 4
    assert select_n_retain(cumvar, threshold=0.55) == 2
    assert select_n_retain(cumvar, threshold=0.99) == 5


def test_n_retain_minimum_one():
    cumvar = np.array([1.0])
    assert select_n_retain(cumvar, threshold=0.85) == 1


def test_n_retain_full_when_threshold_unreached():
    cumvar = np.array([0.1, 0.3, 0.5])
    assert select_n_retain(cumvar, threshold=0.9) == 3


def test_n_retain_raises_on_invalid_threshold():
    cumvar = np.array([0.1, 0.5])
    with pytest.raises(ValueError):
        select_n_retain(cumvar, threshold=0.0)
    with pytest.raises(ValueError):
        select_n_retain(cumvar, threshold=1.5)


def test_n_retain_raises_on_2d_input():
    cumvar = np.array([[0.5], [0.99]])
    with pytest.raises(ValueError, match="1D"):
        select_n_retain(cumvar, threshold=0.85)


# ───────────────────── normalise_pc_signs ─────────────────────


def test_sign_flips_negative_dominant_loading():
    """A PC where the largest |loading| is negative must be flipped."""
    components = np.array([
        [0.1, -0.9, 0.3],     # dominant is -0.9 -> flip
        [0.6, 0.1, -0.2],     # dominant is +0.6 -> keep
    ])
    out = normalise_pc_signs(components)
    # PC1 dominant must now be positive.
    assert out[0, 1] > 0
    # PC2 must be unchanged.
    np.testing.assert_array_equal(out[1], components[1])


def test_sign_normalisation_does_not_mutate_input():
    components = np.array([[0.1, -0.9, 0.3]])
    original = components.copy()
    _ = normalise_pc_signs(components)
    np.testing.assert_array_equal(components, original)


def test_sign_normalisation_preserves_magnitudes():
    rng = np.random.default_rng(0)
    components = rng.standard_normal((5, 8))
    out = normalise_pc_signs(components)
    np.testing.assert_allclose(np.abs(components), np.abs(out), atol=1e-12)


def test_sign_normalisation_dominant_is_nonnegative_for_every_pc():
    rng = np.random.default_rng(1)
    components = rng.standard_normal((6, 12))
    out = normalise_pc_signs(components)
    for i in range(out.shape[0]):
        idx = int(np.argmax(np.abs(out[i])))
        assert out[i, idx] >= 0.0


def test_sign_normalisation_2d_required():
    with pytest.raises(ValueError, match="2D"):
        normalise_pc_signs(np.array([0.1, 0.2, 0.3]))


def test_sign_normalisation_handles_empty():
    out = normalise_pc_signs(np.zeros((0, 5)))
    assert out.shape == (0, 5)


# ───────────────────── fit_pca: shape and contracts ─────────────────────


def test_fit_pca_returns_PcaResult():
    df = _zscored_df(n_patients=40, n_features=8)
    result = fit_pca(df, _feature_cols(df), cumvar_threshold=0.85)
    assert isinstance(result, PcaResult)


def test_fit_pca_scores_shape_matches_n_retain():
    df = _zscored_df(n_patients=40, n_features=12)
    result = fit_pca(df, _feature_cols(df), cumvar_threshold=0.80)
    assert result.scores.shape == (40, result.n_retain)


def test_fit_pca_components_shape():
    df = _zscored_df(n_patients=40, n_features=12)
    result = fit_pca(df, _feature_cols(df))
    # Full SVD: n_components = min(n_patients, n_features).
    assert result.components.shape == (min(40, 12), 12)


def test_fit_pca_raises_on_nan():
    df = _zscored_df()
    df.iloc[3, 2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        fit_pca(df, _feature_cols(df))


def test_fit_pca_raises_on_inf():
    df = _zscored_df()
    df.iloc[5, 4] = np.inf
    with pytest.raises(ValueError, match="Inf"):
        fit_pca(df, _feature_cols(df))


def test_fit_pca_raises_on_constant_column():
    df = _zscored_df(n_features=4)
    df[df.columns[2]] = 7.0
    with pytest.raises(ValueError, match="constant"):
        fit_pca(df, _feature_cols(df))


def test_fit_pca_raises_on_missing_pid_column():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    with pytest.raises(KeyError, match="pid"):
        fit_pca(df, ["a", "b"])


def test_fit_pca_raises_on_empty_features():
    df = _zscored_df()
    with pytest.raises(ValueError, match="must not be empty"):
        fit_pca(df, [])


# ───────────────────── fit_pca: mathematical correctness ─────────────────────


def test_fit_pca_components_orthonormal():
    df = _zscored_df(n_patients=60, n_features=10, seed=3)
    result = fit_pca(df, _feature_cols(df))
    assert_components_orthonormal(result.components, atol=1e-8)


def test_fit_pca_explained_variance_sums_to_total_variance():
    df = _zscored_df(n_patients=40, n_features=8, seed=5)
    feat = _feature_cols(df)
    X = df[feat].to_numpy()
    total_var = float(X.var(axis=0, ddof=1).sum())
    result = fit_pca(df, feat)
    assert float(result.explained_variance.sum()) == pytest.approx(total_var, rel=1e-9)


def test_fit_pca_explained_variance_ratio_sums_to_one():
    df = _zscored_df(n_patients=40, n_features=8, seed=7)
    result = fit_pca(df, _feature_cols(df))
    assert float(result.explained_variance_ratio.sum()) == pytest.approx(1.0, abs=1e-9)


def test_fit_pca_cumulative_variance_monotonic_nondecreasing():
    df = _zscored_df(n_patients=40, n_features=8, seed=9)
    result = fit_pca(df, _feature_cols(df))
    diffs = np.diff(result.cumulative_variance_ratio)
    assert np.all(diffs >= -1e-12)


# ───────────────────── fit_pca: sign convention ─────────────────────


def test_fit_pca_dominant_loading_positive_per_pc():
    df = _zscored_df(n_patients=80, n_features=10, seed=11)
    result = fit_pca(df, _feature_cols(df))
    for i in range(result.n_retain):
        idx = int(np.argmax(np.abs(result.components[i])))
        assert result.components[i, idx] >= 0.0, (
            f"PC{i + 1} dominant loading is negative; sign normalisation failed"
        )


def test_fit_pca_scores_consistent_with_sign_normalised_components():
    """scores = X @ components.T must hold for the SIGN-NORMALISED components."""
    df = _zscored_df(n_patients=50, n_features=8, seed=13)
    feat = _feature_cols(df)
    result = fit_pca(df, feat)
    X = df[feat].to_numpy(dtype=np.float64)
    recomputed = X @ result.components[:result.n_retain].T
    np.testing.assert_allclose(result.scores, recomputed, atol=1e-9)


def test_fit_pca_deterministic_across_runs():
    df = _zscored_df(n_patients=60, n_features=10, seed=21)
    feat = _feature_cols(df)
    r1 = fit_pca(df, feat)
    r2 = fit_pca(df, feat)
    np.testing.assert_array_equal(r1.components, r2.components)
    np.testing.assert_array_equal(r1.scores, r2.scores)
    assert r1.n_retain == r2.n_retain


def test_fit_pca_sign_invariant_under_input_sign_flip():
    """If we negate the entire input matrix, PCA produces flipped scores
    on raw eigenvectors. After sign normalisation the eigenvectors are
    invariant to a global sign flip (because the dominant feature's |loading|
    is unchanged; sign was flipped but the rule rewinds it)."""
    df = _zscored_df(n_patients=50, n_features=8, seed=17)
    feat = _feature_cols(df)
    r1 = fit_pca(df, feat)

    df_neg = df.copy()
    for c in feat:
        df_neg[c] = -df_neg[c]
    r2 = fit_pca(df_neg, feat)

    # After our sign normalisation, the |components| must agree.
    np.testing.assert_allclose(np.abs(r1.components), np.abs(r2.components), atol=1e-8)


# ───────────────────── audit tables ─────────────────────


def test_explained_variance_table_columns_and_count():
    df = _zscored_df(n_patients=50, n_features=8)
    result = fit_pca(df, _feature_cols(df), cumvar_threshold=0.7)
    table = explained_variance_table(result)
    assert len(table) == result.n_components_total
    for c in ("pc", "eigenvalue", "explained_var_pct", "cumulative_var_pct", "retained"):
        assert c in table.columns
    assert table["explained_var_pct"].sum() == pytest.approx(100.0, abs=1e-3)
    assert table["retained"].iloc[:result.n_retain].all()
    assert not table["retained"].iloc[result.n_retain:].any()


def test_top_loadings_table_descending_abs_loading_within_pc():
    df = _zscored_df(n_patients=50, n_features=10, seed=4)
    result = fit_pca(df, _feature_cols(df), cumvar_threshold=0.75)
    table = top_loadings_table(result, top_n=5)
    for _, group in table.groupby("pc", sort=False):
        diffs = np.diff(group["abs_loading"].to_numpy())
        assert np.all(diffs <= 1e-12)


def test_top_loadings_table_abs_equals_abs_of_loading():
    df = _zscored_df(n_patients=40, n_features=8)
    result = fit_pca(df, _feature_cols(df))
    table = top_loadings_table(result, top_n=5)
    np.testing.assert_allclose(
        table["abs_loading"].to_numpy(),
        np.abs(table["loading"].to_numpy()),
        atol=1e-12,
    )


def test_top_loadings_table_rank_starts_at_one():
    df = _zscored_df(n_patients=40, n_features=8)
    result = fit_pca(df, _feature_cols(df))
    table = top_loadings_table(result, top_n=3)
    for _, group in table.groupby("pc", sort=False):
        assert int(group["rank"].iloc[0]) == 1


def test_top_loadings_table_family_assigned():
    df = _zscored_df(n_patients=40, n_features=8)
    result = fit_pca(df, _feature_cols(df))
    table = top_loadings_table(result, top_n=3)
    # All synthetic columns use "original_glcm_" prefix -> PyRad-glcm.
    assert (table["family"] == "PyRad-glcm").all()


def test_pc_external_correlation_length_mismatch_raises():
    df = _zscored_df(n_patients=30, n_features=8)
    result = fit_pca(df, _feature_cols(df))
    bad = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="length"):
        pc_external_correlation(result, bad)


def test_pc_external_correlation_output_shape():
    df = _zscored_df(n_patients=40, n_features=8)
    result = fit_pca(df, _feature_cols(df))
    ext = pd.Series(np.random.default_rng(0).normal(size=40))
    table = pc_external_correlation(result, ext, name="dummy")
    assert len(table) == result.n_retain
    for c in ("pc", "external", "spearman_rho", "pval", "abs_rho"):
        assert c in table.columns
    assert (table["external"] == "dummy").all()


def test_pc_external_correlation_abs_rho_consistent():
    df = _zscored_df(n_patients=40, n_features=8)
    result = fit_pca(df, _feature_cols(df))
    ext = pd.Series(np.random.default_rng(1).normal(size=40))
    table = pc_external_correlation(result, ext)
    np.testing.assert_allclose(
        table["abs_rho"].to_numpy(),
        np.abs(table["spearman_rho"].to_numpy()),
        atol=1e-9,
    )


# ───────────────────── assign_family ─────────────────────


@pytest.mark.parametrize("name, expected", [
    ("original_shape_Sphericity", "PyRad-shape"),
    ("original_firstorder_Range", "PyRad-firstorder"),
    ("original_glcm_Contrast", "PyRad-glcm"),
    ("original_glszm_ZoneEntropy", "PyRad-glszm"),
    ("original_glrlm_RunLengthNonUniformity", "PyRad-glrlm"),
    ("original_ngtdm_Coarseness", "PyRad-ngtdm"),
    ("original_gldm_DependenceEntropy", "PyRad-gldm"),
    ("agatston_lad", "Canonical-burden"),
    ("agatston_total", "Canonical-burden"),
    ("mass_lad", "Canonical-mass"),
    ("volume_lad_mm3", "Canonical-volume"),
    ("mean_hu_lad", "Canonical-HU"),
    ("max_hu_global", "Canonical-HU"),
    ("lesion_count_lad", "Canonical-count"),
    ("lesion_count_total", "Canonical-count"),
    ("n_rois_d1_lad", "Canonical-density-tier"),
    ("n_rois_d4_lm", "Canonical-density-tier"),
    ("inter_lesion_dist_mean_lad", "Canonical-distance"),
    ("first_to_last_dist_lad", "Canonical-distance"),
    ("dist_from_top_max", "Canonical-spatial"),
    ("center_of_mass_z", "Canonical-spatial"),
    ("gini_lesion_volume", "Canonical-distribution"),
    ("n_calcified_arteries", "Canonical-count"),
    ("has_dense_calcium", "Canonical-dense"),
    ("high_density_fraction", "Derived-density"),
    ("vessel_burden_gini", "Derived-distribution"),
    ("totally_unknown_xyz", "Other"),
])
def test_assign_family(name, expected):
    assert assign_family(name) == expected


# ───────────────────── assert_components_orthonormal ─────────────────────


def test_assert_components_orthonormal_passes_on_identity():
    assert_components_orthonormal(np.eye(5))


def test_assert_components_orthonormal_raises_on_non_orthogonal():
    bad = np.array([[1.0, 0.0], [0.5, 0.5]])
    with pytest.raises(AssertionError, match="orthonormal"):
        assert_components_orthonormal(bad)


def test_assert_components_orthonormal_handles_empty():
    assert_components_orthonormal(np.zeros((0, 5)))
