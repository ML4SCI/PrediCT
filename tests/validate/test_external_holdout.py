"""Tests for predict.validate.external_holdout (D029)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import PowerTransformer

from predict.reduce.prepare_matrix import (
    D017_DROPPED_FEATURES,
    D018_BINARISE_SOURCE,
)
from predict.validate.external_holdout import (
    HOLDOUT_RAW_CONTEXT_COLS,
    SPATIAL_FEATURES_FOR_PROJECTION,
    ProductionSettings,
    _yeo_johnson_apply,
    build_holdout_report,
    fit_frozen_pipeline,
    predict_holdout_phenotype,
    project_holdout_to_spatial_pca,
    require_production_parity,
    run_external_holdout_validation,
    transform_holdout_features,
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


# ─────────────────────── fixtures ───────────────────────


def _toy_full_cohort(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic full-cohort raw features with the 13 spatial features +
    everything D019 needs to run end-to-end.

    Two latent clusters injected into n_calcified_arteries + lesion_count_*
    so the spatial-only GMM has a real signal to recover.
    """
    rng = np.random.default_rng(seed)
    n_a = n // 2
    n_b = n - n_a

    def spatial_cluster_a():
        return {
            "lesion_count_lad": rng.poisson(1.2, n_a),
            "lesion_count_rca": rng.poisson(0.6, n_a),
            "lesion_count_lcx": rng.poisson(0.5, n_a),
            "lesion_count_lm": rng.poisson(0.2, n_a),
            "lesion_count_total": rng.poisson(2.5, n_a),
            "n_calcified_arteries": rng.choice([1, 2], n_a, p=[0.7, 0.3]),
            "gini_lesion_volume": rng.beta(2, 5, n_a),
            "dist_from_top_max": rng.uniform(20, 50, n_a),
            "dist_from_top_mean": rng.uniform(15, 40, n_a),
            "center_of_mass_z": rng.uniform(30, 60, n_a),
            "inter_lesion_dist_mean_lad": np.where(
                rng.random(n_a) < 0.35, 0.0, rng.uniform(2, 15, n_a)
            ),
            "inter_lesion_dist_max_lad": np.where(
                rng.random(n_a) < 0.35, 0.0, rng.uniform(5, 25, n_a)
            ),
            "first_to_last_dist_lad": np.where(
                rng.random(n_a) < 0.35, 0.0, rng.uniform(5, 30, n_a)
            ),
        }

    def spatial_cluster_b():
        return {
            "lesion_count_lad": rng.poisson(5.0, n_b),
            "lesion_count_rca": rng.poisson(4.0, n_b),
            "lesion_count_lcx": rng.poisson(3.0, n_b),
            "lesion_count_lm": rng.poisson(1.0, n_b),
            "lesion_count_total": rng.poisson(12.0, n_b),
            "n_calcified_arteries": rng.choice([3, 4], n_b, p=[0.3, 0.7]),
            "gini_lesion_volume": rng.beta(5, 2, n_b),
            "dist_from_top_max": rng.uniform(50, 100, n_b),
            "dist_from_top_mean": rng.uniform(40, 90, n_b),
            "center_of_mass_z": rng.uniform(50, 90, n_b),
            "inter_lesion_dist_mean_lad": rng.uniform(8, 40, n_b),
            "inter_lesion_dist_max_lad": rng.uniform(15, 70, n_b),
            "first_to_last_dist_lad": rng.uniform(15, 80, n_b),
        }

    spatial_a = spatial_cluster_a()
    spatial_b = spatial_cluster_b()
    spatial = {k: np.concatenate([spatial_a[k], spatial_b[k]])
               for k in spatial_a}

    # D017 droppable columns (will be discarded by D019).
    drop_cols = {c: rng.uniform(0, 10, n) for c in D017_DROPPED_FEATURES}
    # D018 source.
    dense = {D018_BINARISE_SOURCE: rng.poisson(0.3, n).astype(float)}
    # Density tiers (16): keep them present.
    tier_cols = {
        f"n_rois_d{t}_{v}": rng.poisson(0.8, n).astype(float)
        for t in (1, 2, 3, 4)
        for v in ("lad", "rca", "lcx", "lm")
    }
    # Texture features (6).
    texture_cols = {
        c: rng.normal(10.0, 2.0, n) for c in TEXTURE_EXAMPLE
    }
    # Other context columns expected by the holdout report.
    context = {
        "agatston_total": rng.gamma(2.0, 200.0, n),
        "mask_voxels": rng.integers(50, 5000, n),
        "low_burden_flag": rng.choice([0, 1], n, p=[0.7, 0.3]),
        "scanner_model": ["Sensation 64"] * n,
    }

    cols = {**spatial, **drop_cols, **dense, **tier_cols,
            **texture_cols, **context}
    df = pd.DataFrame(cols)
    df.insert(0, "pid", [str(i) for i in range(n)])
    df.insert(1, "kernel", rng.choice(["Qr36d/2", "I30f/3"], size=n))
    return df


def _feature_cols(df: pd.DataFrame) -> list[str]:
    excluded = {"pid", "kernel", "scanner_model"}
    return [c for c in df.columns if c not in excluded]


@pytest.fixture
def full_cohort():
    return _toy_full_cohort(n=200, seed=0)


@pytest.fixture
def holdout(full_cohort):
    """4 synthetic holdout rows with GE kernel."""
    rng = np.random.default_rng(42)
    n = 4
    base = _toy_full_cohort(n=n, seed=42)
    base["kernel"] = ["GE_StandardFilter"] * n  # not in {Qr36d/2, I30f/3}
    base["pid"] = ["19", "28", "76", "77"]
    return base


# ─────────────────────── tests ───────────────────────


def test_spatial_features_match_stage_6_list():
    """Regression: the 13 spatial features must exactly match the list in
    scripts/07_discover.py SPATIAL_FEATURES_AFTER_D017."""
    expected = (
        "lesion_count_lad", "lesion_count_rca", "lesion_count_lcx",
        "lesion_count_lm", "lesion_count_total",
        "n_calcified_arteries", "gini_lesion_volume",
        "dist_from_top_max", "dist_from_top_mean", "center_of_mass_z",
        "inter_lesion_dist_mean_lad", "inter_lesion_dist_max_lad",
        "first_to_last_dist_lad",
    )
    assert SPATIAL_FEATURES_FOR_PROJECTION == expected
    assert len(SPATIAL_FEATURES_FOR_PROJECTION) == 13


def test_yeo_johnson_apply_matches_sklearn():
    """_yeo_johnson_apply must reproduce sklearn PowerTransformer.transform
    at a fitted lambda."""
    rng = np.random.default_rng(0)
    train = rng.gamma(2.0, 1.5, 500).reshape(-1, 1)
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    pt.fit(train)
    lam = float(pt.lambdas_[0])

    test_vals = rng.normal(0, 3, 30)  # mix positive + negative
    ours = _yeo_johnson_apply(test_vals, lam)
    theirs = pt.transform(test_vals.reshape(-1, 1)).ravel()
    np.testing.assert_allclose(ours, theirs, atol=1e-10)


def test_fit_frozen_pipeline_captures_zscore_stats(full_cohort):
    frozen = fit_frozen_pipeline(
        full_cohort, _feature_cols(full_cohort),
    )
    # All retained features have a captured mean + std.
    for col in frozen.feature_cols_post_d019:
        assert col in frozen.zscore_means
        assert col in frozen.zscore_stds
    # Spatial features survive D019.
    for col in SPATIAL_FEATURES_FOR_PROJECTION:
        if col in frozen.feature_cols_post_d019:
            assert col in frozen.spatial_feature_cols


def test_fit_frozen_pipeline_focal_label_in_set(full_cohort):
    frozen = fit_frozen_pipeline(
        full_cohort, _feature_cols(full_cohort),
    )
    assert frozen.gmm_focal_label_raw in (0, 1)
    assert "focal" in frozen.full_cohort_focal_diffuse_mapping.values()
    assert "diffuse" in frozen.full_cohort_focal_diffuse_mapping.values()


def test_transform_holdout_does_not_mutate_frozen_pipeline(full_cohort, holdout):
    """Leak invariant at the transform path: transforming the holdout must
    not mutate the captured zscore_means / zscore_stds (which are the
    full-cohort fits) or the GMM."""
    frozen = fit_frozen_pipeline(full_cohort, _feature_cols(full_cohort))
    snapshot_means = dict(frozen.zscore_means)
    snapshot_stds = dict(frozen.zscore_stds)
    snapshot_gmm_means = frozen.gmm.means_.copy()
    _ = transform_holdout_features(holdout, frozen)
    assert frozen.zscore_means == snapshot_means
    assert frozen.zscore_stds == snapshot_stds
    np.testing.assert_array_equal(frozen.gmm.means_, snapshot_gmm_means)


def test_fit_frozen_pipeline_idempotent_under_same_input(full_cohort):
    """Refit guarantee: identical inputs -> identical zscore stats. This
    is the real no-leak property: in production the orchestrator passes
    ONLY the training set, so leak prevention is a property of the caller,
    not of fit_frozen_pipeline itself."""
    a = fit_frozen_pipeline(full_cohort, _feature_cols(full_cohort))
    b = fit_frozen_pipeline(full_cohort, _feature_cols(full_cohort))
    for col in a.feature_cols_post_d019:
        assert a.zscore_means[col] == pytest.approx(b.zscore_means[col], abs=1e-12)
        assert a.zscore_stds[col] == pytest.approx(b.zscore_stds[col], abs=1e-12)


def test_transform_holdout_returns_spatial_cols_only(full_cohort, holdout):
    frozen = fit_frozen_pipeline(full_cohort, _feature_cols(full_cohort))
    out = transform_holdout_features(holdout, frozen)
    # pid + spatial features (subset that survived D019).
    assert "pid" in out.columns
    for col in out.columns:
        if col == "pid":
            continue
        assert col in frozen.spatial_feature_cols
    assert len(out) == len(holdout)


def test_projection_shape_matches_gmm_input(full_cohort, holdout):
    frozen = fit_frozen_pipeline(full_cohort, _feature_cols(full_cohort))
    transformed = transform_holdout_features(holdout, frozen)
    scores = project_holdout_to_spatial_pca(transformed, frozen)
    # Same column count as the GMM expects.
    assert scores.shape == (len(holdout), frozen.gmm.means_.shape[1])


def test_predict_phenotype_raw_in_set(full_cohort, holdout):
    frozen = fit_frozen_pipeline(full_cohort, _feature_cols(full_cohort))
    transformed = transform_holdout_features(holdout, frozen)
    scores = project_holdout_to_spatial_pca(transformed, frozen)
    raw, dists = predict_holdout_phenotype(scores, frozen)
    assert set(raw.tolist()).issubset({0, 1})
    assert dists.shape == (len(holdout), 2)
    assert (dists >= 0).all()


def test_report_columns_and_focal_mapping(full_cohort, holdout):
    frozen = fit_frozen_pipeline(full_cohort, _feature_cols(full_cohort))
    transformed = transform_holdout_features(holdout, frozen)
    scores = project_holdout_to_spatial_pca(transformed, frozen)
    raw, dists = predict_holdout_phenotype(scores, frozen)
    report = build_holdout_report(
        holdout, scores, raw, dists, frozen,
        xml_roundtrip_pass_by_pid={"19": True, "28": True,
                                    "76": True, "77": False},
    )
    assert len(report) == len(holdout)
    # Context columns present.
    for col in HOLDOUT_RAW_CONTEXT_COLS:
        assert col in report.columns
    # Phenotype mapping consistent with frozen.focal_label_raw.
    assert set(report["predicted_phenotype"].unique()).issubset(
        {"focal", "diffuse"}
    )
    # Raw -> string consistency.
    focal_raw = frozen.gmm_focal_label_raw
    str_for_focal = report.loc[
        report["predicted_phenotype_raw"] == focal_raw,
        "predicted_phenotype",
    ]
    if len(str_for_focal):
        assert (str_for_focal == "focal").all()
    # xml_roundtrip column propagated.
    assert "xml_roundtrip_max_pass" in report.columns
    val_77 = report.loc[report["pid"] == "77",
                         "xml_roundtrip_max_pass"].iloc[0]
    assert bool(val_77) is False
    val_19 = report.loc[report["pid"] == "19",
                         "xml_roundtrip_max_pass"].iloc[0]
    assert bool(val_19) is True


def test_run_validation_end_to_end_shape(full_cohort, holdout):
    report, frozen = run_external_holdout_validation(
        full_cohort, holdout, _feature_cols(full_cohort),
    )
    assert len(report) == len(holdout)
    # No PyRadiomics columns in the report (D029.1 ComBat-skip path).
    py_cols = [c for c in report.columns
               if c.startswith("original_") or "_glcm_" in c]
    assert py_cols == []


def test_rank_fallback_columns_are_replayed_with_the_training_mapping(full_cohort):
    """Where the fit falls back to ranks, held-out values must not be
    left raw and z-scored with rank-scale statistics (658.77 instead of
    3.28 in a synthetic example). A skew threshold of 0 forces the rank
    fallback; a held-out copy of a training row must get exactly that row's
    fitted values, and the fit itself checks the replay of all training rows."""
    frozen = fit_frozen_pipeline(full_cohort, _feature_cols(full_cohort),
                                 settings=ProductionSettings(prep_kwargs={"yj_skew_threshold": 0.0}))
    rank_cols = [c for c in frozen.spatial_feature_cols
                 if frozen.transforms.get(c, {}).get("kind") == "rank"]
    assert rank_cols, "the test needs at least one spatial column on the rank fallback"
    copy = full_cohort.iloc[[5]].assign(pid="H1")
    out = transform_holdout_features(copy, frozen)
    row5 = frozen.training_pids.index(str(full_cohort.iloc[5]["pid"]))
    np.testing.assert_allclose(out[frozen.spatial_feature_cols].to_numpy()[0],
                               frozen.training_spatial[row5], atol=1e-9)
    assert np.abs(out[rank_cols].to_numpy()).max() < 10.0


def test_replay_check_catches_a_wrong_transform(full_cohort, monkeypatch):
    """The fit refuses when the held-out transform does not reproduce the
    fitted training values (here: the rank columns left raw, as before)."""
    import predict.validate.external_holdout as eh
    monkeypatch.setattr(eh, "_rank_map", lambda x, ref: x)
    with pytest.raises(RuntimeError, match="replaying the training rows"):
        eh.fit_frozen_pipeline(full_cohort, _feature_cols(full_cohort),
                               settings=ProductionSettings(prep_kwargs={"yj_skew_threshold": 0.0}))


def test_production_parity_passes_and_detects_differences(full_cohort, tmp_path):
    frozen = fit_frozen_pipeline(full_cohort, _feature_cols(full_cohort))
    prep = pd.DataFrame(frozen.training_spatial, columns=frozen.spatial_feature_cols)
    prep.insert(0, "pid", frozen.training_pids)
    labels = pd.DataFrame({"pid": frozen.training_pids,
                           "spatial_only_gmm_k2": frozen.training_labels_raw})
    prep.to_csv(tmp_path / "prepared_matrix.csv", index=False, float_format="%.17g")
    labels.to_csv(tmp_path / "cluster_labels_spatial_k2.csv", index=False)
    assert require_production_parity(frozen, tmp_path)["label_mismatches"] == 0

    bad = labels.copy()
    bad.loc[0, "spatial_only_gmm_k2"] = 1 - bad.loc[0, "spatial_only_gmm_k2"]
    bad.to_csv(tmp_path / "cluster_labels_spatial_k2.csv", index=False)
    with pytest.raises(SystemExit, match="labels differ"):
        require_production_parity(frozen, tmp_path)
    labels.to_csv(tmp_path / "cluster_labels_spatial_k2.csv", index=False)

    shifted = prep.copy()
    shifted[frozen.spatial_feature_cols[0]] += 1e-6
    shifted.to_csv(tmp_path / "prepared_matrix.csv", index=False, float_format="%.17g")
    with pytest.raises(SystemExit, match="transforms differ"):
        require_production_parity(frozen, tmp_path)

    prep.iloc[::-1].to_csv(tmp_path / "prepared_matrix.csv", index=False, float_format="%.17g")
    with pytest.raises(SystemExit, match="cohort differs"):
        require_production_parity(frozen, tmp_path)


def test_frozen_gmm_is_the_production_gmm(full_cohort):
    """The refit used seed 0, 10 inits, 100 iterations while
    stage 6 uses 42, 5, 200. It now uses stage 6's own function and seed."""
    from predict.discover.cluster_discovery import GMM_MAX_ITER, GMM_N_INIT
    from predict.discover.stage6 import PRODUCTION
    frozen = fit_frozen_pipeline(full_cohort, _feature_cols(full_cohort))
    assert frozen.gmm.n_init == GMM_N_INIT
    assert frozen.gmm.max_iter == GMM_MAX_ITER
    assert frozen.gmm.random_state == PRODUCTION["random_state"]


def test_predicted_phenotype_raw_column_preserved(full_cohort, holdout):
    """D029.4: the raw GMM label is kept in predicted_phenotype_raw for audit."""
    report, _ = run_external_holdout_validation(
        full_cohort, holdout, _feature_cols(full_cohort),
    )
    assert "predicted_phenotype_raw" in report.columns
    assert set(report["predicted_phenotype_raw"].unique()).issubset({0, 1})
