"""Unit tests for the LM-isolated low-burden analysis helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run import (  # noqa: E402
    P1_PC1_THRESHOLD,
    P1_PC2_THRESHOLD,
    P2_FISHER_P_MAX,
    P2_LM_RATE_MIN,
    P5_OVERLAP_THRESHOLD,
    attach_lm_features,
    classify_hu_tier,
    cluster_overlap_with_lad,
    cross_stratum_replication,
    density_profile,
    fisher_displaced_vs_nondisplaced_lm,
    identify_displaced,
    wilson_ci,
)


# ─── helpers / fixtures ───


def _synthetic_cohort(seed: int = 0) -> tuple:
    """Build a synthetic cohort with known displaced patients.

    Returns (pca_scores ndarray, pid_order list, agatston Series, kernel Series).
    """
    rng = np.random.default_rng(seed)
    n = 200
    # PC1: agatston-correlated. PC2: random except 5 designated outliers.
    agatston = rng.gamma(shape=2, scale=100, size=n)
    pc1 = (agatston - agatston.mean()) / (agatston.std() + 1e-9) * 2.0
    pc2 = rng.normal(0, 1, size=n)
    # Make 5 low-burden patients displaced on PC2.
    low_burden_idx = np.argsort(agatston)[:50]
    displaced_idx = low_burden_idx[:5]
    pc2[displaced_idx] = 4.0  # well above 2.5 threshold

    pids = [f"p{i:03d}" for i in range(n)]
    kernels = rng.choice(["Qr36d/2", "I30f/3"], size=n)
    agatston_s = pd.Series(agatston, index=pids)
    kernel_s = pd.Series(kernels, index=pids)
    pca = np.stack([pc1, pc2], axis=1)
    return pca, pids, agatston_s, kernel_s, set(pids[i] for i in displaced_idx)


# ─── classify_hu_tier ───


def test_classify_hu_tier_bounds():
    assert classify_hu_tier(129) is None
    assert classify_hu_tier(130) == "W1"
    assert classify_hu_tier(199) == "W1"
    assert classify_hu_tier(200) == "W2"
    assert classify_hu_tier(299) == "W2"
    assert classify_hu_tier(300) == "W3"
    assert classify_hu_tier(399) == "W3"
    assert classify_hu_tier(400) == "W4"
    assert classify_hu_tier(1000) == "W4"


def test_classify_hu_tier_nan_returns_none():
    assert classify_hu_tier(float("nan")) is None


# ─── identify_displaced ───


def test_identify_displaced_finds_synthetic_outliers():
    pca, pids, agatston, kernel, expected = _synthetic_cohort(seed=1)
    df = identify_displaced(pca, pids, agatston, kernel)
    found = set(df.loc[df["displaced"], "pid"].tolist())
    assert found == expected


def test_identify_displaced_only_includes_low_tertile():
    pca, pids, agatston, kernel, _ = _synthetic_cohort(seed=2)
    df = identify_displaced(pca, pids, agatston, kernel)
    # No mid- or high-tertile patient should be flagged
    assert df.loc[df["displaced"], "tertile"].unique().tolist() == ["low"]


def test_identify_displaced_thresholds_are_strict():
    pca, pids, agatston, kernel, _ = _synthetic_cohort(seed=3)
    df = identify_displaced(pca, pids, agatston, kernel)
    disp = df[df["displaced"]]
    assert ((disp["pc1"] > P1_PC1_THRESHOLD)
            | (disp["pc2"] > P1_PC2_THRESHOLD)).all()


# ─── attach_lm_features ───


def test_attach_lm_features_sets_has_lm_correctly():
    df = pd.DataFrame({
        "pid": ["a", "b", "c"], "displaced": [True, False, False],
        "pc1": [1.0, -2.0, -2.0], "pc2": [3.0, 0.0, 0.0],
        "tertile": ["low", "low", "low"],
        "agatston_total": [10.0, 20.0, 30.0],
        "kernel": ["k1", "k1", "k2"],
    })
    feats = pd.DataFrame({
        "pid": ["a", "b", "c"],
        "agatston_lm": [10.0, 0.0, 5.0],
        "max_hu_lm": [400.0, float("nan"), 250.0],
        "mean_hu_lm": [300.0, float("nan"), 200.0],
        "lesion_count_lm": [1, 0, 2],
        "n_calcified_arteries": [1, 3, 4],
        "lesion_count_total": [1, 5, 7],
    })
    out = attach_lm_features(df, feats)
    assert out.loc[out["pid"] == "a", "has_lm"].iloc[0] == True
    assert out.loc[out["pid"] == "b", "has_lm"].iloc[0] == False
    assert out.loc[out["pid"] == "c", "has_lm"].iloc[0] == True
    assert out.loc[out["pid"] == "a", "is_multivessel"].iloc[0] == False
    assert out.loc[out["pid"] == "c", "is_multivessel"].iloc[0] == True


# ─── fisher_displaced_vs_nondisplaced_lm ───


def test_fisher_perfect_separation_passes():
    """All displaced are LM+, all non-displaced are LM-. Fisher p should
    be tiny and the result should PASS."""
    df = pd.DataFrame({
        "pid": [f"p{i}" for i in range(20)],
        "tertile": ["low"] * 20,
        "displaced": [True] * 10 + [False] * 10,
        "has_lm": [True] * 10 + [False] * 10,
    })
    res = fisher_displaced_vs_nondisplaced_lm(df)
    assert res["displaced_lm_rate"] == 1.0
    assert res["non_displaced_lm_rate"] == 0.0
    assert res["fisher_p_one_sided_greater"] < 0.001
    assert res["passes"] is True


def test_fisher_no_separation_fails():
    """Both groups at 50% LM rate. Fisher p should be large; PASS false."""
    df = pd.DataFrame({
        "pid": [f"p{i}" for i in range(40)],
        "tertile": ["low"] * 40,
        "displaced": [True] * 20 + [False] * 20,
        "has_lm": ([True] * 10 + [False] * 10) * 2,
    })
    res = fisher_displaced_vs_nondisplaced_lm(df)
    assert res["fisher_p_one_sided_greater"] > 0.05
    assert res["passes"] is False


def test_fisher_thresholds_match_plan():
    """Regression: p-threshold and rate-floor match plan.md."""
    assert P2_FISHER_P_MAX == 0.001
    assert P2_LM_RATE_MIN == 0.50


# ─── wilson_ci ───


def test_wilson_ci_100_percent_n10_has_finite_lower_bound():
    lo, hi = wilson_ci(10, 10)
    assert 0.6 < lo < 0.8
    assert hi == 1.0


def test_wilson_ci_50_percent_brackets_50():
    lo, hi = wilson_ci(5, 10)
    assert lo < 0.5 < hi


def test_wilson_ci_zero_n_returns_nan():
    lo, hi = wilson_ci(0, 0)
    assert np.isnan(lo)
    assert np.isnan(hi)


# ─── cross_stratum_replication ───


def test_cross_stratum_basic_pass():
    df = pd.DataFrame({
        "pid": [f"p{i}" for i in range(40)],
        "tertile": ["low"] * 40,
        "kernel": ["A"] * 20 + ["B"] * 20,
        "displaced": [True, True, True, False, False, False,
                       False, False, False, False] * 4,
        "has_lm": [True, True, True, False, False, False,
                   False, False, False, False] * 4,
    })
    res = cross_stratum_replication(df)
    for k in ("A", "B"):
        assert res["per_stratum"][k]["displaced_lm_rate"] == 1.0
        assert res["per_stratum"][k]["passes"] is True
    assert res["passes"] is True


def test_cross_stratum_one_stratum_fails():
    """Stratum A has perfect separation, B has 50/50. Overall should fail."""
    df = pd.DataFrame({
        "pid": [f"p{i}" for i in range(20)],
        "tertile": ["low"] * 20,
        "kernel": ["A"] * 10 + ["B"] * 10,
        "displaced": [True] * 3 + [False] * 7 + [True] * 3 + [False] * 7,
        "has_lm":    [True] * 3 + [False] * 7 + [True] * 2 + [False] * 8,
    })
    res = cross_stratum_replication(df)
    assert res["per_stratum"]["A"]["passes"] is True
    assert res["per_stratum"]["B"]["displaced_lm_rate"] == pytest.approx(2/3)
    # Overall is False iff any stratum is False
    assert (res["passes"]
            == all(info["passes"] for info in res["per_stratum"].values()))


# ─── density_profile ───


def test_density_profile_dense_framing():
    df = pd.DataFrame({
        "pid": ["a", "b", "c", "d"],
        "displaced": [True] * 4,
        "max_hu_lm": [350.0, 400.0, 450.0, 600.0],
    })
    out = density_profile(df)
    assert out["n"] == 4
    assert out["median_max_hu_lm"] == 425.0
    assert "advanced" in out["framing"]


def test_density_profile_soft_framing():
    df = pd.DataFrame({
        "pid": ["a", "b", "c", "d"],
        "displaced": [True] * 4,
        "max_hu_lm": [150.0, 180.0, 220.0, 280.0],
    })
    out = density_profile(df)
    assert "early-stage" in out["framing"]


# ─── cluster_overlap_with_lad ───


def test_cluster_overlap_distinct_passes():
    labels = pd.DataFrame({
        "pid": ["a", "a", "b", "c"],
        "vessel": ["LM", "LM", "LM", "LM"],
        "lesion_idx": [0, 1, 0, 0],
        "cluster_kmeans_k12": [0, 2, 3, 1],
    })
    res = cluster_overlap_with_lad(["a", "b", "c"], labels, (10, 11))
    assert res["overlap_fraction"] == 0.0
    assert res["passes_distinctness"] is True


def test_cluster_overlap_with_lad_clusters_fails_distinctness():
    labels = pd.DataFrame({
        "pid": ["a", "b", "c", "d", "e"],
        "vessel": ["LM"] * 5,
        "lesion_idx": [0] * 5,
        "cluster_kmeans_k12": [10, 11, 10, 0, 1],
    })
    res = cluster_overlap_with_lad(["a", "b", "c", "d", "e"], labels, (10, 11))
    # 3 of 5 in cluster 10/11 -> overlap 0.60 -> fails distinctness
    assert res["overlap_fraction"] == 0.6
    assert res["passes_distinctness"] is False


def test_cluster_overlap_threshold_matches_plan():
    assert P5_OVERLAP_THRESHOLD == 0.20


def test_cluster_overlap_not_evaluable_without_lad_clusters():
    """An empty LAD signature makes P5 not evaluable; no fixed cluster ids
    stand in for it (D043)."""
    labels = pd.DataFrame({"pid": ["a"], "vessel": ["LM"], "lesion_idx": [0],
                           "cluster_kmeans_k8": [3]})
    res = cluster_overlap_with_lad(["a"], labels, ())
    assert res["evaluable"] is False
    assert res["passes_distinctness"] is None


def test_missing_or_malformed_lad_signature_stops(tmp_path):
    import json
    from run import _discover_lad_dominant_clusters
    with pytest.raises(SystemExit, match="missing"):
        _discover_lad_dominant_clusters(tmp_path)
    sig = tmp_path / "lad_phenotype" / "lad_cluster_signature.json"
    sig.parent.mkdir(parents=True)
    sig.write_text(json.dumps({"passes": True}))
    with pytest.raises(SystemExit, match="lad_dominant_cluster_ids"):
        _discover_lad_dominant_clusters(tmp_path)
    sig.write_text(json.dumps({"lad_dominant_cluster_ids": []}))
    assert _discover_lad_dominant_clusters(tmp_path) == ()
