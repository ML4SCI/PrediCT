"""Tests for predict.discover.validity (D021 part 3).

Coverage: kernel chi-square on independent vs confounded contingencies;
Jaccard set similarity; Hennig clusterboot recovers stability on clean
clusters and reports low stability on uniform random data; ARI on
identical / random / partial label sequences and on the shared-pid case.
"""
from __future__ import annotations

import numpy as np
import pytest

from predict.discover.cluster_discovery import fit_cluster
from predict.discover.validity import (
    HennigStabilityResult,
    KernelConfounderResult,
    ari,
    ari_on_shared_pids,
    hennig_clusterboot,
    jaccard_similarity,
    kernel_chi_square,
)


# ───────────────────── helpers ─────────────────────


def _three_blobs(n_per: int = 30, d: int = 4, separation: float = 8.0,
                  noise_sd: float = 0.4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres = [-separation, 0.0, +separation]
    parts = []
    for c in centres:
        block = rng.normal(loc=0.0, scale=noise_sd, size=(n_per, d))
        block[:, 0] += c
        parts.append(block)
    return np.vstack(parts)


def _uniform(n: int = 90, d: int = 4, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-5, 5, size=(n, d))


# ───────────────────── kernel_chi_square ─────────────────────


def test_kernel_chi_square_passes_when_independent():
    rng = np.random.default_rng(0)
    n = 300
    labels = rng.integers(0, 3, size=n)
    kernel = rng.choice(["A", "B"], size=n)   # independent of labels
    result = kernel_chi_square(labels, kernel)
    assert isinstance(result, KernelConfounderResult)
    assert result.pval >= 0.05
    assert result.passes is True


def test_kernel_chi_square_fails_when_confounded():
    """Construct cluster-kernel confounding: cluster 0 mostly A, cluster 1
    mostly B."""
    labels = np.array([0] * 60 + [1] * 60 + [2] * 60)
    kernel = (
        ["A"] * 55 + ["B"] * 5
        + ["A"] * 5 + ["B"] * 55
        + ["A"] * 30 + ["B"] * 30
    )
    result = kernel_chi_square(labels, np.array(kernel))
    assert result.pval < 1e-5
    assert result.passes is False


def test_kernel_chi_square_contingency_matches_labels():
    labels = np.array([0, 0, 1, 1, 2, 2])
    kernel = np.array(["A", "B", "A", "A", "B", "B"])
    result = kernel_chi_square(labels, kernel)
    # Expect 3 clusters x 2 kernels
    assert result.contingency.shape == (3, 2)
    assert int(result.contingency.sum()) == 6


def test_kernel_chi_square_raises_on_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        kernel_chi_square(np.array([0, 1]), np.array(["A", "B", "A"]))


def test_kernel_chi_square_raises_on_2d_input():
    with pytest.raises(ValueError, match="1D"):
        kernel_chi_square(np.array([[0, 1], [1, 0]]),
                          np.array([["A", "B"], ["A", "B"]]))


def test_kernel_chi_square_to_dict_serialisable():
    import json
    labels = np.array([0, 0, 1, 1])
    kernel = np.array(["A", "B", "A", "B"])
    json.dumps(kernel_chi_square(labels, kernel).to_dict())


# ───────────────────── jaccard_similarity ─────────────────────


def test_jaccard_identical_one():
    assert jaccard_similarity({1, 2, 3}, {1, 2, 3}) == 1.0


def test_jaccard_disjoint_zero():
    assert jaccard_similarity({1, 2}, {3, 4}) == 0.0


def test_jaccard_half_overlap():
    """|A ∩ B| / |A ∪ B| = 1 / 3 for {1,2} and {2,3}."""
    assert jaccard_similarity({1, 2}, {2, 3}) == pytest.approx(1 / 3)


def test_jaccard_both_empty_one():
    assert jaccard_similarity(set(), set()) == 1.0


def test_jaccard_one_empty_zero():
    assert jaccard_similarity({1}, set()) == 0.0


# ───────────────────── hennig_clusterboot ─────────────────────


def test_hennig_returns_HennigStabilityResult():
    X = _three_blobs()
    labels = fit_cluster(X, k=3, algorithm="kmeans", random_state=0)
    result = hennig_clusterboot(X, labels, k=3, n_bootstrap=5,
                                random_state=0)
    assert isinstance(result, HennigStabilityResult)


def test_hennig_shape_of_jaccard_table():
    X = _three_blobs(n_per=15)
    labels = fit_cluster(X, k=3, algorithm="kmeans", random_state=0)
    result = hennig_clusterboot(X, labels, k=3, n_bootstrap=4,
                                random_state=0)
    assert result.jaccard_all.shape == (3, 4)
    assert result.jaccard_median.shape == (3,)


def test_hennig_high_stability_on_well_separated_clusters():
    X = _three_blobs(n_per=30, separation=12.0, noise_sd=0.2)
    labels = fit_cluster(X, k=3, algorithm="kmeans", random_state=0)
    result = hennig_clusterboot(X, labels, k=3, n_bootstrap=15,
                                random_state=0)
    # All three clusters should be highly stable.
    assert (result.jaccard_median >= 0.85).all()
    assert result.n_stable_clusters == 3


def test_hennig_low_stability_on_uniform_random_data():
    X = _uniform(n=90, d=4)
    labels = fit_cluster(X, k=3, algorithm="kmeans", random_state=0)
    result = hennig_clusterboot(X, labels, k=3, n_bootstrap=15,
                                random_state=0)
    # No real clusters -> low stability.
    assert result.n_stable_clusters < 3
    assert (result.jaccard_median < 0.75).any()


def test_hennig_parallel_byte_identical_to_serial():
    """Parallel Hennig must produce byte-identical Jaccard table."""
    X = _three_blobs(n_per=15)
    labels = fit_cluster(X, k=3, algorithm="kmeans", random_state=0)
    r_serial = hennig_clusterboot(
        X, labels, k=3, n_bootstrap=6, random_state=0, n_jobs=1,
    )
    r_parallel = hennig_clusterboot(
        X, labels, k=3, n_bootstrap=6, random_state=0, n_jobs=2,
    )
    np.testing.assert_array_equal(r_serial.jaccard_all, r_parallel.jaccard_all)


def test_hennig_deterministic_across_runs():
    X = _three_blobs(n_per=20)
    labels = fit_cluster(X, k=3, algorithm="kmeans", random_state=0)
    r1 = hennig_clusterboot(X, labels, k=3, n_bootstrap=5, random_state=33)
    r2 = hennig_clusterboot(X, labels, k=3, n_bootstrap=5, random_state=33)
    np.testing.assert_array_equal(r1.jaccard_all, r2.jaccard_all)


def test_hennig_records_metadata():
    X = _three_blobs(n_per=15)
    labels = fit_cluster(X, k=3, random_state=0)
    result = hennig_clusterboot(X, labels, k=3, algorithm="ward",
                                n_bootstrap=4, threshold=0.80,
                                random_state=77)
    assert result.algorithm == "ward"
    assert result.k == 3
    assert result.threshold == 0.80
    assert result.random_state == 77
    assert result.n_bootstrap == 4


def test_hennig_raises_on_shape_mismatch():
    X = np.zeros((10, 3))
    bad_labels = np.zeros(8, dtype=int)
    with pytest.raises(ValueError, match="row count"):
        hennig_clusterboot(X, bad_labels, k=2, n_bootstrap=2, random_state=0)


def test_hennig_raises_on_invalid_threshold():
    X = _three_blobs(n_per=10)
    labels = fit_cluster(X, k=3, random_state=0)
    with pytest.raises(ValueError, match="threshold"):
        hennig_clusterboot(X, labels, k=3, threshold=1.5, n_bootstrap=2,
                            random_state=0)


def test_hennig_stable_mask_matches_threshold():
    X = _three_blobs(n_per=30, separation=12.0, noise_sd=0.2)
    labels = fit_cluster(X, k=3, random_state=0)
    result = hennig_clusterboot(X, labels, k=3, threshold=0.75,
                                n_bootstrap=10, random_state=0)
    mask = result.stable_mask()
    assert mask.sum() == result.n_stable_clusters
    assert (result.jaccard_median[mask] >= 0.75).all()


# ───────────────────── ari ─────────────────────


def test_ari_identical_one():
    a = np.array([0, 0, 1, 1, 2, 2])
    assert ari(a, a) == pytest.approx(1.0)


def test_ari_relabelled_identical_one():
    """ARI is permutation-invariant by definition."""
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([2, 2, 0, 0, 1, 1])   # same partition, different labels
    assert ari(a, b) == pytest.approx(1.0)


def test_ari_random_near_zero():
    rng = np.random.default_rng(0)
    n = 500
    a = rng.integers(0, 3, size=n)
    b = rng.integers(0, 3, size=n)
    assert abs(ari(a, b)) < 0.05


def test_ari_partial_agreement_positive():
    a = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    b = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2])   # one re-assignment
    assert 0.0 < ari(a, b) < 1.0


def test_ari_raises_on_shape_mismatch():
    with pytest.raises(ValueError, match="mismatch"):
        ari(np.array([0, 1, 2]), np.array([0, 1]))


def test_ari_raises_on_2d_input():
    with pytest.raises(ValueError, match="1D"):
        ari(np.array([[0, 1]]), np.array([[1, 0]]))


# ───────────────────── ari_on_shared_pids ─────────────────────


def test_ari_on_shared_pids_full_overlap():
    pids = ["1", "2", "3", "4"]
    a = np.array([0, 0, 1, 1])
    b = np.array([1, 1, 0, 0])
    score, shared = ari_on_shared_pids(pids, a, pids, b)
    assert score == pytest.approx(1.0)
    assert shared == pids


def test_ari_on_shared_pids_partial_overlap_uses_intersection():
    pids_a = ["1", "2", "3", "4", "5"]
    labels_a = np.array([0, 0, 1, 1, 2])
    pids_b = ["3", "4", "5", "6", "7"]
    labels_b = np.array([1, 1, 2, 0, 0])
    score, shared = ari_on_shared_pids(pids_a, labels_a, pids_b, labels_b)
    assert shared == ["3", "4", "5"]
    # Both sides give cluster {3,4} together and {5} alone -> ARI = 1.
    assert score == pytest.approx(1.0)


def test_ari_on_shared_pids_no_overlap_returns_zero():
    score, shared = ari_on_shared_pids(
        ["1"], np.array([0]), ["2"], np.array([0]),
    )
    assert score == 0.0
    assert shared == []


def test_ari_on_shared_pids_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        ari_on_shared_pids(
            ["1", "2"], np.array([0]),
            ["1"], np.array([0]),
        )


# ───────────── D021.1: Hennig failures raise; kernel policy ─────────────


def test_hennig_failed_refit_raises_instead_of_zero_stability(monkeypatch):
    import predict.discover.validity as v

    def boom(*_a, **_k):
        raise RuntimeError("fit failed")

    X = _three_blobs()
    labels = fit_cluster(X, 3, algorithm="kmeans", random_state=0)
    monkeypatch.setattr(v, "fit_cluster", boom)
    with pytest.raises(RuntimeError, match="fit failed"):
        v.hennig_clusterboot(X, labels, k=3, algorithm="kmeans", n_bootstrap=3, n_jobs=1)


def _kernel_cohort():
    """Two clusters x two kernels, exactly balanced (chi-square 0)."""
    labels = np.repeat([0, 1], 100)
    kernel = np.tile(np.repeat(["A", "B"], 50), 2)
    burden = np.random.default_rng(0).gamma(2.0, 100.0, size=200)
    return labels, kernel, burden


def test_kernel_policy_pass_when_no_association():
    from predict.discover.validity import assess_kernel_confounding
    labels, kernel, burden = _kernel_cohort()
    assert assess_kernel_confounding(labels, kernel, burden)["status"] == "pass"


def test_kernel_policy_confounded_when_cluster_predicts_kernel_beyond_burden():
    from predict.discover.validity import assess_kernel_confounding
    labels, _, burden = _kernel_cohort()
    rng = np.random.default_rng(1)
    kernel = np.where(labels == 0, "A", "B")
    flip = rng.random(200) < 0.1                      # 90% aligned, burden irrelevant
    kernel = np.where(flip, np.where(kernel == "A", "B", "A"), kernel)
    out = assess_kernel_confounding(labels, kernel, burden)
    assert out["status"] == "confounded"
    assert out["p_cluster_given_burden"] < 0.05


def test_kernel_policy_explained_by_burden_branch(monkeypatch):
    """The policy branch itself, with the adjusted test's answer fixed."""
    import predict.discover.validity as v
    labels, _, burden = _kernel_cohort()
    kernel = np.where(labels == 0, "A", "B")
    monkeypatch.setattr(v, "kernel_association_given_burden",
                        lambda *_a, **_k: {"kernels": ["A", "B"], "p_cluster_given_burden": 0.4,
                                           "p_cluster_unadjusted": 1e-9})
    out = v.assess_kernel_confounding(labels, kernel, burden)
    assert out["passes"] is False
    assert out["status"] == "explained_by_burden"
