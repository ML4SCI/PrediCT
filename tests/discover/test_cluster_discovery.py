"""Tests for predict.discover.cluster_discovery (D021 part 2).

Coverage focus:

* fit_cluster runs all three algorithms with correct shape and determinism
* within_cluster_dispersion produces W on hand-computable cases
* gap statistic selects k=3 on three well-separated clusters
* gap statistic selects k=1 on uniform random data
* Monti consensus produces a valid (n, n) matrix and low PAC on clean clusters
* burden_residualise zeroes out the burden-correlated variance

All tests reduce n_bootstrap and n_subsamples from the production defaults
(500 and 100) to keep wall-clock under a minute. The mathematical
correctness invariants are independent of these counts.
"""
from __future__ import annotations

import numpy as np
import pytest

from predict.discover.cluster_discovery import (
    ConsensusResult,
    GapStatisticResult,
    _principal_frame,
    _sample_pca_uniform_null,
    burden_residualise,
    fit_cluster,
    gap_statistic,
    monti_consensus,
    within_cluster_dispersion,
)


# ───────────────────── helpers ─────────────────────


def _three_blobs(n_per: int = 30, d: int = 4, separation: float = 8.0,
                  noise_sd: float = 0.4, seed: int = 0) -> np.ndarray:
    """Well-separated Gaussian blobs along axis 0."""
    rng = np.random.default_rng(seed)
    centres = [-separation, 0.0, +separation]
    parts = []
    for c in centres:
        block = rng.normal(loc=0.0, scale=noise_sd, size=(n_per, d))
        block[:, 0] += c
        parts.append(block)
    return np.vstack(parts)


def _uniform(n: int = 90, d: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(low=-5.0, high=5.0, size=(n, d))


# ───────────────────── fit_cluster ─────────────────────


@pytest.mark.parametrize("algorithm", ["kmeans", "ward", "gmm"])
def test_fit_cluster_returns_integer_labels(algorithm):
    X = _three_blobs()
    labels = fit_cluster(X, k=3, algorithm=algorithm, random_state=0)
    assert labels.dtype.kind == "i"
    assert labels.shape == (X.shape[0],)
    assert set(np.unique(labels)).issubset({0, 1, 2})


@pytest.mark.parametrize("algorithm", ["kmeans", "ward", "gmm"])
def test_fit_cluster_recovers_three_blobs(algorithm):
    X = _three_blobs(n_per=30, separation=8.0)
    labels = fit_cluster(X, k=3, algorithm=algorithm, random_state=0)
    # All three labels should be used.
    assert len(set(labels)) == 3
    # Within-blob purity: each blob's points should mostly share a label.
    for start in (0, 30, 60):
        block_labels = labels[start:start + 30]
        majority = np.bincount(block_labels).max()
        assert majority >= 28   # at most 2 misassigned


def test_fit_cluster_k1_returns_zeros():
    X = _three_blobs()
    labels = fit_cluster(X, k=1, algorithm="kmeans", random_state=0)
    np.testing.assert_array_equal(labels, np.zeros(X.shape[0], dtype=int))


def test_fit_cluster_raises_on_invalid_inputs():
    X = _three_blobs()
    with pytest.raises(ValueError):
        fit_cluster(X, k=0)
    with pytest.raises(ValueError):
        fit_cluster(X, k=X.shape[0] + 1)
    with pytest.raises(ValueError):
        fit_cluster(np.array([1.0, 2.0]), k=2)
    with pytest.raises(ValueError):
        fit_cluster(X, k=3, algorithm="unsupported")


@pytest.mark.parametrize("algorithm", ["kmeans", "gmm"])
def test_fit_cluster_deterministic_for_seeded_algorithms(algorithm):
    """Algorithms that consume the seed must produce identical labels."""
    X = _three_blobs()
    l1 = fit_cluster(X, k=3, algorithm=algorithm, random_state=99)
    l2 = fit_cluster(X, k=3, algorithm=algorithm, random_state=99)
    np.testing.assert_array_equal(l1, l2)


def test_fit_cluster_ward_deterministic_by_construction():
    """Ward is deterministic by construction (no randomness in linkage)."""
    X = _three_blobs()
    l1 = fit_cluster(X, k=3, algorithm="ward")
    l2 = fit_cluster(X, k=3, algorithm="ward")
    np.testing.assert_array_equal(l1, l2)


# ───────────────────── within_cluster_dispersion ─────────────────────


def test_W_zero_on_singletons():
    X = np.array([[0.0, 0.0], [10.0, 10.0]])
    labels = np.array([0, 1])
    assert within_cluster_dispersion(X, labels) == 0.0


def test_W_zero_when_within_cluster_points_identical():
    X = np.array([[1.0, 2.0], [1.0, 2.0], [5.0, 5.0]])
    labels = np.array([0, 0, 1])
    assert within_cluster_dispersion(X, labels) == 0.0


def test_W_hand_computable_two_points_one_cluster():
    """Two points (0, 0) and (2, 0) -> centroid (1, 0); each at distance 1
    from centroid. Sum of squared distances = 1 + 1 = 2."""
    X = np.array([[0.0, 0.0], [2.0, 0.0]])
    labels = np.array([0, 0])
    assert within_cluster_dispersion(X, labels) == pytest.approx(2.0)


def test_W_additive_across_clusters():
    """Two clusters of two points each, both contributing W=2 individually."""
    X = np.array([[0.0, 0.0], [2.0, 0.0],
                   [10.0, 10.0], [12.0, 10.0]])
    labels = np.array([0, 0, 1, 1])
    assert within_cluster_dispersion(X, labels) == pytest.approx(4.0)


def test_W_raises_on_shape_mismatch():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels = np.array([0, 0, 1])
    with pytest.raises(ValueError):
        within_cluster_dispersion(X, labels)


# ───────────────────── gap_statistic ─────────────────────


def test_gap_statistic_returns_dataclass():
    X = _uniform(n=50, d=3)
    result = gap_statistic(X, algorithm="kmeans",
                            k_range=(1, 2, 3),
                            n_bootstrap=10, random_state=0)
    assert isinstance(result, GapStatisticResult)
    assert result.k_range == (1, 2, 3)
    assert len(result.gap_values) == 3
    assert len(result.sk_values) == 3


def test_gap_statistic_selects_k3_on_three_well_separated_clusters():
    X = _three_blobs(n_per=30, separation=10.0, noise_sd=0.3)
    result = gap_statistic(X, algorithm="kmeans",
                            k_range=(1, 2, 3, 4, 5),
                            n_bootstrap=20, random_state=0)
    assert result.selected_k == 3


def test_gap_statistic_selects_k1_on_uniform_random():
    X = _uniform(n=120, d=4)
    result = gap_statistic(X, algorithm="kmeans",
                            k_range=(1, 2, 3, 4),
                            n_bootstrap=30, random_state=0)
    # Tibshirani's rule typically selects k = 1 on truly uniform data.
    assert result.selected_k == 1


def test_gap_statistic_parallel_byte_identical_to_serial():
    """Parallel execution must produce byte-identical gap values to serial."""
    X = _three_blobs(n_per=15)
    r_serial = gap_statistic(
        X, algorithm="kmeans", k_range=(1, 2, 3),
        n_bootstrap=8, random_state=0, n_jobs=1,
    )
    r_parallel = gap_statistic(
        X, algorithm="kmeans", k_range=(1, 2, 3),
        n_bootstrap=8, random_state=0, n_jobs=2,
    )
    np.testing.assert_array_equal(r_serial.gap_values, r_parallel.gap_values)
    np.testing.assert_array_equal(r_serial.sk_values, r_parallel.sk_values)
    assert r_serial.selected_k == r_parallel.selected_k


def test_gap_statistic_deterministic_across_runs():
    X = _three_blobs(n_per=20)
    r1 = gap_statistic(X, algorithm="kmeans",
                        k_range=(1, 2, 3),
                        n_bootstrap=5, random_state=7)
    r2 = gap_statistic(X, algorithm="kmeans",
                        k_range=(1, 2, 3),
                        n_bootstrap=5, random_state=7)
    np.testing.assert_array_equal(r1.gap_values, r2.gap_values)
    assert r1.selected_k == r2.selected_k


def test_gap_statistic_raises_on_invalid_null_reference():
    X = _uniform()
    with pytest.raises(ValueError, match="null_reference"):
        gap_statistic(X, null_reference="bogus", n_bootstrap=2,
                      k_range=(1, 2), random_state=0)


def test_gap_statistic_raises_on_empty_k_range():
    X = _uniform()
    with pytest.raises(ValueError, match="non-empty"):
        gap_statistic(X, k_range=(), n_bootstrap=2, random_state=0)


def test_gap_statistic_records_metadata():
    X = _uniform(n=40, d=3)
    result = gap_statistic(X, algorithm="ward", k_range=(1, 2),
                            n_bootstrap=4, random_state=11)
    assert result.algorithm == "ward"
    assert result.null_reference == "pca_uniform"
    assert result.n_bootstrap == 4
    assert result.random_state == 11


# ───────────────────── monti_consensus ─────────────────────


def test_monti_consensus_matrix_shape_and_diagonal():
    X = _three_blobs(n_per=20)
    result = monti_consensus(X, k=3, n_subsamples=20,
                             subsample_frac=0.80, random_state=0)
    assert result.consensus_matrix.shape == (X.shape[0], X.shape[0])
    np.testing.assert_array_equal(
        np.diag(result.consensus_matrix), np.ones(X.shape[0]),
    )


def test_monti_consensus_matrix_symmetric():
    X = _three_blobs(n_per=20)
    result = monti_consensus(X, k=3, n_subsamples=10,
                             subsample_frac=0.80, random_state=0)
    np.testing.assert_allclose(
        result.consensus_matrix, result.consensus_matrix.T, atol=1e-12,
    )


def test_monti_consensus_values_in_unit_interval():
    X = _three_blobs(n_per=20)
    result = monti_consensus(X, k=3, n_subsamples=10,
                             subsample_frac=0.80, random_state=0)
    cm = result.consensus_matrix
    assert cm.min() >= 0.0
    assert cm.max() <= 1.0


def test_monti_consensus_low_pac_on_well_separated_clusters():
    X = _three_blobs(n_per=20, separation=10.0, noise_sd=0.2)
    result = monti_consensus(X, k=3, n_subsamples=30,
                             subsample_frac=0.80, random_state=0)
    # Sharp clusters -> few consensus entries in the ambiguous band.
    assert result.pac_score < 0.10


def test_monti_consensus_high_pac_on_uniform_random():
    X = _uniform(n=60, d=3)
    result = monti_consensus(X, k=3, n_subsamples=30,
                             subsample_frac=0.80, random_state=0)
    # Random data forced into 3 clusters -> very ambiguous.
    assert result.pac_score > 0.20


def test_monti_consensus_parallel_byte_identical_to_serial():
    """Parallel Monti must produce byte-identical consensus matrix."""
    X = _three_blobs(n_per=12)
    r_serial = monti_consensus(
        X, k=3, n_subsamples=8, subsample_frac=0.80,
        random_state=0, n_jobs=1,
    )
    r_parallel = monti_consensus(
        X, k=3, n_subsamples=8, subsample_frac=0.80,
        random_state=0, n_jobs=2,
    )
    np.testing.assert_array_equal(
        r_serial.consensus_matrix, r_parallel.consensus_matrix,
    )
    assert r_serial.pac_score == r_parallel.pac_score


def test_monti_consensus_deterministic_across_runs():
    X = _three_blobs(n_per=15)
    r1 = monti_consensus(X, k=3, n_subsamples=10, random_state=5)
    r2 = monti_consensus(X, k=3, n_subsamples=10, random_state=5)
    np.testing.assert_array_equal(r1.consensus_matrix, r2.consensus_matrix)
    assert r1.pac_score == r2.pac_score


def test_monti_consensus_records_metadata():
    X = _three_blobs(n_per=10)
    result = monti_consensus(X, k=3, algorithm="ward",
                             n_subsamples=5, subsample_frac=0.75,
                             random_state=33)
    assert result.k == 3
    assert result.algorithm == "ward"
    assert result.n_subsamples == 5
    assert result.subsample_frac == 0.75
    assert result.random_state == 33


def test_monti_consensus_invalid_subsample_frac_raises():
    X = _three_blobs(n_per=15)
    with pytest.raises(ValueError, match="subsample_frac"):
        monti_consensus(X, k=2, subsample_frac=0.0, n_subsamples=2,
                        random_state=0)
    with pytest.raises(ValueError, match="subsample_frac"):
        monti_consensus(X, k=2, subsample_frac=1.5, n_subsamples=2,
                        random_state=0)


# ───────────────────── burden_residualise ─────────────────────


def test_burden_residualise_zeros_out_burden_correlated_column():
    """If a feature is exactly linear in log(burden + 1), the residual is ~0."""
    rng = np.random.default_rng(0)
    n = 100
    burden = rng.uniform(0, 1000, size=n)
    log_b = np.log1p(burden)
    X = np.column_stack([2.0 * log_b + 3.0, rng.normal(size=n)])
    R = burden_residualise(X, burden, log_transform=True)
    # First column was a perfect linear function of log_b -> residual ~ 0.
    assert np.max(np.abs(R[:, 0])) < 1e-9
    # Second column was independent -> residual should be similar to original.
    assert np.std(R[:, 1]) == pytest.approx(np.std(X[:, 1]), rel=1e-3)


def test_burden_residualise_preserves_shape():
    rng = np.random.default_rng(1)
    n, p = 50, 8
    X = rng.standard_normal((n, p))
    burden = rng.uniform(0, 100, size=n)
    R = burden_residualise(X, burden)
    assert R.shape == X.shape


def test_burden_residualise_log1p_handles_zero_burden():
    n = 30
    X = np.random.default_rng(0).normal(size=(n, 3))
    burden = np.zeros(n)
    R = burden_residualise(X, burden, log_transform=True)
    # Burden all zero -> only intercept removed -> R is column-mean-centred X.
    np.testing.assert_allclose(R.mean(axis=0), 0.0, atol=1e-9)


def test_burden_residualise_raises_on_length_mismatch():
    X = np.zeros((10, 3))
    burden = np.zeros(5)
    with pytest.raises(ValueError):
        burden_residualise(X, burden)


def test_burden_residualise_linear_vs_log_changes_behaviour():
    rng = np.random.default_rng(2)
    n = 80
    burden = rng.uniform(0, 1000, size=n)
    # Make feature linear in burden, not log(burden).
    X = np.column_stack([3.0 * burden + 5.0, rng.normal(size=n)])
    R_lin = burden_residualise(X, burden, log_transform=False)
    R_log = burden_residualise(X, burden, log_transform=True)
    # Linear should zero column 0; log should leave residual.
    assert np.max(np.abs(R_lin[:, 0])) < 1e-9
    assert np.max(np.abs(R_log[:, 0])) > 1.0


# ───────────── D021.1: reference box in the input's own principal axes ─────────────


def test_gap_reference_follows_the_inputs_principal_axes():
    """Points along a diagonal line (thin band). The PCA-uniform reference
    must fill the band's own principal-axes box, not the square spanned by
    the raw columns (the wrong null; D021.1). This checks the reference itself:
    the selected k on one null realisation is noisy (a 20-seed probe on this
    band picked k = 1 in 8/20 with this box, k = 5 in 20/20 with the
    raw-column box)."""
    rng = np.random.default_rng(0)
    t = rng.uniform(-5.0, 5.0, size=300)
    X = np.column_stack([t, t + rng.normal(0.0, 0.05, size=300)])
    frame = _principal_frame(X)
    center, Vt, low, high = frame
    ref = _sample_pca_uniform_null(frame, 2000, random_state=1)
    R = (ref - center) @ Vt.T
    assert np.all(R >= low - 1e-9) and np.all(R <= high + 1e-9)
    # Along the band the reference spans the data's extent ...
    assert np.ptp(R[:, 0]) > 0.95 * (high[0] - low[0])
    # ... and across it only the band's thickness (about 0.1), whereas the
    # raw-column box reaches the square's corners, about 7 away.
    perp_ref = np.abs((ref - center) @ Vt[1]).max()
    assert perp_ref <= max(abs(low[1]), abs(high[1])) + 1e-9 < 0.5
    corner = np.array([X[:, 0].min(), X[:, 1].max()])
    assert abs((corner - center) @ Vt[1]) > 5.0


def test_gap_selection_is_rotation_invariant():
    """Rotating the data (pairwise distances unchanged) must not change the
    selected k, now that the reference lives in the data's principal axes."""
    X = _three_blobs(n_per=30, separation=10.0, noise_sd=0.3)
    Q, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(X.shape[1], X.shape[1])))
    a = gap_statistic(X, algorithm="kmeans", k_range=(1, 2, 3, 4, 5), n_bootstrap=20, random_state=0)
    b = gap_statistic(X @ Q, algorithm="kmeans", k_range=(1, 2, 3, 4, 5), n_bootstrap=20, random_state=0)
    assert a.selected_k == b.selected_k == 3
