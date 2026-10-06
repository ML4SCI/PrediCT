"""Tests for the D036 synthetic cluster injection helpers."""
from __future__ import annotations

import numpy as np
import pytest

from predict.discover.detectability import (
    assign_groups,
    group_counts,
    plant,
    planting_directions,
    simplex_vertices,
    zscore,
)


@pytest.mark.parametrize("k", [2, 3, 4, 6])
def test_simplex_vertices_equidistant_and_centred(k):
    V = simplex_vertices(k, 2.5)
    assert V.shape == (k, k - 1)
    np.testing.assert_allclose(V.mean(axis=0), 0.0, atol=1e-12)
    d = np.linalg.norm(V[:, None, :] - V[None, :, :], axis=2)
    off = d[~np.eye(k, dtype=bool)]
    np.testing.assert_allclose(off, 2.5, rtol=1e-12)


def test_simplex_rejects_k1():
    with pytest.raises(ValueError):
        simplex_vertices(1, 1.0)


@pytest.mark.parametrize("kind", ["balanced", "imbalanced"])
@pytest.mark.parametrize("k", [2, 3, 4])
def test_group_counts_sum_to_n(k, kind):
    c = group_counts(440, k, kind)
    assert c.sum() == 440 and len(c) == k and (c > 0).all()
    if kind == "imbalanced":
        assert (np.diff(c) < 0).all()


def test_assign_groups_has_requested_sizes():
    rng = np.random.default_rng(0)
    lab = assign_groups(rng, 440, 3, "imbalanced")
    np.testing.assert_array_equal(np.bincount(lab), group_counts(440, 3, "imbalanced"))


def test_planting_directions_orthonormal_and_avoid():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(20)
    U = planting_directions(rng, 20, 3, avoid=a)
    np.testing.assert_allclose(U.T @ U, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(U.T @ (a / np.linalg.norm(a)), 0.0, atol=1e-12)


def test_plant_separation_in_background_sd_units():
    rng = np.random.default_rng(2)
    d, n, k, delta = 10, 6000, 3, 3.0
    A = rng.standard_normal((d, d))
    cov = A @ A.T / d + np.eye(d)
    X = rng.multivariate_normal(np.zeros(d), cov, size=n)
    lab = assign_groups(rng, n, k, "balanced")
    U = planting_directions(rng, d, k - 1)
    Xp = plant(X, lab, delta, U, cov)
    # Group means projected on U, in units of the background SD along U.
    sd = np.sqrt(np.diag(U.T @ cov @ U))
    means = np.stack([(Xp[lab == g] - X[lab == g]).mean(axis=0) @ U / sd for g in range(k)])
    dist = np.linalg.norm(means[:, None] - means[None], axis=2)[~np.eye(k, dtype=bool)]
    np.testing.assert_allclose(dist, delta, rtol=1e-10)
    assert not np.shares_memory(X, Xp)


def test_plant_delta_zero_is_identity():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((50, 5))
    lab = assign_groups(rng, 50, 2, "balanced")
    U = planting_directions(rng, 5, 1)
    np.testing.assert_array_equal(plant(X, lab, 0.0, U, np.eye(5)), X)


def test_zscore():
    rng = np.random.default_rng(4)
    Z = zscore(rng.normal(3, 2, size=(100, 4)))
    np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(Z.std(axis=0, ddof=1), 1, atol=1e-12)
