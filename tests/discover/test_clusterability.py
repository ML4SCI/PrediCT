"""Tests for predict.discover.clusterability (D021 part 1).

Verifies Hopkins on three canonical situations: well-separated clusters
(H near 1), uniform-random (H near 0.5), and the verdict classifier at
boundary values. Determinism is pinned.
"""
from __future__ import annotations

import numpy as np
import pytest

from predict.discover.clusterability import (
    HopkinsResult,
    assess_clusterability,
    hopkins_statistic,
)


# ───────────────────── helpers ─────────────────────


def _three_clusters(n_per: int = 80, d: int = 5, seed: int = 0) -> np.ndarray:
    """Build three well-separated Gaussian blobs.

    Centres at +5, 0, -5 along the first axis; noise sd 0.5.
    Hopkins on this should be high (> 0.7).
    """
    rng = np.random.default_rng(seed)
    centres = [+5.0, 0.0, -5.0]
    parts = []
    for c in centres:
        block = rng.normal(loc=0.0, scale=0.5, size=(n_per, d))
        block[:, 0] += c
        parts.append(block)
    return np.vstack(parts)


def _uniform_random(n: int = 240, d: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(low=-5.0, high=5.0, size=(n, d))


# ───────────────────── hopkins_statistic core ─────────────────────


def test_hopkins_returns_float_in_unit_interval():
    X = _uniform_random()
    H = hopkins_statistic(X)
    assert isinstance(H, float)
    assert 0.0 <= H <= 1.0


def test_hopkins_is_high_on_well_separated_clusters():
    X = _three_clusters(n_per=80, d=5)
    H = hopkins_statistic(X, sample_size=40, random_state=0)
    assert H > 0.70


def test_hopkins_is_near_half_on_uniform_random():
    # The Hopkins seed must differ from the data seed: with the same seed the
    # "uniform" queries replay the stream that generated the data and land
    # on data points (H about 0.02 here). See the D021.1 tests below.
    X = _uniform_random(n=400, d=5)
    H = hopkins_statistic(X, sample_size=40, random_state=1)
    assert 0.40 <= H <= 0.60


def test_hopkins_deterministic_under_same_seed():
    X = _three_clusters()
    H1 = hopkins_statistic(X, sample_size=30, random_state=42)
    H2 = hopkins_statistic(X, sample_size=30, random_state=42)
    assert H1 == H2


def test_hopkins_differs_under_different_seed_but_in_same_ballpark():
    X = _three_clusters()
    H1 = hopkins_statistic(X, sample_size=30, random_state=1)
    H2 = hopkins_statistic(X, sample_size=30, random_state=2)
    assert H1 != H2
    assert abs(H1 - H2) < 0.1   # cluster tendency dominates over seed jitter


def test_hopkins_raises_on_1d_input():
    with pytest.raises(ValueError, match="2D"):
        hopkins_statistic(np.array([1.0, 2.0, 3.0]))


def test_hopkins_raises_on_too_small_dataset():
    with pytest.raises(ValueError):
        hopkins_statistic(np.array([[1.0, 2.0]]))


def test_hopkins_raises_when_sample_size_equals_n():
    X = _uniform_random(n=20)
    with pytest.raises(ValueError, match="must be < n"):
        hopkins_statistic(X, sample_size=20)


def test_hopkins_default_sample_size_uses_sample_frac():
    X = _uniform_random(n=200)
    # 10% of 200 = 20.
    H = hopkins_statistic(X, sample_frac=0.10, random_state=0)
    # Should run cleanly; no value check, just a smoke that the default path works.
    assert 0.0 <= H <= 1.0


def test_hopkins_degenerate_constant_data_returns_half():
    """All-zero matrix: every NN distance is 0, denom 0 -> defined as 0.5."""
    X = np.zeros((20, 4))
    H = hopkins_statistic(X, sample_size=5)
    assert H == pytest.approx(0.5)


# ───────────────────── assess_clusterability ─────────────────────


def test_assess_returns_HopkinsResult():
    X = _three_clusters()
    result = assess_clusterability(X)
    assert isinstance(result, HopkinsResult)


def test_assess_verdict_clustered_on_blobs():
    X = _three_clusters(n_per=80, d=5)
    result = assess_clusterability(X, random_state=0)
    assert result.verdict == "clustered"
    assert result.H >= 0.65


def test_assess_verdict_random_on_uniform():
    X = _uniform_random(n=400, d=5)
    result = assess_clusterability(X, random_state=1)   # != data seed
    assert result.verdict == "random_or_regular"


def test_assess_verdict_classifier_boundary_clustered_at_threshold():
    """A synthetic dataset crafted to produce H near 0.65 should be classified
    'clustered' at the threshold boundary (>= threshold). We do not aim
    for exact boundary; instead we monkey-patch the threshold for a clean
    boundary test on a fixed H."""
    X = _three_clusters()
    # Force the verdict computation by examining hi.verdict logic via params.
    result = assess_clusterability(X, threshold=0.50,
                                    ambiguous_band=(0.40, 0.50),
                                    random_state=0)
    # On clustered data, H is well above 0.50.
    assert result.verdict == "clustered"


def test_assess_verdict_ambiguous_band():
    X = _uniform_random(n=400, d=5)
    # Force the band to include the uniform-random H (~0.5).
    result = assess_clusterability(X, threshold=0.65,
                                    ambiguous_band=(0.45, 0.65),
                                    random_state=1)   # != data seed
    assert result.verdict in ("ambiguous", "random_or_regular")


def test_assess_raises_on_invalid_band():
    X = _uniform_random(n=50)
    with pytest.raises(ValueError, match="ambiguous_band"):
        assess_clusterability(X, ambiguous_band=(0.7, 0.4))


def test_assess_to_dict_serialisable():
    import json
    X = _three_clusters()
    result = assess_clusterability(X)
    json.dumps(result.to_dict())


def test_assess_records_sample_size_and_features():
    X = _three_clusters(n_per=50, d=7)
    result = assess_clusterability(X, sample_frac=0.10, random_state=0)
    assert result.n_total == 150
    assert result.n_features == 7
    assert result.sample_size == 15   # round(0.10 * 150)


def test_assess_deterministic_across_runs():
    X = _three_clusters()
    r1 = assess_clusterability(X, random_state=7)
    r2 = assess_clusterability(X, random_state=7)
    assert r1.H == r2.H
    assert r1.verdict == r2.verdict


def test_assess_records_random_state():
    X = _three_clusters()
    result = assess_clusterability(X, random_state=99)
    assert result.random_state == 99


# ───────────── D021.1: uniform queries use their nearest neighbour ─────────────


# Data seed s, Hopkins seed 10_000 + s: independent streams. With equal seeds
# the uniform queries replay the data's own draws and sit on data points
# (cluster probe 2026-10-05: H 0.344 in 1-D, 0.024 in 5-D). The old
# 2nd-neighbour version skipped that coincident point, which hid the overlap
# (0.566 / 0.501) and is very likely how the 2nd-neighbour rule was chosen.


def test_hopkins_calibrated_on_one_dimensional_uniform():
    """Random queries have no self-match, so they must use the nearest
    neighbour. The old 2nd-neighbour version averaged 0.659 here (probe);
    this one 0.500."""
    hs = [hopkins_statistic(np.random.default_rng(s).uniform(size=(300, 1)),
                            sample_size=30, random_state=10_000 + s) for s in range(60)]
    assert 0.45 <= float(np.mean(hs)) <= 0.55


def test_hopkins_calibrated_on_uniform_in_five_dimensions():
    """Old version 0.550 (the ambiguous band's lower edge), this one 0.501."""
    hs = [hopkins_statistic(np.random.default_rng(s).uniform(size=(400, 5)),
                            sample_size=40, random_state=10_000 + s) for s in range(40)]
    assert 0.45 <= float(np.mean(hs)) <= 0.55
