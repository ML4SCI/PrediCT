"""Tests for the D038 taxometric comparison (experimental).

The two textbook cases: a clear two-group taxon must read categorical, a
skewed single-factor continuum must read dimensional.
"""
from __future__ import annotations

import numpy as np
import pytest

from predict.discover.taxometric import (
    ccfi,
    generate_comparison,
    generate_dimensional_skewed,
    latent_moments,
    lmode_curve,
    mambac_curve,
    maxeig_curve,
    nearest_correlation,
    su_moments,
    verdict,
)


def _taxon(n=600, d=6, sep=3.0, base_rate=0.3, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.random(n) < base_rate
    return rng.standard_normal((n, d)) + sep * g[:, None]


def _continuum(n=600, d=6, seed=1, skewed=True):
    rng = np.random.default_rng(seed)
    f = rng.lognormal(0.0, 0.6, n) if skewed else rng.standard_normal(n)
    f = (f - f.mean()) / f.std()
    return 0.75 * f[:, None] + np.sqrt(1 - 0.75 ** 2) * rng.standard_normal((n, d))


def test_nearest_correlation_is_psd_unit_diagonal():
    R = np.array([[1, .9, -.9], [.9, 1, .9], [-.9, .9, 1.0]])
    C = nearest_correlation(R)
    assert np.all(np.linalg.eigvalsh(C) > 0)
    np.testing.assert_allclose(np.diag(C), 1.0)


def test_generate_comparison_keeps_marginals_and_correlations():
    X = _continuum(n=500)
    Y = generate_comparison(X, np.random.default_rng(3))
    np.testing.assert_array_equal(np.sort(Y, axis=0), np.sort(X, axis=0))
    iu = np.triu_indices(X.shape[1], 1)
    err = np.corrcoef(Y, rowvar=False)[iu] - np.corrcoef(X, rowvar=False)[iu]
    assert np.sqrt(np.mean(err ** 2)) < 0.06


def test_generate_dimensional_skewed_keeps_marginals_and_correlations():
    X = _continuum(n=500)
    Y = generate_dimensional_skewed(X, np.random.default_rng(4))
    np.testing.assert_array_equal(np.sort(Y, axis=0), np.sort(X, axis=0))
    iu = np.triu_indices(X.shape[1], 1)
    err = np.corrcoef(Y, rowvar=False)[iu] - np.corrcoef(X, rowvar=False)[iu]
    assert np.sqrt(np.mean(err ** 2)) < 0.06


def test_su_moments_match_simulation():
    rng = np.random.default_rng(6)
    a, b = -1.2, 1.5
    x = np.sinh((rng.standard_normal(2_000_000) - a) / b)
    m, v, sk, ku = su_moments(a, b)
    assert abs(x.mean() - m) < 0.01 and abs(x.var() - v) / v < 0.02
    xs = (x - x.mean()) / x.std()
    assert abs(np.mean(xs ** 3) - sk) < 0.05
    assert abs(np.mean(xs ** 4) - 3 - ku) / max(1, abs(ku)) < 0.1


def test_latent_moments_recover_skew_of_a_noisy_skewed_factor():
    # lognormal(0, 0.6) has skewness (e^.36 + 2) * sqrt(e^.36 - 1) = 2.26; the
    # indicators carry it diluted by noise, the estimate must undo that.
    _lam, _psi, skew_f, _ = latent_moments(_continuum(n=20000, seed=5))
    assert abs(skew_f - 2.26) < 0.5, skew_f


@pytest.mark.parametrize("f", [mambac_curve, maxeig_curve, lmode_curve])
def test_curves_have_fixed_length_and_are_finite(f):
    c = f(_continuum(n=400))
    assert c.shape == (50,) and np.all(np.isfinite(c))


def test_verdict_thresholds():
    assert verdict(0.40) == "dimensional"
    assert verdict(0.50) == "ambiguous"
    assert verdict(0.60) == "categorical"


@pytest.mark.parametrize("model", ["skewed_factor", "copula"])
def test_clear_taxon_reads_categorical(model):
    r = ccfi(_taxon(), n_comparison=20, seed=0, dimensional_model=model)
    assert r.verdict == "categorical", r


@pytest.mark.parametrize("model", ["skewed_factor", "copula"])
def test_normal_continuum_reads_dimensional(model):
    r = ccfi(_continuum(skewed=False), n_comparison=20, seed=0, dimensional_model=model)
    assert r.verdict == "dimensional", r


def test_skewed_continuum_reads_dimensional():
    # The Ruscio-style copula model reads this case categorical (CCFI 0.72,
    # 2026-10-03); D038 uses the skewed-factor dimensional model for that reason.
    r = ccfi(_continuum(), n_comparison=20, seed=0, dimensional_model="skewed_factor")
    assert r.verdict == "dimensional", r
