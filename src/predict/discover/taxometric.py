"""Taxometric comparison of dimensional vs categorical structure (D038, experimental).

Implements the comparison-curve approach of Ruscio and colleagues: the
observed taxometric curves (MAMBAC, MAXEIG, L-Mode) are compared with the
average curves of simulated comparison data that reproduce the observed
marginal distributions and correlations under a dimensional model (one
population) and under a categorical model (two groups, each with its own
marginals and correlations). The comparison curve fit index

    CCFI = RMSR_dim / (RMSR_dim + RMSR_cat)

is near 0 when the data look dimensional and near 1 when they look
categorical. Conventional reading: <= 0.45 dimensional, >= 0.55 categorical.

References: Ruscio and Kaczetow 2008 (comparison data generation); Ruscio,
Walters, Marcus and Kaczetow 2010 (CCFI). Simplifications here are recorded
in decisions/D038-taxometric-continuum-test.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_CURVE = 50          # cuts (MAMBAC), windows (MAXEIG), grid points (L-Mode)
END_CASES = 25        # MAMBAC: cases kept beyond the first / last cut
WINDOW_OVERLAP = 0.90  # MAXEIG
LMODE_GRID = np.linspace(-2.5, 2.5, N_CURVE)
PROCEDURES = ("mambac", "maxeig", "lmode")


# ─────────────────────────── comparison data ───────────────────────────


def nearest_correlation(R: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Clip eigenvalues to eps and rescale to unit diagonal."""
    R = (R + R.T) / 2
    w, V = np.linalg.eigh(R)
    R2 = (V * np.clip(w, eps, None)) @ V.T
    d = np.sqrt(np.diag(R2))
    R2 = R2 / np.outer(d, d)
    np.fill_diagonal(R2, 1.0)
    return R2


def _rank_match(Z: np.ndarray, sorted_cols: np.ndarray) -> np.ndarray:
    """Replace each column of Z by sorted_cols[:, j] in the rank order of Z[:, j]."""
    Y = np.empty_like(sorted_cols)
    for j in range(Z.shape[1]):
        Y[np.argsort(Z[:, j], kind="stable"), j] = sorted_cols[:, j]
    return Y


def generate_comparison(
    X: np.ndarray, rng: np.random.Generator, *, n_pop: int = 10_000, n_iter: int = 20,
) -> np.ndarray:
    """Data with X's exact marginals (per column) and approximately X's correlations.

    The intermediate normal correlation is calibrated on a population of
    n_pop rows (marginals bootstrapped from X), then one sample of len(X)
    rows is drawn and rank-matched to X's own sorted columns.
    """
    n, d = X.shape
    target = np.corrcoef(X, rowvar=False)
    pop_sorted = np.sort(X[rng.integers(0, n, size=n_pop)], axis=0)
    R_int = target.copy()
    best, best_err = R_int, np.inf
    iu = np.triu_indices(d, 1)
    for _ in range(n_iter):
        Rn = nearest_correlation(R_int)
        Z = rng.multivariate_normal(np.zeros(d), Rn, size=n_pop)
        err_mat = target - np.corrcoef(_rank_match(Z, pop_sorted), rowvar=False)
        err = float(np.sqrt(np.mean(err_mat[iu] ** 2)))
        if err < best_err:
            best, best_err = Rn, err
        R_int = R_int + err_mat
    Z = rng.multivariate_normal(np.zeros(d), best, size=n)
    return _rank_match(Z, np.sort(X, axis=0))


def su_moments(a: float, b: float) -> tuple[float, float, float, float]:
    """Exact mean, variance, skewness, excess kurtosis of a standard Johnson SU.

    X = sinh(Y) with Y ~ N(mu, s^2), mu = -a / b, s = 1 / b. Raw moments follow
    from sinh^n(Y) = 2^-n sum_j C(n, j) (-1)^j exp((n - 2j) Y) and
    E exp(k Y) = exp(k mu + k^2 s^2 / 2). Returns NaNs on overflow.
    """
    from math import comb
    mu, s2 = -a / b, 1.0 / b ** 2
    with np.errstate(over="ignore", invalid="ignore"):
        raw = []
        for n in (1, 2, 3, 4):
            raw.append(sum(comb(n, j) * (-1) ** j * np.exp((n - 2 * j) * mu + (n - 2 * j) ** 2 * s2 / 2)
                           for j in range(n + 1)) / 2 ** n)
        m1, m2r, m3r, m4r = raw
        var = m2r - m1 ** 2
        c3 = m3r - 3 * m1 * m2r + 2 * m1 ** 3
        c4 = m4r - 4 * m1 * m3r + 6 * m1 ** 2 * m2r - 3 * m1 ** 4
        out = (m1, var, c3 / var ** 1.5, c4 / var ** 2 - 3.0)
    return tuple(float(v) if np.isfinite(v) else float("nan") for v in out)


def _su_from_moments(skew: float, exkurt: float) -> tuple[float, float]:
    """Johnson SU shape (a, b) whose skewness / excess kurtosis best match the
    targets. SU is unimodal and leptokurtic; a platykurtic target (e.g. a
    two-group latent) cannot be matched and falls back toward normal."""
    from scipy.optimize import minimize

    def loss(p):
        _m, var, sk, ku = su_moments(p[0], float(np.exp(p[1])))
        if not (np.isfinite(sk) and np.isfinite(ku)) or var <= 0:
            return 1e9
        return (sk - skew) ** 2 + (ku - exkurt) ** 2 / (1.0 + abs(exkurt))

    starts = [(-1.0, 0.0), (1.0, 0.0), (0.0, 0.5), (-2.0, -0.5), (2.0, -0.5), (-0.5, -1.0), (0.5, -1.0)]
    best = min((minimize(loss, x0, method="Nelder-Mead",
                         options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 4000}) for x0 in starts),
               key=lambda r: r.fun)
    return float(best.x[0]), float(np.exp(best.x[1]))


def latent_moments(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """One-factor model of z-scored X: loadings, uniquenesses, and the latent
    factor's skewness and excess kurtosis.

    Bartlett scores are the factor plus normal noise of variance
    1 / (lambda' Psi^-1 lambda); normal noise adds nothing to the third and
    fourth cumulants, so the latent's own cumulants are recovered from the
    scores' cumulants and the noise-corrected variance.
    """
    from sklearn.decomposition import FactorAnalysis
    Zx = (X - X.mean(0)) / X.std(0, ddof=1)
    fa = FactorAnalysis(n_components=1, random_state=0).fit(Zx)
    lam = fa.components_[0].copy()
    if lam.sum() < 0:
        lam = -lam
    psi = np.clip(fa.noise_variance_, 1e-6, None)
    denom = float(lam @ (lam / psi))
    fh = Zx @ (lam / psi) / denom
    m = fh - fh.mean()
    m2 = float(np.mean(m ** 2))
    var_f = max(m2 - 1.0 / denom, 0.05 * m2)
    k3 = float(np.mean(m ** 3))
    k4 = float(np.mean(m ** 4)) - 3.0 * m2 ** 2
    return lam, psi, k3 / var_f ** 1.5, k4 / var_f ** 2


_LATENT_CACHE: dict = {}


def _latent_model(X: np.ndarray):
    """(loadings, SU a, SU b, SU mean, SU sd) for X; cached because every
    comparison dataset of one ccfi() call reuses the same fit."""
    key = (X.shape, hash(np.ascontiguousarray(X).tobytes()))
    if key not in _LATENT_CACHE:
        if len(_LATENT_CACHE) > 64:
            _LATENT_CACHE.clear()
        lam, _psi, skew_f, exk_f = latent_moments(X)
        a, b = _su_from_moments(skew_f, exk_f)
        f_mu, f_var, _, _ = su_moments(a, b)
        _LATENT_CACHE[key] = (lam, a, b, f_mu, float(np.sqrt(f_var)))
    return _LATENT_CACHE[key]


def generate_dimensional_skewed(
    X: np.ndarray, rng: np.random.Generator, *, n_pop: int = 10_000, n_iter: int = 20,
) -> np.ndarray:
    """Dimensional comparison data: one unimodal (possibly skewed) latent factor
    plus normal residual structure (D038 revision 2).

    Ruscio-style comparison data (``generate_comparison``) join the observed
    marginals through a normal copula, which misses the tail co-movement of
    indicators sharing one skewed latent dimension (the tail reads as a
    taxon). Here: a one-factor model gives loadings; the latent's skewness
    and excess kurtosis are recovered from Bartlett scores (noise-corrected,
    ``latent_moments``) and matched by a Johnson SU (unimodal); data are
    lambda * f + E with E normal, cov(E) = R_int - lambda lambda' (clipped
    PSD), R_int calibrated so the rank-matched data reproduce the observed
    correlations; marginals are rank-matched exactly.
    """
    n, d = X.shape
    target = np.corrcoef(X, rowvar=False)
    lam, su_a, su_b, f_mu, f_sd = _latent_model(X)

    def draw(R, m):
        resid = R - np.outer(lam, lam)
        rw, rV = np.linalg.eigh((resid + resid.T) / 2)
        resid = (rV * np.clip(rw, 0, None)) @ rV.T
        f = (np.sinh((rng.standard_normal(m) - su_a) / su_b) - f_mu) / f_sd
        E = rng.multivariate_normal(np.zeros(d), resid + 1e-9 * np.eye(d), size=m)
        return f[:, None] * lam[None, :] + E

    pop_sorted = np.sort(X[rng.integers(0, n, size=n_pop)], axis=0)
    R_int, best, best_err = target.copy(), target.copy(), np.inf
    iu = np.triu_indices(d, 1)
    for _ in range(n_iter):
        Y = _rank_match(draw(R_int, n_pop), pop_sorted)
        err_mat = target - np.corrcoef(Y, rowvar=False)
        err = float(np.sqrt(np.mean(err_mat[iu] ** 2)))
        if err < best_err:
            best, best_err = R_int.copy(), err
        R_int = R_int + err_mat
    return _rank_match(draw(best, n), np.sort(X, axis=0))


def two_group_labels(X: np.ndarray, seed: int) -> np.ndarray:
    """Categorical assignment for the comparison model: k-means k=2 on z-scores."""
    from sklearn.cluster import KMeans
    Z = (X - X.mean(0)) / X.std(0, ddof=1)
    return KMeans(n_clusters=2, n_init=20, random_state=seed).fit_predict(Z)


def generate_categorical(X: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Each group regenerated with its own marginals and correlations, then stacked."""
    parts = [generate_comparison(X[labels == g], rng) for g in np.unique(labels)]
    return np.vstack(parts)


# ─────────────────────────── procedures ───────────────────────────


def _z(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(0)) / X.std(0, ddof=1)


def mambac_curve(X: np.ndarray) -> np.ndarray:
    """Mean-above-minus-below-a-cut, averaged over 'one output, sum of rest as input'."""
    Z = _z(X)
    n, d = Z.shape
    cuts = np.linspace(END_CASES, n - END_CASES, N_CURVE).astype(int)
    curves = []
    for j in range(d):
        inp = Z.sum(axis=1) - Z[:, j]
        out = Z[np.argsort(inp, kind="stable"), j]
        csum = np.cumsum(out)
        total = csum[-1]
        below = csum[cuts - 1] / cuts
        above = (total - csum[cuts - 1]) / (n - cuts)
        curves.append(above - below)
    return np.mean(curves, axis=0)


def maxeig_curve(X: np.ndarray) -> np.ndarray:
    """Largest eigenvalue of the zero-diagonal covariance of the other indicators
    in overlapping windows along one input indicator, averaged over inputs."""
    Z = _z(X)
    n, d = Z.shape
    w = int(n / (1 + (N_CURVE - 1) * (1 - WINDOW_OVERLAP)))
    starts = np.linspace(0, n - w, N_CURVE).astype(int)
    curves = []
    for j in range(d):
        order = np.argsort(Z[:, j], kind="stable")
        out = np.delete(Z[order], j, axis=1)
        vals = []
        for s in starts:
            C = np.cov(out[s:s + w], rowvar=False)
            np.fill_diagonal(C, 0.0)
            vals.append(float(np.linalg.eigvalsh(C)[-1]))
        curves.append(vals)
    return np.mean(curves, axis=0)


def lmode_curve(X: np.ndarray) -> np.ndarray:
    """Density of first-principal-component scores (z-scored) on a fixed grid."""
    from scipy.stats import gaussian_kde
    Z = _z(X)
    w, V = np.linalg.eigh(np.corrcoef(Z, rowvar=False))
    s = Z @ V[:, -1]
    s = (s - s.mean()) / s.std(ddof=1)
    return gaussian_kde(s)(LMODE_GRID)


CURVES = {"mambac": mambac_curve, "maxeig": maxeig_curve, "lmode": lmode_curve}


# ─────────────────────────── CCFI ───────────────────────────


@dataclass
class TaxometricResult:
    ccfi: dict[str, float]          # per procedure
    mean_ccfi: float
    verdict: str                    # dimensional | categorical | ambiguous
    base_rate: float                # smaller group share in the categorical model
    n_comparison: int


def verdict(mean_ccfi: float, lo: float = 0.45, hi: float = 0.55) -> str:
    if mean_ccfi <= lo:
        return "dimensional"
    if mean_ccfi >= hi:
        return "categorical"
    return "ambiguous"


DIMENSIONAL_MODELS = ("skewed_factor", "copula")


def ccfi(X: np.ndarray, *, n_comparison: int = 100, seed: int = 0,
         dimensional_model: str = "skewed_factor") -> TaxometricResult:
    """CCFI per procedure and their mean for indicator matrix X (n x d, d >= 3).

    dimensional_model: "skewed_factor" (D038 revision, default) or "copula"
    (Ruscio-style normal copula, kept for the diagnostic comparison).
    """
    X = np.asarray(X, float)
    if X.shape[1] < 3:
        raise ValueError("taxometric analysis needs at least 3 indicators")
    if dimensional_model not in DIMENSIONAL_MODELS:
        raise ValueError(f"dimensional_model must be one of {DIMENSIONAL_MODELS}")
    gen_dim = generate_dimensional_skewed if dimensional_model == "skewed_factor" else generate_comparison
    rng = np.random.default_rng(seed)
    labels = two_group_labels(X, seed)
    obs = {p: f(X) for p, f in CURVES.items()}
    dim = {p: [] for p in CURVES}
    cat = {p: [] for p in CURVES}
    for _ in range(n_comparison):
        D = gen_dim(X, rng)
        C = generate_categorical(X, labels, rng)
        for p, f in CURVES.items():
            dim[p].append(f(D))
            cat[p].append(f(C))
    out = {}
    for p in CURVES:
        r_dim = float(np.sqrt(np.mean((obs[p] - np.mean(dim[p], axis=0)) ** 2)))
        r_cat = float(np.sqrt(np.mean((obs[p] - np.mean(cat[p], axis=0)) ** 2)))
        out[p] = r_dim / (r_dim + r_cat) if (r_dim + r_cat) > 0 else 0.5
    m = float(np.mean(list(out.values())))
    br = float(min(np.mean(labels), 1 - np.mean(labels)))
    return TaxometricResult(ccfi=out, mean_ccfi=m, verdict=verdict(m), base_rate=br,
                            n_comparison=n_comparison)


__all__ = [
    "CURVES", "DIMENSIONAL_MODELS", "PROCEDURES", "TaxometricResult", "ccfi",
    "generate_categorical", "generate_comparison", "generate_dimensional_skewed",
    "latent_moments", "su_moments",
    "lmode_curve", "mambac_curve", "maxeig_curve",
    "nearest_correlation", "two_group_labels", "verdict",
]
