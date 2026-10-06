"""Stage 5 cluster discovery (D021 part 2).

Implements three independent answers to "how many clusters are in this data":

  1. **Gap statistic** (Tibshirani 2001 with Maitra and Ramler 2010 1-SE-
     from-argmax variant): for each k in [1, K_max], compare within-cluster
     dispersion W_k of the observed data against the expected W_k under a
     uniform null reference. Selected k is the smallest k within one
     standard error of the maximum gap. See ``_select_k_tibshirani`` for
     the rationale (the classic Tibshirani rule misbehaves when the gap
     curve is non-monotonic at low k, which happens on highly-separated
     clusters where gap is negative for k < k_true).
  2. **Monti consensus clustering** (Monti 2003): subsample-stability of
     cluster assignments. For each k, runs the algorithm on
     n_subsamples bootstrap subsamples of size subsample_frac * n,
     accumulates a (n, n) co-clustering matrix, and reports the
     proportion of ambiguous clusterings (PAC).
  3. **Forced-k characterisation**: at a user-specified k (default 3), runs
     the algorithm on the full data and returns cluster labels for
     downstream profiling.

Three clustering algorithms supported (D021 lock):
  * **k-means**: spherical clusters, centroid-based. sklearn KMeans with
    n_init=50 to avoid local minima.
  * **Ward linkage**: variance-minimising agglomerative. sklearn
    AgglomerativeClustering linkage='ward' on Euclidean.
  * **GMM**: Gaussian-mixture model with full covariance. sklearn
    GaussianMixture.

Spectral clustering is intentionally NOT supported (D021 dropped it; the
bandwidth hyperparameter cannot be principledly chosen).

Determinism: every stochastic step (k-means init, GMM EM init, gap-
statistic bootstrap draws, Monti subsampling) is fed a derived seed from
``random_state``. Same input + same random_state -> byte-identical result.

Decisions referencing this module:
    D021 - cluster discovery framework
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture


ClusterAlgorithm = Literal["kmeans", "ward", "gmm"]
NullReference = Literal["pca_uniform"]


# ─────────────────────────── result containers ───────────────────────────


@dataclass(frozen=True)
class GapStatisticResult:
    """One run of the gap statistic on a single algorithm.

    ``selected_k`` follows the 1-SE-from-argmax rule in
    ``_select_k_tibshirani``: the smallest k whose gap is within one
    standard error of the maximum gap.
    """
    k_range: tuple[int, ...]
    gap_values: np.ndarray
    sk_values: np.ndarray
    selected_k: int
    log_Wk_observed: np.ndarray
    log_Wk_ref_mean: np.ndarray
    log_Wk_ref_std: np.ndarray
    algorithm: str
    null_reference: str
    n_bootstrap: int
    random_state: int


@dataclass(frozen=True)
class ConsensusResult:
    """Consensus clustering at one (algorithm, k) combination.

    `consensus_matrix` entry (i, j) is the empirical probability that
    patients i and j are assigned the same cluster across subsamples in
    which they were both drawn. ``pac_score`` is the proportion of
    consensus entries in the ambiguous band (0.1, 0.9); low PAC means
    sharp clusters.

    `cluster_labels` is the assignment on the full data (no subsampling).
    """
    k: int
    consensus_matrix: np.ndarray
    cdf_x: np.ndarray
    cdf_y: np.ndarray
    pac_score: float
    cluster_labels: np.ndarray
    algorithm: str
    n_subsamples: int
    subsample_frac: float
    random_state: int


# ─────────────────────────── clustering algorithms ───────────────────────────

# The production GMM (D021). Stage 8 refits the spatial-only GMM with this
# same function so the "frozen" model is the production procedure, not a
# re-implementation with other settings.
GMM_COVARIANCE_TYPE = "full"
GMM_N_INIT = 5
GMM_MAX_ITER = 200


def fit_gmm(X: np.ndarray, k: int, *, random_state: int) -> GaussianMixture:
    """Fit the production Gaussian mixture and return the fitted model."""
    model = GaussianMixture(
        n_components=k, covariance_type=GMM_COVARIANCE_TYPE,
        random_state=random_state, n_init=GMM_N_INIT, max_iter=GMM_MAX_ITER,
    )
    return model.fit(X)


def fit_kmeans(X: np.ndarray, k: int, *, random_state: int, n_init: int = 50) -> KMeans:
    """Fit the production k-means and return the fitted model (the lesion
    stability check rebuilds the stored lesion partition with it and assigns
    perturbed lesions with ``predict``; D043)."""
    return KMeans(n_clusters=k, n_init=n_init, random_state=random_state).fit(X)


def fit_cluster(
    X: np.ndarray,
    k: int,
    *,
    algorithm: ClusterAlgorithm = "kmeans",
    random_state: int = 42,
    n_init_kmeans: int = 50,
) -> np.ndarray:
    """Cluster ``X`` into ``k`` groups with the chosen algorithm.

    Returns integer labels of shape (n,), values in [0, k - 1]. Determinism
    is guaranteed for fixed inputs and random_state.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > X.shape[0]:
        raise ValueError(f"k = {k} cannot exceed n = {X.shape[0]}")

    if k == 1:
        return np.zeros(X.shape[0], dtype=int)

    if algorithm == "kmeans":
        model = fit_kmeans(X, k, random_state=random_state, n_init=n_init_kmeans)
        return np.asarray(model.labels_, dtype=int)
    if algorithm == "ward":
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        return np.asarray(model.fit_predict(X), dtype=int)
    if algorithm == "gmm":
        return np.asarray(fit_gmm(X, k, random_state=random_state).predict(X), dtype=int)
    raise ValueError(f"unknown algorithm {algorithm!r}")


# ─────────────────────────── gap statistic ───────────────────────────


def within_cluster_dispersion(X: np.ndarray, labels: np.ndarray) -> float:
    """Tibshirani's W_k: sum over clusters of within-cluster sum of squared
    distances from each point to the cluster centroid.

    Equivalent to ``sum_r 0.5 / n_r * D_r`` where D_r is the within-cluster
    pairwise squared-distance sum. We use the centroid form for stability.
    Returns 0.0 if all points are in a single point (no within-cluster
    variation).
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if X.shape[0] != labels.shape[0]:
        raise ValueError("X and labels must have the same row count")
    W = 0.0
    unique = np.unique(labels)
    for c in unique:
        mask = labels == c
        n_c = int(mask.sum())
        if n_c < 2:
            continue
        block = X[mask]
        centroid = block.mean(axis=0)
        diffs = block - centroid
        W += float(np.sum(diffs * diffs))
    return W


def _principal_frame(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The input's own principal axes (Tibshirani's PCA-uniform reference).

    Returns ``(center, Vt, low, high)``: the column means, the right singular
    vectors of the centred data (all d axes kept; a rotation, not a
    reduction), and the bounding box of the data in those axes. The input's
    columns need not be principal axes: burden-residualised PC scores are
    correlated, and a box on their columns is the wrong null (D021.1).
    """
    center = X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X - center, full_matrices=False)
    R = (X - center) @ Vt.T
    return center, Vt, R.min(axis=0), R.max(axis=0)


def _sample_pca_uniform_null(
    frame: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    n: int,
    *,
    random_state: int,
) -> np.ndarray:
    """Tibshirani's PCA-uniform null: n points uniform in the box spanned by
    the data's principal axes, mapped back to the input coordinates."""
    center, Vt, low, high = frame
    rng = np.random.default_rng(random_state)
    Z = rng.uniform(low=low, high=high, size=(n, len(low)))
    return Z @ Vt + center


def _select_k_tibshirani(
    k_range: tuple[int, ...],
    gap_values: np.ndarray,
    sk_values: np.ndarray,
) -> int:
    """Gap-statistic k selection using the "1-SE from argmax" variant.

    Returns the smallest k in k_range such that
        gap_values[k] >= gap_values[k_argmax] - sk_values[k_argmax]

    Tibshirani's original 2001 rule ("smallest k where gap does not
    significantly decrease going to k+1") works for monotone-increasing
    gap curves but is misled when the gap curve is non-monotonic at low k.
    This happens routinely on highly-separated clusters: forcing the data
    into k < k_true clusters produces a within-cluster dispersion larger
    than the uniform reference, so gap is NEGATIVE at low k. The classic
    rule then incorrectly selects k=1 because gap(1) and gap(2) are both
    similarly negative.

    The "1-SE from argmax" variant (used by the R factoextra package and
    recommended in Maitra and Ramler 2010) finds the smallest k whose gap
    is within one standard error of the maximum gap. This:
      * selects the true k for clustered data (peak gap defines the cluster
        count, and only k near k_true sit within 1 SE of it)
      * tends to select k=1 for random data (the gap curve is roughly flat),
        but not reliably when the data match the uniform reference: on 300
        points uniform along a line it picked k=1 in 8 of 20 datasets
        (cluster probe, 2026-10-05); sk is the spread of the reference
        draws and does not include the observed curve's sampling noise
    """
    if len(gap_values) == 0:
        raise ValueError("gap_values must be non-empty")
    i_max = int(np.argmax(gap_values))
    threshold = float(gap_values[i_max]) - float(sk_values[i_max])
    for i in range(len(k_range)):
        if gap_values[i] >= threshold:
            return int(k_range[i])
    # Defensive fallback (should not occur because gap[i_max] >= threshold).
    return int(k_range[i_max])


def _gap_bootstrap_iter(
    b: int, k_idx: int, k: int, algorithm: str,
    frame: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], n: int, random_state: int,
) -> float:
    """One bootstrap iteration of the gap statistic. Pure function of its
    arguments so it can be parallelised by joblib without race conditions.

    The seed formula is identical to the serial loop (``random_state *
    10_000 + k_idx * 1_000 + b``), so the parallel result is byte-identical
    to the serial result for any n_jobs.
    """
    seed = random_state * 10_000 + k_idx * 1_000 + b
    X_ref = _sample_pca_uniform_null(frame, n, random_state=seed)
    labels_ref = fit_cluster(X_ref, k, algorithm=algorithm, random_state=seed)
    W_ref = within_cluster_dispersion(X_ref, labels_ref)
    return float(np.log(W_ref)) if W_ref > 0 else float("-inf")


def gap_statistic(
    X: np.ndarray,
    *,
    algorithm: ClusterAlgorithm = "kmeans",
    k_range: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8),
    n_bootstrap: int = 500,
    null_reference: NullReference = "pca_uniform",
    random_state: int = 42,
    n_jobs: int = 1,
) -> GapStatisticResult:
    """Compute the Tibshirani gap statistic and return the selected k.

    For each k, fits ``algorithm`` on the observed data to get W_k_obs.
    For each bootstrap b, draws a null sample uniformly in the box spanned
    by X's own principal axes (all dimensions kept) and fits the same
    algorithm to get W_k_ref(b). The gap is

        gap(k) = mean_b log(W_k_ref(b)) - log(W_k_obs)

    and the standard error is

        sk = std_b(log(W_k_ref(b))) * sqrt(1 + 1/B)

    Selected k follows Maitra-Ramler 2010's 1-SE-from-argmax variant of
    Tibshirani's rule (see ``_select_k_tibshirani``).

    Parallelism: ``n_jobs`` controls the number of joblib workers used for
    the bootstrap iterations within each k. Output is byte-identical to
    the serial version because each iteration uses a deterministic seed
    derived from (random_state, k_idx, b) and results are accumulated in
    fixed index order.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if null_reference != "pca_uniform":
        raise ValueError(f"null_reference {null_reference!r} not supported")
    k_range = tuple(int(k) for k in k_range)
    if not k_range:
        raise ValueError("k_range must be non-empty")

    n, _d = X.shape
    frame = _principal_frame(X)

    log_Wk_observed = np.zeros(len(k_range), dtype=np.float64)
    log_Wk_ref = np.zeros((len(k_range), n_bootstrap), dtype=np.float64)

    for i, k in enumerate(k_range):
        labels_obs = fit_cluster(X, k, algorithm=algorithm,
                                  random_state=random_state)
        W_obs = within_cluster_dispersion(X, labels_obs)
        log_Wk_observed[i] = np.log(W_obs) if W_obs > 0 else -np.inf

        if n_jobs == 1:
            log_Wk_ref[i] = np.array([
                _gap_bootstrap_iter(b, i, k, algorithm,
                                     frame, n, random_state)
                for b in range(n_bootstrap)
            ], dtype=np.float64)
        else:
            log_Wk_ref[i] = np.array(Parallel(n_jobs=n_jobs)(
                delayed(_gap_bootstrap_iter)(
                    b, i, k, algorithm,
                    frame, n, random_state,
                )
                for b in range(n_bootstrap)
            ), dtype=np.float64)

    log_Wk_ref_mean = np.mean(log_Wk_ref, axis=1)
    log_Wk_ref_std = np.std(log_Wk_ref, axis=1, ddof=0)

    gap_values = log_Wk_ref_mean - log_Wk_observed
    sk_values = log_Wk_ref_std * np.sqrt(1.0 + 1.0 / float(n_bootstrap))

    selected_k = _select_k_tibshirani(k_range, gap_values, sk_values)

    return GapStatisticResult(
        k_range=k_range,
        gap_values=gap_values,
        sk_values=sk_values,
        selected_k=selected_k,
        log_Wk_observed=log_Wk_observed,
        log_Wk_ref_mean=log_Wk_ref_mean,
        log_Wk_ref_std=log_Wk_ref_std,
        algorithm=str(algorithm),
        null_reference=str(null_reference),
        n_bootstrap=int(n_bootstrap),
        random_state=int(random_state),
    )


# ─────────────────────────── consensus clustering ───────────────────────────


def _consensus_cdf(consensus_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empirical CDF of the upper-triangular consensus entries.

    Returns (cdf_x, cdf_y) where cdf_x is the unique sorted consensus values
    and cdf_y is the empirical CDF evaluated at those points. Used by PAC
    and by stage-5's plots/.
    """
    n = consensus_matrix.shape[0]
    triu = consensus_matrix[np.triu_indices(n, k=1)]
    if triu.size == 0:
        return np.array([0.0]), np.array([1.0])
    x_sorted = np.sort(triu)
    cdf_y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    return x_sorted, cdf_y


def _pac_score(
    consensus_matrix: np.ndarray,
    *,
    low: float = 0.1,
    high: float = 0.9,
) -> float:
    """Proportion of Ambiguous Clusterings (Senbabaoglu 2014).

    Fraction of upper-triangular consensus entries strictly between low and
    high. Lower PAC = sharper clusters.
    """
    n = consensus_matrix.shape[0]
    if n < 2:
        return 0.0
    triu = consensus_matrix[np.triu_indices(n, k=1)]
    ambiguous = ((triu > low) & (triu < high)).sum()
    return float(ambiguous / triu.size) if triu.size > 0 else 0.0


def _monti_subsample_iter(
    b: int, k: int, algorithm: str, X: np.ndarray,
    sub_n: int, n_total: int, random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """One Monti subsample iteration. Returns (boot_idx, sub_labels).

    Pure function. Seed is ``random_state * 100_000 + b``, byte-identical
    to the serial implementation across n_jobs.
    """
    seed = random_state * 100_000 + b
    sub_rng = np.random.default_rng(seed)
    idx = sub_rng.choice(n_total, size=sub_n, replace=False)
    sub_X = X[idx]
    sub_labels = fit_cluster(sub_X, k, algorithm=algorithm, random_state=seed)
    return idx, sub_labels


def monti_consensus(
    X: np.ndarray,
    *,
    k: int,
    algorithm: ClusterAlgorithm = "kmeans",
    n_subsamples: int = 100,
    subsample_frac: float = 0.80,
    random_state: int = 42,
    n_jobs: int = 1,
) -> ConsensusResult:
    """Monti consensus clustering at a fixed k.

    On each of n_subsamples iterations, draws a row subsample of size
    ceil(subsample_frac * n), clusters it, and updates two (n, n) tally
    matrices: ``M`` counts how many times each pair was assigned to the
    same cluster among the iterations both rows were sampled; ``I`` counts
    how many times both rows were sampled. consensus_matrix = M / I
    with safe handling for I == 0.

    Returns the consensus matrix, its CDF, PAC, and the labels from a
    final fit on the full data.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not (0.0 < subsample_frac <= 1.0):
        raise ValueError(f"subsample_frac must be in (0, 1], got {subsample_frac}")

    n = X.shape[0]
    sub_n = max(2, int(np.ceil(subsample_frac * n)))
    M = np.zeros((n, n), dtype=np.float64)
    I = np.zeros((n, n), dtype=np.float64)

    # Run the subsample clusterings (parallelisable).
    if n_jobs == 1:
        subsample_results = [
            _monti_subsample_iter(b, k, algorithm, X, sub_n, n, random_state)
            for b in range(n_subsamples)
        ]
    else:
        subsample_results = Parallel(n_jobs=n_jobs)(
            delayed(_monti_subsample_iter)(
                b, k, algorithm, X, sub_n, n, random_state,
            )
            for b in range(n_subsamples)
        )

    # Accumulate M (same-cluster) and I (both-sampled) tallies serially.
    # The accumulation order does not change the final matrices because
    # addition is commutative; we accumulate by iteration index for
    # determinism in floating-point summation order.
    for idx, sub_labels in subsample_results:
        # Same-cluster tally.
        for c in np.unique(sub_labels):
            members = idx[sub_labels == c]
            if len(members) < 2:
                continue
            grid_rows, grid_cols = np.meshgrid(members, members, indexing="ij")
            M[grid_rows, grid_cols] += 1.0
        # Both-sampled tally.
        grid_rows, grid_cols = np.meshgrid(idx, idx, indexing="ij")
        I[grid_rows, grid_cols] += 1.0

    # Safe division: where I == 0 set consensus to 0 (pair never co-sampled).
    consensus = np.where(I > 0, M / np.maximum(I, 1.0), 0.0)
    # Diagonal is meaningless for clustering interpretation; set to 1 for cleanliness.
    np.fill_diagonal(consensus, 1.0)

    cdf_x, cdf_y = _consensus_cdf(consensus)
    pac = _pac_score(consensus)

    full_labels = fit_cluster(X, k, algorithm=algorithm,
                                random_state=random_state)

    return ConsensusResult(
        k=int(k),
        consensus_matrix=consensus,
        cdf_x=cdf_x,
        cdf_y=cdf_y,
        pac_score=pac,
        cluster_labels=full_labels,
        algorithm=str(algorithm),
        n_subsamples=int(n_subsamples),
        subsample_frac=float(subsample_frac),
        random_state=int(random_state),
    )


# ─────────────────────────── burden residualisation ───────────────────────────


def burden_residualise(
    X: np.ndarray,
    burden: np.ndarray,
    *,
    log_transform: bool = True,
) -> np.ndarray:
    """Regress out a 1-D burden covariate from every column of X via OLS.

    If ``log_transform`` is True, uses ``log(burden + 1)`` as the regressor
    (recommended for Agatston which spans 0 to ~3000). Returns the
    residualised X with the same shape. The burden axis remains in the
    matrix as the OLS residual = 0 contribution; other features have their
    burden-correlated variance removed.
    """
    X = np.asarray(X, dtype=np.float64)
    burden = np.asarray(burden, dtype=np.float64).reshape(-1)
    if X.shape[0] != burden.shape[0]:
        raise ValueError("X and burden row counts must match")

    z = np.log1p(burden) if log_transform else burden
    Z = np.column_stack([np.ones_like(z), z])
    # beta_j = (Z^T Z)^-1 Z^T x_j for each column j
    # residuals_j = x_j - Z @ beta_j
    beta, *_ = np.linalg.lstsq(Z, X, rcond=None)
    return X - Z @ beta


__all__ = [
    "ClusterAlgorithm",
    "ConsensusResult",
    "GapStatisticResult",
    "NullReference",
    "burden_residualise",
    "fit_cluster",
    "fit_gmm",
    "fit_kmeans",
    "gap_statistic",
    "monti_consensus",
    "within_cluster_dispersion",
]
