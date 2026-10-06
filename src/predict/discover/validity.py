"""Stage 5 cluster validity checks (D021 part 3).

Three independent validity tests on a candidate cluster solution:

  1. **Kernel chi-square confounder test**: was ComBat enough? Test the
     null "cluster labels and kernel are independent" via a Pearson
     chi-square on the cluster x kernel contingency table. p < 0.05 is a
     HARD FAIL: the clustering is dominated by scanner identity, not
     biology. The script must raise / investigate before publishing.

  2. **Hennig clusterboot bootstrap stability** (Hennig 2007): for each
     original cluster, repeatedly draw a bootstrap row-resample of the
     same size with replacement, recluster, and find the bootstrap cluster
     with maximum Jaccard similarity to the original cluster (restricted
     to rows that appear in the bootstrap). Median Jaccard per cluster
     across the bootstraps tells us how reproducible each cluster is.
     Threshold for "stable cluster" is median Jaccard >= 0.75 (Hennig 2007
     recommendation).

  3. **Adjusted Rand Index between full and robust cohorts**: after running
     stage 5 on the full N=422 and again on the robust N~280 (D015), we
     compare cluster assignments on the overlapping patient set. ARI close
     to 1 means clusters are stable to low-burden patient removal; ARI
     near 0 means clusters depended on those patients.

NaN handling: no NaN allowed in labels or input arrays; we assert and
fail loud. Determinism: every bootstrap uses a derived seed.

Decisions referencing this module:
    D021 - validity checks and sensitivity reruns
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import chi2_contingency
from sklearn.metrics import adjusted_rand_score

from predict.discover.cluster_discovery import ClusterAlgorithm, fit_cluster


# ─────────────────────────── containers ───────────────────────────


@dataclass(frozen=True)
class KernelConfounderResult:
    """Chi-square test on a cluster x kernel contingency table."""
    chi2: float
    pval: float
    dof: int
    contingency: np.ndarray
    cluster_names: list[str]
    kernel_names: list[str]
    expected: np.ndarray
    passes: bool   # True iff pval >= 0.05 (no significant confounding)

    def to_dict(self) -> dict:
        return {
            "chi2": self.chi2,
            "pval": self.pval,
            "dof": self.dof,
            "contingency": self.contingency.tolist(),
            "cluster_names": list(self.cluster_names),
            "kernel_names": list(self.kernel_names),
            "expected": self.expected.tolist(),
            "passes": self.passes,
        }


@dataclass(frozen=True)
class HennigStabilityResult:
    """Per-cluster bootstrap stability via the Hennig 2007 clusterboot rule."""
    cluster_ids: list[int]
    jaccard_median: np.ndarray
    jaccard_mean: np.ndarray
    jaccard_all: np.ndarray              # (n_clusters, n_bootstrap)
    n_bootstrap: int
    threshold: float
    n_stable_clusters: int
    n_total_clusters: int
    algorithm: str
    k: int
    random_state: int

    def stable_mask(self) -> np.ndarray:
        return self.jaccard_median >= self.threshold


# ─────────────────────────── kernel chi-square ───────────────────────────


def kernel_chi_square(
    labels: np.ndarray,
    kernel: np.ndarray,
    *,
    alpha: float = 0.05,
) -> KernelConfounderResult:
    """Pearson chi-square test that cluster labels and kernel are independent.

    A small p (< alpha) is a HARD FAIL: the cluster solution co-varies with
    scanner identity, meaning ComBat did not fully harmonise. Stage 5
    should NOT be considered done until this test passes.

    NaN in either array raises. Both arrays must have the same length.
    """
    labels = np.asarray(labels)
    kernel = np.asarray(kernel)
    if labels.shape != kernel.shape:
        raise ValueError(
            f"labels {labels.shape} vs kernel {kernel.shape} shape mismatch"
        )
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")
    if pd.isna(labels).any() or pd.isna(kernel).any():
        raise ValueError("NaN in labels or kernel array")

    df = pd.DataFrame({"cluster": labels, "kernel": kernel})
    ct = pd.crosstab(df["cluster"], df["kernel"])
    chi2, pval, dof, expected = chi2_contingency(ct.to_numpy())

    return KernelConfounderResult(
        chi2=float(chi2),
        pval=float(pval),
        dof=int(dof),
        contingency=ct.to_numpy(),
        cluster_names=[str(c) for c in ct.index.tolist()],
        kernel_names=[str(k) for k in ct.columns.tolist()],
        expected=np.asarray(expected, dtype=np.float64),
        passes=bool(pval >= alpha),
    )


def _loglik_logit(X: np.ndarray, y: np.ndarray) -> float:
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1e10, max_iter=10000).fit(X, y)
    p = np.clip(m.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    return float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())


def kernel_association_given_burden(
    labels: np.ndarray, kernel: np.ndarray, agatston: np.ndarray,
) -> dict:
    """Likelihood-ratio tests on the two most frequent kernels: does the
    cluster label predict the kernel, unadjusted and given log1p(Agatston)?

    kernel ~ cluster vs kernel ~ 1 (unadjusted) and kernel ~ cluster +
    log1p(agatston) vs kernel ~ log1p(agatston) (burden-adjusted). The one
    implementation behind D021.1's scanner-bias assessment and
    scripts/diagnose_kernel_chi2.py.
    """
    from scipy.stats import chi2 as chi2_dist
    kernel = np.asarray(kernel).astype(str)
    labels = np.asarray(labels)
    main2 = pd.Series(kernel).value_counts().index[:2].tolist()
    if len(main2) < 2:
        raise ValueError("kernel_association_given_burden needs two kernels")
    keep = np.isin(kernel, main2)
    y = (kernel[keep] == main2[0]).astype(int)
    lb = np.log1p(np.asarray(agatston, dtype=float)[keep])[:, None]
    cl = pd.get_dummies(labels[keep], drop_first=True).to_numpy(float)
    ll0 = _loglik_logit(lb, y)
    ll1 = _loglik_logit(np.column_stack([lb, cl]), y)
    ll_nb = _loglik_logit(cl, y)
    ybar = y.mean()
    ll_null = float(len(y) * (ybar * np.log(ybar) + (1 - ybar) * np.log(1 - ybar)))
    df = cl.shape[1]
    return {
        "kernels": main2,
        "lr_cluster_unadjusted": 2 * (ll_nb - ll_null),
        "p_cluster_unadjusted": float(chi2_dist.sf(2 * (ll_nb - ll_null), df)),
        "lr_cluster_given_burden": 2 * (ll1 - ll0),
        "p_cluster_given_burden": float(chi2_dist.sf(2 * (ll1 - ll0), df)),
        "p_burden_alone": float(chi2_dist.sf(2 * (ll0 - ll_null), 1)),
    }


def assess_kernel_confounding(
    labels: np.ndarray, kernel: np.ndarray, agatston: np.ndarray, *, alpha: float = 0.05,
) -> dict:
    """D021.1 kernel policy for one clustering.

    ``status`` is "pass" (chi-square p >= alpha), "explained_by_burden"
    (chi-square fails but the cluster no longer predicts the kernel once
    log1p(Agatston) is in the model, p_cluster_given_burden >= alpha), or
    "confounded" (it still does). "confounded" stops stage 6.
    """
    chi = kernel_chi_square(labels, kernel, alpha=alpha)
    out = {"chi2": chi.chi2, "pval": chi.pval, "dof": chi.dof, "passes": chi.passes}
    if chi.passes:
        out["status"] = "pass"
        return out
    adj = kernel_association_given_burden(labels, kernel, agatston)
    out.update(adj)
    out["status"] = ("explained_by_burden" if adj["p_cluster_given_burden"] >= alpha
                     else "confounded")
    return out


# ─────────────────────────── Hennig bootstrap stability ───────────────────────────


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """|A ∩ B| / |A ∪ B|. Returns 1.0 if both empty, 0.0 if one empty."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _hennig_bootstrap_iter(
    b: int, k: int, algorithm: str, X: np.ndarray,
    cluster_ids: list[int], orig_sets: dict[int, set[int]],
    n_clusters: int, random_state: int,
) -> np.ndarray:
    """One Hennig bootstrap iteration. Returns a (n_clusters,) vector of
    per-cluster Jaccard scores. Pure function so safe to parallelise.

    Seed is ``random_state * 1_000_000 + b``, byte-identical to the serial
    version across n_jobs.
    """
    n = X.shape[0]
    seed = random_state * 1_000_000 + b
    sub_rng = np.random.default_rng(seed)
    boot_idx = sub_rng.integers(low=0, high=n, size=n)
    boot_X = X[boot_idx]
    # No catch: a failed refit is a software or numerical error, not
    # cluster instability. Scoring it as Jaccard 0 would hide every such
    # failure (D021.1).
    boot_labels = fit_cluster(
        boot_X, k, algorithm=algorithm, random_state=seed,
    )

    unique_in_boot = set(int(i) for i in np.unique(boot_idx))
    out = np.zeros(n_clusters, dtype=np.float64)
    for ci, c in enumerate(cluster_ids):
        A = orig_sets[c] & unique_in_boot
        best = 0.0
        for cp in np.unique(boot_labels):
            boot_in_cp = boot_idx[boot_labels == cp]
            B = set(int(i) for i in np.unique(boot_in_cp))
            j = jaccard_similarity(A, B)
            if j > best:
                best = j
        out[ci] = best
    return out


def hennig_clusterboot(
    X: np.ndarray,
    original_labels: np.ndarray,
    *,
    k: int,
    algorithm: ClusterAlgorithm = "kmeans",
    n_bootstrap: int = 100,
    threshold: float = 0.75,
    random_state: int = 42,
    n_jobs: int = 1,
) -> HennigStabilityResult:
    """Hennig 2007 clusterboot: per-cluster Jaccard stability.

    For each bootstrap:
      1. Resample n rows with replacement (Hennig's recommendation; standard
         bootstrap, not subsampling).
      2. Recluster the bootstrap with the same algorithm and k.
      3. For each original cluster c:
           - Define A = set of unique original-row indices that were in
             cluster c AND were sampled in the bootstrap.
           - For each bootstrap cluster c': define B = set of unique
             original-row indices assigned to c' (collapsing duplicates).
           - Jaccard(c, b) = max over c' of |A ∩ B| / |A ∪ B|.
      4. Record Jaccard(c, b) for each original cluster c.

    Across all bootstraps, the per-cluster MEDIAN Jaccard is the headline
    stability number. Hennig 2007 calls median >= 0.75 "highly stable",
    0.6 to 0.75 "moderately stable", < 0.6 "unstable".

    Returns ``HennigStabilityResult``.
    """
    X = np.asarray(X, dtype=np.float64)
    original_labels = np.asarray(original_labels, dtype=int)
    if X.shape[0] != original_labels.shape[0]:
        raise ValueError("X and original_labels must have same row count")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    cluster_ids = sorted(int(c) for c in np.unique(original_labels))
    n_clusters = len(cluster_ids)

    # Pre-compute the set of row indices for each original cluster.
    orig_sets: dict[int, set[int]] = {
        c: set(int(i) for i in np.where(original_labels == c)[0])
        for c in cluster_ids
    }

    # Run the bootstrap iterations (parallelisable).
    if n_jobs == 1:
        rows = [
            _hennig_bootstrap_iter(b, k, algorithm, X, cluster_ids,
                                    orig_sets, n_clusters, random_state)
            for b in range(n_bootstrap)
        ]
    else:
        rows = Parallel(n_jobs=n_jobs)(
            delayed(_hennig_bootstrap_iter)(
                b, k, algorithm, X, cluster_ids,
                orig_sets, n_clusters, random_state,
            )
            for b in range(n_bootstrap)
        )
    jaccard = np.column_stack(rows).astype(np.float64)

    jaccard_median = np.median(jaccard, axis=1)
    jaccard_mean = np.mean(jaccard, axis=1)
    n_stable = int(np.sum(jaccard_median >= threshold))

    return HennigStabilityResult(
        cluster_ids=cluster_ids,
        jaccard_median=jaccard_median,
        jaccard_mean=jaccard_mean,
        jaccard_all=jaccard,
        n_bootstrap=int(n_bootstrap),
        threshold=float(threshold),
        n_stable_clusters=n_stable,
        n_total_clusters=int(n_clusters),
        algorithm=str(algorithm),
        k=int(k),
        random_state=int(random_state),
    )


# ─────────────────────────── ARI between cohorts ───────────────────────────


def ari(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Adjusted Rand Index between two label sequences of equal length.

    sklearn's implementation. ARI = 1 for identical clustering (up to
    relabel), 0 for random agreement, can be slightly negative.
    """
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    if labels_a.shape != labels_b.shape:
        raise ValueError(
            f"labels_a {labels_a.shape} vs labels_b {labels_b.shape} mismatch"
        )
    if labels_a.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels_a.shape}")
    return float(adjusted_rand_score(labels_a, labels_b))


def ari_on_shared_pids(
    pids_a: list[str],
    labels_a: np.ndarray,
    pids_b: list[str],
    labels_b: np.ndarray,
) -> tuple[float, list[str]]:
    """Compute ARI on the intersection of pids between two cohort runs.

    Used for the D021 sensitivity comparison: stage 5 on the full N=422
    cohort vs the robust N~280 cohort. Returns (ari, shared_pids).
    """
    if len(pids_a) != len(labels_a):
        raise ValueError("pids_a and labels_a length mismatch")
    if len(pids_b) != len(labels_b):
        raise ValueError("pids_b and labels_b length mismatch")
    a_lookup = dict(zip(pids_a, labels_a))
    b_lookup = dict(zip(pids_b, labels_b))
    shared = sorted(set(a_lookup) & set(b_lookup))
    if not shared:
        return 0.0, []
    la = np.array([a_lookup[p] for p in shared])
    lb = np.array([b_lookup[p] for p in shared])
    return ari(la, lb), shared


__all__ = [
    "HennigStabilityResult",
    "KernelConfounderResult",
    "ari",
    "ari_on_shared_pids",
    "hennig_clusterboot",
    "jaccard_similarity",
    "kernel_chi_square",
]
