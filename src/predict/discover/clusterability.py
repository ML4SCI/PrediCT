"""Stage 5 cluster tendency check via Hopkins statistic (D021 part 1).

Hopkins (Hopkins 1954, Lawson and Jurs 1990, Banerjee 2004) compares the
nearest-neighbour distance distribution of:
  * real data points (sampled at random from the data, distance to their
    2nd-nearest neighbour, which skips self), denoted ``u_i``
  * synthetic points sampled uniformly inside the data's bounding box,
    distance to their nearest neighbour in the data, denoted ``w_i``

Hopkins H = sum(w_i) / (sum(w_i) + sum(u_i))

A real query is a row of X, so its 2nd neighbour skips itself; a uniform
query is not in X and uses its 1st neighbour (as pyclustertend does). This
gives H ~ 0.5 on uniform random data (D021.1; using the 2nd neighbour for
uniform queries too would bias H upward). Tests must seed the
data and this function independently: with equal seeds the uniform queries
replay the data's own random stream and land on data points. Plain
distances are used; the "d-th power" convention of the original Hopkins
paper is unstable at d >= 5.

Interpretation (Banerjee 2004 ranges):
  H near 1.0  -> strong clustering (real points are tightly grouped relative
                 to the bounding box, so uniform samples are far from data)
  H around 0.5 -> random distribution
  H below 0.5  -> regular spacing (anti-clustering)

We adopt thresholds from the D021 config:
  H >= 0.65       -> clustering tendency confirmed
  0.55 <= H < 0.65 -> ambiguous (report both downstream outcomes)
  H < 0.55       -> no cluster tendency

This module is a fast guard. If H is below the ambiguous band the
downstream gap statistic + consensus clustering are still run, but the
expected outcome (k = 1) is documented up front.

Determinism: seeded random sample of m data rows and m uniform points; same
seed plus same data give a byte-identical H.

Decisions referencing this module:
    D021 - Hopkins threshold 0.65, ambiguous band 0.55 to 0.65, sample 10%.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial import cKDTree


HopkinsVerdict = Literal["clustered", "ambiguous", "random_or_regular"]


@dataclass(frozen=True)
class HopkinsResult:
    """Structured Hopkins output for the audit log.

    Fields:
        H                 the statistic in [0, 1]
        n_total           full dataset size
        n_features        dimensionality
        sample_size       number of (real, uniform) pairs used
        threshold         cluster-tendency threshold (default 0.65)
        ambiguous_band    (low, high) range that triggers the "ambiguous"
                          verdict (default (0.55, 0.65))
        verdict           "clustered" | "ambiguous" | "random_or_regular"
        random_state      seed used for the sample draw
    """
    H: float
    n_total: int
    n_features: int
    sample_size: int
    threshold: float
    ambiguous_band: tuple[float, float]
    verdict: HopkinsVerdict
    random_state: int

    def to_dict(self) -> dict:
        return {
            "H": self.H,
            "n_total": self.n_total,
            "n_features": self.n_features,
            "sample_size": self.sample_size,
            "threshold": self.threshold,
            "ambiguous_band": list(self.ambiguous_band),
            "verdict": self.verdict,
            "random_state": self.random_state,
        }


def hopkins_statistic(
    X: np.ndarray,
    *,
    sample_size: int | None = None,
    sample_frac: float = 0.10,
    random_state: int = 42,
) -> float:
    """Compute the Hopkins statistic on data matrix ``X`` of shape (n, d).

    If ``sample_size`` is None, uses ``max(1, int(sample_frac * n))``
    (default 10% of the rows). Returns H in [0, 1].

    Numerical notes:
      * Distances are Euclidean.
      * Real queries (sampled from X) use the 2nd-nearest neighbour in X,
        skipping themselves at distance 0. Uniform queries are not in X, so
        they use the 1st-nearest neighbour (as pyclustertend does). This
        makes H ~ 0.5 on uniform random data. Letting uniform queries skip
        their nearest neighbour too would inflate H on random data (mean
        0.667 on 1-D uniform; D021.1).
      * Uniform samples are drawn in the per-feature bounding box of X.
      * The function is fully deterministic for fixed ``random_state``.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    n, d = X.shape
    if n < 3:
        raise ValueError(
            f"need at least 3 rows for 2nd-NN queries on both real and "
            f"uniform samples, got n = {n}"
        )

    if sample_size is None:
        sample_size = max(1, int(round(sample_frac * n)))
    if sample_size < 1:
        raise ValueError(f"sample_size must be >= 1, got {sample_size}")
    if sample_size >= n:
        raise ValueError(
            f"sample_size {sample_size} must be < n = {n} so the 2nd-nearest "
            "neighbour exists when sampling real points"
        )

    rng = np.random.default_rng(random_state)

    # m random points from the real data (without replacement).
    sample_idx = rng.choice(n, size=sample_size, replace=False)
    real_sample = X[sample_idx]

    # m uniform points in the per-feature bounding box of X.
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    uniform_sample = rng.uniform(low=mins, high=maxs, size=(sample_size, d))

    # Build kd-tree on the full dataset.
    tree = cKDTree(X)

    # For each real sample: 2nd-NN (skip self at distance 0).
    u_dist, _ = tree.query(real_sample, k=2)
    u_distances = u_dist[:, 1]

    # For each uniform sample: 1st-NN (a generated point has no self-match).
    w_dist, _ = tree.query(uniform_sample, k=1)
    w_distances = np.asarray(w_dist, dtype=np.float64).reshape(-1)

    u_sum = float(u_distances.sum())
    w_sum = float(w_distances.sum())
    denom = u_sum + w_sum
    if denom == 0.0:
        # Degenerate: all distances are zero. Treat as random.
        return 0.5
    return float(w_sum / denom)


def assess_clusterability(
    X: np.ndarray,
    *,
    sample_frac: float = 0.10,
    threshold: float = 0.65,
    ambiguous_band: tuple[float, float] = (0.55, 0.65),
    random_state: int = 42,
) -> HopkinsResult:
    """Compute Hopkins and classify the verdict against the D021 thresholds.

    Returns ``HopkinsResult`` for downstream logging. Verdict logic:
      H >= threshold (default 0.65)              -> "clustered"
      ambiguous_band[0] <= H < threshold         -> "ambiguous"
      H < ambiguous_band[0]                      -> "random_or_regular"
    """
    if not (0.0 <= ambiguous_band[0] <= ambiguous_band[1] <= 1.0):
        raise ValueError(
            f"ambiguous_band must satisfy 0 <= low <= high <= 1, "
            f"got {ambiguous_band}"
        )
    n, d = X.shape
    sample_size = max(1, int(round(sample_frac * n)))
    H = hopkins_statistic(
        X, sample_size=sample_size, random_state=random_state,
    )
    if H >= threshold:
        verdict: HopkinsVerdict = "clustered"
    elif H >= ambiguous_band[0]:
        verdict = "ambiguous"
    else:
        verdict = "random_or_regular"

    return HopkinsResult(
        H=H,
        n_total=int(n),
        n_features=int(d),
        sample_size=int(sample_size),
        threshold=float(threshold),
        ambiguous_band=(float(ambiguous_band[0]), float(ambiguous_band[1])),
        verdict=verdict,
        random_state=int(random_state),
    )


__all__ = [
    "HopkinsResult",
    "HopkinsVerdict",
    "assess_clusterability",
    "hopkins_statistic",
]
