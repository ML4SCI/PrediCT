"""Synthetic cluster injection for the detectability floor (D036).

Plants k groups with a known separation into a feature matrix so the
unchanged D020 / D021 pipeline can be asked whether it recovers them. Without
this, "the data form a continuum" and "the pipeline could not have found
clusters" are observationally identical.

Separation is expressed in units of the background's own standard deviation
along each planting direction: Delta = 2 means adjacent group means sit two
background SDs apart along that axis. Groups sit on the vertices of a regular
simplex, so every pair of groups is equally separated.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

Proportions = Literal["balanced", "imbalanced"]


def simplex_vertices(k: int, delta: float) -> np.ndarray:
    """Vertices of a centred regular simplex in R^(k-1), pairwise distance delta.

    Returns shape (k, k-1). Rows are group means in planting-direction
    coordinates.
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    E = np.eye(k) - 1.0 / k                     # centred standard basis
    # Orthonormal basis of the (k-1)-dim subspace orthogonal to ones.
    U, _s, _vt = np.linalg.svd(E)
    coords = E @ U[:, : k - 1]                  # (k, k-1), pairwise dist sqrt(2)
    return coords * (delta / np.sqrt(2.0))


def group_counts(n: int, k: int, kind: Proportions) -> np.ndarray:
    """Integer group sizes summing to n.

    balanced: equal; imbalanced: proportional to 1, 1/2, 1/4, ... (k=3 gives
    roughly 57 / 29 / 14 %).
    """
    if kind == "balanced":
        w = np.ones(k)
    elif kind == "imbalanced":
        w = 0.5 ** np.arange(k)
    else:
        raise ValueError(f"unknown proportions {kind!r}")
    raw = w / w.sum() * n
    counts = np.floor(raw).astype(int)
    counts[np.argsort(raw - counts)[::-1][: n - counts.sum()]] += 1
    return counts


def assign_groups(rng: np.random.Generator, n: int, k: int, kind: Proportions) -> np.ndarray:
    """Random group labels with the requested sizes, independent of the data."""
    labels = np.repeat(np.arange(k), group_counts(n, k, kind))
    return rng.permutation(labels)


def planting_directions(
    rng: np.random.Generator, d: int, m: int, avoid: np.ndarray | None = None,
) -> np.ndarray:
    """m orthonormal random directions in R^d, shape (d, m).

    If ``avoid`` (shape (d,)) is given, the directions are orthogonal to it
    (used to plant structure orthogonal to the burden axis, PC1).
    """
    if m < 1 or m > d - (avoid is not None):
        raise ValueError(f"cannot draw {m} directions in R^{d}")
    G = rng.standard_normal((d, m))
    if avoid is not None:
        a = np.asarray(avoid, float).reshape(d, 1)
        a = a / np.linalg.norm(a)
        G = G - a @ (a.T @ G)
    Q, R = np.linalg.qr(G)
    return Q * np.sign(np.diag(R))              # deterministic sign


def plant(
    X: np.ndarray, labels: np.ndarray, delta: float, U: np.ndarray, cov: np.ndarray,
) -> np.ndarray:
    """Add group mean shifts to X.

    U: (d, k-1) orthonormal planting directions. cov: (d, d) covariance of the
    background, used to express delta in background-SD units along each
    direction. Returns a new array; X is not modified.
    """
    k = int(labels.max()) + 1
    if U.shape[1] != k - 1:
        raise ValueError(f"need {k - 1} directions for k={k}, got {U.shape[1]}")
    sd_along = np.sqrt(np.einsum("ij,jk,ki->i", U.T, cov, U))
    V = simplex_vertices(k, delta)              # (k, k-1)
    shifts = (V * sd_along) @ U.T               # (k, d)
    return X + shifts[labels]


def zscore(X: np.ndarray) -> np.ndarray:
    """Column z-score (ddof=1), as D019 applies before PCA."""
    sd = X.std(axis=0, ddof=1)
    if not (sd > 0).all():
        raise ValueError("constant column after planting")
    return (X - X.mean(axis=0)) / sd


__all__ = [
    "Proportions", "assign_groups", "group_counts", "plant",
    "planting_directions", "simplex_vertices", "zscore",
]
