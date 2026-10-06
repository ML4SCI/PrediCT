"""Stage 5 PCA on the representative-feature subset (D020 part 2).

Input is the prepared analysis matrix from ``prepare_matrix`` restricted to
the representative features chosen by ``redundancy.run_redundancy_clustering``
(so the matrix has already been z-scored, ComBat-harmonised, and Yeo-Johnson
transformed). Output is a ``PcaResult`` dataclass exposing components,
explained variance, cumulative variance, retained-component count, and the
projected scores.

CRITICAL SIGN CONVENTION (D020):

PCA eigenvectors are sign-arbitrary: multiplying any eigenvector by -1
produces an equally valid solution, but downstream interpretation ("high PC1
means high burden") flips with the sign. We enforce a deterministic sign
rule: for each PC, the feature with the largest absolute loading must have a
non-negative loading. ``normalise_pc_signs`` applies this rule in place on
the components array, and is called inside ``fit_pca`` before scores are
projected. The scores returned by ``fit_pca`` therefore obey "higher score
on PC_i means more of PC_i's dominant feature."

Skipping this step would mean that running stage 5 on Tuesday and running it
on Wednesday could produce sign-flipped scores, and downstream clustering
(which depends on relative score signs) could pick opposite cluster labels
on the two days. This is the "single negative sign destroys results"
failure mode.

NaN / Inf policy:

* ``fit_pca`` asserts no NaN, no Inf, and no constant columns at entry. The
  contract is that prepare_matrix delivers finite z-scored data; failure to
  meet this contract is a hard error.

Determinism:

* sklearn PCA with ``svd_solver='full'`` is deterministic on a given input.
  We also set a random_state for explicit safety even though full SVD does
  not use it. Test ``test_fit_pca_deterministic_across_runs`` verifies this.

Decisions referencing this module:
    D020 - PCA cumvar 0.85, sign convention, scoring
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA


# Feature family map for the loadings audit table.
# Evaluated left-to-right; first matching prefix wins. The "==" entries are
# exact-match families that have no prefix structure. Keep in lockstep with
# the feature_schema.feature_names() output.
FAMILY_MAP: tuple[tuple[str, str], ...] = (
    ("original_shape_",       "PyRad-shape"),
    ("original_firstorder_",  "PyRad-firstorder"),
    ("original_glcm_",        "PyRad-glcm"),
    ("original_glszm_",       "PyRad-glszm"),
    ("original_glrlm_",       "PyRad-glrlm"),
    ("original_ngtdm_",       "PyRad-ngtdm"),
    ("original_gldm_",        "PyRad-gldm"),
    ("agatston_",             "Canonical-burden"),
    ("mass_",                 "Canonical-mass"),
    ("volume_",               "Canonical-volume"),
    ("mean_hu_",              "Canonical-HU"),
    ("max_hu_",               "Canonical-HU"),
    ("lesion_count_",         "Canonical-count"),
    ("n_rois_d",              "Canonical-density-tier"),
    ("inter_lesion_dist_",    "Canonical-distance"),
    ("first_to_last_dist_",   "Canonical-distance"),
    ("dist_from_top_",        "Canonical-spatial"),
    ("center_of_mass_",       "Canonical-spatial"),
    ("gini_",                 "Canonical-distribution"),
    ("n_calcified_arteries",  "Canonical-count"),
    ("has_dense_calcium",     "Canonical-dense"),
    ("high_density_fraction", "Derived-density"),
    ("vessel_burden_gini",    "Derived-distribution"),
)


@dataclass(frozen=True)
class PcaResult:
    """Full PCA output.

    components are sign-normalised per D034. components.shape =
    (n_components_total, n_features). scores.shape = (n_patients, n_retain).
    The first ``n_retain`` PCs satisfy cumulative variance >= cumvar_threshold.
    """
    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_variance_ratio: np.ndarray
    scores: np.ndarray
    feature_names: list[str]
    pid_order: list[str]
    n_retain: int
    n_components_total: int
    cumvar_threshold: float


# ───────────────────────── helpers ─────────────────────────


def assign_family(feature_name: str) -> str:
    """Return the feature family label. Prefix-matched left-to-right; first
    hit wins. Returns ``"Other"`` if no prefix matches.
    """
    for prefix, family in FAMILY_MAP:
        if feature_name == prefix or feature_name.startswith(prefix):
            return family
    return "Other"


def normalise_pc_signs(components: np.ndarray) -> np.ndarray:
    """Flip the sign of any row of ``components`` whose largest-|loading|
    entry is negative. Returns a NEW array (does not mutate input).

    On ties of |loadings|, np.argmax returns the first index; this is rare
    in practice on floating-point data and produces a deterministic answer
    either way.
    """
    if components.ndim != 2:
        raise ValueError(f"components must be 2D, got shape {components.shape}")
    out = components.copy()
    for i in range(out.shape[0]):
        row = out[i]
        if row.size == 0:
            continue
        max_abs_idx = int(np.argmax(np.abs(row)))
        if row[max_abs_idx] < 0.0:
            out[i] = -row
    return out


def select_n_retain(cumvar_ratio: np.ndarray, threshold: float) -> int:
    """First k such that ``cumvar_ratio[k-1] >= threshold``.

    Always returns at least 1. Returns the full length if no element reaches
    the threshold (defensive; should not happen on real data where the last
    cumvar is exactly 1.0).
    """
    if cumvar_ratio.ndim != 1:
        raise ValueError("cumvar_ratio must be 1D")
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    if len(cumvar_ratio) == 0:
        return 0
    mask = cumvar_ratio >= threshold
    if not mask.any():
        return int(len(cumvar_ratio))
    return int(int(np.argmax(mask)) + 1)


# ───────────────────────── core fit ─────────────────────────


def fit_pca(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    cumvar_threshold: float = 0.85,
    pid_col: str = "pid",
    random_state: int = 42,
) -> PcaResult:
    """Fit full SVD PCA on the z-scored feature matrix and return scores +
    components + variance tables.

    The function asserts:
      * no NaN in feature_cols
      * no Inf in feature_cols
      * no constant column in feature_cols (sd > 0)

    Sign-normalisation is applied before scores are projected. Scores are
    therefore consistent with the orientation of the components in the
    returned result.
    """
    if pid_col not in df.columns:
        raise KeyError(f"dataframe missing pid column {pid_col!r}")
    if not feature_cols:
        raise ValueError("feature_cols must not be empty")

    X = df[feature_cols].to_numpy(dtype=np.float64)
    if np.isnan(X).any():
        nan_cols = [c for c, has in zip(feature_cols, np.isnan(X).any(axis=0)) if has]
        raise ValueError(f"NaN in PCA input columns: {nan_cols[:5]}")
    if np.isinf(X).any():
        inf_cols = [c for c, has in zip(feature_cols, np.isinf(X).any(axis=0)) if has]
        raise ValueError(f"Inf in PCA input columns: {inf_cols[:5]}")
    sds = X.std(axis=0, ddof=1)
    if not (sds > 0).all():
        const = [c for c, s in zip(feature_cols, sds) if s <= 0.0]
        raise ValueError(f"constant column(s) at PCA entry: {const}")

    n_components = min(X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, svd_solver="full", random_state=random_state)
    pca.fit(X)

    components = normalise_pc_signs(pca.components_)
    explained_variance = np.asarray(pca.explained_variance_, dtype=np.float64)
    explained_variance_ratio = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    n_retain = select_n_retain(cumulative_variance_ratio, cumvar_threshold)
    scores = X @ components[:n_retain].T

    return PcaResult(
        components=components,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_variance_ratio,
        cumulative_variance_ratio=cumulative_variance_ratio,
        scores=scores,
        feature_names=list(feature_cols),
        pid_order=[str(p) for p in df[pid_col].tolist()],
        n_retain=n_retain,
        n_components_total=int(n_components),
        cumvar_threshold=float(cumvar_threshold),
    )


# ───────────────────────── audit tables ─────────────────────────


def explained_variance_table(result: PcaResult) -> pd.DataFrame:
    """Per-PC table: pc, eigenvalue, explained_var_pct, cumulative_var_pct, retained."""
    rows = []
    for i in range(result.n_components_total):
        rows.append({
            "pc": f"PC{i + 1}",
            "eigenvalue": float(result.explained_variance[i]),
            "explained_var_pct": float(result.explained_variance_ratio[i] * 100.0),
            "cumulative_var_pct": float(result.cumulative_variance_ratio[i] * 100.0),
            "retained": bool(i < result.n_retain),
        })
    return pd.DataFrame(rows)


def top_loadings_table(
    result: PcaResult,
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    """Per-(PC, feature) table of the top |loadings| within each retained PC.

    Columns: pc, rank (1-indexed), feature, loading (signed), abs_loading,
    family.
    """
    rows = []
    for i in range(result.n_retain):
        loadings = result.components[i]
        top_idx = np.argsort(np.abs(loadings))[::-1][:top_n]
        for rank, idx in enumerate(top_idx, start=1):
            name = result.feature_names[idx]
            rows.append({
                "pc": f"PC{i + 1}",
                "rank": rank,
                "feature": name,
                "loading": float(loadings[idx]),
                "abs_loading": float(abs(loadings[idx])),
                "family": assign_family(name),
            })
    return pd.DataFrame(rows)


def pc_external_correlation(
    result: PcaResult,
    external: pd.Series,
    *,
    name: str = "external",
) -> pd.DataFrame:
    """Spearman rho and p-value of each retained PC score vs an external
    reference variable.

    Used to verify "PC1 is a burden axis" by passing ``agatston_total`` as
    the external reference. ``external`` must align positionally to
    ``result.pid_order`` (same length, same order). The function asserts.

    Returns a dataframe with one row per retained PC:
        pc, spearman_rho, pval, abs_rho.
    """
    if len(external) != len(result.pid_order):
        raise ValueError(
            f"external length {len(external)} != pid_order length "
            f"{len(result.pid_order)}; align upstream."
        )
    ext_arr = np.asarray(external, dtype=float)
    rows = []
    for i in range(result.n_retain):
        rho, pval = stats.spearmanr(result.scores[:, i], ext_arr)
        rows.append({
            "pc": f"PC{i + 1}",
            "external": name,
            "spearman_rho": float(rho) if not np.isnan(rho) else float("nan"),
            "pval": float(pval) if not np.isnan(pval) else float("nan"),
            "abs_rho": float(abs(rho)) if not np.isnan(rho) else float("nan"),
        })
    return pd.DataFrame(rows)


# ───────────────────────── orthogonality / sanity check ─────────────────────────


def assert_components_orthonormal(
    components: np.ndarray, *, atol: float = 1e-8,
) -> None:
    """Hard assertion that the rows of ``components`` form an orthonormal
    set: ``components @ components.T == I``. Used in tests and as an
    optional self-check before downstream consumers trust the result.
    """
    if components.ndim != 2:
        raise ValueError("components must be 2D")
    k = components.shape[0]
    if k == 0:
        return
    gram = components @ components.T
    diff = gram - np.eye(k)
    if not np.allclose(diff, 0.0, atol=atol):
        max_err = float(np.max(np.abs(diff)))
        raise AssertionError(
            f"components not orthonormal: max |G - I| = {max_err:.2e}"
        )


__all__ = [
    "FAMILY_MAP",
    "PcaResult",
    "assert_components_orthonormal",
    "assign_family",
    "explained_variance_table",
    "fit_pca",
    "normalise_pc_signs",
    "pc_external_correlation",
    "select_n_retain",
    "top_loadings_table",
]
