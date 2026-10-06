"""Stage 5 analysis-matrix preparation (D019).

Named ``prepare_matrix`` (not ``preprocessing``) to avoid collision with
``predict.preprocess`` from stage 2. The module takes the gated 88-feature
set from stage 4 and produces the analysis-ready matrix that downstream
PCA / clustering consume.

Seven composable transforms applied in order:

  1. ``apply_d017_drops``           - drop 13 sentinel-prone per-vessel features
  2. ``apply_d018_binarisation``    - dense_calcium_count -> has_dense_calcium
  3. ``maybe_add_derived_features`` - high_density_fraction, vessel_burden_gini
                                      (conditional on R^2 < 0.95 non-redundancy gate)
  4. ``variance_filter``            - drop columns with sd < threshold
  5. ``combat_harmonise``           - ComBat on every gated non-shape PyRadiomics
                                      column (D035), kernel as covariate; the
                                      rest pass through
  6. ``yeo_johnson_with_fallback``  - Yeo-Johnson on 19 sparse columns; if
                                      |post-YJ skewness| > 1.0 falls back to
                                      rank transform on that column
  7. ``global_zscore``              - z-score across the full matrix

``run_matrix_prep`` chains all of them and returns the final (n_patients,
n_features) numpy array plus a structured log of what each step did.

CONTRACTS (publication-grade, do not weaken):

* Every transform takes a pandas DataFrame indexed by ``pid`` (str) and a
  list of feature column names. It returns (transformed_df, new_columns,
  diagnostics_dict). Patient order is preserved exactly. Feature columns
  not in the active feature list pass through unchanged.
* Every transform asserts NO NaN at entry on the active feature columns.
  This is correct because the stage-4 gate guarantees the 422-patient
  ``radiomics_status == "ok"`` subset has full data. NaN at entry means
  upstream contract violation; we fail loudly.
* Every transform that has a sign convention (ComBat shift direction,
  Yeo-Johnson lambda) is unit-tested for sign correctness.
* Every transform that uses randomness logs the seed in diagnostics.
* The seven steps are individually reproducible: rerunning a step on its own
  input produces byte-identical output (no global state).

Decisions referencing this module:
    D017 - sentinel-prone feature drops
    D018 - density tier + dense_calcium handling
    D019 - this preparation pipeline
    D040 - vessel_burden_gini over all four vessels; stage-5 transformed
           features carry measured D033 reliability
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer

from predict.config import VESSEL_NAMES


# ───────────────────────── D017 / D018 fixed transforms ─────────────────────────


D017_DROPPED_FEATURES: tuple[str, ...] = (
    # Diffusivity (4): provably degenerate, perfectly equals 1{count==1}.
    "diffusivity_lad", "diffusivity_rca", "diffusivity_lcx", "diffusivity_lm",
    # Distance for non-LAD vessels (9): sentinel rates 56-93%.
    "inter_lesion_dist_mean_rca", "inter_lesion_dist_mean_lcx", "inter_lesion_dist_mean_lm",
    "inter_lesion_dist_max_rca",  "inter_lesion_dist_max_lcx",  "inter_lesion_dist_max_lm",
    "first_to_last_dist_rca",     "first_to_last_dist_lcx",     "first_to_last_dist_lm",
)


D018_BINARISE_SOURCE: str = "dense_calcium_count"
D018_BINARISE_TARGET: str = "has_dense_calcium"

# Features stage 5 creates from canonical inputs (D018 binarisation, D019
# derived features). Stage 5b computes each one per perturbation draw with
# ``stage5_transformed_columns`` and gates it with the D033 rule (D040), so
# their reliability is measured, never assumed. All are canonical-lineage
# (XML-derived) features.
STAGE5_TRANSFORMED: tuple[str, ...] = (
    D018_BINARISE_TARGET, "high_density_fraction", "vessel_burden_gini",
)


# Columns that need the Yeo-Johnson treatment with rank fallback.
# 16 density-tier counts + 3 LAD distance features.
SPARSE_COLUMNS: tuple[str, ...] = tuple(
    f"n_rois_d{tier}_{v}"
    for tier in (1, 2, 3, 4)
    for v in ("lad", "rca", "lcx", "lm")
) + (
    "inter_lesion_dist_mean_lad",
    "inter_lesion_dist_max_lad",
    "first_to_last_dist_lad",
)


def harmonisable_columns(feature_cols: Iterable[str]) -> list[str]:
    """Columns ComBat harmonises (D035): every PyRadiomics feature in the
    active set except shape, in input order.

    Intensity-derived PyRadiomics features (firstorder and texture) carry
    reconstruction-kernel effects; shape features read only the binary mask
    and pass through. Canonical XML-derived features are never harmonised
    (D019). Derived from the gated set at run time, so the list cannot go
    stale when the gate changes (D035).
    """
    return [
        c for c in feature_cols
        if c.startswith("original_") and not c.startswith("original_shape_")
    ]


# ───────────────────────── diagnostics container ─────────────────────────


@dataclass
class MatrixPrepLog:
    """Structured per-step record for ``outputs/06_reduce/matrix_prep_log.json``."""
    n_patients_in: int = 0
    n_features_in: int = 0
    d017_dropped: list[str] = field(default_factory=list)
    d018_target_present: bool = False
    derived_accepted: list[dict] = field(default_factory=list)
    derived_rejected: list[dict] = field(default_factory=list)
    d018_dropped_unreliable: bool = False
    variance_dropped: list[dict] = field(default_factory=list)
    combat_audit: list[dict] = field(default_factory=list)
    yj_per_column: list[dict] = field(default_factory=list)
    zscore_columns: list[str] = field(default_factory=list)
    # D029 / stage 8: captured per-column means + stds from global_zscore.
    # Required so stage 8 holdout / leave-k-out projections can reapply the
    # IDENTICAL fits without leakage. Each is {col -> float}.
    zscore_means: dict = field(default_factory=dict)
    zscore_stds: dict = field(default_factory=dict)
    # Full precision (the rounded pair above is for reading); stage 8
    # replays the fit with these.
    zscore_means_exact: dict = field(default_factory=dict)
    zscore_stds_exact: dict = field(default_factory=dict)
    n_patients_out: int = 0
    n_features_out: int = 0

    def to_dict(self) -> dict:
        return {
            "n_patients_in": self.n_patients_in,
            "n_features_in": self.n_features_in,
            "d017_dropped": list(self.d017_dropped),
            "d018_target_present": bool(self.d018_target_present),
            "derived_accepted": list(self.derived_accepted),
            "derived_rejected": list(self.derived_rejected),
            "variance_dropped": list(self.variance_dropped),
            "combat_audit": list(self.combat_audit),
            "yj_per_column": list(self.yj_per_column),
            "zscore_columns": list(self.zscore_columns),
            "zscore_means": dict(self.zscore_means),
            "zscore_stds": dict(self.zscore_stds),
            "zscore_means_exact": dict(self.zscore_means_exact),
            "zscore_stds_exact": dict(self.zscore_stds_exact),
            "n_patients_out": self.n_patients_out,
            "n_features_out": self.n_features_out,
        }


# ───────────────────────── helpers ─────────────────────────


def _assert_no_nan(df: pd.DataFrame, columns: Iterable[str], step: str) -> None:
    cols = list(columns)
    if not cols:
        return
    sub = df.loc[:, cols]
    if sub.isna().any().any():
        nan_cols = sub.columns[sub.isna().any()].tolist()
        raise ValueError(
            f"[{step}] NaN found in feature columns: {nan_cols[:5]} "
            f"(and {max(0, len(nan_cols) - 5)} more). "
            "Stage 5 contract requires no NaN at preprocessing entry; "
            "upstream stages must filter to radiomics_status == 'ok' first."
        )


def _assert_pid_column(df: pd.DataFrame, step: str) -> None:
    if "pid" not in df.columns:
        raise KeyError(f"[{step}] dataframe missing 'pid' column")


# ───────────────────────── D017 drops ─────────────────────────


def apply_d017_drops(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Drop the 13 D017 sentinel-prone per-vessel features if present.

    Idempotent (rerunning on already-dropped frame is a no-op).
    Returns (new_df, new_feature_cols, dropped_names).
    """
    _assert_pid_column(df, "apply_d017_drops")
    drops = [c for c in D017_DROPPED_FEATURES if c in df.columns]
    new_df = df.drop(columns=drops, errors="ignore")
    new_feature_cols = [c for c in feature_cols if c not in drops]
    return new_df, new_feature_cols, drops


# ───────────────────────── D018 binarisation ─────────────────────────


def apply_d018_binarisation(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[str], bool]:
    """Replace ``dense_calcium_count`` with ``has_dense_calcium`` (binary 0/1).

    No-op if the source column is absent. The new binary column inherits the
    source's position in the feature_cols list. Source column is dropped.
    """
    _assert_pid_column(df, "apply_d018_binarisation")
    new_df = df.copy()
    if D018_BINARISE_SOURCE not in new_df.columns:
        return new_df, list(feature_cols), False

    new_df[D018_BINARISE_TARGET] = (new_df[D018_BINARISE_SOURCE] > 0).astype(int)
    new_df = new_df.drop(columns=[D018_BINARISE_SOURCE])
    new_feature_cols = [
        D018_BINARISE_TARGET if c == D018_BINARISE_SOURCE else c
        for c in feature_cols
    ]
    return new_df, new_feature_cols, True


# ───────────────────────── derived features ─────────────────────────


def compute_high_density_fraction(df: pd.DataFrame) -> pd.Series | None:
    """(d3 + d4 count) / (d1 + d2 + d3 + d4 count), summed across all 4 vessels.

    Returns None if any required column is missing. Patients with zero total
    tier-bin count get value 0.0 (avoids division by zero).
    """
    d12 = [f"n_rois_d{t}_{v}" for t in (1, 2) for v in ("lad", "rca", "lcx", "lm")]
    d34 = [f"n_rois_d{t}_{v}" for t in (3, 4) for v in ("lad", "rca", "lcx", "lm")]
    needed = d12 + d34
    if not all(c in df.columns for c in needed):
        return None
    d12_sum = df[d12].sum(axis=1)
    d34_sum = df[d34].sum(axis=1)
    total = d12_sum + d34_sum
    out = pd.Series(0.0, index=df.index)
    mask = total > 0
    out.loc[mask] = d34_sum.loc[mask] / total.loc[mask]
    return out


def compute_vessel_burden_gini(df: pd.DataFrame) -> pd.Series | None:
    """Gini coefficient across the 4 per-vessel Agatston values (D019),
    calcium-free vessels included (D040).

    Range 0 (burden spread evenly over LAD, RCA, LCx, LM) to 0.75 (all
    burden in one vessel). A patient with no calcium returns 0.0 (does not
    occur in COCA). Until D040 the calcium-free vessels were dropped first,
    which gave a single-vessel patient 0, the value of perfectly even
    disease: the opposite of D019's "across four vessels" and of D025's
    focal-concentration hypothesis.
    """
    cols = [f"agatston_{v.lower()}" for v in VESSEL_NAMES]
    if not all(c in df.columns for c in cols):
        return None
    out = pd.Series(0.0, index=df.index)
    arr = df[cols].to_numpy(dtype=float)
    n = arr.shape[1]
    ranks = np.arange(1, n + 1)
    for i in range(arr.shape[0]):
        sorted_v = np.sort(arr[i])
        total = sorted_v.sum()
        if total <= 0:
            continue
        cum = np.sum(ranks * sorted_v)
        out.iloc[i] = float((2.0 * cum) / (n * total) - (n + 1) / n)
    return out


def stage5_transformed_columns(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Every STAGE5_TRANSFORMED feature computable from ``df``'s canonical
    columns, with the same functions the matrix preparation uses. Stage 5b
    applies this to each perturbation draw to measure their reliability."""
    out: dict[str, pd.Series] = {}
    if D018_BINARISE_SOURCE in df.columns:
        out[D018_BINARISE_TARGET] = (df[D018_BINARISE_SOURCE] > 0).astype(int)
    for name, fn in (("high_density_fraction", compute_high_density_fraction),
                     ("vessel_burden_gini", compute_vessel_burden_gini)):
        s = fn(df)
        if s is not None:
            out[name] = s
    return out


def _r2_against_existing(
    df: pd.DataFrame,
    new_series: pd.Series,
    existing_cols: list[str],
) -> float:
    """R^2 of the new series predicted by a linear combination of existing
    feature columns. Used as the non-redundancy gate at < 0.95.

    Both inputs are assumed to be NaN-free over the same index.
    """
    X = df[existing_cols].to_numpy(dtype=float)
    y = new_series.to_numpy(dtype=float)
    if X.shape[0] != y.shape[0]:
        raise ValueError("row counts must match")
    if np.var(y) == 0.0:
        return 1.0  # constant target is perfectly "explained" by any constant
    model = LinearRegression().fit(X, y)
    y_hat = model.predict(X)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0
    return float(1.0 - ss_res / ss_tot)


def maybe_add_derived_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    r2_redundancy_threshold: float = 0.95,
    reliable: set[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[dict], list[dict]]:
    """Conditionally append ``high_density_fraction`` and ``vessel_burden_gini``.

    Each candidate is added only if its R^2 against linear combinations of the
    existing feature columns is below ``r2_redundancy_threshold`` (default 0.95).
    ``count_volume_ratio`` is intentionally not offered (D019 alternatives).

    ``reliable`` (D040): when given, a candidate not in it is rejected
    before the R^2 check because it failed the D033 reliability gate on the
    perturbation draws (stage 5b ``derived_icc.csv``).

    Returns (new_df, new_feature_cols, accepted_records, rejected_records).
    Each record is {name, r2, threshold, accepted}.
    """
    _assert_pid_column(df, "maybe_add_derived_features")
    _assert_no_nan(df, feature_cols, "maybe_add_derived_features")

    new_df = df.copy()
    new_cols = list(feature_cols)
    accepted: list[dict] = []
    rejected: list[dict] = []

    candidates: list[tuple[str, pd.Series | None]] = [
        ("high_density_fraction", compute_high_density_fraction(df)),
        ("vessel_burden_gini", compute_vessel_burden_gini(df)),
    ]

    for name, series in candidates:
        if series is None:
            rejected.append({
                "name": name,
                "r2": None,
                "threshold": r2_redundancy_threshold,
                "accepted": False,
                "reason": "input columns missing",
            })
            continue
        if series.isna().any():
            rejected.append({
                "name": name,
                "r2": None,
                "threshold": r2_redundancy_threshold,
                "accepted": False,
                "reason": "NaN in computed values",
            })
            continue
        if reliable is not None and name not in reliable:
            rejected.append({
                "name": name,
                "r2": None,
                "threshold": r2_redundancy_threshold,
                "accepted": False,
                "reason": "fails the D033 reliability gate (stage 5b derived_icc.csv)",
            })
            continue
        # Compute R^2 against the CURRENT state of the matrix (new_df), not
        # the original df, so earlier-accepted derived features participate
        # in the redundancy check for later candidates.
        r2 = _r2_against_existing(new_df, series, new_cols)
        rec = {
            "name": name,
            "r2": round(r2, 4),
            "threshold": r2_redundancy_threshold,
            "accepted": bool(r2 < r2_redundancy_threshold),
        }
        if rec["accepted"]:
            new_df[name] = series.values
            new_cols.append(name)
            accepted.append(rec)
        else:
            rec["reason"] = (
                f"R^2 = {r2:.4f} >= {r2_redundancy_threshold:.2f} "
                "(redundant with existing features)"
            )
            rejected.append(rec)

    return new_df, new_cols, accepted, rejected


# ───────────────────────── variance filter ─────────────────────────


def variance_filter(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    threshold: float = 0.01,
) -> tuple[pd.DataFrame, list[str], list[dict]]:
    """Drop columns whose standard deviation is below ``threshold``.

    Returns (new_df, new_feature_cols, dropped_records).
    """
    _assert_pid_column(df, "variance_filter")
    _assert_no_nan(df, feature_cols, "variance_filter")

    sds = df[feature_cols].std(ddof=1)
    keep = [c for c in feature_cols if sds[c] >= threshold]
    dropped_names = [c for c in feature_cols if sds[c] < threshold]
    dropped_records = [
        {"feature": c, "sd": round(float(sds[c]), 6), "threshold": threshold}
        for c in dropped_names
    ]
    other_cols = [c for c in df.columns if c not in feature_cols]
    return df[other_cols + keep].copy(), keep, dropped_records


# ───────────────────────── ComBat ─────────────────────────


def _explained_variance_by_kernel(
    df: pd.DataFrame, columns: list[str], kernel_col: str = "kernel",
) -> dict[str, float]:
    """R^2 of OLS regressing each feature column on the kernel one-hot indicator.

    Used pre/post ComBat to verify the harmonisation worked.
    """
    if kernel_col not in df.columns:
        raise KeyError(f"kernel column {kernel_col!r} missing from df")
    if not columns:
        return {}
    dummies = pd.get_dummies(df[kernel_col], drop_first=True).to_numpy(dtype=float)
    if dummies.shape[1] == 0:
        return {c: 0.0 for c in columns}
    out: dict[str, float] = {}
    for col in columns:
        y = df[col].to_numpy(dtype=float)
        if np.var(y) == 0.0:
            out[col] = 0.0
            continue
        # Closed-form OLS R^2.
        X = np.column_stack([np.ones(len(y)), dummies])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        out[col] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return out


def combat_harmonise(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    columns_to_harmonise: Iterable[str] | None = None,
    kernel_col: str = "kernel",
    acceptance_max_post_r2: float = 0.02,
) -> tuple[pd.DataFrame, list[str], list[dict]]:
    """Apply ComBat (Johnson 2007) to the listed columns with kernel as the
    batch covariate. Columns not in ``columns_to_harmonise`` pass through.
    ``columns_to_harmonise=None`` (the default) means
    :func:`harmonisable_columns` of ``feature_cols`` (D035).

    Uses neuroCombat under the hood. Wrapper handles patient-ID alignment and
    produces the pre/post scanner R^2 audit.

    Returns (new_df, feature_cols, audit_records). Audit record:
        {feature, kernel_r2_pre, kernel_r2_post, passes_threshold}

    Raises if any post-ComBat R^2 exceeds ``acceptance_max_post_r2`` (default
    0.02 per Lin 2022 Table S3). Caller can catch and inspect the audit.
    """
    _assert_pid_column(df, "combat_harmonise")
    _assert_no_nan(df, feature_cols, "combat_harmonise")
    if kernel_col not in df.columns:
        raise KeyError(f"kernel column {kernel_col!r} required for ComBat")

    if columns_to_harmonise is None:
        columns_to_harmonise = harmonisable_columns(feature_cols)
    cols = [c for c in columns_to_harmonise if c in df.columns]
    if not cols:
        return df.copy(), list(feature_cols), []

    # Single-kernel cohort: ComBat is a no-op because there are no batches
    # to harmonise across. This happens in the D021 kernel-stratified
    # sensitivity reruns where we restrict to one kernel at a time. We
    # short-circuit and emit a single audit record per harmonisable column
    # with pre = post = 0 so downstream code sees an explicit pass.
    kernel_counts = df[kernel_col].value_counts()
    if len(kernel_counts) <= 1:
        audit = [
            {
                "feature": c,
                "kernel_r2_pre": 0.0,
                "kernel_r2_post": 0.0,
                "passes_threshold": True,
                "skipped": True,
                "reason": "single-kernel cohort; no batches to harmonise",
            }
            for c in cols
        ]
        return df.copy(), list(feature_cols), audit

    # Defense in depth: ComBat needs >= 2 samples per batch to estimate the
    # within-batch variance. Fail loud if a singleton-kernel slipped through
    # the cohort filter; the orchestrator is supposed to drop these.
    singletons = kernel_counts[kernel_counts < 2]
    if not singletons.empty:
        raise ValueError(
            f"[combat_harmonise] singleton-sample kernel group(s) "
            f"{singletons.to_dict()} cannot be ComBat-harmonised. Filter "
            f"cohort to kernels with >= 2 samples before calling."
        )

    # Pre-ComBat audit.
    pre = _explained_variance_by_kernel(df, cols, kernel_col=kernel_col)

    # Apply ComBat. neuroCombat expects (features x samples) input.
    try:
        from neuroCombat import neuroCombat
    except ImportError as exc:
        raise ImportError(
            "ComBat requires the neuroCombat package. Install with: "
            "pip install neuroCombat"
        ) from exc

    data = df[cols].to_numpy(dtype=float).T   # (n_features, n_samples)
    covars = pd.DataFrame({kernel_col: df[kernel_col].values})
    result = neuroCombat(
        dat=data, covars=covars, batch_col=kernel_col,
        categorical_cols=None, continuous_cols=None,
    )
    harmonised = np.asarray(result["data"]).T   # back to (n_samples, n_features)

    new_df = df.copy()
    for j, col in enumerate(cols):
        new_df[col] = harmonised[:, j]

    # Post-ComBat audit.
    post = _explained_variance_by_kernel(new_df, cols, kernel_col=kernel_col)
    audit = [
        {
            "feature": c,
            "kernel_r2_pre": round(pre[c], 6),
            "kernel_r2_post": round(post[c], 6),
            "passes_threshold": bool(post[c] <= acceptance_max_post_r2),
        }
        for c in cols
    ]
    failing = [rec for rec in audit if not rec["passes_threshold"]]
    if failing:
        names = ", ".join(f"{r['feature']}({r['kernel_r2_post']:.4f})" for r in failing)
        raise RuntimeError(
            f"ComBat post-correction R^2 exceeds {acceptance_max_post_r2} on: {names}. "
            "Investigate kernel imbalance or non-linear kernel effects."
        )
    return new_df, list(feature_cols), audit


# ───────────────────────── Yeo-Johnson with rank fallback ─────────────────────────


def _rank_transform(values: np.ndarray) -> np.ndarray:
    """Rank values to [0, 1] (average rank for ties). NaN propagates."""
    return stats.rankdata(values, method="average", nan_policy="propagate") / len(values)


def yeo_johnson_with_fallback(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    sparse_columns: Iterable[str] = SPARSE_COLUMNS,
    skew_threshold: float = 1.0,
) -> tuple[pd.DataFrame, list[str], list[dict]]:
    """Apply Yeo-Johnson per column on ``sparse_columns``. If the post-YJ
    |skewness| exceeds ``skew_threshold``, fall back to rank-transform on
    that column.

    Columns not in ``sparse_columns`` pass through untouched. Binary 0/1
    columns (any column whose unique values are subsets of {0, 1}) are
    skipped automatically.

    Returns (new_df, feature_cols, per_column_records). Record fields:
        {feature, pre_skew, lambda, post_yj_skew, transform_used,
         post_final_skew, fallback_triggered}
    """
    _assert_pid_column(df, "yeo_johnson_with_fallback")
    _assert_no_nan(df, feature_cols, "yeo_johnson_with_fallback")

    new_df = df.copy()
    records: list[dict] = []
    cols = [c for c in sparse_columns if c in df.columns]

    for col in cols:
        values = new_df[col].to_numpy(dtype=float)
        # Binary column? Skip.
        uniq = set(np.unique(values))
        if uniq.issubset({0.0, 1.0}):
            records.append({
                "feature": col,
                "pre_skew": 0.0,
                "lambda": None,
                "post_yj_skew": 0.0,
                "transform_used": "binary_skipped",
                "post_final_skew": 0.0,
                "fallback_triggered": False,
            })
            continue

        pre_skew = float(stats.skew(values, bias=False))
        # Constant column safety net (variance_filter should have caught this).
        if np.std(values) == 0.0:
            records.append({
                "feature": col,
                "pre_skew": pre_skew,
                "lambda": None,
                "post_yj_skew": 0.0,
                "transform_used": "constant_skipped",
                "post_final_skew": 0.0,
                "fallback_triggered": False,
            })
            continue

        # Yeo-Johnson via sklearn PowerTransformer (lambda fit by MLE).
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        try:
            yj_values = pt.fit_transform(values.reshape(-1, 1)).ravel()
            lam = float(pt.lambdas_[0])
            post_yj_skew = float(stats.skew(yj_values, bias=False))
        except Exception:
            # Defensive: if YJ fails, treat as failure and fall back.
            yj_values = values
            lam = None
            post_yj_skew = pre_skew

        if abs(post_yj_skew) <= skew_threshold:
            final = yj_values
            transform_used = "yeo-johnson"
            fallback = False
        else:
            final = _rank_transform(values)
            transform_used = "rank"
            fallback = True

        post_final_skew = float(stats.skew(final, bias=False))
        new_df[col] = final
        records.append({
            "feature": col,
            "pre_skew": round(pre_skew, 4),
            "lambda": round(lam, 4) if lam is not None else None,
            # Full precision, for replaying the fit on new rows (stage 8).
            "lambda_exact": lam,
            "post_yj_skew": round(post_yj_skew, 4),
            "transform_used": transform_used,
            "post_final_skew": round(post_final_skew, 4),
            "fallback_triggered": fallback,
        })

    return new_df, list(feature_cols), records


# ───────────────────────── global z-score ─────────────────────────


def global_zscore(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[str], dict]:
    """Centre and unit-variance scale each feature column independently.

    Uses sample standard deviation (ddof=1). Constant columns (std==0) are
    left at 0 (subtract mean, do not divide). This should not happen because
    variance_filter runs first; this is a safety net.

    Returns (new_df, feature_cols, {means, stds}). The means / stds are
    keyed by feature name for reproducibility (serialisable as JSON).
    """
    _assert_pid_column(df, "global_zscore")
    _assert_no_nan(df, feature_cols, "global_zscore")

    new_df = df.copy()
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    means_exact: dict[str, float] = {}
    stds_exact: dict[str, float] = {}
    for col in feature_cols:
        v = new_df[col].to_numpy(dtype=float)
        mean = float(v.mean())
        std = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        means[col] = round(mean, 6)
        stds[col] = round(std, 6)
        means_exact[col] = mean
        stds_exact[col] = std
        centred = v - mean
        if std == 0.0:
            new_df[col] = centred
        else:
            new_df[col] = centred / std
    return new_df, list(feature_cols), {"means": means, "stds": stds,
                                        "means_exact": means_exact, "stds_exact": stds_exact}


# ───────────────────────── orchestrator ─────────────────────────


def run_matrix_prep(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    variance_threshold: float = 0.01,
    derived_r2_threshold: float = 0.95,
    yj_skew_threshold: float = 1.0,
    combat_max_post_r2: float = 0.02,
    add_derived: bool = True,
    reliable_transformed: set[str] | None = None,
) -> tuple[pd.DataFrame, list[str], MatrixPrepLog]:
    """Run the full D019 matrix-preparation pipeline.

    Input is a dataframe with a 'pid' column, optional 'kernel' column
    (required if ComBat will run), and ``feature_cols`` columns matching
    the gated feature set from stage 4.

    Output is the transformed dataframe, the final feature column list, and
    a ``MatrixPrepLog`` describing every step.

    ``reliable_transformed`` (D040): the STAGE5_TRANSFORMED features that
    passed the D033 gate on the perturbation draws. When given, an
    unreliable binarised or derived feature is left out (and logged). None
    keeps every transformed feature (D014 semantics, where canonical inputs
    are invariant by construction).
    """
    _assert_pid_column(df, "run_matrix_prep")
    log = MatrixPrepLog(
        n_patients_in=len(df),
        n_features_in=len(feature_cols),
    )

    # Step 1: D017 drops.
    df, feature_cols, dropped = apply_d017_drops(df, feature_cols)
    log.d017_dropped = list(dropped)

    # Step 2: D018 binarise.
    df, feature_cols, binarised = apply_d018_binarisation(df, feature_cols)
    log.d018_target_present = bool(binarised)
    if (binarised and reliable_transformed is not None
            and D018_BINARISE_TARGET not in reliable_transformed):
        df = df.drop(columns=[D018_BINARISE_TARGET])
        feature_cols = [c for c in feature_cols if c != D018_BINARISE_TARGET]
        log.d018_dropped_unreliable = True

    # Step 3: derived features.
    if add_derived:
        df, feature_cols, accepted, rejected = maybe_add_derived_features(
            df, feature_cols, r2_redundancy_threshold=derived_r2_threshold,
            reliable=reliable_transformed,
        )
        log.derived_accepted = accepted
        log.derived_rejected = rejected

    # Step 4: variance filter.
    df, feature_cols, dropped_var = variance_filter(
        df, feature_cols, threshold=variance_threshold,
    )
    log.variance_dropped = dropped_var

    # Step 5: ComBat on every gated non-shape PyRadiomics feature (D035).
    df, feature_cols, audit = combat_harmonise(
        df, feature_cols,
        columns_to_harmonise=harmonisable_columns(feature_cols),
        acceptance_max_post_r2=combat_max_post_r2,
    )
    log.combat_audit = audit

    # Step 6: Yeo-Johnson with rank fallback.
    df, feature_cols, yj_records = yeo_johnson_with_fallback(
        df, feature_cols, skew_threshold=yj_skew_threshold,
    )
    log.yj_per_column = yj_records

    # Step 7: global z-score.
    df, feature_cols, zinfo = global_zscore(df, feature_cols)
    log.zscore_columns = list(feature_cols)
    log.zscore_means = dict(zinfo.get("means", {}))
    log.zscore_stds = dict(zinfo.get("stds", {}))
    log.zscore_means_exact = dict(zinfo.get("means_exact", {}))
    log.zscore_stds_exact = dict(zinfo.get("stds_exact", {}))

    log.n_patients_out = len(df)
    log.n_features_out = len(feature_cols)
    return df, feature_cols, log


def production_prep_kwargs(cfg) -> dict:
    """The ``run_matrix_prep`` settings of the production stage 5 run.

    Stage 5 and stage 8 (which refits the production pipeline on the
    published cohort and on cross-validation folds) both call this, so the
    two cannot use different settings.
    ``reliable_transformed`` (D040) comes from the gate's derived_icc.csv;
    the D033 (realistic) gate requires it, the D014 gate has none.
    """
    from predict.config import gate_dir

    derived_icc_csv = gate_dir(cfg) / "derived_icc.csv"
    if derived_icc_csv.exists():
        derived = pd.read_csv(derived_icc_csv)
        reliable: set[str] | None = set(
            derived.loc[derived["passes_gate"].astype(bool), "feature"])
    elif cfg.stability.primary_gate == "realistic":
        raise SystemExit(f"{derived_icc_csv} missing; rerun stage 5b (D040)")
    else:
        reliable = None
    return {
        "variance_threshold": cfg.reduce.variance_threshold,
        "yj_skew_threshold": cfg.raw["reduce"].get("yeo_johnson_skewness_fallback", 1.0),
        "combat_max_post_r2": 0.02,
        "add_derived": True,
        "reliable_transformed": reliable,
    }


__all__ = [
    "D017_DROPPED_FEATURES",
    "D018_BINARISE_SOURCE",
    "D018_BINARISE_TARGET",
    "STAGE5_TRANSFORMED",
    "stage5_transformed_columns",
    "SPARSE_COLUMNS",
    "production_prep_kwargs",
    "harmonisable_columns",
    "MatrixPrepLog",
    "apply_d017_drops",
    "apply_d018_binarisation",
    "compute_high_density_fraction",
    "compute_vessel_burden_gini",
    "maybe_add_derived_features",
    "variance_filter",
    "combat_harmonise",
    "yeo_johnson_with_fallback",
    "global_zscore",
    "run_matrix_prep",
]
