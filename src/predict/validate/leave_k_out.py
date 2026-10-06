"""Stage 8 D030: leave-k-out CV for spatial-only k=2 phenotype robustness.

10-fold kernel-stratified CV with full per-fold refit of the production
pipeline:

  D019 prepare_matrix
    -> spatial-only PCA on the 13 SPATIAL_FEATURES_FOR_PROJECTION
    -> GMM(k=2) on the spatial PC scores
  with the production settings (``ProductionSettings``; D041: stage 6's
  random_state and GMM function, stage 5's prep settings).

Integrity (D041): the cohort must be exactly
the published cohort, every cohort patient must have a reference label,
and every planned fold must produce a row; reference class counts use the
published (full-cohort D023) focal/diffuse mapping, not the fold model's.

Each fold:

  1. Train on ~378 pids; test on ~42.
  2. Refit the entire pipeline on the training rows.
  3. Project the held-out rows onto the train-fold's spatial PCA.
  4. GMM.predict to assign each held-out pid a raw cluster label in {0, 1}.
  5. Compute ARI between predicted held-out labels and the FULL-COHORT
     reference labels (from outputs/06_reduce/cluster_labels_spatial_k2.csv)
     restricted to that fold's held-out pids. RAW LABELS, no focal/diffuse
     mapping: ARI is permutation invariant.
  6. Compute a fold-specific PASS threshold T_fold = 5th-percentile of a
     null simulation at K=10% per-patient disagreement, with class
     proportions matching the fold.

Overall PASS = median fold ARI >= median T_fold.

See decisions/D030 for the full rationale.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedKFold

from predict.validate.external_holdout import (
    FrozenPipeline,
    ProductionSettings,
    SPATIAL_FEATURES_FOR_PROJECTION,
    fit_frozen_pipeline,
    predict_holdout_phenotype,
    project_holdout_to_spatial_pca,
    transform_holdout_features,
)
from predict.validate.label_alignment import apply_mapping, determine_mapping

_log = logging.getLogger(__name__)


# Locked per D030
N_SPLITS_DEFAULT: int = 10
SEED_DEFAULT: int = 42
DISAGREEMENT_RATE_DEFAULT: float = 0.10   # K = 10% (Wrinkle 7 locked)
N_SIM_DEFAULT: int = 10_000
PERCENTILE_DEFAULT: float = 5.0           # 5th-percentile threshold


# ─────────────────────── splitting ───────────────────────


@dataclass(frozen=True)
class FoldSplit:
    fold: int
    train_pids: list[str]
    test_pids: list[str]
    train_kernel_counts: dict[str, int]
    test_kernel_counts: dict[str, int]


def kernel_stratified_kfold_split(
    features_df: pd.DataFrame,
    n_splits: int = N_SPLITS_DEFAULT,
    seed: int = SEED_DEFAULT,
    kernel_col: str = "kernel",
    pid_col: str = "pid",
) -> list[FoldSplit]:
    """Generate ``n_splits`` kernel-stratified folds.

    Returns a list of :class:`FoldSplit`. Pid lists are in the order
    StratifiedKFold yielded them; train/test are disjoint and cover all
    pids exactly once.
    """
    if pid_col not in features_df.columns:
        raise KeyError(f"missing pid column {pid_col!r}")
    if kernel_col not in features_df.columns:
        raise KeyError(f"missing kernel column {kernel_col!r}")

    pids = features_df[pid_col].astype(str).to_numpy()
    kernels = features_df[kernel_col].astype(str).to_numpy()

    # Stratify on kernel: every fold must contain both Qr36d/2 and I30f/3
    # so per-fold ComBat fits remain valid.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits: list[FoldSplit] = []
    for fold_idx, (train_ix, test_ix) in enumerate(skf.split(pids, kernels)):
        train_pids = pids[train_ix].tolist()
        test_pids = pids[test_ix].tolist()
        splits.append(FoldSplit(
            fold=fold_idx,
            train_pids=list(train_pids),
            test_pids=list(test_pids),
            train_kernel_counts=_counts(kernels[train_ix]),
            test_kernel_counts=_counts(kernels[test_ix]),
        ))
    return splits


def _counts(arr: np.ndarray) -> dict[str, int]:
    vals, counts = np.unique(arr, return_counts=True)
    return {str(v): int(c) for v, c in zip(vals, counts)}


# ─────────────────────── per-fold fit / predict ───────────────────────


def predict_fold(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    gated_feature_cols: Iterable[str],
    *,
    settings: ProductionSettings | None = None,
) -> tuple[np.ndarray, FrozenPipeline]:
    """Refit the full pipeline on ``train_features`` (production settings)
    and predict ``test_features``.

    Returns ``(raw_test_labels, frozen)``. Raw labels are in ``{0, 1}``.
    """
    frozen = fit_frozen_pipeline(train_features, gated_feature_cols, settings=settings)
    transformed = transform_holdout_features(test_features, frozen)
    scores = project_holdout_to_spatial_pca(transformed, frozen)
    raw_labels, _ = predict_holdout_phenotype(scores, frozen)
    return raw_labels, frozen


# ─────────────────────── simulation-based threshold ───────────────────────


def simulate_ari_threshold(
    reference_labels: np.ndarray,
    *,
    disagreement_rate: float = DISAGREEMENT_RATE_DEFAULT,
    n_simulations: int = N_SIM_DEFAULT,
    percentile: float = PERCENTILE_DEFAULT,
    rng_seed: int = 0,
) -> float:
    """Compute the per-fold PASS threshold T_fold (D030.4).

    For each iteration:
      - start from ``reference_labels`` (the per-fold reference vector)
      - flip ``floor(disagreement_rate * N)`` randomly chosen indices
      - compute ARI(reference, perturbed)
    Returns the ``percentile``-th percentile of the simulated ARI
    distribution (default 5th percentile -> conservative lower bound).
    """
    reference = np.asarray(reference_labels).astype(int)
    n = len(reference)
    n_flip = int(np.floor(disagreement_rate * n))
    if n_flip == 0 or n <= 1:
        return float("nan")

    rng = np.random.default_rng(rng_seed)
    aris = np.empty(n_simulations, dtype=float)
    for i in range(n_simulations):
        flip_idx = rng.choice(n, size=n_flip, replace=False)
        perturbed = reference.copy()
        perturbed[flip_idx] = 1 - perturbed[flip_idx]
        aris[i] = adjusted_rand_score(reference, perturbed)
    return float(np.percentile(aris, percentile))


# ─────────────────────── orchestrator ───────────────────────


def run_leave_k_out(
    full_cohort_features: pd.DataFrame,
    gated_feature_cols: Iterable[str],
    reference_labels: pd.Series,
    *,
    n_splits: int = N_SPLITS_DEFAULT,
    seed: int = SEED_DEFAULT,
    disagreement_rate: float = DISAGREEMENT_RATE_DEFAULT,
    n_simulations: int = N_SIM_DEFAULT,
    settings: ProductionSettings | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run the D030 leave-k-out cross-validation. Returns ``(per_fold_df,
    summary)``.

    Parameters
    ----------
    full_cohort_features  : raw features for exactly the published cohort
    gated_feature_cols    : features that passed the stage-4 ICC gate
    reference_labels      : Series indexed by pid; the full-cohort GMM
                            spatial-k=2 labels written by stage 6
                            (raw {0,1}, NO focal/diffuse mapping); must
                            cover exactly the cohort's pids

    Returns
    -------
    per_fold_df : DataFrame with one row per fold (D030 output schema)
    summary     : dict with median ARI, median T_fold, overall PASS verdict
    """
    pid_col = "pid"
    reference_labels = reference_labels.copy()
    reference_labels.index = reference_labels.index.astype(str)
    cohort_pids = full_cohort_features[pid_col].astype(str)
    if cohort_pids.duplicated().any() or reference_labels.index.duplicated().any():
        raise ValueError("duplicate pids in the cohort or the reference labels")
    if set(cohort_pids) != set(reference_labels.index):
        raise ValueError(
            "reference labels must cover exactly the cohort: "
            f"{len(set(cohort_pids) - set(reference_labels.index))} cohort pids without a label, "
            f"{len(set(reference_labels.index) - set(cohort_pids))} labels outside the cohort")
    # Published focal/diffuse mapping of the reference labels (D023 on the
    # full cohort, strict), for the reference class counts.
    ref_mapping = determine_mapping(
        full_cohort_features.set_index(cohort_pids).loc[reference_labels.index],
        reference_labels, tie_break="raise")
    focal_ids = {k for k, v in ref_mapping.items() if v == "focal"}

    splits = kernel_stratified_kfold_split(
        full_cohort_features, n_splits=n_splits, seed=seed,
    )
    rows: list[dict] = []
    for split in splits:
        train_df = full_cohort_features[
            full_cohort_features[pid_col].astype(str).isin(split.train_pids)
        ].reset_index(drop=True)
        test_df = full_cohort_features[
            full_cohort_features[pid_col].astype(str).isin(split.test_pids)
        ].reset_index(drop=True)
        test_pids = test_df[pid_col].astype(str).tolist()

        # Reference labels of this fold's held-out pids (all present: the
        # coverage check above).
        ref = reference_labels.loc[test_pids].astype(int)

        # Predict.
        raw_test_labels, frozen = predict_fold(
            train_df, test_df, gated_feature_cols, settings=settings,
        )
        pred_series = pd.Series(raw_test_labels, index=test_pids).astype(int)

        # ARI on RAW labels (D030.5).
        ari = float(adjusted_rand_score(ref.to_numpy(), pred_series.to_numpy()))

        # Per-fold simulation threshold.
        T_fold = simulate_ari_threshold(
            ref.to_numpy(),
            disagreement_rate=disagreement_rate,
            n_simulations=n_simulations,
            rng_seed=split.fold + 1,
        )

        # Reference class balance under the published mapping (the fold
        # model's label numbers can be swapped relative to the reference).
        n_focal_ref = int(ref.isin(focal_ids).sum())

        rows.append({
            "fold": split.fold,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_test_with_reference": len(ref),
            **{f"n_test_kernel_{k}": v
               for k, v in split.test_kernel_counts.items()},
            "n_test_focal_reference": n_focal_ref,
            "n_test_diffuse_reference": int(len(ref) - n_focal_ref),
            "ari": ari,
            "T_fold": T_fold,
            "pass_fold": bool(ari >= T_fold) if not np.isnan(T_fold) else False,
        })

    per_fold_df = pd.DataFrame(rows)
    if len(per_fold_df) != n_splits:
        raise RuntimeError(f"run_leave_k_out produced {len(per_fold_df)} of {n_splits} planned folds")

    median_ari = float(np.median(per_fold_df["ari"]))
    median_T = float(np.median(per_fold_df["T_fold"]))
    iqr_ari = float(
        np.percentile(per_fold_df["ari"], 75)
        - np.percentile(per_fold_df["ari"], 25)
    )
    overall_pass = bool(median_ari >= median_T)

    summary = {
        "n_folds": int(len(per_fold_df)),
        "median_ari": median_ari,
        "iqr_ari": iqr_ari,
        "median_T_fold": median_T,
        "disagreement_rate_K": float(disagreement_rate),
        "n_simulations": int(n_simulations),
        "percentile": float(PERCENTILE_DEFAULT),
        "overall_pass": overall_pass,
    }
    return per_fold_df, summary


def attach_summary_row(
    per_fold_df: pd.DataFrame, summary: dict,
) -> pd.DataFrame:
    """Append a SUMMARY row to a per-fold dataframe so the on-disk CSV is
    a single-table deliverable.
    """
    if per_fold_df.empty:
        return per_fold_df
    row = {col: "" for col in per_fold_df.columns}
    row["fold"] = "SUMMARY"
    row["ari"] = summary["median_ari"]
    row["T_fold"] = summary["median_T_fold"]
    row["pass_fold"] = summary["overall_pass"]
    return pd.concat([per_fold_df, pd.DataFrame([row])], ignore_index=True)


# ─────────────────────── interpretive helper ───────────────────────


def apply_focal_diffuse_mapping_to_predictions(
    raw_test_labels: pd.Series,
    frozen: FrozenPipeline,
) -> pd.Series:
    """Map per-fold raw GMM labels to focal/diffuse strings.

    D030.5: this mapping is for INTERPRETIVE columns only and must NOT be
    applied to the ARI input.
    """
    return apply_mapping(
        raw_test_labels,
        frozen.full_cohort_focal_diffuse_mapping,
    ).astype("string")


__all__ = [
    "N_SPLITS_DEFAULT",
    "SEED_DEFAULT",
    "DISAGREEMENT_RATE_DEFAULT",
    "N_SIM_DEFAULT",
    "PERCENTILE_DEFAULT",
    "FoldSplit",
    "kernel_stratified_kfold_split",
    "predict_fold",
    "simulate_ari_threshold",
    "run_leave_k_out",
    "attach_summary_row",
    "apply_focal_diffuse_mapping_to_predictions",
]
