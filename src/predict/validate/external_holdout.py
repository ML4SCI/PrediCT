"""Stage 8 D029: external GE-scanner holdout validation.

Applies the frozen production pipeline to the GE-manufacturer patients that
D004 excluded from the main cohort (pids 19, 28, 76, 77), and reports their
projected spatial PC scores + GMM-predicted phenotype.

Critical: the holdout is NEVER used to refit any model. D019 transforms,
spatial PCA, and GMM are all fit on the published production cohort; only
transform and predict paths touch the holdout.

ComBat is skipped (D029.1): the 13 spatial features used for the k=2
phenotype are canonical XML features, which ComBat never touches (it
harmonises PyRadiomics non-shape features only, D035), so the spatial-only
projection is ComBat-free by construction. The holdout report does not
include PyRadiomics texture summaries.

Production fidelity (D041):

* The fit uses the production settings (``ProductionSettings``, from
  ``production_settings(cfg)``): stage 5's ``production_prep_kwargs``, the
  configured PCA variance, and stage 6's random_state with stage 6's own
  GMM function (``fit_gmm``). Cross-validation folds use the same.
* Fitted transforms are kept at full precision, including the training
  values behind a rank fallback, and replaying the training rows through
  the transform path must reproduce the fitted training matrix
  (``fit_frozen_pipeline`` checks this on every fit).
* For the published cohort, ``require_production_parity`` requires the
  refit to reproduce stage 5's prepared spatial matrix and stage 6's
  spatial labels before any holdout projection.

The public entry point is :func:`run_external_holdout_validation`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from predict.discover.cluster_discovery import fit_gmm
from predict.reduce.pca import fit_pca
from predict.reduce.prepare_matrix import (
    D017_DROPPED_FEATURES,
    D018_BINARISE_SOURCE,
    D018_BINARISE_TARGET,
    SPARSE_COLUMNS,
    harmonisable_columns,
    run_matrix_prep,
)
from predict.validate.label_alignment import determine_mapping

_log = logging.getLogger(__name__)


# The 13 spatial features used for the spatial-only k=2 GMM (D021 /
# Finding 3). Mirrors SPATIAL_FEATURES_AFTER_D017 in scripts/07_discover.py.
SPATIAL_FEATURES_FOR_PROJECTION: tuple[str, ...] = (
    "lesion_count_lad", "lesion_count_rca", "lesion_count_lcx",
    "lesion_count_lm", "lesion_count_total",
    "n_calcified_arteries",
    "gini_lesion_volume",
    "dist_from_top_max", "dist_from_top_mean",
    "center_of_mass_z",
    "inter_lesion_dist_mean_lad", "inter_lesion_dist_max_lad",
    "first_to_last_dist_lad",
)

# Canonical raw context columns the holdout report always carries.
HOLDOUT_RAW_CONTEXT_COLS: tuple[str, ...] = (
    "kernel", "scanner_model",
    "agatston_total", "n_calcified_arteries", "lesion_count_total",
    "mask_voxels", "low_burden_flag",
)

# Replay of the training rows must reproduce the fitted matrix to this.
REPLAY_ATOL = 1e-9


# ─────────────────────── settings + data classes ───────────────────────


@dataclass(frozen=True)
class ProductionSettings:
    """How the production stage 5-6 spatial pipeline is fitted."""
    prep_kwargs: dict = field(default_factory=dict)
    pca_cumvar: float = 0.85
    pca_random_state: int = 42
    gmm_random_state: int = 42


def production_settings(cfg: Any) -> ProductionSettings:
    """The settings stage 5 (``production_prep_kwargs``) and stage 6
    (configured PCA variance, PRODUCTION random_state) used."""
    from predict.discover.stage6 import PRODUCTION
    from predict.reduce.prepare_matrix import production_prep_kwargs

    rs = int(PRODUCTION["random_state"])
    return ProductionSettings(
        prep_kwargs=production_prep_kwargs(cfg),
        pca_cumvar=float(cfg.reduce.pca_cumvar),
        pca_random_state=rs,
        gmm_random_state=rs,
    )


@dataclass(frozen=True)
class FrozenPipeline:
    """Fit state from the training cohort, used to transform new rows
    WITHOUT leakage. Every parameter is full precision."""
    feature_cols_post_d019: list[str]
    zscore_means: dict[str, float]
    zscore_stds: dict[str, float]
    # Per SPARSE column: {"kind": "yeo-johnson", "lambda": float} |
    # {"kind": "rank", "reference": sorted training values} |
    # {"kind": "binary_skipped" | "constant_skipped"}.
    transforms: dict[str, dict]
    yj_records: list[dict]
    spatial_feature_cols: list[str]
    # Sign-normalised components from fit_pca (n_retain, n_features).
    spatial_pca_components: np.ndarray
    spatial_pca_means: np.ndarray
    gmm: GaussianMixture
    gmm_focal_label_raw: int  # the cluster id mapped to "focal"
    full_cohort_focal_diffuse_mapping: dict
    training_pids: list[str]
    training_spatial: np.ndarray        # fitted (post-D019) spatial matrix
    training_labels_raw: np.ndarray     # GMM labels of the training rows


# ─────────────────────── fit ───────────────────────


def _filter_to_combat_compatible(
    df: pd.DataFrame, kernel_col: str = "kernel",
) -> pd.DataFrame:
    """Drop singleton-kernel patients per D019 ComBat constraint."""
    if kernel_col not in df.columns:
        return df
    counts = df[kernel_col].value_counts()
    keep_kernels = set(counts[counts >= 2].index)
    return df[df[kernel_col].isin(keep_kernels)].reset_index(drop=True)


def _fitted_transforms(df_in: pd.DataFrame, feature_cols: list[str],
                       yj_records: list[dict]) -> dict[str, dict]:
    harmonised = set(harmonisable_columns(feature_cols))
    out: dict[str, dict] = {}
    for rec in yj_records:
        col, kind = rec["feature"], rec["transform_used"]
        if col in harmonised:
            raise ValueError(f"{col} is both ComBat-harmonised and Yeo-Johnson transformed; "
                             "the replay below assumes YJ sees raw values")
        if kind == "yeo-johnson":
            out[col] = {"kind": kind, "lambda": float(rec["lambda_exact"])}
        elif kind == "rank":
            out[col] = {"kind": kind,
                        "reference": np.sort(df_in[col].to_numpy(dtype=float))}
        else:
            out[col] = {"kind": kind}
    return out


def fit_frozen_pipeline(
    full_cohort_features: pd.DataFrame,
    gated_feature_cols: Iterable[str],
    *,
    settings: ProductionSettings | None = None,
    spatial_cols: Iterable[str] = SPATIAL_FEATURES_FOR_PROJECTION,
) -> FrozenPipeline:
    """Fit the D019 + spatial PCA + GMM chain on the training cohort with
    the production settings, then check that replaying the training rows
    through :func:`transform_holdout_features` reproduces the fit."""
    settings = settings or ProductionSettings()
    df = _filter_to_combat_compatible(full_cohort_features.copy())
    feature_cols = list(gated_feature_cols)

    _log.info("D019 on training cohort: N=%d, %d features", len(df), len(feature_cols))
    prep_df, prep_feat_cols, log = run_matrix_prep(df, feature_cols, **settings.prep_kwargs)

    spatial_kept = [c for c in spatial_cols if c in prep_feat_cols]
    if len(spatial_kept) < 2:
        raise ValueError(
            f"fit_frozen_pipeline: only {len(spatial_kept)} of "
            f"{len(SPATIAL_FEATURES_FOR_PROJECTION)} spatial features "
            f"survived D019 (need >= 2)."
        )

    spatial_pca = fit_pca(prep_df, spatial_kept, cumvar_threshold=settings.pca_cumvar,
                          random_state=settings.pca_random_state)
    training_spatial = prep_df[spatial_kept].to_numpy(dtype=float)
    spatial_means = training_spatial.mean(axis=0)
    spatial_scores = spatial_pca.scores  # (N, n_retain)
    spatial_components_retained = spatial_pca.components[:spatial_pca.n_retain]

    gmm = fit_gmm(spatial_scores, 2, random_state=settings.gmm_random_state)
    gmm_labels_raw = gmm.predict(spatial_scores)

    pids = prep_df["pid"].astype(str).tolist()
    labels_series = pd.Series(gmm_labels_raw, index=pids)
    raw_feats_for_mapping = full_cohort_features.set_index(
        full_cohort_features["pid"].astype(str)
    ).reindex(labels_series.index)
    mapping = determine_mapping(raw_feats_for_mapping, labels_series)
    focal_raw_label = next(k for k, v in mapping.items() if v == "focal")

    frozen = FrozenPipeline(
        feature_cols_post_d019=list(prep_feat_cols),
        zscore_means=dict(log.zscore_means_exact),
        zscore_stds=dict(log.zscore_stds_exact),
        transforms=_fitted_transforms(df, list(prep_feat_cols), log.yj_per_column),
        yj_records=list(log.yj_per_column),
        spatial_feature_cols=spatial_kept,
        spatial_pca_components=spatial_components_retained,
        spatial_pca_means=spatial_means,
        gmm=gmm,
        gmm_focal_label_raw=int(focal_raw_label),
        full_cohort_focal_diffuse_mapping=mapping,
        training_pids=pids,
        training_spatial=training_spatial,
        training_labels_raw=np.asarray(gmm_labels_raw, dtype=int),
    )
    _require_replay(frozen, df)
    return frozen


def _require_replay(frozen: FrozenPipeline, training_raw: pd.DataFrame) -> None:
    """The transform path applied to the training rows must reproduce the
    fitted training matrix, and the GMM their labels."""
    replay = transform_holdout_features(training_raw, frozen)
    replay = replay.set_index(replay["pid"].astype(str)).loc[frozen.training_pids]
    got = replay[frozen.spatial_feature_cols].to_numpy(dtype=float)
    err = float(np.max(np.abs(got - frozen.training_spatial))) if got.size else 0.0
    if not err <= REPLAY_ATOL:
        raise RuntimeError(f"replaying the training rows differs from the fit by {err:.3g} "
                           f"(> {REPLAY_ATOL}); the held-out transform is not the fitted one")
    labels, _ = predict_holdout_phenotype(project_holdout_to_spatial_pca(replay, frozen), frozen)
    if not np.array_equal(labels, frozen.training_labels_raw):
        raise RuntimeError("replayed training rows get different GMM labels than the fit")


def require_production_parity(frozen: FrozenPipeline, cohort_dir: Path) -> dict:
    """The refit on the published cohort must reproduce stage 5's prepared
    spatial matrix and stage 6's spatial labels (``cohort_dir`` is the
    verified ``06_reduce`` folder). Raises SystemExit otherwise."""
    cohort_dir = Path(cohort_dir)
    prep = pd.read_csv(cohort_dir / "prepared_matrix.csv", dtype={"pid": str})
    labels = pd.read_csv(cohort_dir / "cluster_labels_spatial_k2.csv", dtype={"pid": str})
    if prep["pid"].tolist() != frozen.training_pids:
        raise SystemExit("hard stop: stage 8 refit cohort differs from stage 5's published cohort "
                         f"({len(frozen.training_pids)} vs {len(prep)} patients or another order)")
    missing = [c for c in frozen.spatial_feature_cols if c not in prep.columns]
    if missing:
        raise SystemExit(f"hard stop: spatial features {missing} not in stage 5's prepared matrix")
    published = prep[frozen.spatial_feature_cols].to_numpy(dtype=float)
    err = float(np.max(np.abs(published - frozen.training_spatial)))
    if not err <= 1e-9:
        raise SystemExit(f"hard stop: stage 8 refit transforms differ from stage 5's prepared "
                         f"matrix by {err:.3g}")
    pub_labels = labels.set_index("pid").loc[frozen.training_pids, "spatial_only_gmm_k2"].to_numpy(int)
    n_diff = int((pub_labels != frozen.training_labels_raw).sum())
    if n_diff:
        raise SystemExit(f"hard stop: stage 8 refit GMM labels differ from stage 6's published "
                         f"spatial labels for {n_diff} of {len(pub_labels)} patients")
    return {"n_patients": len(pub_labels), "max_abs_transform_diff": err, "label_mismatches": 0}


# ─────────────────────── transform ───────────────────────


def _rank_map(x: np.ndarray, reference_sorted: np.ndarray) -> np.ndarray:
    """The training rank transform applied to new values: average rank among
    the training values divided by N, so a value equal to a training value
    gets exactly that value's training rank (scipy rankdata 'average')."""
    n = len(reference_sorted)
    lo = np.searchsorted(reference_sorted, x, side="left")
    hi = np.searchsorted(reference_sorted, x, side="right")
    return (lo + ((hi - lo) + 1) / 2.0) / n


def transform_holdout_features(
    holdout_raw: pd.DataFrame,
    frozen: FrozenPipeline,
) -> pd.DataFrame:
    """Apply the captured D019 transforms to new RAW rows.

    Steps (ComBat skipped per D029.1):
      1. Drop D017 columns.
      2. Apply D018 binarisation if source present.
      3. SPARSE columns: Yeo-Johnson with the fitted lambda, or the
         training rank mapping where the fit fell back to ranks (before
         2026-10-05 rank columns were left raw and then z-scored with
         rank-scale statistics). Binary and constant columns pass through.
      4. Per-column z-score with the fitted means + stds.

    Returns the rows restricted to ``frozen.spatial_feature_cols`` plus
    'pid', ready for spatial-PCA projection.
    """
    df = holdout_raw.copy()

    for col in D017_DROPPED_FEATURES:
        if col in df.columns:
            df = df.drop(columns=col)

    if (D018_BINARISE_SOURCE in df.columns
            and D018_BINARISE_TARGET not in df.columns):
        df[D018_BINARISE_TARGET] = (df[D018_BINARISE_SOURCE] > 0).astype(int)

    for col in SPARSE_COLUMNS:
        if col not in df.columns or col not in frozen.transforms:
            continue
        spec = frozen.transforms[col]
        v = df[col].to_numpy(dtype=float)
        if spec["kind"] == "yeo-johnson":
            df[col] = _yeo_johnson_apply(v, spec["lambda"])
        elif spec["kind"] == "rank":
            df[col] = _rank_map(v, spec["reference"])
        elif spec["kind"] not in ("binary_skipped", "constant_skipped"):
            raise ValueError(f"unknown fitted transform {spec['kind']!r} for {col}")

    for col in frozen.feature_cols_post_d019:
        if col not in df.columns:
            continue
        mean = frozen.zscore_means[col]
        std = frozen.zscore_stds[col]
        v = df[col].to_numpy(dtype=float)
        df[col] = v - mean if std == 0.0 else (v - mean) / std

    missing = [c for c in frozen.spatial_feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"new rows lack spatial features {missing}")
    out = df[["pid", *frozen.spatial_feature_cols]].reset_index(drop=True)
    if not np.isfinite(out[frozen.spatial_feature_cols].to_numpy(dtype=float)).all():
        raise ValueError("non-finite transformed spatial features (missing raw values?)")
    return out


def _yeo_johnson_apply(x: np.ndarray, lam: float) -> np.ndarray:
    """Forward Yeo-Johnson transform with a fixed lambda (no fitting).

    Mirrors sklearn's PowerTransformer(yeo-johnson, standardize=False)
    transform branch.
    """
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    if abs(lam) < 1e-9:
        out[pos] = np.log1p(x[pos])
    else:
        out[pos] = (np.power(x[pos] + 1.0, lam) - 1.0) / lam
    if abs(lam - 2.0) < 1e-9:
        out[neg] = -np.log1p(-x[neg])
    else:
        out[neg] = -(np.power(-x[neg] + 1.0, 2.0 - lam) - 1.0) / (2.0 - lam)
    return out


def project_holdout_to_spatial_pca(
    holdout_transformed: pd.DataFrame,
    frozen: FrozenPipeline,
) -> np.ndarray:
    """Project pre-transformed rows onto the frozen spatial PCA.

    Returns an array of shape (n_rows, n_pcs).
    """
    X = holdout_transformed[frozen.spatial_feature_cols].to_numpy(dtype=float)
    centred = X - frozen.spatial_pca_means
    return centred @ frozen.spatial_pca_components.T


# ─────────────────────── predict + report ───────────────────────


def predict_holdout_phenotype(
    spatial_scores: np.ndarray,
    frozen: FrozenPipeline,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(raw_labels, distance_to_centroids)``.

    ``raw_labels`` are the unmapped GMM cluster ids in ``{0, 1}``.
    ``distance_to_centroids`` is shape ``(n, 2)`` giving Euclidean
    distance to each GMM mean (column order = GMM component order).
    """
    raw_labels = frozen.gmm.predict(spatial_scores)
    means = frozen.gmm.means_  # (2, n_pcs)
    diffs = spatial_scores[:, None, :] - means[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    return raw_labels.astype(int), dists


def build_holdout_report(
    holdout_raw: pd.DataFrame,
    spatial_scores: np.ndarray,
    raw_labels: np.ndarray,
    centroid_dists: np.ndarray,
    frozen: FrozenPipeline,
    xml_roundtrip_pass_by_pid: dict[str, bool] | None = None,
) -> pd.DataFrame:
    """Assemble the per-pid report described in D029.

    Output columns:
      pid, kernel, scanner_model, mask_voxels, low_burden_flag,
      agatston_total, n_calcified_arteries, lesion_count_total,
      spatial_pc1, spatial_pc2, ..., spatial_pcK,
      predicted_phenotype_raw, predicted_phenotype,
      distance_to_focal_centroid, distance_to_diffuse_centroid,
      xml_roundtrip_max_pass.
    """
    n_pcs = spatial_scores.shape[1]
    rows: list[dict] = []
    focal_raw = frozen.gmm_focal_label_raw
    diffuse_raw = 1 - focal_raw if focal_raw in (0, 1) else None
    if diffuse_raw is None:
        raise ValueError("focal_raw must be 0 or 1")

    label_to_str = {focal_raw: "focal", diffuse_raw: "diffuse"}

    for i, (_, row) in enumerate(holdout_raw.iterrows()):
        pid = str(row["pid"])
        rec: dict = {"pid": pid}
        for col in HOLDOUT_RAW_CONTEXT_COLS:
            rec[col] = row.get(col, None)
        for k in range(n_pcs):
            rec[f"spatial_pc{k + 1}"] = float(spatial_scores[i, k])
        rec["predicted_phenotype_raw"] = int(raw_labels[i])
        rec["predicted_phenotype"] = label_to_str[int(raw_labels[i])]
        rec["distance_to_focal_centroid"] = float(centroid_dists[i, focal_raw])
        rec["distance_to_diffuse_centroid"] = float(centroid_dists[i, diffuse_raw])
        if xml_roundtrip_pass_by_pid is not None:
            rec["xml_roundtrip_max_pass"] = bool(
                xml_roundtrip_pass_by_pid.get(pid, False)
            )
        else:
            rec["xml_roundtrip_max_pass"] = None
        rows.append(rec)
    return pd.DataFrame(rows)


def holdout_report(
    holdout_features: pd.DataFrame,
    frozen: FrozenPipeline,
    xml_roundtrip_pass_by_pid: dict[str, bool] | None = None,
) -> pd.DataFrame:
    """Transform, project, predict and report the holdout rows."""
    transformed = transform_holdout_features(holdout_features, frozen)
    scores = project_holdout_to_spatial_pca(transformed, frozen)
    raw_labels, dists = predict_holdout_phenotype(scores, frozen)
    return build_holdout_report(holdout_features, scores, raw_labels, dists, frozen,
                                xml_roundtrip_pass_by_pid=xml_roundtrip_pass_by_pid)


# ─────────────────────── public entry point ───────────────────────


def run_external_holdout_validation(
    full_cohort_features: pd.DataFrame,
    holdout_features: pd.DataFrame,
    gated_feature_cols: Iterable[str],
    *,
    settings: ProductionSettings | None = None,
    xml_roundtrip_pass_by_pid: dict[str, bool] | None = None,
) -> tuple[pd.DataFrame, FrozenPipeline]:
    """Fit on the cohort, report the holdout. Returns ``(report, frozen)``.

    Stage 8 itself fits, checks ``require_production_parity`` and then calls
    :func:`holdout_report`; this wrapper skips the parity check (no
    published files) and is used by tests.
    """
    frozen = fit_frozen_pipeline(full_cohort_features, gated_feature_cols, settings=settings)
    report = holdout_report(holdout_features, frozen, xml_roundtrip_pass_by_pid)
    return report, frozen


__all__ = [
    "SPATIAL_FEATURES_FOR_PROJECTION",
    "HOLDOUT_RAW_CONTEXT_COLS",
    "ProductionSettings",
    "production_settings",
    "FrozenPipeline",
    "fit_frozen_pipeline",
    "require_production_parity",
    "transform_holdout_features",
    "project_holdout_to_spatial_pca",
    "predict_holdout_phenotype",
    "build_holdout_report",
    "holdout_report",
    "run_external_holdout_validation",
]
