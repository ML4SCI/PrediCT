#!/usr/bin/env python
"""Stage 6 orchestrator (D021 cluster discovery, validity, sensitivity).

Reads the seam files written by ``scripts/06_reduce.py`` and runs the full
discovery + validity pipeline, keeping the reduce stage focused on
dimensionality reduction.

Consumes from ``--cohort-dir`` (typically ``outputs/06_reduce/`` or a
stratified subdir):
  * pca_scores.npy + pca_scores_pid_order.csv (preferred, byte-exact seam)
    or pca_scores.csv (lossy fallback; CSV float roundtrip drifts Hopkins
    by ~0.03)
  * prepared_matrix.csv - post-D019 features (used for spatial-only PCA)
  * cohort_metadata.csv - RAW agatston_total, kernel, low_burden_flag,
                          category (needed for burden residualisation and
                          forced-k crosstabs at raw scale)

Pipeline:
  1. Hopkins clusterability on X_full
  2. Gap statistic on 3 algorithms x 3 feature spaces (full,
     burden-residualised via log(agatston_total + 1), spatial-only)
  3. Monti consensus clustering at gap-selected k (from full space) for each algorithm
  4. Forced k=3 characterisation (descriptive only) + crosstabs
  5. Validity: kernel chi-square + Hennig clusterboot at forced k
  6. Secondary finding probe: Hennig on spatial-only x GMM x k=2
  7. ARI across algorithms at forced k

Publication: results are written to
``<cohort-dir>.partial/`` together with copies of the stage 5 files listed in
``COMPLETE_stage5.json``, plus ``COMPLETE_stage6.json`` (input hashes, stage 6
code/config fingerprint, settings, output hashes). Only a complete,
production-setting run replaces the cohort folder (atomic swap that keeps the
other cohorts' sub-folders). Stages 7 and 8 refuse a cohort without a
matching record (``predict.discover.stage6.require_stage6_complete``).

A run with any non-production setting (fewer bootstraps, another k range,
forced k or seed) is a smoke run: it writes to
``outputs/07_discover_smoke/<cohort>/``, marked PARTIAL_SMOKE_DO_NOT_USE,
never to the cohort folder.

Kernel policy (D021.1): the cluster x kernel chi-square is assessed per
algorithm; a failure is followed by the burden-adjusted test. "confounded"
(the cluster still predicts the kernel given log1p(Agatston)) stops the stage
(exit 1, nothing published); "explained_by_burden" is recorded in the
validity table and the completion record, and must be reported downstream.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from predict.completion import file_sha, files_manifest, write_marker
from predict.provenance import code_revision
from predict.config import Config, default_n_jobs, load_config
from predict.discover.stage6 import (
    PRODUCTION as STAGE6_PRODUCTION,
    STAGE5_MARKER, STAGE6_MARKER, require_stage5_complete, stage6_fingerprint,
)
from predict.publish import fresh_partial, publish_keeping_subdirs
from predict.discover.cluster_discovery import (
    burden_residualise,
    fit_cluster,
    gap_statistic,
    monti_consensus,
)
from predict.discover.clusterability import assess_clusterability
from predict.discover.validity import (
    ari,
    assess_kernel_confounding,
    hennig_clusterboot,
)
from predict.reduce.pca import fit_pca


# Spatial-distribution feature subset for finding-3 (the focal-vs-diffuse
# k=2 partition that replicated across all three production cohorts under
# spatial-only x GMM). Same list as the pre-split 06_reduce.py.
SPATIAL_FEATURES_AFTER_D017: tuple[str, ...] = (
    "lesion_count_lad", "lesion_count_rca", "lesion_count_lcx", "lesion_count_lm",
    "lesion_count_total",
    "n_calcified_arteries",
    "gini_lesion_volume",
    "dist_from_top_max", "dist_from_top_mean",
    "center_of_mass_z",
    "inter_lesion_dist_mean_lad", "inter_lesion_dist_max_lad",
    "first_to_last_dist_lad",
)

ALGORITHMS: tuple[str, ...] = ("kmeans", "ward", "gmm")

# Production settings (D021), defined in predict.discover.stage6 (stage 8
# refits with them). Any other value makes the run a smoke run that can
# never publish to the cohort folder.
PRODUCTION: dict = dict(STAGE6_PRODUCTION)


# ─────────────────────── reproducibility breadcrumbs ───────────────────────


def _file_sha(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def build_run_header(
    repo_root: Path, cfg: Config, args: argparse.Namespace, cohort_dir: Path,
) -> dict:
    info: dict = {
        "stage": "discover",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": code_revision(repo_root),
        "config_yaml_sha": _file_sha(repo_root / "configs" / "default.yaml"),
        "pca_scores_npy_sha": _file_sha(cohort_dir / "pca_scores.npy"),
        "pca_scores_csv_sha": _file_sha(cohort_dir / "pca_scores.csv"),
        "prepared_matrix_sha": _file_sha(cohort_dir / "prepared_matrix.csv"),
        "cohort_metadata_sha": _file_sha(cohort_dir / "cohort_metadata.csv"),
        "python_version": sys.version.split()[0],
        "args": vars(args),
    }
    for mod in ("numpy", "pandas", "scipy", "sklearn"):
        try:
            m = __import__(mod)
            info[f"{mod}_version"] = getattr(m, "__version__", "unknown")
        except Exception:
            info[f"{mod}_version"] = "n/a"
    return info


# ─────────────────────── helpers ───────────────────────


def _save_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _load_seam(cohort_dir: Path) -> tuple[pd.DataFrame, np.ndarray, list[str], pd.DataFrame]:
    """Load the 06_reduce.py seam files.

    Returns
    -------
    prep_df       : post-D019 features matrix (pid + metadata + features)
    pca_scores    : N x n_retain PC matrix in pid_order
    pid_order     : list of pids in the order pca_scores rows go in
    cohort_meta   : raw kernel / agatston_total / low_burden_flag / category
    """
    pca_npy = cohort_dir / "pca_scores.npy"
    pca_pid_csv = cohort_dir / "pca_scores_pid_order.csv"
    pca_csv = cohort_dir / "pca_scores.csv"
    prep_csv = cohort_dir / "prepared_matrix.csv"
    meta_csv = cohort_dir / "cohort_metadata.csv"

    for p in (prep_csv, meta_csv):
        if not p.exists():
            raise FileNotFoundError(
                f"missing seam file {p.name} in {cohort_dir}; "
                f"run scripts/06_reduce.py first"
            )

    # Prefer the byte-exact NPY seam. The CSV path is a lossy fallback
    # for cohort directories produced before the NPY seam was added.
    if pca_npy.exists() and pca_pid_csv.exists():
        pca_scores = np.load(pca_npy)
        pid_order = pd.read_csv(pca_pid_csv, dtype={"pid": str})["pid"].tolist()
        if len(pid_order) != pca_scores.shape[0]:
            raise ValueError(
                f"pca_scores.npy ({pca_scores.shape[0]} rows) and "
                f"pca_scores_pid_order.csv ({len(pid_order)} rows) disagree"
            )
    elif pca_csv.exists():
        pca_df = pd.read_csv(pca_csv, dtype={"pid": str})
        pid_order = pca_df["pid"].tolist()
        pc_cols = [c for c in pca_df.columns if c != "pid"]
        pca_scores = pca_df[pc_cols].to_numpy(dtype=float)
    else:
        raise FileNotFoundError(
            f"missing pca_scores seam in {cohort_dir}; expected "
            f"pca_scores.npy + pca_scores_pid_order.csv (preferred) "
            f"or pca_scores.csv (fallback). Run scripts/06_reduce.py first."
        )

    prep_df = pd.read_csv(prep_csv, dtype={"pid": str})
    cohort_meta = pd.read_csv(meta_csv, dtype={"pid": str})

    return prep_df, pca_scores, pid_order, cohort_meta


def _gap_for_algorithm_and_space(
    X: np.ndarray, algorithm: str, k_range: tuple[int, ...],
    n_bootstrap: int, random_state: int, label: str,
    log: logging.Logger, n_jobs: int = 1,
) -> dict:
    t0 = time.perf_counter()
    result = gap_statistic(
        X, algorithm=algorithm, k_range=k_range,
        n_bootstrap=n_bootstrap, random_state=random_state,
        n_jobs=n_jobs,
    )
    log.info(
        "gap[%s][%s] selected k=%d (gap=%.3f, sk=%.3f) in %.1fs",
        label, algorithm, result.selected_k,
        float(result.gap_values[k_range.index(result.selected_k)]),
        float(result.sk_values[k_range.index(result.selected_k)]),
        time.perf_counter() - t0,
    )
    return {
        "feature_space": label,
        "algorithm": algorithm,
        "k_range": list(result.k_range),
        "gap_values": result.gap_values.tolist(),
        "sk_values": result.sk_values.tolist(),
        "log_Wk_observed": result.log_Wk_observed.tolist(),
        "log_Wk_ref_mean": result.log_Wk_ref_mean.tolist(),
        "log_Wk_ref_std": result.log_Wk_ref_std.tolist(),
        "selected_k": int(result.selected_k),
        "n_bootstrap": int(result.n_bootstrap),
        "random_state": int(result.random_state),
    }


# ─────────────────────── main ───────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--cohort-dir", type=Path, required=True,
                        help="Directory written by scripts/06_reduce.py "
                             "(e.g., outputs/06_reduce/ or "
                             "outputs/06_reduce/stratified_Qr36d_2/).")
    parser.add_argument("--k-range", type=int, nargs="+",
                        default=list(PRODUCTION["k_range"]),
                        help="Gap-statistic k values (D021 extended to 12 "
                             "after smoke runs selected the boundary k=8).")
    parser.add_argument("--gap-bootstraps", type=int, default=PRODUCTION["gap_bootstraps"],
                        help="D021 default; any other value is a smoke run.")
    parser.add_argument("--consensus-subsamples", type=int,
                        default=PRODUCTION["consensus_subsamples"])
    parser.add_argument("--hennig-bootstraps", type=int, default=PRODUCTION["hennig_bootstraps"])
    parser.add_argument("--random-state", type=int, default=PRODUCTION["random_state"])
    parser.add_argument("--forced-k", type=int, default=PRODUCTION["forced_k"])
    parser.add_argument("--n-jobs", type=int, default=default_n_jobs(),
                        help="joblib parallelism for gap statistic, Monti "
                             "consensus, and Hennig clusterboot. Defaults to "
                             "the scheduler allocation (PBS_NCPUS / "
                             "PBS_NODEFILE).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log = logging.getLogger("discover")

    cfg = load_config(args.config)
    cohort_dir = Path(args.cohort_dir)
    if not cohort_dir.exists():
        raise SystemExit(f"cohort directory does not exist: {cohort_dir}")
    repo_root = cfg.paths.outputs.parent

    # Stage 5 must have finished cleanly; its listed files are verified.
    stage5 = require_stage5_complete(cohort_dir)

    settings = {k: getattr(args, k) for k in PRODUCTION}
    smoke = settings != PRODUCTION
    if smoke:
        label = str(cohort_dir.resolve().relative_to(cfg.paths.outputs.resolve())).replace(
            "\\", "/").replace("/", "__")
        out_dir = cfg.paths.outputs / "07_discover_smoke" / label
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        (out_dir / "PARTIAL_SMOKE_DO_NOT_USE").write_text(
            f"smoke run, settings {settings}\n", encoding="utf-8")
        log.warning("smoke run (non-production settings): writing to %s, never to %s",
                    out_dir, cohort_dir)
    else:
        out_dir = fresh_partial(cohort_dir)
        for name in [*stage5["files"], STAGE5_MARKER]:
            shutil.copy2(cohort_dir / name, out_dir / name)

    # Run header
    header = build_run_header(repo_root, cfg, args, cohort_dir)
    _save_json(out_dir / "run_header_discover.json", header)
    log.info("run_header: git=%s, pca_scores_npy_sha=%s",
             header["git_commit"], header["pca_scores_npy_sha"])

    # ── Load seam files ───────────────────────────────────────────
    prep_df, X_full, pid_order, cohort_meta = _load_seam(cohort_dir)
    n_patients, n_retain = X_full.shape
    log.info("loaded seam: N=%d, n_retain=%d PCs, prep_df shape=%s, "
             "metadata cols=%s",
             n_patients, n_retain, prep_df.shape, cohort_meta.shape)

    # Diagnostic: hash X_full bytes so the comparison vs 06_reduce.py's
    # in-memory hash is direct.
    X_full_sha = hashlib.sha256(
        np.ascontiguousarray(X_full, dtype=np.float64).tobytes()
    ).hexdigest()[:16]
    log.info("DIAG: X_full float64-bytes sha256[:16] = %s "
             "(should match the DIAG line in 06_reduce.py if NPY seam is exact)",
             X_full_sha)

    # Align metadata to pid_order (pca_scores order).
    cohort_meta = (cohort_meta.set_index("pid")
                   .loc[pid_order]
                   .reset_index())

    # ── 1. D021 Hopkins ───────────────────────────────────────────
    log.info("D021 Hopkins clusterability check...")
    hopkins_result = assess_clusterability(
        X_full,
        sample_frac=cfg.raw["hopkins"]["sample_frac"],
        threshold=cfg.raw["hopkins"]["cluster_tendency_threshold"],
        ambiguous_band=tuple(cfg.raw["hopkins"]["ambiguous_band"]),
        random_state=args.random_state,
    )
    _save_json(out_dir / "hopkins.json", hopkins_result.to_dict())
    log.info("Hopkins H=%.3f (%s)", hopkins_result.H, hopkins_result.verdict)

    # ── 2. D021 three-run gap statistic ───────────────────────────
    log.info("D021 gap statistic on 3 algorithms x 3 feature spaces...")

    # Feature space B: burden-residualised PC scores (raw agatston_total).
    # Required: a missing input must stop the stage, not skip the analysis.
    if "agatston_total" not in cohort_meta.columns:
        raise SystemExit("agatston_total missing from cohort_metadata.csv; the "
                         "burden-residualised analysis is required (D021)")
    burden = cohort_meta["agatston_total"].to_numpy(dtype=float)
    X_burden = burden_residualise(X_full, burden, log_transform=True)

    # Feature space C: spatial-only features (re-PCA'd from prepared_matrix).
    spatial_cols = [c for c in SPATIAL_FEATURES_AFTER_D017 if c in prep_df.columns]
    missing_spatial = [c for c in SPATIAL_FEATURES_AFTER_D017 if c not in spatial_cols]
    if missing_spatial:
        log.warning("spatial features not in prepared_matrix (recorded): %s", missing_spatial)
    if len(spatial_cols) < 2:
        raise SystemExit(f"only {len(spatial_cols)} spatial features in prepared_matrix; "
                         "the spatial-only analysis is required (D021)")
    spatial_pca = fit_pca(prep_df, spatial_cols,
                          cumvar_threshold=cfg.reduce.pca_cumvar,
                          random_state=args.random_state)
    # Align spatial PC scores to pid_order (fit_pca sorts on pid order
    # from prep_df, which is the same as 06_reduce.py wrote it; assert).
    if spatial_pca.pid_order != pid_order:
        spatial_score_df = pd.DataFrame(
            spatial_pca.scores, index=spatial_pca.pid_order,
        ).loc[pid_order]
        X_spatial = spatial_score_df.to_numpy(dtype=float)
    else:
        X_spatial = spatial_pca.scores
    log.info("spatial-only feature space: %d features -> %d PCs",
             len(spatial_cols), X_spatial.shape[1])

    k_range_tuple = tuple(int(k) for k in args.k_range)
    gap_records: list[dict] = []
    for label, X in (("full", X_full),
                     ("burden_residualised", X_burden),
                     ("spatial_only", X_spatial)):
        if X is None:
            continue
        for algo in ALGORITHMS:
            rec = _gap_for_algorithm_and_space(
                X, algo, k_range_tuple, args.gap_bootstraps,
                args.random_state, label, log, n_jobs=args.n_jobs,
            )
            gap_records.append(rec)

    _save_json(out_dir / "gap_statistic.json", gap_records)
    pd.DataFrame([
        {
            "feature_space": r["feature_space"],
            "algorithm": r["algorithm"],
            "selected_k": r["selected_k"],
            **{f"gap_k{r['k_range'][i]}": r["gap_values"][i]
               for i in range(len(r["k_range"]))},
        }
        for r in gap_records
    ]).to_csv(out_dir / "gap_statistic_summary.csv", index=False)

    # ── 3. Monti consensus clustering ─────────────────────────────
    log.info("D021 Monti consensus clustering...")
    consensus_records: list[dict] = []
    consensus_matrices: dict[str, np.ndarray] = {}
    for algo in ALGORITHMS:
        full_rec = next(r for r in gap_records
                        if r["feature_space"] == "full" and r["algorithm"] == algo)
        k = int(full_rec["selected_k"])
        cons = monti_consensus(
            X_full, k=k, algorithm=algo,
            n_subsamples=args.consensus_subsamples,
            subsample_frac=cfg.raw["consensus"]["subsample_frac"],
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )
        consensus_matrices[f"{algo}_k{k}"] = cons.consensus_matrix
        consensus_records.append({
            "algorithm": algo,
            "k": k,
            "pac_score": cons.pac_score,
            "n_subsamples": cons.n_subsamples,
            "subsample_frac": cons.subsample_frac,
        })
    _save_json(out_dir / "consensus_summary.json", consensus_records)
    np.savez_compressed(out_dir / "consensus_matrices.npz", **consensus_matrices)

    # ── 4. Forced k=3 characterisation ────────────────────────────
    log.info("D021 forced k=%d characterisation...", args.forced_k)
    forced_records: list[dict] = []
    labels_per_algo: dict[str, np.ndarray] = {}
    for algo in ALGORITHMS:
        labels = fit_cluster(X_full, k=args.forced_k, algorithm=algo,
                             random_state=args.random_state)
        labels_per_algo[algo] = labels
        forced_records.append({
            "algorithm": algo,
            "k": args.forced_k,
            "cluster_sizes": np.bincount(labels).tolist(),
        })
    _save_json(out_dir / "forced_k_characterisation.json", forced_records)

    forced_labels_df = pd.DataFrame({"pid": pid_order})
    for algo, labels in labels_per_algo.items():
        forced_labels_df[f"forced_k{args.forced_k}_{algo}"] = labels
    forced_labels_df.to_csv(out_dir / "cluster_labels_forced.csv", index=False)

    # Crosstabs against burden tertile, kernel, low_burden_flag, category.
    crosstab_rows: list[dict] = []
    burden_tertile = (pd.qcut(cohort_meta["agatston_total"], q=3,
                              labels=["low", "mid", "high"],
                              duplicates="drop")
                      if "agatston_total" in cohort_meta.columns else None)
    for algo, labels in labels_per_algo.items():
        for covar_name, covar in (
            ("burden_tertile", burden_tertile),
            ("kernel", cohort_meta.get("kernel")),
            ("low_burden_flag", cohort_meta.get("low_burden_flag")),
            ("category", cohort_meta.get("category")),
        ):
            if covar is None:
                continue
            ct = pd.crosstab(labels, covar)
            for cluster_id in ct.index:
                for covar_value in ct.columns:
                    crosstab_rows.append({
                        "algorithm": algo,
                        "k": args.forced_k,
                        "cluster_id": int(cluster_id),
                        "covariate": covar_name,
                        "covariate_value": str(covar_value),
                        "count": int(ct.loc[cluster_id, covar_value]),
                    })
    pd.DataFrame(crosstab_rows).to_csv(
        out_dir / "forced_k_crosstabs.csv", index=False,
    )

    # ── 5. Validity checks ────────────────────────────────────────
    log.info("D021 validity checks...")
    validity_records: list[dict] = []

    # Kernel chi-square per algorithm, D021.1 policy.
    if "kernel" not in cohort_meta.columns:
        raise SystemExit("kernel missing from cohort_metadata.csv; the D021 kernel check is required")
    kernels = cohort_meta["kernel"].astype(str).to_numpy()
    kernel_status: dict[str, str] = {}
    for algo, labels in labels_per_algo.items():
        if len(set(kernels)) < 2:
            kernel_status[algo] = "not_applicable_single_kernel"
            continue
        kc = assess_kernel_confounding(labels, kernels, burden)
        kernel_status[algo] = kc["status"]
        validity_records.append({
            "test": "kernel_chi_square",
            "algorithm": algo,
            "k": args.forced_k,
            "chi2": kc["chi2"],
            "pval": kc["pval"],
            "dof": kc["dof"],
            "passes": kc["passes"],
            "kernel_status": kc["status"],
            "p_cluster_given_burden": kc.get("p_cluster_given_burden"),
            "p_cluster_unadjusted_lr": kc.get("p_cluster_unadjusted"),
        })
        if kc["status"] != "pass":
            log.warning("kernel chi-square FAIL on %s @ k=%d (p=%.3g); burden-adjusted "
                        "p=%.3g -> %s (D021.1)", algo, args.forced_k, kc["pval"],
                        kc["p_cluster_given_burden"], kc["status"])

    # Hennig bootstrap stability per algorithm at forced_k.
    for algo, labels in labels_per_algo.items():
        hen = hennig_clusterboot(
            X_full, labels, k=args.forced_k, algorithm=algo,
            n_bootstrap=args.hennig_bootstraps,
            threshold=0.75, random_state=args.random_state,
            n_jobs=args.n_jobs,
        )
        for cluster_id, median, mean_ in zip(
            hen.cluster_ids, hen.jaccard_median, hen.jaccard_mean,
        ):
            validity_records.append({
                "test": "hennig_clusterboot",
                "algorithm": algo,
                "k": args.forced_k,
                "feature_space": "full",
                "cluster_id": int(cluster_id),
                "jaccard_median": float(median),
                "jaccard_mean": float(mean_),
                "stable": bool(median >= 0.75),
            })

    # Secondary finding probe: Hennig on spatial-only x GMM x k=2.
    spatial_gmm_k2_labels = fit_cluster(
        X_spatial, k=2, algorithm="gmm",
        random_state=args.random_state,
    )
    log.info("Hennig on spatial-only x GMM x k=2 (secondary finding probe)...")
    hen_spatial = hennig_clusterboot(
        X_spatial, spatial_gmm_k2_labels, k=2, algorithm="gmm",
        n_bootstrap=args.hennig_bootstraps,
        threshold=0.75, random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
    for cluster_id, median, mean_ in zip(
        hen_spatial.cluster_ids, hen_spatial.jaccard_median,
        hen_spatial.jaccard_mean,
    ):
        validity_records.append({
            "test": "hennig_clusterboot",
            "algorithm": "gmm",
            "k": 2,
            "feature_space": "spatial_only",
            "cluster_id": int(cluster_id),
            "jaccard_median": float(median),
            "jaccard_mean": float(mean_),
            "stable": bool(median >= 0.75),
        })

    # Also persist spatial-only k=2 labels for downstream analyse stage.
    pd.DataFrame({
        "pid": pid_order,
        "spatial_only_gmm_k2": spatial_gmm_k2_labels,
    }).to_csv(out_dir / "cluster_labels_spatial_k2.csv", index=False)

    pd.DataFrame(validity_records).to_csv(
        out_dir / "validity_checks.csv", index=False,
    )

    # ── 6. ARI cross-algorithm ────────────────────────────────────
    ari_records: list[dict] = []
    algos = list(labels_per_algo.keys())
    for i, a in enumerate(algos):
        for j, b in enumerate(algos):
            if j <= i:
                continue
            ari_records.append({
                "algo_a": a, "algo_b": b, "k": args.forced_k,
                "ari": ari(labels_per_algo[a], labels_per_algo[b]),
            })
    pd.DataFrame(ari_records).to_csv(
        out_dir / "ari_across_algorithms.csv", index=False,
    )

    stage6_outputs = sorted(f.name for f in out_dir.iterdir() if f.is_file()
                            and f.name not in stage5["files"] and f.name != STAGE5_MARKER
                            and f.name != "PARTIAL_SMOKE_DO_NOT_USE")
    if smoke:
        log.info("smoke run complete; outputs in %s (not published)", out_dir)
        return 0
    confounded = sorted(a for a, st in kernel_status.items() if st == "confounded")
    if confounded:
        log.error("D021.1 hard stop: kernel-confounded clustering (%s); %s NOT updated, "
                  "outputs kept in %s for investigation", ", ".join(confounded),
                  cohort_dir, out_dir)
        return 1
    write_marker(out_dir, {
        "stage": 6,
        "stage5_marker_sha": file_sha(out_dir / STAGE5_MARKER),
        "stage6_fingerprint": stage6_fingerprint(cfg),
        "settings": settings,
        "inputs": files_manifest(out_dir, ["pca_scores.npy", "pca_scores_pid_order.csv",
                                           "prepared_matrix.csv", "cohort_metadata.csv"]),
        "spatial_features": spatial_cols,
        "kernel_status": kernel_status,
        "outputs": files_manifest(out_dir, stage6_outputs),
        "revision": code_revision(repo_root),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, name=STAGE6_MARKER)
    publish_keeping_subdirs(out_dir, cohort_dir)
    log.info("stage 6 (discover) complete. outputs in %s", cohort_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
