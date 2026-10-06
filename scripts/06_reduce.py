#!/usr/bin/env python
"""Stage 5 orchestrator (D019, D020, D022).

End-to-end pipeline for the dimensionality-reduction stage:

  1. Load stage-3 features.csv + stage-4 icc_report.csv
  2. Filter to eligible cohort (radiomics_status == "ok"; D015 N=422)
  3. Restrict to the gated 88-feature set from stage 4 (gated_features.csv)
  4. Apply ComBat kernel filter (drop singleton-kernel patients; D019);
     optional `--kernel-filter` for the kernel-stratified sensitivity rerun
  5. Run matrix preparation (D019): D017 drops + D018 binarisation +
     derived features + variance filter + ComBat + Yeo-Johnson + z-score
  6. Spearman r^2 redundancy clustering: multi-block by default (D022),
     single-matrix sensitivity via --block-mode single (D020)
  7. PCA on the representative feature subset, sign-normalised (D020)
  8. Persist everything under outputs/06_reduce/ (or stratified subdir)

Cluster discovery, gap statistic, Monti consensus, forced-k characterisation,
and Hennig stability checks (D021) are in ``scripts/07_discover.py``. This
script writes the seam files that 07_discover.py consumes:

  * pca_scores.csv            -> X_full (PC matrix)
  * prepared_matrix.csv       -> post-D019 features (for spatial PCA)
  * cohort_metadata.csv       -> RAW kernel, low_burden_flag, category,
                                 agatston_total (for burden residualisation
                                 and forced-k crosstabs)
  * representative_features.csv, multi_block_assignments.csv,
    pca_loadings.csv, pca_top_loadings.csv, pca_explained_variance.csv,
    pc_agatston_correlation.csv, combat_audit.csv, matrix_prep_log.json,
    run_header.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from predict.provenance import code_revision
from predict.config import Config, default_n_jobs, gate_dir, load_config
from predict.discover.clusterability import assess_clusterability
from predict.features.feature_schema import feature_names
from predict.reduce.pca import (
    explained_variance_table,
    fit_pca,
    pc_external_correlation,
    top_loadings_table,
)
from predict.completion import file_sha, files_manifest, write_marker
from predict.discover.stage6 import STAGE5_MARKER
from predict.publish import fresh_partial, publish_keeping_subdirs
from predict.reduce.prepare_matrix import (
    STAGE5_TRANSFORMED,
    production_prep_kwargs,
    run_matrix_prep,
)
from predict.reduce.redundancy import (
    DEFAULT_BLOCKS,
    build_icc_lookup,
    run_multi_block_redundancy_clustering,
    run_redundancy_clustering,
)


# Features stage 5 creates (D018 binarisation + D019 derived); all have
# canonical (XML) lineage. Their reliability comes from stage 5b's
# derived_icc.csv under the D033 gate (D040).
DERIVED_FEATURE_CANDIDATES: tuple[str, ...] = STAGE5_TRANSFORMED

# Raw metadata columns to persist for downstream stages (07_discover.py
# needs raw agatston_total for burden residualisation; the z-scored feature
# version is destroyed by D019).
METADATA_COLS_FOR_DOWNSTREAM: tuple[str, ...] = (
    "pid", "kernel", "scanner_model", "mask_voxels", "low_burden_flag",
    "roundtrip_quality", "category", "radiomics_status", "radiomics_reason",
    "agatston_total",
)


# ─────────────────────── reproducibility breadcrumbs ───────────────────────


def _file_sha(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def build_run_header(
    repo_root: Path, cfg: Config, args: argparse.Namespace,
) -> dict:
    info: dict = {
        "stage": "reduce",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": code_revision(repo_root),
        "config_yaml_sha": _file_sha(repo_root / "configs" / "default.yaml"),
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


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _load_inputs(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (eligible features dataframe, ICC report, gated feature list)."""
    features_csv = cfg.paths.outputs / "03_features" / "features.csv"
    icc_csv = gate_dir(cfg) / "icc_report.csv"
    gated_csv = gate_dir(cfg) / "gated_features.csv"

    if not features_csv.exists():
        raise FileNotFoundError(f"missing {features_csv}")
    if not icc_csv.exists():
        raise FileNotFoundError(f"missing {icc_csv}")
    if not gated_csv.exists():
        raise FileNotFoundError(f"missing {gated_csv}")

    features = pd.read_csv(features_csv, dtype={"pid": str})
    icc = pd.read_csv(icc_csv)
    gated = pd.read_csv(gated_csv)["feature"].tolist()

    eligible = features[features["radiomics_status"] == "ok"].copy().reset_index(drop=True)
    return eligible, icc, gated


# ─────────────────────── main ───────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--kernel-filter", type=str, default=None,
                        help="Restrict the cohort to a single kernel for the "
                             "D021 kernel-stratified sensitivity rerun. "
                             "Outputs land in outputs/06_reduce/stratified_<kernel>/.")
    parser.add_argument("--n-jobs", type=int, default=default_n_jobs(),
                        help="Reserved for future parallel sections; reduce "
                             "stage itself is fast and single-threaded. "
                             "Defaults to the scheduler allocation "
                             "(PBS_NCPUS / PBS_NODEFILE).")
    parser.add_argument("--block-mode", choices=("multi", "single"),
                        default="multi",
                        help="D022: multi-block redundancy clustering is the "
                             "primary (default). 'single' runs the D020 "
                             "single-matrix variant as a sensitivity check.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log = logging.getLogger("reduce")

    cfg = load_config(args.config)
    base_out = cfg.paths.outputs / "06_reduce"
    # The D020 single-block sensitivity never writes over the D022 primary.
    if args.block_mode == "single":
        base_out = base_out / "sensitivity_single_block"
    if args.kernel_filter is not None:
        kernel_slug = args.kernel_filter.replace("/", "_")
        publish_dir = base_out / f"stratified_{kernel_slug}"
    else:
        publish_dir = base_out
    # Everything is written to <cohort>.partial/ and replaces the cohort's
    # files only when the whole run succeeds.
    out_dir = fresh_partial(publish_dir)
    repo_root = cfg.paths.outputs.parent

    # ── Run header ────────────────────────────────────────────────
    header = build_run_header(repo_root, cfg, args)
    _save_json(out_dir / "run_header.json", header)
    log.info("run_header: git=%s, config_sha=%s",
             header["git_commit"], header["config_yaml_sha"])

    # ── 1 / 2. Load inputs + eligible cohort ──────────────────────
    features_sha = file_sha(cfg.paths.outputs / "03_features" / "features.csv")
    eligible, icc_report, gated_features = _load_inputs(cfg)
    log.info("eligible cohort: N=%d (D015)", len(eligible))
    log.info("gated features from stage 4: %d", len(gated_features))

    # Restrict columns to gated + metadata.
    metadata_cols = [c for c in
                     ("pid", "kernel", "scanner_model", "mask_voxels",
                      "low_burden_flag", "roundtrip_quality", "category",
                      "radiomics_status", "radiomics_reason")
                     if c in eligible.columns]
    missing_gated = [c for c in gated_features if c not in eligible.columns]
    if missing_gated:
        raise SystemExit(f"{len(missing_gated)} gated feature(s) missing from features.csv "
                         f"(e.g. {missing_gated[:5]}); the gate and stage 3 disagree")
    keep_cols = metadata_cols + list(gated_features)
    eligible = eligible[keep_cols].copy()

    # D040: measured reliability of the stage-5 transformed features. The
    # D033 gate (5b) writes derived_icc.csv; a D014 classic gate has none,
    # and there canonical inputs are invariant by construction.
    derived_icc_csv = gate_dir(cfg) / "derived_icc.csv"
    if derived_icc_csv.exists():
        derived_icc = pd.read_csv(derived_icc_csv)
        missing_tf = sorted(set(DERIVED_FEATURE_CANDIDATES) - set(derived_icc["feature"]))
        if missing_tf:
            raise SystemExit(f"{derived_icc_csv} lacks {missing_tf}; rerun stage 5b")
        reliable_transformed = set(derived_icc.loc[derived_icc["passes_gate"].astype(bool), "feature"])
        log.info("D040 transformed-feature reliability: %s", {
            r.feature: round(float(r.icc), 4) for r in derived_icc.itertuples()})
    elif cfg.stability.primary_gate == "realistic":
        raise SystemExit(f"{derived_icc_csv} missing; rerun stage 5b (D040)")
    else:
        derived_icc = None
        reliable_transformed = None
        log.info("classic (D014) gate: transformed features invariant by construction")

    # Stash RAW agatston_total BEFORE D019 z-scores it. 07_discover.py needs
    # the raw value for burden residualisation.
    if "agatston_total" in eligible.columns:
        raw_agatston = eligible[["pid", "agatston_total"]].copy()
    else:
        raw_agatston = None
        log.warning("agatston_total not in eligible columns; downstream "
                    "burden residualisation will be unavailable")

    # ── ComBat kernel filter (D019) ──────────────────────────────
    # Patients with rare kernels (singleton-kernel groups in COCA: B35f and
    # I36f/3 with one patient each) are dropped because ComBat cannot estimate
    # within-batch variance on a single sample. The --kernel-filter flag
    # further restricts to one kernel for the D021 stratified rerun.
    combat_groups = set(cfg.raw["reduce"].get("combat_kernel_groups", []))
    if args.kernel_filter is not None:
        target = {args.kernel_filter}
        log.info("kernel-stratified rerun: restricting cohort to kernel %s",
                 args.kernel_filter)
    elif combat_groups:
        target = combat_groups
    else:
        target = None

    if target is not None:
        kernel_counts = eligible["kernel"].value_counts().to_dict()
        log.info("kernel distribution in eligible cohort: %s", kernel_counts)
        before = len(eligible)
        eligible = eligible[eligible["kernel"].isin(target)].copy().reset_index(drop=True)
        dropped = before - len(eligible)
        if dropped > 0:
            log.info(
                "dropped %d patient(s) with kernel outside %s",
                dropped, sorted(target),
            )
        log.info("post-kernel-filter cohort: N=%d", len(eligible))
        if len(eligible) < 2:
            raise SystemExit(
                f"after --kernel-filter {args.kernel_filter}, only {len(eligible)} "
                f"patients remain; cannot run stage 5"
            )

    # ── 3. D019 matrix prep ───────────────────────────────────────
    log.info("D019 matrix preparation...")
    prep_kwargs = production_prep_kwargs(cfg)    # shared with stage 8
    if prep_kwargs["reliable_transformed"] != reliable_transformed:
        raise SystemExit("D040 reliable set differs between 06_reduce and production_prep_kwargs")
    prep_df, prep_features, prep_log = run_matrix_prep(
        eligible, list(gated_features), **prep_kwargs,
    )
    _save_json(out_dir / "matrix_prep_log.json", prep_log.to_dict())
    if prep_log.combat_audit:
        pd.DataFrame(prep_log.combat_audit).to_csv(
            out_dir / "combat_audit.csv", index=False,
        )
    log.info("prep: %d -> %d features (D019 complete)",
             prep_log.n_features_in, prep_log.n_features_out)

    # ── 4. D020 / D022 redundancy clustering ──────────────────────
    log.info("D020/D022 redundancy clustering (mode=%s)...", args.block_mode)
    icc_lookup = build_icc_lookup(
        icc_report, derived_report_df=derived_icc,
        invariant_names=DERIVED_FEATURE_CANDIDATES if derived_icc is None else (),
    )
    no_icc = [f for f in prep_features if f not in icc_lookup]
    if no_icc:
        raise SystemExit(f"features without an ICC entry after matrix prep: {no_icc}")
    canonical_set = set(feature_names()) | set(DERIVED_FEATURE_CANDIDATES)

    if args.block_mode == "multi":
        # D022 multi-block (primary).
        multi = run_multi_block_redundancy_clustering(
            prep_df, prep_features, icc_lookup, canonical_set,
            blocks=DEFAULT_BLOCKS,
            primary_method=cfg.reduce.r2_linkage,
            sensitivity_methods=(),
            min_gap=cfg.reduce.r2_elbow_min_gap,
            fallback_distance=cfg.reduce.r2_fallback_distance,
        )
        representatives = multi.representatives
        multi.assignments_dataframe().to_csv(
            out_dir / "multi_block_assignments.csv", index=False,
        )
        per_block_summary = {
            name: {
                "n_input_features": len(result.feature_order),
                "n_clusters": result.n_clusters(),
                "cut_threshold": result.cut_threshold,
                "representatives": list(result.representatives),
            }
            for name, result in multi.per_block.items()
        }
        _save_json(out_dir / "multi_block_summary.json", {
            "block_partition": {
                b.name: {
                    "prefixes": list(b.prefixes),
                    "exact_names": list(b.exact_names),
                }
                for b in DEFAULT_BLOCKS
            },
            "per_block": per_block_summary,
            "unassigned_features": list(multi.unassigned),
            "total_representatives": len(representatives),
        })
        pd.DataFrame({"feature": representatives}).to_csv(
            out_dir / "representative_features.csv", index=False,
        )
        log.info("multi-block redundancy: %d features -> %d representatives "
                 "across %d non-empty blocks",
                 len(prep_features), len(representatives),
                 len(multi.per_block))
        for name, info in per_block_summary.items():
            log.info("  block[%s]: %d -> %d", name,
                     info["n_input_features"], info["n_clusters"])
    else:
        # D020 single-matrix (sensitivity).
        redundancy = run_redundancy_clustering(
            prep_df, prep_features, icc_lookup, canonical_set,
            primary_method=cfg.reduce.r2_linkage,
            sensitivity_methods=("ward", "complete"),
            min_gap=cfg.reduce.r2_elbow_min_gap,
            fallback_distance=cfg.reduce.r2_fallback_distance,
        )
        representatives = redundancy.representatives
        redundancy.cluster_assignments.to_csv(
            out_dir / "redundancy_clusters.csv", index=False,
        )
        pd.DataFrame({"feature": representatives}).to_csv(
            out_dir / "representative_features.csv", index=False,
        )
        _save_json(out_dir / "redundancy_sensitivity.json", {
            "primary": {
                "method": cfg.reduce.r2_linkage,
                "n_clusters": redundancy.n_clusters(),
                "cut_threshold": redundancy.cut_threshold,
                "representatives": redundancy.representatives,
            },
            "sensitivity_per_method": redundancy.sensitivity_per_method,
        })
        log.info("single-matrix redundancy: %d features -> %d representatives",
                 len(prep_features), len(representatives))

    # ── 5. D020 PCA on representatives ────────────────────────────
    log.info("D020 PCA...")
    pca = fit_pca(prep_df, representatives,
                  cumvar_threshold=cfg.reduce.pca_cumvar,
                  random_state=args.random_state)
    log.info("PCA: n_retain=%d at %.0f%% cumvar",
             pca.n_retain, 100 * cfg.reduce.pca_cumvar)

    # ── Diagnostic: in-memory Hopkins on pca.scores ────────────────
    # If this matches 07_discover.py's reported H exactly, the NPY/CSV seam
    # is byte-exact. If it differs, the drift is upstream (D019 or PCA),
    # not the seam. Same function and sample-size
    # rule as stage 6 (int() here gave 43 vs stage 6's 44 rows at N=436).
    diag_H = assess_clusterability(
        pca.scores,
        sample_frac=cfg.raw["hopkins"]["sample_frac"],
        random_state=args.random_state,
    ).H
    log.info("DIAG: in-memory Hopkins on pca.scores = %.6f (N=%d, n_retain=%d)",
             diag_H, pca.scores.shape[0], pca.n_retain)

    # Audit hash of pca.scores in-memory bytes (helps confirm npy seam
    # roundtrip is lossless).
    pca_scores_sha = hashlib.sha256(
        np.ascontiguousarray(pca.scores, dtype=np.float64).tobytes()
    ).hexdigest()[:16]
    log.info("DIAG: pca.scores float64-bytes sha256[:16] = %s", pca_scores_sha)

    explained_variance_table(pca).to_csv(
        out_dir / "pca_explained_variance.csv", index=False,
    )
    top_loadings_table(pca).to_csv(
        out_dir / "pca_top_loadings.csv", index=False,
    )
    pd.DataFrame(
        pca.components,
        index=[f"PC{i + 1}" for i in range(pca.components.shape[0])],
        columns=pca.feature_names,
    ).to_csv(out_dir / "pca_loadings.csv")
    pd.DataFrame(
        pca.scores,
        columns=[f"PC{i + 1}" for i in range(pca.n_retain)],
        index=pca.pid_order,
    ).rename_axis("pid").to_csv(out_dir / "pca_scores.csv")
    # Byte-exact seam for 07_discover.py. CSV roundtrips lose ~12-digit
    # float precision which is enough to shift Hopkins by ~0.03 on the
    # full cohort. NPY preserves float64 exactly so the downstream
    # discover stage is reproducible against the in-memory pca.scores.
    np.save(out_dir / "pca_scores.npy", pca.scores)
    pd.Series(pca.pid_order, name="pid").to_csv(
        out_dir / "pca_scores_pid_order.csv", index=False,
    )

    # PC vs Agatston external sanity check.
    if raw_agatston is not None:
        agatston_aligned = (raw_agatston.set_index("pid")
                            .loc[pca.pid_order, "agatston_total"])
        pc_external_correlation(pca, agatston_aligned, name="agatston_total").to_csv(
            out_dir / "pc_agatston_correlation.csv", index=False,
        )

    # ── 6. Persist seam files for 07_discover.py ──────────────────
    # prepared_matrix.csv: post-D019 prep_df (z-scored features + metadata).
    # 07_discover.py reads this for the spatial-only PCA reconstruction.
    prep_df.to_csv(out_dir / "prepared_matrix.csv", index=False)

    # cohort_metadata.csv: RAW values 07_discover.py needs (agatston_total
    # before z-scoring; kernel / low_burden_flag / category for crosstabs).
    # Rebuild from features.csv to get raw agatston_total.
    raw_features_csv = cfg.paths.outputs / "03_features" / "features.csv"
    raw_features = pd.read_csv(raw_features_csv, dtype={"pid": str})
    meta_cols_present = [c for c in METADATA_COLS_FOR_DOWNSTREAM
                         if c in raw_features.columns]
    cohort_meta = (raw_features[meta_cols_present]
                   .set_index("pid")
                   .loc[prep_df["pid"].tolist()]
                   .reset_index())
    cohort_meta.to_csv(out_dir / "cohort_metadata.csv", index=False)
    log.info("seam files written: prepared_matrix.csv (%d rows x %d cols), "
             "cohort_metadata.csv (%d rows x %d cols)",
             len(prep_df), prep_df.shape[1],
             len(cohort_meta), cohort_meta.shape[1])

    # Completion record (stage 6 copies exactly these files and checks them).
    if file_sha(raw_features_csv) != features_sha:
        raise SystemExit(f"hard stop: {raw_features_csv} changed during the stage 5 run")
    stage5_files = sorted(f.name for f in out_dir.iterdir() if f.is_file())
    write_marker(out_dir, {
        "stage": 5,
        "files": files_manifest(out_dir, stage5_files),
        "cohort": str(publish_dir.relative_to(cfg.paths.outputs)).replace("\\", "/"),
        "kernel_filter": args.kernel_filter,
        "block_mode": args.block_mode,
        "gated_features_sha": file_sha(gate_dir(cfg) / "gated_features.csv"),
        "derived_icc_sha": file_sha(gate_dir(cfg) / "derived_icc.csv"),
        # Stage 7 joins raw features.csv measurements to these cohorts'
        # labels and refuses a features.csv other than this one.
        "features_csv_sha": features_sha,
        "revision": code_revision(repo_root),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, name=STAGE5_MARKER)
    publish_keeping_subdirs(out_dir, publish_dir)
    log.info("stage 5 (reduce) complete. outputs in %s", publish_dir)
    log.info("next: python scripts/07_discover.py --cohort-dir %s", publish_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
