#!/usr/bin/env python
"""Stage 8: validate (D029 + D030 + D031).

Orchestrator for the three stage-8 deliverables:

  D029  external GE-scanner holdout: project pids {19, 28, 76, 77} onto
        the frozen production spatial-only k=2 phenotype. Descriptive
        only (N=4). Writes external_holdout_report.csv +
        xml_roundtrip_holdout.csv.

  D030  leave-k-out cross-validation: 10-fold kernel-stratified;
        full per-fold refit; per-fold ARI vs full-cohort reference;
        runtime PASS threshold T = 5th-percentile of K=10% disagreement
        simulation. Writes leave_k_out_ari.csv (10 fold rows + SUMMARY).

  D031  cross-cohort ARI consolidation: pure re-export of
        outputs/07_analyse/cross_cohort_ari.csv with PASS columns.
        Writes cross_cohort_ari_consolidated.csv.

All outputs land in ``outputs/08_validate/``. The run header
``run_header_validate.json`` records seam SHAs + the computed LOO
threshold T + per-deliverable verdicts.

Safeguards (D041): inputs are verified first
(stage 7 record, gate files vs stage 5); the D029 holdout must process
every expected patient (processing failures are written to
holdout_failures.csv and stop the stage; D002 round-trip failures stay
descriptive); D029 and D030 refit with the production settings on exactly
the published cohort, and D029 requires parity with stage 5-6 before
projecting; outputs are written to 08_validate.partial/ and published with
COMPLETE_stage8.json only when everything completed. Any failure exits
non-zero and publishes nothing. Skipped deliverables or non-production CV
settings write to outputs/08_validate_smoke/ (PARTIAL_SMOKE_DO_NOT_USE).

Usage::

    python scripts/09_validate.py
    python scripts/09_validate.py --skip-holdout --skip-loo   # smoke output only
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from predict.provenance import code_revision
from predict.completion import files_manifest, write_marker
from predict.publish import fresh_partial, publish
from predict.features.canonical import grouping_from_config
from predict.preprocess.slice_matcher import Z_MATCH_TOLERANCE_MM
from predict.config import Config, gate_dir, load_config
from predict.io import load_patient_metadata
from predict.io.spacing import load_spacing_metadata
from predict.validate.cross_cohort_ari import (
    ARI_PASS_THRESHOLD,
    write as write_cross_cohort_ari,
)
from predict.validate.external_holdout import (
    SPATIAL_FEATURES_FOR_PROJECTION,
    fit_frozen_pipeline,
    holdout_report,
    production_settings,
    require_production_parity,
)
from predict.validate.stage8 import STAGE8_MARKER, stage8_fingerprint, stage8_inputs
from predict.validate.leave_k_out import (
    DISAGREEMENT_RATE_DEFAULT,
    N_SIM_DEFAULT,
    N_SPLITS_DEFAULT,
    PERCENTILE_DEFAULT,
    SEED_DEFAULT,
    attach_summary_row,
    run_leave_k_out,
)


# ─────────────────────── helpers ───────────────────────


def _file_sha(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _save_json(path: Path, payload: dict | list) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8",
    )


def _load_script_module(path: Path, mod_name: str):
    """Import a numerically-named script under scripts/ as a module."""
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _discover_holdout_pids(cfg: Config) -> list[str]:
    """GE holdout pids from stage 1's outputs/01_manifest/exclusions.csv.

    The manifest is required: a fixed fallback list would hide a missing
    or changed stage 1."""
    excl = cfg.paths.outputs / "01_manifest" / "exclusions.csv"
    if not excl.exists():
        raise SystemExit(f"hard stop: {excl} missing; stage 8 takes the D029 holdout from stage 1")
    df = pd.read_csv(excl, dtype={"pid": str})
    ge = df[df["reason"] == "ge_scanner"]["pid"].astype(str).tolist()
    if not ge:
        raise SystemExit(f"hard stop: no ge_scanner pids in {excl}; the D029 holdout is empty")
    return sorted(ge, key=lambda s: int(s) if s.isdigit() else s)


# ─────────────────────── D029: holdout pipeline ───────────────────────


def _preprocess_holdout(
    pids: list[str],
    cfg: Config,
    out_dir: Path,
    log: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Run scripts/02_preprocess._process_one on each holdout pid.

    Returns ``(preprocess_report_df, xml_roundtrip_df, failures)``.
    """
    preprocess_mod = _load_script_module(
        Path(__file__).resolve().parent / "02_preprocess.py",
        "scripts_02_preprocess",
    )
    data_raw_root = str(cfg.paths.data_raw)
    xml_dir = str(cfg.paths.data_xml)
    target_spacing = cfg.resample.target_spacing
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save the spacing.json so the stage 3 reader can pick it up.
    from predict.io.spacing import save_spacing_metadata
    save_spacing_metadata(out_dir / "spacing.json", target_spacing)

    rows: list[dict] = []
    trips_all: list[dict] = []
    failures: list[dict] = []
    for pid in pids:
        log.info("D029 preprocess: pid %s", pid)
        result = preprocess_mod._process_one(
            pid, data_raw_root, xml_dir, str(out_dir),
            target_spacing,
            cfg.hu.clip_min, cfg.hu.clip_max,
            cfg.hu.metal_artifact_threshold,
            Z_MATCH_TOLERANCE_MM,
        )
        if result["status"] != "ok":
            log.error("preprocess failed for %s: %s", pid, result.get("error"))
            failures.append({"pid": pid, "step": "preprocess",
                             "status": result["status"], "error": result.get("error", "")})
            continue
        trips = result.pop("trips", [])
        rows.append({k: v for k, v in result.items() if k != "trips"})
        for t in trips:
            t["pid"] = pid
            trips_all.append(t)
    return pd.DataFrame(rows), pd.DataFrame(trips_all), failures


def _extract_holdout_features(
    pids: list[str],
    cfg: Config,
    holdout_preproc_dir: Path,
    quality_by_pid: dict[str, str],
    excluded_by_pid: dict[str, set],
    log: logging.Logger,
) -> tuple[pd.DataFrame, list[dict]]:
    """Run scripts/03_features._process_one (sans per-vessel PyRadiomics)
    on each holdout pid. Returns ``(features_df, failures)``."""
    features_mod = _load_script_module(
        Path(__file__).resolve().parent / "03_features.py",
        "scripts_03_features",
    )
    target_spacing = (
        load_spacing_metadata(holdout_preproc_dir / "spacing.json")
        if (holdout_preproc_dir / "spacing.json").exists()
        else cfg.resample.target_spacing
    )

    extractor_holder: dict = {}
    rows: list[dict] = []
    failures: list[dict] = []
    for pid in pids:
        log.info("D029 features: pid %s", pid)
        result = features_mod._process_one(
            pid, str(cfg.paths.data_raw.parent), str(cfg.paths.data_xml),
            str(holdout_preproc_dir),
            excluded_by_pid.get(pid, set()),
            target_spacing,
            quality_by_pid.get(pid, "ok"),
            grouping_from_config(cfg),
            extractor_holder,
        )
        if result["status"] != "ok":
            log.error("features failed for %s: %s",
                      pid, result.get("error"))
            failures.append({"pid": pid, "step": "features",
                             "status": result["status"], "error": result.get("error", "")})
            continue
        rows.append(result["row"])
    return pd.DataFrame(rows), failures


def _xml_pass_by_pid_from_trips(trips_df: pd.DataFrame) -> dict[str, bool]:
    """D029.3: per-pid D002 PASS = all non-dirty ROIs pass Max-exact.

    Pids whose ROIs are ALL ``matched_via="dirty"`` have nothing to gate
    on; they PASS by default (consistent with stage 2's behaviour for the
    same condition).
    """
    if trips_df.empty:
        return {}
    out: dict[str, bool] = {}
    for pid, grp in trips_df.groupby("pid"):
        non_dirty = grp[grp["matched_via"] != "dirty"]
        if non_dirty.empty:
            out[str(pid)] = True
        else:
            out[str(pid)] = bool(non_dirty["passes"].astype(bool).all())
    return out


def _gated_feature_cols(cfg: Config) -> list[str]:
    gated_csv = gate_dir(cfg) / "gated_features.csv"
    return pd.read_csv(gated_csv)["feature"].tolist()


def _published_cohort(cfg: Config) -> pd.DataFrame:
    """Raw features.csv rows of exactly stage 5's published cohort, in its
    order (re-filtering features.csv would give 437 rows; the published
    cohort has 436)."""
    pids = pd.read_csv(cfg.paths.outputs / "06_reduce" / "prepared_matrix.csv",
                       dtype={"pid": str}, usecols=["pid"])["pid"].tolist()
    df = pd.read_csv(cfg.paths.outputs / "03_features" / "features.csv", dtype={"pid": str})
    df = df.set_index("pid")
    missing = [p for p in pids if p not in df.index]
    if missing:
        raise SystemExit(f"hard stop: {len(missing)} published cohort pids not in features.csv "
                         f"(e.g. {missing[:5]})")
    return df.loc[pids].reset_index()


def _stop_on_failures(failures: list[dict], expected: list[str], out_dir: Path, step: str) -> None:
    if failures:
        pd.DataFrame(failures).to_csv(out_dir / "holdout_failures.csv", index=False)
        raise SystemExit(
            f"hard stop: D029 {step} failed for {len(failures)} of {len(expected)} holdout "
            f"patients ({', '.join(f['pid'] for f in failures)}); see "
            f"{out_dir / 'holdout_failures.csv'}. Nothing published.")


def run_d029_holdout(cfg: Config, out_dir: Path, log: logging.Logger,
                     settings) -> dict:
    """Run the D029 external GE-holdout pipeline. Returns a summary dict;
    any processing failure stops the stage."""
    holdout_pids = _discover_holdout_pids(cfg)
    log.info("D029 holdout pids: %s", holdout_pids)

    holdout_preproc_dir = out_dir / "holdout_preprocessed"
    pre_report, trips, failures = _preprocess_holdout(
        holdout_pids, cfg, holdout_preproc_dir, log,
    )
    _stop_on_failures(failures, holdout_pids, out_dir, "preprocessing")
    if trips.empty:
        raise SystemExit("hard stop: D029 preprocessing produced no XML round-trip rows")

    # D002 XML round-trip audit on the holdout.
    trips.to_csv(out_dir / "xml_roundtrip_holdout.csv", index=False)
    pre_report.to_csv(out_dir / "holdout_preprocess_report.csv", index=False)
    xml_pass_by_pid = _xml_pass_by_pid_from_trips(trips)

    # quality + excluded ROIs for stage 3.
    quality_by_pid = dict(zip(
        pre_report["pid"].astype(str), pre_report["roundtrip_quality"]
    )) if not pre_report.empty else {}
    fails = trips[
        (~trips["passes"].astype(bool)) & (trips["matched_via"] != "dirty")
    ]
    excluded_by_pid: dict[str, set] = {}
    for _, r in fails.iterrows():
        excluded_by_pid.setdefault(str(r["pid"]), set()).add(
            (int(r["image_index"]), int(r["roi_idx_in_slice"]))
        )

    holdout_features, failures = _extract_holdout_features(
        holdout_pids, cfg, holdout_preproc_dir,
        quality_by_pid, excluded_by_pid, log,
    )
    _stop_on_failures(failures, holdout_pids, out_dir, "feature extraction")
    holdout_features.to_csv(
        out_dir / "holdout_features.csv", index=False,
    )
    if sorted(holdout_features["pid"].astype(str)) != sorted(holdout_pids):
        raise SystemExit("hard stop: D029 holdout features do not cover the expected patients")

    # Refit the production pipeline on the published cohort, require parity
    # with stage 5-6, then project + predict the holdout.
    full_cohort = _published_cohort(cfg)
    gated_cols = _gated_feature_cols(cfg)
    frozen = fit_frozen_pipeline(full_cohort, gated_cols, settings=settings)
    parity = require_production_parity(frozen, cfg.paths.outputs / "06_reduce")
    log.info("D029 production parity: %s", parity)
    report = holdout_report(holdout_features, frozen, xml_pass_by_pid)
    if sorted(report["pid"].astype(str)) != sorted(holdout_pids):
        raise SystemExit("hard stop: D029 report does not cover the expected patients")
    report.to_csv(out_dir / "external_holdout_report.csv", index=False)
    log.info("D029 wrote %d holdout rows", len(report))

    return {
        "verdict": "ok",
        "n_holdout": int(len(report)),
        "holdout_pids": holdout_pids,
        "xml_pass_by_pid": xml_pass_by_pid,
        "production_parity": parity,
        "phenotype_counts": (
            report["predicted_phenotype"].value_counts().to_dict()
        ),
    }


# ─────────────────────── D030: leave-k-out ───────────────────────


def _load_reference_labels(cfg: Config) -> pd.Series:
    """Load full-cohort spatial-k=2 labels from the stage 6 seam (verified
    by stage8_inputs before anything runs)."""
    p = (cfg.paths.outputs / "06_reduce"
         / "cluster_labels_spatial_k2.csv")
    df = pd.read_csv(p, dtype={"pid": str})
    # The seam writes a single label column; pick the GMM one.
    label_col = next(
        (c for c in df.columns if c.startswith("spatial_k2") or "gmm" in c.lower()),
        None,
    )
    if label_col is None:
        candidates = [c for c in df.columns if c != "pid"]
        if len(candidates) != 1:
            raise ValueError(
                f"D030: cannot infer label column in {p}; got {candidates}"
            )
        label_col = candidates[0]
    return pd.Series(
        df[label_col].astype(int).to_numpy(),
        index=df["pid"].astype(str),
        name="spatial_k2_label",
    )


def run_d030_loo(
    cfg: Config, out_dir: Path, log: logging.Logger,
    n_splits: int, n_simulations: int, seed: int, settings,
) -> dict:
    """Run the D030 LOO CV. Returns a summary dict including threshold T."""
    full_cohort = _published_cohort(cfg)
    gated_cols = _gated_feature_cols(cfg)
    reference = _load_reference_labels(cfg)

    log.info("D030 LOO: N_cohort=%d, n_splits=%d, n_sim=%d",
             len(full_cohort), n_splits, n_simulations)
    per_fold, summary = run_leave_k_out(
        full_cohort, gated_cols, reference,
        n_splits=n_splits, seed=seed,
        n_simulations=n_simulations,
        disagreement_rate=DISAGREEMENT_RATE_DEFAULT,
        settings=settings,
    )
    out_df = attach_summary_row(per_fold, summary)
    out_df.to_csv(out_dir / "leave_k_out_ari.csv", index=False)
    log.info("D030 wrote %d fold rows + 1 summary", len(per_fold))
    log.info("D030 median ARI = %.3f, median T = %.3f, PASS = %s",
             summary["median_ari"], summary["median_T_fold"],
             summary["overall_pass"])
    return summary


# ─────────────────────── D031: ARI consolidation ───────────────────────


def run_d031_consolidation(
    cfg: Config, out_dir: Path, log: logging.Logger,
) -> dict:
    # Stage 7 output: only from a complete, current stage 7 run.
    from predict.analyse.stage7 import require_stage7_complete
    require_stage7_complete(cfg.paths.outputs / "07_analyse", cfg)
    src = cfg.paths.outputs / "07_analyse" / "cross_cohort_ari.csv"
    out_path = out_dir / "cross_cohort_ari_consolidated.csv"
    df = write_cross_cohort_ari(src, out_path)
    log.info("D031 wrote %d rows", len(df))
    return {
        "verdict": "ok",
        "n_rows": int(len(df)),
        "pass_threshold": ARI_PASS_THRESHOLD,
        "n_passing": int(df["pass_verdict"].sum()),
    }


# ─────────────────────── main ───────────────────────


def build_run_header(repo_root: Path, cfg: Config,
                     args: argparse.Namespace) -> dict:
    out = cfg.paths.outputs
    info: dict = {
        "stage": "validate",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": code_revision(repo_root),
        "config_yaml_sha": _file_sha(repo_root / "configs" / "default.yaml"),
        "features_csv_sha": _file_sha(out / "03_features" / "features.csv"),
        "gated_features_csv_sha": _file_sha(gate_dir(cfg) / "gated_features.csv"),
        "primary_gate": cfg.stability.primary_gate,
        "cluster_labels_spatial_k2_sha": _file_sha(
            out / "06_reduce" / "cluster_labels_spatial_k2.csv"
        ),
        "cross_cohort_ari_sha": _file_sha(
            out / "07_analyse" / "cross_cohort_ari.csv"
        ),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--skip-holdout", action="store_true",
                        help="Skip D029 external holdout.")
    parser.add_argument("--skip-loo", action="store_true",
                        help="Skip D030 leave-k-out (slow ~5 min).")
    parser.add_argument("--skip-ari-consolidation", action="store_true",
                        help="Skip D031 cross-cohort ARI consolidation.")
    parser.add_argument("--n-splits", type=int, default=N_SPLITS_DEFAULT)
    parser.add_argument("--n-simulations", type=int, default=N_SIM_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("validate")

    cfg = load_config(args.config)
    repo_root = cfg.paths.outputs.parent
    t_overall = time.perf_counter()

    # Inputs first (stage 7 record, gate files vs stage 5), then a fresh
    # partial folder. A reduced run is a smoke run with its own folder.
    inputs = stage8_inputs(cfg)
    run_settings = {
        "skip_holdout": bool(args.skip_holdout), "skip_loo": bool(args.skip_loo),
        "skip_ari_consolidation": bool(args.skip_ari_consolidation),
        "n_splits": int(args.n_splits), "n_simulations": int(args.n_simulations),
        "seed": int(args.seed),
    }
    production = run_settings == {
        "skip_holdout": False, "skip_loo": False, "skip_ari_consolidation": False,
        "n_splits": N_SPLITS_DEFAULT, "n_simulations": N_SIM_DEFAULT, "seed": SEED_DEFAULT,
    }
    target = cfg.paths.outputs / ("08_validate" if production else "08_validate_smoke")
    out_dir = fresh_partial(target)
    if not production:
        (out_dir / "PARTIAL_SMOKE_DO_NOT_USE").write_text("reduced stage 8 run\n", encoding="utf-8")
        log.warning("reduced run (%s): outputs go to %s and are never published as stage 8",
                    run_settings, target)
    settings = production_settings(cfg)

    summary: dict = {
        "stage": "validate",
        "config": dict(vars(args)),
        "thresholds": {
            "ari_pass_threshold_d031": ARI_PASS_THRESHOLD,
            "disagreement_rate_K_d030": DISAGREEMENT_RATE_DEFAULT,
            "percentile_d030": PERCENTILE_DEFAULT,
            "n_simulations_d030": int(args.n_simulations),
        },
    }

    if not args.skip_holdout:
        t0 = time.perf_counter()
        summary["d029"] = run_d029_holdout(cfg, out_dir, log, settings)
        summary["d029"]["runtime_sec"] = round(time.perf_counter() - t0, 2)
    else:
        log.info("D029 skipped by flag")
        summary["d029"] = {"verdict": "skipped"}

    if not args.skip_loo:
        t0 = time.perf_counter()
        summary["d030"] = run_d030_loo(
            cfg, out_dir, log,
            n_splits=args.n_splits,
            n_simulations=args.n_simulations,
            seed=args.seed,
            settings=settings,
        )
        summary["d030"]["runtime_sec"] = round(time.perf_counter() - t0, 2)
    else:
        log.info("D030 skipped by flag")
        summary["d030"] = {"verdict": "skipped"}

    if not args.skip_ari_consolidation:
        t0 = time.perf_counter()
        summary["d031"] = run_d031_consolidation(cfg, out_dir, log)
        summary["d031"]["runtime_sec"] = round(time.perf_counter() - t0, 2)
    else:
        log.info("D031 skipped by flag")
        summary["d031"] = {"verdict": "skipped"}

    header = build_run_header(repo_root, cfg, args)
    header["summary"] = summary
    header["inputs"] = inputs
    header["total_runtime_sec"] = round(time.perf_counter() - t_overall, 2)
    _save_json(out_dir / "run_header_validate.json", header)

    if production:
        if stage8_inputs(cfg) != inputs:
            raise SystemExit("hard stop: stage 8 inputs changed during the run; nothing published")
        write_marker(out_dir, {
            "stage": 8,
            "stage8_fingerprint": stage8_fingerprint(),
            "inputs": inputs,
            "settings": run_settings,
            "outputs": files_manifest(out_dir, [f.name for f in out_dir.iterdir() if f.is_file()]),
            "revision": code_revision(repo_root),
            "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, name=STAGE8_MARKER)
    publish(out_dir, target)
    log.info("stage 8 complete in %.1f s; outputs in %s",
             header["total_runtime_sec"], target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
