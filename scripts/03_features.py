#!/usr/bin/env python
"""Stage 3: feature extraction.

Reads ``outputs/01_manifest/manifest.csv`` plus stage-2 outputs
(``{pid}_ct.npy``, ``{pid}_mask.npy``, ``spacing.json``,
``xml_roundtrip.csv``, ``preprocess_report.csv``) and produces one
``features.csv`` row per patient.

Per-patient pipeline (each worker):

  1. Load native DICOM headers via ``load_patient_metadata`` for
     ``slice_positions`` and native pixel spacing.
  2. Parse XML (cheap).
  3. Derive ``excluded_roi_ids`` for this patient from the cohort
     ``xml_roundtrip.csv`` (passes=False, matched_via!=dirty).
  4. Compute Agatston (D011), per-vessel aggregates, density tiers.
  5. Group ROIs into 3D lesions (D007) with the configured
     ``lesion_grouping`` parameters -> spatial features.
  6. Run PyRadiomics on the resampled whole-mask (D009). A mask below
     PyRadiomics' own minimumROISize / minimumROIDimensions is skipped
     (decided before extraction); any exception from the extractor fails
     the patient.
  7. Aggregate into one row + a list of lesion rows for the audit CSV.

Per-vessel PyRadiomics was removed: it paired
masks rasterised in native pixel coordinates with the resampled CT (Dice 0.28
with the saved whole mask on a real scan), it was off by default and nothing
used its output. A correct version would build native per-vessel masks and
resample them exactly as stage 2 does.

Everything is written to ``outputs/03_features.partial/``. Only a run in
which every manifest patient succeeds replaces ``outputs/03_features/``;
otherwise the script exits 1 and the published folder is untouched
(``predict.publish``, same policy as stage 2). ``--limit`` never publishes.

Outputs (under ``outputs/03_features/``):

  - ``features.csv``        : main artefact, one row per patient.
  - ``lesions.csv``         : one row per 3D lesion (audit trail).
  - ``feature_extraction_log.csv``: per-patient runtime + counts.
  - ``errors.csv``          : patients that failed (if any).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from predict.provenance import code_revision
from predict.config import Config, default_n_jobs, load_config
from predict.features.canonical import compute_canonical_features, grouping_from_config
from predict.features.feature_schema import zero_features
from predict.features.radiomics import create_extractor, extract_pyradiomics, roi_too_small
from predict.io import (
    PatientMetadata,
    load_patient_metadata,
    load_spacing_metadata,
    parse_calcium_xml,
)
from predict.publish import fresh_partial, publish


LOW_BURDEN_VOXEL_THRESHOLD: int = 100


# ───────────────────── reproducibility breadcrumbs ─────────────────────


def _file_sha(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def build_run_header(repo_root: Path, params_yaml: Path, grouping: dict) -> dict:
    """Return a dict of reproducibility breadcrumbs for the run."""
    import sys as _sys
    info = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": code_revision(repo_root),
        "params_yaml_sha": _file_sha(params_yaml),
        "config_yaml_sha": _file_sha(repo_root / "configs" / "default.yaml"),
        "lesion_grouping": grouping,
        "python_version": _sys.version.split()[0],
    }
    # Library versions (lazy imports so this still runs without them).
    for mod in ("numpy", "pandas", "SimpleITK", "pydicom", "radiomics", "cv2"):
        try:
            m = __import__(mod)
            info[f"{mod}_version"] = getattr(m, "__version__", "unknown")
        except Exception:
            info[f"{mod}_version"] = "n/a"
    return info


# ───────────────────── per-patient worker ─────────────────────


def _process_one(
    pid: str,
    data_root_str: str,
    xml_dir_str: str,
    preproc_dir_str: str,
    excluded_set: set,
    target_spacing: tuple,
    quality_label: str,
    grouping: dict,
    extractor_holder: dict,
) -> dict:
    """Compute every feature for one patient."""
    t0 = time.perf_counter()
    try:
        data_root = Path(data_root_str)
        preproc_dir = Path(preproc_dir_str)

        meta: PatientMetadata = load_patient_metadata(pid, data_root)
        parse_result = parse_calcium_xml(pid, Path(xml_dir_str))

        # Stage-2 artefacts.
        ct = np.load(preproc_dir / f"{pid}_ct.npy")
        mask = np.load(preproc_dir / f"{pid}_mask.npy")

        # ── Feature dict starts from schema zeros ─────────────────────
        row = zero_features()
        row["pid"] = pid
        row["kernel"] = meta.kernel
        row["scanner_model"] = meta.scanner_model
        row["mask_voxels"] = int(mask.sum())
        row["low_burden_flag"] = int(mask.sum()) < LOW_BURDEN_VOXEL_THRESHOLD
        row["roundtrip_quality"] = quality_label

        # ── Canonical XML features: Agatston (D011), per-vessel
        #    aggregates, density tiers, lesion CCL (D007) + spatial ─────
        canonical, category, lesions_per_vessel = compute_canonical_features(
            parse_result, meta, excluded_set, **grouping,
        )
        row.update(canonical)
        row["category"] = category

        # ── PyRadiomics on whole-mask ─────────────────────────────────
        # Graceful degrade per D010, decided BEFORE extraction: a mask that
        # PyRadiomics' own minimumROISize / minimumROIDimensions would reject
        # keeps its XML-derived features and leaves the PyRadiomics columns
        # missing (NaN in the CSV). Anything the extractor raises (input
        # guardrail, shape mismatch, PyRadiomics bug) fails the patient
        # instead of passing as a skip.
        extractor = extractor_holder.get("e")
        if extractor is None:
            extractor = create_extractor()
            extractor_holder["e"] = extractor

        if mask.sum() == 0:
            raise ValueError("empty mask (stage 2 guarantees a non-empty mask)")
        skip_reason = roi_too_small(mask, extractor)
        if skip_reason is None:
            radiomics = extract_pyradiomics(ct, mask, target_spacing, extractor, pid=pid)
            row.update(radiomics)
            radiomics_status, radiomics_reason = "ok", "ok"
        else:
            radiomics = {}
            radiomics_status, radiomics_reason = "skipped", skip_reason

        row["radiomics_status"] = radiomics_status
        row["radiomics_reason"] = radiomics_reason

        # ── Lesion audit rows ─────────────────────────────────────────
        lesion_rows = [
            {
                "pid": pid,
                "vessel": vessel,
                "lesion_idx": i,
                "n_rois": l.n_rois,
                "slice_indices": ";".join(str(s) for s in l.slice_indices),
                "centroid_x_mm": l.centroid_mm[0],
                "centroid_y_mm": l.centroid_mm[1],
                "centroid_z_mm": l.centroid_mm[2],
                "total_area_mm2": l.total_area_mm2,
                "volume_mm3": l.volume_mm3,
                "mean_hu_weighted": l.mean_hu_weighted,
                "max_hu": l.max_hu,
            }
            for vessel, lesions in lesions_per_vessel.items()
            for i, l in enumerate(lesions)
        ]

        runtime = time.perf_counter() - t0
        log_row = {
            "pid": pid,
            "runtime_sec": round(runtime, 2),
            "n_radiomics_features": len(radiomics),
            "radiomics_status": radiomics_status,
            "radiomics_reason": radiomics_reason,
            "n_lesions": sum(len(v) for v in lesions_per_vessel.values()),
            "mask_voxels": int(mask.sum()),
            "low_burden_flag": row["low_burden_flag"],
        }

        return {"status": "ok", "row": row, "lesions": lesion_rows, "log": log_row}

    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "pid": pid,
            "error": f"{type(exc).__name__}: {exc}",
            "runtime_sec": round(time.perf_counter() - t0, 2),
        }




# Worker-process scope: cache the extractor across patients to avoid
# reinitialising it for every call.
_worker_extractor_holder: dict = {}


def _worker(args) -> dict:
    return _process_one(*args, _worker_extractor_holder)


# ───────────────────── main ─────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--n-workers", type=int, default=default_n_jobs(cap=16),
                        help="Worker processes. Defaults to the scheduler allocation "
                             "(PBS_NCPUS / PBS_NODEFILE), capped at 16 because each "
                             "worker holds a full CT volume in RAM.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N patients.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("features")

    cfg: Config = load_config(args.config)
    publish_dir = cfg.paths.outputs / "03_features"
    out_dir = fresh_partial(publish_dir)
    preproc_dir = cfg.paths.outputs / "02_preprocessed"
    # D007 lesion grouping, from the config (never the function defaults).
    grouping = grouping_from_config(cfg)

    # Write a run header for reproducibility (D-feature provenance).
    header = build_run_header(
        repo_root=cfg.paths.outputs.parent,
        params_yaml=cfg.paths.outputs.parent / "params.yaml",
        grouping=grouping,
    )
    (out_dir / "run_header.json").write_text(json.dumps(header, indent=2) + "\n",
                                              encoding="utf-8")
    log.info("Run header: %s", header)

    manifest_path = cfg.paths.outputs / "01_manifest" / "manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype={"pid": str})
    pids = manifest["pid"].astype(str).tolist()
    if args.limit:
        pids = pids[: args.limit]

    # Load preprocess report for quality labels.
    preproc_report = pd.read_csv(
        preproc_dir / "preprocess_report.csv", dtype={"pid": str},
    )
    quality_by_pid = dict(zip(preproc_report["pid"].astype(str),
                              preproc_report["roundtrip_quality"]))
    missing = [p for p in pids if p not in quality_by_pid]
    if missing:
        log.error("%d manifest patient(s) missing from stage 2 (%s); rerun stage 2.",
                  len(missing), ", ".join(missing[:10]))
        return 1

    # Load roundtrip CSV and group failures per patient.
    trips = pd.read_csv(preproc_dir / "xml_roundtrip.csv", dtype={"pid": str})
    fail_mask = (~trips["passes"].astype(bool)) & (trips["matched_via"] != "dirty")
    fails = trips[fail_mask]
    excluded_by_pid: dict[str, set] = {}
    for _, r in fails.iterrows():
        excluded_by_pid.setdefault(str(r["pid"]), set()).add(
            (int(r["image_index"]), int(r["roi_idx_in_slice"]))
        )

    target_spacing = load_spacing_metadata(preproc_dir / "spacing.json")

    log.info(
        "Features: %d patients | workers=%d | lesion_grouping=%s",
        len(pids), args.n_workers, grouping,
    )

    rows: list[dict] = []
    lesion_rows: list[dict] = []
    log_rows: list[dict] = []
    errors: list[dict] = []

    tasks = [
        (
            pid,
            str(cfg.paths.data_raw.parent),
            str(cfg.paths.data_xml),
            str(preproc_dir),
            excluded_by_pid.get(pid, set()),
            target_spacing,
            quality_by_pid[pid],
            grouping,
        )
        for pid in pids
    ]

    if args.n_workers == 1:
        for t in tqdm(tasks, desc="features"):
            _consume(_worker(t), rows, lesion_rows, log_rows, errors)
    else:
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            futures = {pool.submit(_worker, t): t[0] for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="features"):
                _consume(fut.result(), rows, lesion_rows, log_rows, errors)

    pd.DataFrame(rows).to_csv(out_dir / "features.csv", index=False)
    pd.DataFrame(lesion_rows).to_csv(out_dir / "lesions.csv", index=False)
    pd.DataFrame(log_rows).to_csv(out_dir / "feature_extraction_log.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "errors.csv", index=False)

    log.info(
        "Done. ok=%d | low_burden=%d | errors=%d | features.csv columns=%d",
        len(rows),
        sum(1 for r in rows if r.get("low_burden_flag")),
        len(errors),
        len(rows[0]) if rows else 0,
    )

    if errors:
        log.error(
            "%d patient(s) failed (%s); see %s. %s NOT updated.",
            len(errors), ", ".join(sorted(str(e["pid"]) for e in errors)),
            out_dir / "errors.csv", publish_dir,
        )
        return 1
    if args.limit:
        log.warning("--limit smoke run: outputs left in %s, not published.", out_dir)
        return 0
    publish(out_dir, publish_dir)
    log.info("Published %s", publish_dir)
    return 0


def _consume(result, rows, lesion_rows, log_rows, errors) -> None:
    if result["status"] == "ok":
        rows.append(result["row"])
        lesion_rows.extend(result["lesions"])
        log_rows.append(result["log"])
    else:
        errors.append(result)


if __name__ == "__main__":
    sys.exit(main())
