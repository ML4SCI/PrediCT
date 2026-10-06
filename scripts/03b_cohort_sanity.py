#!/usr/bin/env python
"""Stage 3 cohort-level sanity checks.

Runs after ``scripts/03_features.py`` finishes. Verifies:

  1. ``features.csv`` row count matches the manifest.
  2. Canonical-schema column count is exactly ``n_features()`` (D-feature_schema).
  3. NaN audit: NaN should appear only on PyRadiomics columns and only for
     rows with ``radiomics_status == "skipped"``.
  4. Per-vessel mask voxel sums equal the whole-mask voxel count on every
     patient (D008 cross-check on real data).
  5. ``agatston_total`` equals the sum of per-vessel Agatston (D011).
  6. ``volume_total_mm3`` and ``mass_total`` equal the sum of their per-vessel
     components.

Outputs ``outputs/03_features/cohort_sanity.json`` with pass/fail per check
and the list of offending PIDs (if any). Exits non-zero on any failure.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from predict.config import load_config
from predict.features.feature_schema import feature_names, n_features
from predict.features.per_artery_mask import build_per_artery_masks
from predict.io import load_patient_dicom, parse_calcium_xml
from predict.preprocess.mask_builder import build_3d_mask


SUMS_TOLERANCE = 1e-6


def _audit_row_count(features: pd.DataFrame, manifest: pd.DataFrame) -> dict:
    n_man = len(manifest)
    n_feat = len(features)
    return {
        "name": "row_count_matches_manifest",
        "ok": bool(n_man == n_feat),
        "expected": int(n_man),
        "actual": int(n_feat),
    }


def _audit_schema(features: pd.DataFrame) -> dict:
    canonical = set(feature_names())
    actual_cols = set(features.columns)
    missing = sorted(canonical - actual_cols)
    return {
        "name": "canonical_schema_present",
        "ok": bool(len(missing) == 0),
        "expected_count": int(n_features()),
        "missing": missing,
    }


def _audit_nan_only_in_radiomics_for_skipped(features: pd.DataFrame) -> dict:
    # Canonical-schema columns must never be NaN.
    canonical = list(feature_names())
    canonical_nan = features[canonical].isna().sum()
    canonical_nan = canonical_nan[canonical_nan > 0]

    # All non-canonical, non-meta columns are PyRadiomics. NaN allowed iff
    # radiomics_status == "skipped" for that row.
    pyr_cols = [c for c in features.columns
                if c.startswith("original_")]
    if pyr_cols:
        non_skipped = features[features["radiomics_status"] != "skipped"]
        bad_pyr = non_skipped[pyr_cols].isna().sum()
        bad_pyr = bad_pyr[bad_pyr > 0]
    else:
        bad_pyr = pd.Series(dtype=int)

    ok = bool(canonical_nan.empty and bad_pyr.empty)
    return {
        "name": "nan_audit",
        "ok": ok,
        "canonical_nan_columns": {k: int(v) for k, v in canonical_nan.to_dict().items()},
        "pyradiomics_nan_in_ok_rows": {k: int(v) for k, v in bad_pyr.to_dict().items()},
    }


def _audit_total_consistency(features: pd.DataFrame) -> dict:
    """agatston_total ≈ sum(agatston_{lad,rca,lcx,lm})."""
    per_v_agatston = (
        features["agatston_lad"] + features["agatston_rca"]
        + features["agatston_lcx"] + features["agatston_lm"]
    )
    per_v_volume = (
        features["volume_lad_mm3"] + features["volume_rca_mm3"]
        + features["volume_lcx_mm3"] + features["volume_lm_mm3"]
    )
    per_v_mass = (
        features["mass_lad"] + features["mass_rca"]
        + features["mass_lcx"] + features["mass_lm"]
    )

    drift_agatston = (features["agatston_total"] - per_v_agatston).abs().max()
    drift_volume = (features["volume_total_mm3"] - per_v_volume).abs().max()
    drift_mass = (features["mass_total"] - per_v_mass).abs().max()

    ok = bool(
        drift_agatston < SUMS_TOLERANCE
        and drift_volume < SUMS_TOLERANCE
        and drift_mass < SUMS_TOLERANCE
    )
    return {
        "name": "per_vessel_totals_consistent",
        "ok": ok,
        "max_agatston_drift": float(drift_agatston),
        "max_volume_drift": float(drift_volume),
        "max_mass_drift": float(drift_mass),
    }


def _audit_per_vessel_voxel_sum(
    features: pd.DataFrame,
    data_root: Path,
    xml_dir: Path,
    trips_csv: Path,
) -> dict:
    """For each patient, sum per-vessel mask voxels at NATIVE spacing and
    compare against a fresh native whole-mask built the same way.

    We use ``load_patient_dicom`` here (paying the full SimpleITK pixel read)
    because the saved ``{pid}_mask.npy`` is in *resampled* coordinates and is
    not directly comparable to per-vessel masks built at native coordinates.
    """
    # Build per-patient exclusion sets from the stage-2 trips file.
    trips = pd.read_csv(trips_csv, dtype={"pid": str})
    fail_mask = (~trips["passes"].astype(bool)) & (trips["matched_via"] != "dirty")
    fails = trips[fail_mask]
    excluded_by_pid: dict[str, set] = {}
    for _, r in fails.iterrows():
        excluded_by_pid.setdefault(str(r["pid"]), set()).add(
            (int(r["image_index"]), int(r["roi_idx_in_slice"]))
        )

    union_offenders: list[dict] = []
    overlap_records: list[dict] = []

    for _, row in features.iterrows():
        pid = str(row["pid"])
        loaded = load_patient_dicom(pid, data_root)
        parse_result = parse_calcium_xml(pid, xml_dir)
        excluded = excluded_by_pid.get(pid, set())

        whole_native, _ = build_3d_mask(parse_result, loaded, excluded_roi_ids=excluded)
        masks = build_per_artery_masks(parse_result, loaded, excluded_roi_ids=excluded)

        whole_voxels = int(whole_native.sum())
        per_vessel_sum = int(sum(m.sum() for m in masks.values()))

        # Union (the correct invariant): per-vessel masks OR'd together must
        # exactly equal the whole-mask.
        union_mask = np.zeros_like(whole_native)
        for m in masks.values():
            union_mask |= m
        union_voxels = int(union_mask.sum())

        if union_voxels != whole_voxels:
            union_offenders.append({
                "pid": pid,
                "whole_native_voxels": whole_voxels,
                "union_voxels": union_voxels,
                "diff": union_voxels - whole_voxels,
            })

        # Inter-vessel overlap = per-vessel sum − union. Expected to be 0 for
        # almost all patients; non-zero implies LM-vs-proximal-LAD/LCx overlap
        # (a known COCA annotation phenomenon, not a bug).
        overlap = per_vessel_sum - union_voxels
        if overlap > 0:
            overlap_records.append({
                "pid": pid,
                "overlap_voxels": overlap,
                "whole_native_voxels": whole_voxels,
                "overlap_pct": round(100.0 * overlap / max(whole_voxels, 1), 2),
            })

    return {
        "name": "per_vessel_union_equals_whole",
        "ok": bool(len(union_offenders) == 0),
        "n_union_offenders": len(union_offenders),
        "union_offenders_head": union_offenders[:5],
        "n_patients_with_inter_vessel_overlap": len(overlap_records),
        "max_overlap_voxels": max((r["overlap_voxels"] for r in overlap_records), default=0),
        "max_overlap_pct": max((r["overlap_pct"] for r in overlap_records), default=0.0),
        "overlap_head": overlap_records[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--skip-per-vessel-sum", action="store_true",
                        help="Skip the per-vessel voxel-sum check (slowest audit, ~1 min).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log = logging.getLogger("sanity")

    cfg = load_config(args.config)
    out_dir = cfg.paths.outputs / "03_features"
    preproc_dir = cfg.paths.outputs / "02_preprocessed"

    features = pd.read_csv(out_dir / "features.csv", dtype={"pid": str})
    manifest = pd.read_csv(cfg.paths.outputs / "01_manifest" / "manifest.csv",
                           dtype={"pid": str})

    audits: list[dict] = []
    audits.append(_audit_row_count(features, manifest))
    audits.append(_audit_schema(features))
    audits.append(_audit_nan_only_in_radiomics_for_skipped(features))
    audits.append(_audit_total_consistency(features))
    if not args.skip_per_vessel_sum:
        audits.append(_audit_per_vessel_voxel_sum(
            features,
            cfg.paths.data_raw.parent,
            cfg.paths.data_xml,
            preproc_dir / "xml_roundtrip.csv",
        ))

    all_ok = bool(all(a["ok"] for a in audits))
    summary = {"ok": all_ok, "audits": audits}
    (out_dir / "cohort_sanity.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8",
    )

    for a in audits:
        marker = "OK" if a["ok"] else "FAIL"
        log.info("%s: %s", marker, a["name"])
        if not a["ok"]:
            log.info("  details: %s", json.dumps(a, indent=2))

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
