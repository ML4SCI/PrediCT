#!/usr/bin/env python
"""Slice-geometry and series-consistency audit (header only, no pixels).

For every patient folder that has both DICOM and XML (before any exclusion),
reads all DICOM headers, applies the D006 series selection used by the
loaders, and reports:

  * z regularity of the selected series: sorted ImagePositionPatient[2]
    differences, their median, every difference that deviates from the
    median by more than ``--tol`` mm, the estimated number of missing
    slices, duplicate z, and the uniform spacing SimpleITK would assign
    ((z_last - z_first) / (n - 1)) versus SliceThickness;
  * orientation: ImageOrientationPatient of the selected series (non-axial
    or tilted series are flagged);
  * in-plane grid (D039.1): distinct PixelSpacing and matrix values and the
    largest (x, y) origin deviation from the first slice, per series;
  * calcium exposure: XML ROIs whose z lies inside an irregular gap span;
  * discovery consistency: the single header discovery read before D039
    (the lexicographically first .dcm) versus the selected series, field by
    field (manufacturer, model, kernel, thickness, series UID). Kept as the
    audit record for that change; discovery now uses the selected series.

Writes ``outputs/checks/slice_geometry/per_patient.csv`` and prints a
summary. Diagnostic only; changes nothing.

Usage::

    python scripts/diagnose_slice_geometry.py [--config ...] [--tol 0.05]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from joblib import Parallel, delayed

from predict.config import default_n_jobs, load_config
from predict.io.dicom_loader import (
    _read_header,
    _select_single_series,
    find_dicom_subfolder,
)
from predict.io.xml_parser import parse_calcium_xml

AXIAL = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def _roi_z(roi) -> float | None:
    if roi.points_mm:
        return float(np.mean([p[2] for p in roi.points_mm]))
    if roi.center_xyz is not None:
        return float(roi.center_xyz[2])
    return None


def _first_file_header(dcm_dir: Path) -> dict:
    """What discovery read before D039: the lexicographically first .dcm."""
    first = sorted(f for f in os.listdir(dcm_dir) if f.endswith(".dcm"))[0]
    return _read_header(str(dcm_dir / first))


def _audit(pid: str, data_root: Path, xml_dir: Path, tol: float) -> dict:
    row: dict = {"pid": pid, "error": ""}
    try:
        dcm_dir = find_dicom_subfolder(data_root / "raw" / pid)
        paths = [str(dcm_dir / f) for f in os.listdir(dcm_dir) if f.endswith(".dcm")]
        all_h = [_read_header(p) for p in paths]
        sel = _select_single_series(all_h, pid)
        sel.sort(key=lambda m: (m["z_position"], m["instance_number"]))
        z = np.array([m["z_position"] for m in sel], dtype=float)
        n = len(z)
        row.update(
            n_files=len(all_h),
            n_series=len({m["series_uid"] for m in all_h}),
            n_slices=n,
            sel_series_uid=sel[0]["series_uid"],
        )

        d = np.diff(z) if n > 1 else np.array([])
        med = float(np.median(d)) if d.size else float("nan")
        irregular = np.flatnonzero(np.abs(d - med) > tol) if d.size else np.array([], int)
        missing = int(sum(max(0, int(round(d[i] / med)) - 1) for i in irregular)) if med > 0 else 0
        thick = sorted({m["slice_thickness"] for m in sel if m["slice_thickness"]})
        row.update(
            dz_median=med,
            dz_min=float(d.min()) if d.size else float("nan"),
            dz_max=float(d.max()) if d.size else float("nan"),
            n_irregular_gaps=int(irregular.size),
            irregular_gaps_mm=";".join(f"{d[i]:.3f}" for i in irregular),
            n_missing_est=missing,
            n_duplicate_z=int((d < 1e-3).sum()) if d.size else 0,
            sitk_implied_dz=float((z[-1] - z[0]) / (n - 1)) if n > 1 else float("nan"),
            slice_thickness=";".join(f"{t:g}" for t in thick),
        )
        row["thickness_ne_dz"] = bool(thick and med > 0 and abs(thick[0] - med) > tol)
        xy = np.array([m["ipp_xy"] for m in sel if m["ipp_xy"] is not None], dtype=float)
        row.update(
            n_pixel_spacings=len({m["pixel_spacing"] for m in sel}),
            pixel_spacings=";".join(sorted({str(m["pixel_spacing"]) for m in sel})),
            n_matrices=len({m["matrix"] for m in sel}),
            ipp_xy_max_dev=float(np.abs(xy - xy[0]).max()) if len(xy) else float("nan"),
        )

        ds = pydicom.dcmread(sel[0]["path"], stop_before_pixels=True)
        iop = getattr(ds, "ImageOrientationPatient", None)
        if iop is None:
            row.update(iop="", non_axial=True)
        else:
            v = np.array([float(x) for x in iop])
            row.update(iop=";".join(f"{x:.4f}" for x in v),
                       non_axial=bool(np.abs(v - AXIAL).max() > 1e-3))

        # Calcium ROIs inside an irregular gap span [z_i, z_{i+1}].
        n_roi, n_roi_gap = 0, 0
        try:
            parse = parse_calcium_xml(pid, xml_dir)
            spans = [(z[i], z[i + 1]) for i in irregular]
            for s in parse.slices:
                for roi in s.rois:
                    rz = _roi_z(roi)
                    if rz is None:
                        continue
                    n_roi += 1
                    if any(lo - tol <= rz <= hi + tol for lo, hi in spans):
                        n_roi_gap += 1
        except Exception as e:  # XML problems are reported, not fatal
            row["error"] = f"xml: {e}"
        row.update(n_rois=n_roi, n_rois_in_gap_span=n_roi_gap)

        peek = _first_file_header(dcm_dir)
        head = sel[0]
        for f in ("manufacturer", "scanner_model", "kernel", "slice_thickness", "series_uid"):
            row[f"peek_{f}"] = peek[f]
            row[f"sel_{f}"] = head[f]
            row[f"mismatch_{f}"] = peek[f] != head[f]
    except Exception as e:
        row["error"] = str(e)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--tol", type=float, default=0.05,
                    help="mm deviation from the median z step that counts as irregular")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_root = cfg.paths.data_raw.parent
    raw_dir, xml_dir = cfg.paths.data_raw, cfg.paths.data_xml
    raw_ids = {n for n in os.listdir(raw_dir) if (raw_dir / n).is_dir()}
    xml_ids = {n[:-4] for n in os.listdir(xml_dir) if n.endswith(".xml")}
    pids = sorted(raw_ids & xml_ids, key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else 0, s))

    manifest_path = cfg.paths.outputs / "01_manifest" / "manifest.csv"
    in_manifest = set()
    if manifest_path.exists():
        in_manifest = set(pd.read_csv(manifest_path, dtype={"pid": str})["pid"])

    rows = Parallel(n_jobs=default_n_jobs())(
        delayed(_audit)(p, data_root, xml_dir, args.tol) for p in pids
    )
    df = pd.DataFrame(rows)
    df["in_manifest"] = df["pid"].isin(in_manifest)

    out = cfg.paths.outputs / "checks" / "slice_geometry"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_patient.csv", index=False)

    ok = df[df["error"].fillna("") == ""]
    m = ok[ok["in_manifest"]]
    print(f"folders audited: {len(df)}  errors: {len(df) - len(ok)}  in manifest: {len(m)}")
    for _, r in df[df["error"].fillna("") != ""].iterrows():
        print(f"  ERROR pid {r['pid']}: {r['error']}")

    print(f"\n[1] z regularity (tol {args.tol} mm), manifest patients")
    print(f"  dz_median values: {m['dz_median'].round(3).value_counts().to_dict()}")
    irr = m[m["n_irregular_gaps"] > 0]
    print(f"  irregular: {len(irr)} / {len(m)}")
    cols = ["pid", "n_slices", "dz_median", "irregular_gaps_mm", "n_missing_est",
            "sitk_implied_dz", "n_rois", "n_rois_in_gap_span"]
    if len(irr):
        print(irr[cols].to_string(index=False))
    print(f"  duplicate z: {int((m['n_duplicate_z'] > 0).sum())}")
    print(f"  thickness != dz_median: {int(m['thickness_ne_dz'].sum())}")
    print(f"  non-axial / tilted: {int(m['non_axial'].sum())}")
    mixed = m[(m["n_pixel_spacings"] > 1) | (m["n_matrices"] > 1) | (m["ipp_xy_max_dev"] > 0.05)]
    print(f"  mixed in-plane grid (D039.1): {len(mixed)}")
    if len(mixed):
        print(mixed[["pid", "pixel_spacings", "n_matrices", "ipp_xy_max_dev"]].to_string(index=False))
    rest = m.drop(mixed.index)["ipp_xy_max_dev"]
    print(f"  (x, y) origin max deviation among the rest: max {rest.max():.4g} mm, "
          f"n > 0: {int((rest > 0).sum())}")
    if m["non_axial"].any():
        print(m.loc[m["non_axial"], ["pid", "iop"]].to_string(index=False))

    print("\n[2] discovery header vs selected series, all audited folders")
    for f in ("manufacturer", "scanner_model", "kernel", "slice_thickness", "series_uid"):
        bad = ok[ok[f"mismatch_{f}"]]
        print(f"  {f}: {len(bad)} mismatch")
        if len(bad) and f != "series_uid":
            print(bad[["pid", f"peek_{f}", f"sel_{f}", "in_manifest"]].to_string(index=False))
    print(f"\nwrote {out / 'per_patient.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
