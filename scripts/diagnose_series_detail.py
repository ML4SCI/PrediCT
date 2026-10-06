#!/usr/bin/env python
"""Per-series detail for patients flagged by diagnose_slice_geometry.py.

Question it answers: are the slices missing from an irregular selected
series present in the folder under another series (an export split), and are
those slices on the same grid (orientation, pixel spacing, in-plane origin,
matrix, kernel, thickness) so they belong to the same acquisition?

Default patients: every row of outputs/checks/slice_geometry/per_patient.csv
with an irregular gap or SliceThickness != median z step. Override with
--pids. Prints only; writes nothing.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

from predict.config import load_config
from predict.io.dicom_loader import _select_single_series, _read_header, find_dicom_subfolder
from predict.io.xml_parser import parse_calcium_xml


def _grid(path: str) -> dict:
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    g = lambda k, d="": getattr(ds, k, d)
    ipp = [float(x) for x in g("ImagePositionPatient", [np.nan] * 3)]
    return {
        "iop": tuple(round(float(x), 4) for x in g("ImageOrientationPatient", [])),
        "spacing": tuple(round(float(x), 5) for x in g("PixelSpacing", [])),
        "xy": (round(ipp[0], 3), round(ipp[1], 3)),
        "matrix": (int(g("Rows", 0)), int(g("Columns", 0))),
        "image_type": "\\".join(str(x) for x in g("ImageType", [])),
        "has_uid": hasattr(ds, "SeriesInstanceUID"),
    }


def _fmt_z(z: list[float]) -> str:
    z = sorted(z)
    return ", ".join(f"{v:.2f}" for v in z) if len(z) <= 12 else (
        f"{z[0]:.2f} .. {z[-1]:.2f} ({len(z)} values)")


def _detail(pid: str, data_root: Path, xml_dir: Path) -> None:
    dcm_dir = find_dicom_subfolder(data_root / "raw" / pid)
    hdrs = [_read_header(str(dcm_dir / f)) for f in os.listdir(dcm_dir) if f.endswith(".dcm")]
    sel = _select_single_series(list(hdrs), pid)
    sel_paths = {m["path"] for m in sel}
    zs = np.sort([m["z_position"] for m in sel])
    dz = float(np.median(np.diff(zs)))
    sel_grid = _grid(sel[0]["path"])

    # Expected positions on the selected series' own regular grid.
    expected = zs[0] + dz * np.arange(int(round((zs[-1] - zs[0]) / dz)) + 1)
    missing = [e for e in expected if np.abs(zs - e).min() > 0.05]

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for m in hdrs:
        groups[(m["series_uid"], m["series_number"], m["path"] in sel_paths)].append(m)

    print(f"\n=== pid {pid}: {len(hdrs)} files, {len(groups)} group(s); selected "
          f"{len(sel)} slices, dz {dz:g}, span {zs[0]:.2f}..{zs[-1]:.2f}")
    print(f"  selected grid: {sel_grid}")
    print(f"  missing z on the selected grid ({len(missing)}): {_fmt_z(missing)}")
    for (uid, snum, is_sel), g in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        z = [m["z_position"] for m in g]
        gr = _grid(g[0]["path"])
        tag = "SELECTED" if is_sel else "alternate"
        print(f"  [{tag}] series #{snum} uid={'<missing>' if not uid else uid[-12:]} "
              f"n={len(g)} kernel={g[0]['kernel']} thick={g[0]['slice_thickness']} "
              f"modality={g[0]['modality']}")
        print(f"      image_type={gr['image_type']}  has_uid={gr['has_uid']}")
        print(f"      z: {_fmt_z(z)}")
        if not is_sel:
            fills = sum(1 for v in z if missing and min(abs(v - e) for e in missing) <= 0.05)
            dup = sum(1 for v in z if np.abs(zs - v).min() <= 0.05)
            same = {k: gr[k] == sel_grid[k] for k in ("iop", "spacing", "xy", "matrix")}
            print(f"      fills a missing z: {fills}/{len(z)}   duplicates a selected z: {dup}/{len(z)}")
            print(f"      same as selected: {same}  kernel_same={g[0]['kernel'] == sel[0]['kernel']}")

    try:
        parse = parse_calcium_xml(pid, xml_dir)
        rz = [float(np.mean([p[2] for p in r.points_mm])) for s in parse.slices for r in s.rois if r.points_mm]
        on_missing = sum(1 for v in rz if missing and min(abs(v - e) for e in missing) <= 0.5)
        print(f"  XML ROIs: {len(rz)}; on a missing z (within 0.5 mm): {on_missing}")
    except Exception as e:
        print(f"  XML: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--pids", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    pids = args.pids
    if not pids:
        df = pd.read_csv(cfg.paths.outputs / "checks" / "slice_geometry" / "per_patient.csv",
                         dtype={"pid": str})
        flag = (df["n_irregular_gaps"] > 0) | (df["thickness_ne_dz"] == True)  # noqa: E712
        print(df.loc[flag, ["pid", "n_slices", "dz_median", "slice_thickness",
                            "n_irregular_gaps", "in_manifest"]].to_string(index=False))
        pids = df.loc[flag, "pid"].tolist()
    for pid in pids:
        _detail(pid, cfg.paths.data_raw.parent, cfg.paths.data_xml)
    return 0


if __name__ == "__main__":
    sys.exit(main())
