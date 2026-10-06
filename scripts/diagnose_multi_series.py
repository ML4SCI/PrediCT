#!/usr/bin/env python
"""Step 0 of D033: what are the extra series in multi-series COCA folders?

Motivation
----------
A realistic perturbation set (open item D033) needs its image-side magnitudes
(noise, reconstruction kernel) calibrated against real variation, not assumed.
About 14 COCA folders hold a second CT series with the same z positions.
If a second series is an alternate
reconstruction of the SAME acquisition on the SAME grid, the expert mask
transfers voxel for voxel and the pair is a real re-reconstruction test-retest
sample. This script finds out what those series are. It computes no radiomics
and reads no ICC output.

What it does
------------
1. Scans every patient folder under paths.data_raw. Groups CT headers by
   SeriesInstanceUID exactly as the loader does (production _read_header), and
   marks the series D006 selects (production _select_single_series).
2. For every multi-series patient, one row per series: slice count, kernel,
   description, image type, thickness, spacing, matrix, FOV, first-slice IPP,
   acquisition / series / content time, acquisition number, frame of
   reference, kVp, mA, exposure, CTDIvol.
3. For every non-selected CT series with >= 2 slices, the relation to the
   selected one: same z set? same in-plane grid? same acquisition?
4. For every alternate with >= 2 slices: the D002 XML round-trip pass rate on
   EACH series. The annotation was drawn on the series whose Max matches
   exactly, so this verifies the D006 selection; mismatches are reported
   loudly.
5. For pairs on an identical grid (same z set, matrix, spacing, in-plane IPP)
   also: identical pixels?, noise sigma per series (Immerkaer 1996, soft-tissue
   band [-100, 200] HU, median over slices), and inside the expert mask
   (production build_3d_mask on the selected series): mean / max HU, voxels
   >= 130 HU, in each.
6. Classifies each pair:
     duplicate        identical grid, identical pixels
     identical_grid   identical grid, pixels differ
     same_z_refov     same slices, different in-plane grid (FOV size/centre)
     partial_z        different z coverage
     short_series     < 2 slices (scout / dose report)
   Acquisition time is anonymised to 0 in COCA, so same_acquisition is
   "unknown" unless both times are non-zero.

Usage
-----
    python scripts/diagnose_multi_series.py

Outputs (into --out-dir, default outputs/diagnose_multi_series/):
    series_headers.csv     one row per (patient, series)
    pair_comparison.csv    one row per (patient, non-selected series)
    multi_series_summary.txt
"""
from __future__ import annotations

import argparse
import logging
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

_log = logging.getLogger("diagnose_multi_series")

SOFT_BAND = (-100.0, 200.0)
CALC_HU = 130.0
GEOM_TOL = 1e-3


# ───────────────────────── header helpers ─────────────────────────


def _g(ds, name, default=""):
    v = getattr(ds, name, default)
    if v is None:
        return default
    try:
        if not isinstance(v, (str, bytes)) and hasattr(v, "__iter__"):
            return "/".join(str(x) for x in v)
    except TypeError:
        pass
    return str(v).strip()


def _series_info(first_path: str) -> dict:
    import pydicom
    ds = pydicom.dcmread(first_path, stop_before_pixels=True)
    ps = getattr(ds, "PixelSpacing", None)
    return {
        "series_description": _g(ds, "SeriesDescription"),
        "image_type": _g(ds, "ImageType"),
        "rows": int(getattr(ds, "Rows", 0) or 0),
        "cols": int(getattr(ds, "Columns", 0) or 0),
        "pixel_spacing_row": float(ps[0]) if ps is not None else float("nan"),
        "pixel_spacing_col": float(ps[1]) if ps is not None else float("nan"),
        "recon_diameter": _g(ds, "ReconstructionDiameter"),
        "acquisition_number": _g(ds, "AcquisitionNumber"),
        "acquisition_time": _g(ds, "AcquisitionTime"),
        "series_time": _g(ds, "SeriesTime"),
        "content_time": _g(ds, "ContentTime"),
        "frame_of_reference": _g(ds, "FrameOfReferenceUID"),
        "study_uid": _g(ds, "StudyInstanceUID"),
        "kvp": _g(ds, "KVP"),
        "tube_current": _g(ds, "XRayTubeCurrent"),
        "exposure": _g(ds, "Exposure"),
        "ctdi_vol": _g(ds, "CTDIvol"),
    }


def _ipp_xy(path: str) -> tuple[float, float]:
    import pydicom
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is None:
        return (float("nan"), float("nan"))
    return (float(ipp[0]), float(ipp[1]))


def _read_volume(headers: list[dict]):
    import SimpleITK as sitk
    hs = sorted(headers, key=lambda m: (m["z_position"], m["instance_number"]))
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([m["path"] for m in hs])
    img = reader.Execute()
    return img, tuple(m["z_position"] for m in hs)


# ───────────────────────── image measures ─────────────────────────


def immerkaer_sigma(vol: np.ndarray) -> float:
    """Noise sd; the one implementation lives in predict.stability.realistic (D033 C1)."""
    from predict.stability.realistic import immerkaer_sigma as _imm
    return _imm(vol, band=SOFT_BAND)


def _roundtrip(parse_result, ct_arr, positions, tol):
    from predict.validate.xml_roundtrip import (
        failed_roi_ids, pass_rate, xml_roundtrip_check,
    )
    trips = xml_roundtrip_check(parse_result, ct_arr, positions, tolerance_mm=tol)
    n_elig = sum(1 for t in trips if t.matched_via != "dirty")
    return pass_rate(trips), n_elig, failed_roi_ids(trips)


def _mask_stats(ct: np.ndarray, mask: np.ndarray, prefix: str) -> dict:
    v = ct[mask > 0].astype(np.float64)
    return {
        f"{prefix}_mask_mean_hu": float(v.mean()) if v.size else float("nan"),
        f"{prefix}_mask_max_hu": float(v.max()) if v.size else float("nan"),
        f"{prefix}_mask_n_ge130": int((v >= CALC_HU).sum()),
    }


# ───────────────────────── per-patient worker ─────────────────────────


def _nonzero_time(v) -> bool:
    try:
        return float(v) != 0.0
    except (TypeError, ValueError):
        return bool(str(v).strip())


def _patient_worker(pid: str, data_raw: str, xml_dir: str, tol: float) -> dict:
    from predict.io.dicom_loader import (
        LoadedPatient, _read_header, _select_single_series, find_dicom_subfolder,
    )
    out = {"pid": pid, "series": [], "pairs": [], "error": ""}
    try:
        dcm_dir = find_dicom_subfolder(Path(data_raw) / pid)
        paths = [str(dcm_dir / f) for f in os.listdir(dcm_dir) if f.endswith(".dcm")]
        headers = [_read_header(p) for p in paths]
        ct = [h for h in headers if str(h["modality"]).upper() == "CT"] or headers
        groups: dict[str, list[dict]] = defaultdict(list)
        for h in ct:
            groups[h["series_uid"]].append(h)
        out["n_series"] = len(groups)
        if len(groups) < 2:
            return out

        chosen = _select_single_series(headers, pid)
        sel_uid = chosen[0]["series_uid"]

        info: dict[str, dict] = {}
        for uid, g in groups.items():
            g_sorted = sorted(g, key=lambda m: (m["z_position"], m["instance_number"]))
            si = _series_info(g_sorted[0]["path"])
            si.update({
                "pid": pid, "series_uid": uid,
                "series_number": g_sorted[0]["series_number"],
                "selected_by_d006": uid == sel_uid,
                "n_slices": len(g),
                "kernel": g_sorted[0]["kernel"],
                "slice_thickness": g_sorted[0]["slice_thickness"],
                "scanner_model": g_sorted[0]["scanner_model"],
                "z_min": g_sorted[0]["z_position"], "z_max": g_sorted[-1]["z_position"],
            })
            si["ipp_x"], si["ipp_y"] = _ipp_xy(g_sorted[0]["path"])
            info[uid] = si
            out["series"].append(si)

        sel = info[sel_uid]
        sel_z = sorted(round(h["z_position"], 3) for h in groups[sel_uid])
        sel_vol = None
        mask = None
        for uid, g in groups.items():
            if uid == sel_uid:
                continue
            alt = info[uid]
            row = {
                "pid": pid, "selected_uid": sel_uid, "alt_uid": uid,
                "sel_series_number": sel["series_number"],
                "alt_series_number": alt["series_number"],
                "sel_n": sel["n_slices"], "alt_n": alt["n_slices"],
                "sel_kernel": sel["kernel"], "alt_kernel": alt["kernel"],
                "sel_desc": sel["series_description"],
                "alt_desc": alt["series_description"],
            }
            if len(g) < 2:
                row["relation"] = "short_series"
                out["pairs"].append(row)
                continue
            alt_z = sorted(round(h["z_position"], 3) for h in g)
            same_z = alt_z == sel_z
            same_inplane = (
                sel["rows"] == alt["rows"] and sel["cols"] == alt["cols"]
                and abs(sel["pixel_spacing_row"] - alt["pixel_spacing_row"]) < GEOM_TOL
                and abs(sel["pixel_spacing_col"] - alt["pixel_spacing_col"]) < GEOM_TOL
                and abs(sel["ipp_x"] - alt["ipp_x"]) < GEOM_TOL
                and abs(sel["ipp_y"] - alt["ipp_y"]) < GEOM_TOL
            )
            # Acquisition time is anonymised to 0 in COCA, so it cannot tell
            # acquisitions apart; only a non-zero time counts as evidence.
            t_sel, t_alt = sel["acquisition_time"], alt["acquisition_time"]
            if _nonzero_time(t_sel) and _nonzero_time(t_alt):
                same_acq = "yes" if t_sel == t_alt else "no"
            else:
                same_acq = "unknown"
            row.update({
                "same_z": same_z, "same_inplane": same_inplane,
                "same_acquisition": same_acq,
                "sel_spacing": sel["pixel_spacing_row"],
                "alt_spacing": alt["pixel_spacing_row"],
                "ipp_dx": alt["ipp_x"] - sel["ipp_x"],
                "ipp_dy": alt["ipp_y"] - sel["ipp_y"],
            })

            import SimpleITK as sitk
            from predict.io.xml_parser import parse_calcium_xml
            from predict.preprocess.mask_builder import build_3d_mask
            if sel_vol is None:
                sel_img, sel_pos = _read_volume(groups[sel_uid])
                sel_vol = sitk.GetArrayFromImage(sel_img)
                parse = parse_calcium_xml(pid, Path(xml_dir))
                sel_rate, n_elig, excluded = _roundtrip(parse, sel_vol, sel_pos, tol)
            alt_img, alt_pos = _read_volume(g)
            alt_vol = sitk.GetArrayFromImage(alt_img)
            # D002 round-trip on BOTH series, for every alternate with >= 2
            # slices: XML Max matches exactly only on the series the annotator
            # drew on. This is the check that the D006 selection was right.
            alt_rate, _n, _ex = _roundtrip(parse, alt_vol, alt_pos, tol)
            row.update({
                "roundtrip_n_rois": n_elig,
                "sel_roundtrip_pass": sel_rate,
                "alt_roundtrip_pass": alt_rate,
                "annotated_on": ("selected" if sel_rate > alt_rate else
                                 "alternate" if alt_rate > sel_rate else "tie"),
            })
            if not same_z:
                row["relation"] = "partial_z"
            elif not same_inplane:
                # Same slices, same kernel, different FOV size or centre: a
                # re-reconstruction at another in-plane grid.
                row["relation"] = "same_z_refov"
            else:
                if mask is None:
                    loaded = LoadedPatient(
                        pid=pid, ct_sitk=sel_img, slice_positions=sel_pos,
                        pixel_spacing=sel_img.GetSpacing(),
                        slice_thickness=sel["slice_thickness"] or sel_img.GetSpacing()[2],
                        n_slices=len(sel_pos), scanner_model=sel["scanner_model"],
                        kernel=sel["kernel"], manufacturer="", series_uid=sel_uid,
                    )
                    mask, _rep = build_3d_mask(parse, loaded, tolerance_mm=tol,
                                               excluded_roi_ids=excluded)
                    sel_sigma = immerkaer_sigma(sel_vol)
                identical = bool(np.array_equal(sel_vol, alt_vol))
                d = alt_vol.astype(np.float64) - sel_vol.astype(np.float64)
                body = (sel_vol >= SOFT_BAND[0]) & (sel_vol <= SOFT_BAND[1])
                row.update({
                    "identical_pixels": identical,
                    "sel_noise_sigma": sel_sigma,
                    "alt_noise_sigma": sel_sigma if identical else immerkaer_sigma(alt_vol),
                    "softband_diff_mean": float(d[body].mean()),
                    "softband_diff_sd": float(d[body].std()),
                    "mask_voxels": int(mask.sum()),
                })
                row.update(_mask_stats(sel_vol, mask, "sel"))
                row.update(_mask_stats(alt_vol, mask, "alt"))
                row["relation"] = "duplicate" if identical else "identical_grid"
            out["pairs"].append(row)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


# ───────────────────────── report ─────────────────────────


def _pid_key(p) -> tuple:
    s = str(p)
    return (0, int(s), "") if s.isdigit() else (1, 0, s)


def render(n_scanned: int, n_multi: int, pairs: pd.DataFrame, errors: list) -> str:
    L: list[str] = []
    add = L.append
    add("=" * 88)
    add("MULTI-SERIES DIAGNOSTIC (D033 step 0; headers + pixels, no radiomics, no ICC)")
    add("=" * 88)
    add(f"patient folders scanned {n_scanned}   with >= 2 CT series {n_multi}   "
        f"errors {len(errors)}")
    if pairs.empty:
        add("no non-selected series found")
        return "\n".join(L)
    add("")
    add("relation counts (one row per non-selected series):")
    for rel, n in pairs["relation"].value_counts().items():
        add(f"  {rel:<22}{n:>4}")
    full = pairs[pairs["relation"] != "short_series"]
    if not full.empty:
        add("")
        add("D006 selection check (D002 round-trip on every series with >= 2 slices):")
        add(f"{'pid':>5} {'relation':<16}{'spacing sel/alt':>18}{'IPP dx,dy mm':>16}"
            f"{'n_roi':>7}{'rt sel/alt':>14}  annotated_on")
        for _, r in full.sort_values("pid", key=lambda s: s.map(_pid_key)).iterrows():
            add(f"{r['pid']:>5} {r['relation']:<16}"
                f"{r['sel_spacing']:>9.3f}/{r['alt_spacing']:<8.3f}"
                f"{r['ipp_dx']:>7.1f},{r['ipp_dy']:<7.1f}"
                f"{r['roundtrip_n_rois']:>7.0f}"
                f"{r['sel_roundtrip_pass']:>7.2f}/{r['alt_roundtrip_pass']:<6.2f}  "
                f"{r['annotated_on']}")
        wrong = full[full["annotated_on"] != "selected"]
        add("")
        if len(wrong):
            add(f"WARNING: {len(wrong)} series where the XML does NOT match the selected "
                f"series better: pids {sorted(wrong['pid'].tolist(), key=_pid_key)}. "
                "Inspect before trusting D006 for them.")
        else:
            add("D006 check PASSED: on every multi-series patient the XML matches the "
                "selected series better than every alternate.")
        n_ref = int((full["relation"] == "same_z_refov").sum())
        add(f"same_z_refov (same slices, other FOV grid; candidate real re-recon "
            f"pairs via physical-space mask transfer): {n_ref}")
    grid = pairs[pairs["relation"].isin(["duplicate", "identical_grid"])]
    if not grid.empty:
        add("")
        add(f"{'pid':>5} {'relation':<16}{'sigma sel/alt':>15}"
            f"{'maskHU sel/alt':>17}{'>=130 sel/alt':>16}")
        for _, r in grid.iterrows():
            add(f"{r['pid']:>5} {r['relation']:<16}"
                f"{r['sel_noise_sigma']:>7.1f}/{r['alt_noise_sigma']:<7.1f}"
                f"{r['sel_mask_mean_hu']:>8.0f}/{r['alt_mask_mean_hu']:<8.0f}"
                f"{r['sel_mask_n_ge130']:>7.0f}/{r['alt_mask_n_ge130']:<8.0f}")
    if errors:
        add("")
        add("errors:")
        for e in errors[:20]:
            add(f"  pid {e['pid']}: {e['error']}")
    add("=" * 88)
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("outputs/diagnose_multi_series"))
    ap.add_argument("--n-jobs", type=int, default=None,
                    help="default: predict.config.default_n_jobs(cap=16)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    from predict.config import default_n_jobs, load_config
    cfg = load_config()
    data_raw = cfg.paths.data_raw
    tol = cfg.resample.slice_mm / 2.0  # same tolerance as stage 2
    pids = sorted((p.name for p in data_raw.iterdir() if p.is_dir()),
                  key=lambda s: (len(s), s))
    n_jobs = args.n_jobs or default_n_jobs(cap=16)
    _log.info("scanning %d folders under %s, n_jobs %d", len(pids), data_raw, n_jobs)

    results = []
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futs = [pool.submit(_patient_worker, p, str(data_raw),
                            str(cfg.paths.data_xml), tol) for p in pids]
        for i, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if i % 50 == 0 or i == len(futs):
                _log.info("folders done %d/%d", i, len(futs))

    series = pd.DataFrame([s for r in results for s in r["series"]])
    pairs = pd.DataFrame([p for r in results for p in r["pairs"]])
    errors = [r for r in results if r["error"]]
    n_multi = sum(1 for r in results if r.get("n_series", 0) >= 2)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    series.to_csv(args.out_dir / "series_headers.csv", index=False)
    pairs.to_csv(args.out_dir / "pair_comparison.csv", index=False)
    report = render(len(pids), n_multi, pairs, errors)
    (args.out_dir / "multi_series_summary.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    _log.info("wrote %s", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
