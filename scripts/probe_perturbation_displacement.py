#!/usr/bin/env python
"""Geometry probe: how far does each D014 perturbation move the calcium?

Motivation
----------
The perturbation-subset probe showed the four +/-5 mm translations dominate
the ICC gate. Dropping them on that result would be choosing a rule after
seeing its outcome. D033 (open) instead proposes a geometry-based rule, e.g.
"keep perturbations whose median lesion displacement <= median lesion
radius". This script produces the geometry that rule would be locked on.

It reads NO ICC output (nothing under outputs/05_icc or
outputs/probe_perturbation_subsets). The numbers here depend only on lesion
positions, lesion sizes, volume geometry and the locked perturbation specs.

What it measures
----------------
1. Per lesion x spatial perturbation (lesions.csv, D007 lesions):
     rotation theta about the volume centre: chord = 2 r sin(|theta| / 2),
       r = in-plane distance from the rotation centre (production
       _volume_center_physical, so the centre is exactly the one stage 4 uses);
     translation: |t| (z is never translated, D014).
   Two lesion radius definitions, reported side by side (neither is chosen
   here; D033 must pick one before any ICC is looked at):
     r_inplane = sqrt(mean per-slice area / pi)   (equivalent circle; the
                 perturbations are in-plane, so this is the relevant extent)
     r_sphere  = (3 V / (4 pi)) ** (1/3)
2. Per patient x perturbation (preprocessed mask, unless --skip-retention):
     retained = |M and T(M)| / |M|, where T is the production rotate/translate
     applied to the mask. This is the fraction of mask voxels that still hold
     calcium after the CT moves under the fixed mask: exactly what the
     fixed-mask features see.
3. Frame check: fraction of lesion centroids within 1 mm of the patient's
   mask footprint (max projection over z). Lesion centroids come from XML
   pixel coords x native spacing; the preprocessed grid shares the native
   origin (resampling.py:61), so both are in one frame. A low hit rate means
   the frames disagree and part 1 is invalid; the script stops below 90%.

Self-check (always runs first): a small off-centre blob is rotated with the
production rotate(); its measured centroid shift must match the chord formula
within one voxel, and translate() must shift it by |t|. If not, stop.

Usage
-----
    python scripts/probe_perturbation_displacement.py                 # full
    python scripts/probe_perturbation_displacement.py --skip-retention  # fast, CSV only

Outputs (into --out-dir, default outputs/probe_perturbation_displacement/):
    lesion_displacement.csv    one row per (lesion, perturbation)
    patient_retention.csv      one row per (patient, perturbation)
    perturbation_summary.csv   one row per perturbation
    displacement_summary.txt   human-readable report
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parents[1]
_log = logging.getLogger("probe_perturbation_displacement")

FRAME_TOL_MM = 1.0
FRAME_MIN_HIT = 0.90


def _load_script_module(rel_path: str, alias: str):
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


# Eligible cohort (D015) from the production loader. It reads features.csv
# only; nothing ICC-derived is touched.
_gate = _load_script_module("scripts/05_icc_gate.py", "_prod_icc_gate")


# ───────────────────────── geometry ─────────────────────────


def spec_displacement_mm(spec, r_mm: np.ndarray) -> np.ndarray:
    """In-plane displacement of points at distance r_mm from the rotation centre."""
    if spec.kind == "rotate":
        theta = math.radians(abs(spec.rotation_deg))
        return 2.0 * r_mm * math.sin(theta / 2.0)
    if spec.kind == "translate":
        return np.full_like(r_mm, math.hypot(spec.tx_mm, spec.ty_mm), dtype=float)
    return np.zeros_like(r_mm, dtype=float)


def _centre_xy(shape_zyx: tuple[int, int, int], spacing: tuple[float, float, float]):
    """Rotation centre (x, y) in mm, via the production helper."""
    from predict.stability.perturbations import _volume_center_physical
    nz, ny, nx = shape_zyx
    img = sitk.Image([int(nx), int(ny), int(nz)], sitk.sitkUInt8)
    img.SetSpacing(spacing)
    c = _volume_center_physical(img)
    return float(c[0]), float(c[1])


def _move_mask(mask_img: sitk.Image, spec) -> np.ndarray:
    """Apply the production spatial transform to the mask; return moved bool array."""
    from predict.stability.perturbations import rotate, translate
    if spec.kind == "rotate":
        out = rotate(mask_img, spec.rotation_deg, background_hu=0.0)
    elif spec.kind == "translate":
        out = translate(mask_img, spec.tx_mm, spec.ty_mm, background_hu=0.0)
    else:
        return sitk.GetArrayFromImage(mask_img) > 0.5
    return sitk.GetArrayFromImage(out) > 0.5


def self_check(spacing: tuple[float, float, float]) -> None:
    """Production rotate/translate must move a blob by the chord / |t|."""
    from predict.stability.perturbations import PerturbationSpec
    nx, ny, nz = 401, 401, 3
    arr = np.zeros((nz, ny, nx), dtype=np.float32)
    # Off-centre 5x5 blob, 120 voxels right and 40 up of centre.
    cx, cy = 200 + 120, 200 - 40
    arr[:, cy - 2:cy + 3, cx - 2:cx + 3] = 1.0
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)

    def centroid_mm(a):
        zz, yy, xx = np.nonzero(a)
        return xx.mean() * spacing[0], yy.mean() * spacing[1]

    x0, y0 = centroid_mm(arr > 0.5)
    ccx, ccy = _centre_xy(arr.shape, spacing)
    r = math.hypot(x0 - ccx, y0 - ccy)
    # One voxel: thresholding the interpolated blob moves its centroid by a
    # fraction of a voxel; a wrong centre or frame errs by millimetres.
    tol = max(spacing[0], spacing[1])
    checks = [
        PerturbationSpec(name="chk_rot", kind="rotate", rotation_deg=10.0),
        PerturbationSpec(name="chk_rot_neg", kind="rotate", rotation_deg=-5.0),
        PerturbationSpec(name="chk_tx", kind="translate", tx_mm=5.0),
        PerturbationSpec(name="chk_ty", kind="translate", ty_mm=-2.0),
    ]
    for spec in checks:
        moved = _move_mask(img, spec)
        if not moved.any():
            raise RuntimeError(f"self-check {spec.name}: blob vanished")
        x1, y1 = centroid_mm(moved)
        got = math.hypot(x1 - x0, y1 - y0)
        want = float(spec_displacement_mm(spec, np.array([r]))[0])
        if abs(got - want) > tol:
            raise RuntimeError(
                f"self-check {spec.name}: measured shift {got:.3f} mm, "
                f"formula {want:.3f} mm (tol {tol:.3f}). Frame or centre wrong.")
        _log.info("self-check %-12s shift %.3f mm vs formula %.3f mm  ok",
                  spec.name, got, want)


# ───────────────────────── per-patient worker ─────────────────────────


def _patient_worker(pid: str, preproc: str, spacing, specs, lesions_xy,
                    do_retention: bool) -> dict:
    mask_np = np.load(Path(preproc) / f"{pid}_mask.npy", mmap_mode="r")
    shape = tuple(int(s) for s in mask_np.shape)
    centre = _centre_xy(shape, spacing)

    # Frame check: lesion centroid distance to the mask footprint.
    from scipy.ndimage import distance_transform_edt
    footprint = np.asarray(mask_np).max(axis=0) > 0
    dist = distance_transform_edt(~footprint, sampling=(spacing[1], spacing[0]))
    frame = []
    ny, nx = footprint.shape
    for x_mm, y_mm in lesions_xy:
        col = int(round(x_mm / spacing[0]))
        row = int(round(y_mm / spacing[1]))
        inside = 0 <= row < ny and 0 <= col < nx
        frame.append(float(dist[row, col]) if inside else float("inf"))

    retention = []
    if do_retention:
        m = np.asarray(mask_np) > 0
        n_mask = int(m.sum())
        mimg = sitk.GetImageFromArray(m.astype(np.float32))
        mimg.SetSpacing(spacing)
        for spec in specs:
            moved = _move_mask(mimg, spec)
            kept = int((m & moved).sum())
            retention.append({
                "pid": pid, "perturbation": spec.name, "kind": spec.kind,
                "mask_voxels": n_mask,
                "retained_fraction": kept / n_mask if n_mask else float("nan"),
            })
    return {"pid": pid, "centre": centre, "frame_dist": frame,
            "retention": retention}


# ───────────────────────── report ─────────────────────────


def _q(s: pd.Series, q: float) -> float:
    return float(s.quantile(q)) if len(s) else float("nan")


def summarise(les: pd.DataFrame, ret: pd.DataFrame, specs) -> pd.DataFrame:
    rows = []
    for spec in specs:
        d = les[les["perturbation"] == spec.name]
        row = {
            "perturbation": spec.name, "kind": spec.kind,
            "n_lesions": len(d),
            "disp_median_mm": _q(d["displacement_mm"], 0.5),
            "disp_p25_mm": _q(d["displacement_mm"], 0.25),
            "disp_p75_mm": _q(d["displacement_mm"], 0.75),
            "disp_p90_mm": _q(d["displacement_mm"], 0.90),
            "frac_disp_gt_r_inplane": float((d["displacement_mm"] > d["r_inplane_mm"]).mean()),
            "frac_disp_gt_diam_inplane": float((d["displacement_mm"] > 2 * d["r_inplane_mm"]).mean()),
            "frac_disp_gt_r_sphere": float((d["displacement_mm"] > d["r_sphere_mm"]).mean()),
            # Area-weighted: share of calcified area whose lesion moves > own radius.
            "area_frac_disp_gt_r_inplane": float(
                (d["total_area_mm2"] * (d["displacement_mm"] > d["r_inplane_mm"])).sum()
                / d["total_area_mm2"].sum()) if len(d) else float("nan"),
        }
        if len(ret):
            r = ret[ret["perturbation"] == spec.name]["retained_fraction"]
            row.update({
                "n_patients": int(r.notna().sum()),
                "retained_median": _q(r, 0.5),
                "retained_p10": _q(r, 0.10),
                "retained_p25": _q(r, 0.25),
                "frac_patients_retained_lt_0p5": float((r < 0.5).mean()),
                "frac_patients_retained_0": float((r == 0).mean()),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def render(summ: pd.DataFrame, les: pd.DataFrame, n_pat: int,
           frame_hit: float, has_ret: bool) -> str:
    L: list[str] = []
    add = L.append
    one = les.drop_duplicates(["pid", "vessel", "lesion_idx"])
    med_ri = float(one["r_inplane_mm"].median())
    med_rs = float(one["r_sphere_mm"].median())
    add("=" * 88)
    add("PERTURBATION DISPLACEMENT PROBE (geometry only; no ICC output read)")
    add("=" * 88)
    add(f"patients {n_pat}   lesions {len(one)}   frame check: "
        f"{frame_hit:.1%} of centroids within {FRAME_TOL_MM:g} mm of mask")
    add(f"lesion r_inplane median {med_ri:.2f} mm "
        f"[IQR {one['r_inplane_mm'].quantile(.25):.2f}-{one['r_inplane_mm'].quantile(.75):.2f}]"
        f"   r_sphere median {med_rs:.2f} mm")
    add(f"distance from rotation centre median {one['r_centre_mm'].median():.1f} mm "
        f"[IQR {one['r_centre_mm'].quantile(.25):.1f}-{one['r_centre_mm'].quantile(.75):.1f}]")
    add("")
    hdr = (f"{'perturbation':<18}{'disp med [IQR] mm':>22}{'>r_in':>8}{'>2r_in':>8}"
           f"{'area>r':>8}")
    if has_ret:
        hdr += f"{'ret med':>9}{'ret p10':>9}{'ret<.5':>8}"
    add(hdr)
    add("-" * len(hdr))
    for _, r in summ.iterrows():
        s = (f"{r['perturbation']:<18}"
             f"{r['disp_median_mm']:>7.2f} [{r['disp_p25_mm']:5.2f}-{r['disp_p75_mm']:5.2f}]  "
             f"{r['frac_disp_gt_r_inplane']:>7.0%} {r['frac_disp_gt_diam_inplane']:>7.0%} "
             f"{r['area_frac_disp_gt_r_inplane']:>7.0%}")
        if has_ret:
            s += (f"{r['retained_median']:>9.2f}{r['retained_p10']:>9.2f}"
                  f"{r['frac_patients_retained_lt_0p5']:>8.0%}")
        add(s)
    add("")
    add("Candidate D033 readout (DESCRIPTIVE ONLY; the rule is not locked here):")
    add("  rule 'median lesion displacement <= cohort median lesion radius'")
    for label, med in (("r_inplane", med_ri), ("r_sphere", med_rs)):
        keep = summ.loc[summ["disp_median_mm"] <= med, "perturbation"].tolist()
        drop = summ.loc[summ["disp_median_mm"] > med, "perturbation"].tolist()
        add(f"  {label} ({med:.2f} mm): keep {len(keep)}  drop {drop}")
    add("")
    add("Columns: >r_in / >2r_in = share of lesions displaced beyond their own")
    add("in-plane radius / diameter; area>r = same, weighted by lesion area;")
    add("ret = fraction of mask voxels still holding calcium after the move.")
    add("=" * 88)
    return "\n".join(L)


# ───────────────────────── main ─────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("outputs/probe_perturbation_displacement"))
    ap.add_argument("--skip-retention", action="store_true",
                    help="skip mask transforms (fast; lesion geometry + frame check only)")
    ap.add_argument("--n-jobs", type=int, default=None,
                    help="default: predict.config.default_n_jobs()")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    from predict.config import default_n_jobs, load_config
    from predict.io.spacing import load_spacing_metadata
    from predict.stability.perturbations import enumerate_perturbations

    cfg = load_config()
    outputs = cfg.paths.outputs
    preproc = outputs / "02_preprocessed"
    spacing = tuple(float(s) for s in load_spacing_metadata(preproc / "spacing.json"))
    specs = enumerate_perturbations(cfg)
    n_jobs = args.n_jobs or default_n_jobs()
    _log.info("spacing %s, perturbations %d, n_jobs %d", spacing, len(specs), n_jobs)

    self_check(spacing)

    eligible = set(_gate._load_baseline(outputs / "03_features" / "features.csv")
                   ["pid"].astype(str))
    lesions = pd.read_csv(outputs / "03_features" / "lesions.csv")
    lesions["pid"] = lesions["pid"].astype(str)
    lesions = lesions[lesions["pid"].isin(eligible)].reset_index(drop=True)
    missing = sorted(p for p in eligible
                     if not (preproc / f"{p}_mask.npy").exists())
    if missing:
        raise FileNotFoundError(f"{len(missing)} eligible pids lack a mask, e.g. {missing[:5]}")
    pids = sorted(eligible, key=lambda p: (len(p), p))
    _log.info("eligible N=%d, lesions=%d", len(pids), len(lesions))

    n_sl = lesions["slice_indices"].astype(str).str.count(";") + 1
    lesions["r_inplane_mm"] = np.sqrt(lesions["total_area_mm2"] / n_sl / math.pi)
    lesions["r_sphere_mm"] = np.cbrt(3.0 * lesions["volume_mm3"] / (4.0 * math.pi))

    by_pid = {p: g for p, g in lesions.groupby("pid")}
    tasks = []
    for p in pids:
        g = by_pid.get(p)
        xy = [] if g is None else list(zip(g["centroid_x_mm"], g["centroid_y_mm"]))
        tasks.append((p, str(preproc), spacing, specs, xy, not args.skip_retention))

    results: dict[str, dict] = {}
    if n_jobs == 1:
        for t in tasks:
            results[t[0]] = _patient_worker(*t)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futs = {pool.submit(_patient_worker, *t): t[0] for t in tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                res = fut.result()
                results[res["pid"]] = res
                if i % 50 == 0 or i == len(futs):
                    _log.info("patients done %d/%d", i, len(futs))

    # Frame check before anything is reported.
    fd = np.array([d for p in pids for d in results[p]["frame_dist"]], dtype=float)
    frame_hit = float((fd <= FRAME_TOL_MM).mean()) if fd.size else float("nan")
    _log.info("frame check: %.1f%% of %d centroids within %g mm of mask",
              100 * frame_hit, fd.size, FRAME_TOL_MM)
    if not fd.size or frame_hit < FRAME_MIN_HIT:
        raise RuntimeError(
            f"frame check failed ({frame_hit:.1%} < {FRAME_MIN_HIT:.0%}): lesion "
            "centroids and the preprocessed grid disagree; displacements invalid.")

    rows = []
    for p in pids:
        g = by_pid.get(p)
        if g is None:
            continue
        ccx, ccy = results[p]["centre"]
        r_c = np.hypot(g["centroid_x_mm"].to_numpy() - ccx,
                       g["centroid_y_mm"].to_numpy() - ccy)
        for spec in specs:
            disp = spec_displacement_mm(spec, r_c)
            for j, (_, le) in enumerate(g.iterrows()):
                rows.append({
                    "pid": p, "vessel": le["vessel"], "lesion_idx": le["lesion_idx"],
                    "perturbation": spec.name, "kind": spec.kind,
                    "r_centre_mm": r_c[j], "displacement_mm": disp[j],
                    "r_inplane_mm": le["r_inplane_mm"], "r_sphere_mm": le["r_sphere_mm"],
                    "total_area_mm2": le["total_area_mm2"],
                    "frame_dist_mm": results[p]["frame_dist"][j],
                })
    les = pd.DataFrame(rows)
    ret = pd.DataFrame([r for p in pids for r in results[p]["retention"]])

    summ = summarise(les, ret, specs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    les.to_csv(args.out_dir / "lesion_displacement.csv", index=False)
    if len(ret):
        ret.to_csv(args.out_dir / "patient_retention.csv", index=False)
    summ.to_csv(args.out_dir / "perturbation_summary.csv", index=False)
    report = render(summ, les, len(pids), frame_hit, bool(len(ret)))
    (args.out_dir / "displacement_summary.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    _log.info("wrote %s (no ICC output read or modified)", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
