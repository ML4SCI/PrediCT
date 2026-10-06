#!/usr/bin/env python
"""Stage 4b, D033 realistic composite perturbations.

For every eligible patient (D015) and every draw of the requested arms
(D033: 20 primary, 10 per secondary factor, 10 per attribution factor):

  1. M1 jitters the XML polygons (predict.stability.realistic).
  2. The 68 canonical features are recomputed from the jittered annotation
     (predict.features.canonical, the same function stage 3 uses).
  3. The mask is rebuilt at native resolution (production build_3d_mask) and
     resampled to the target grid (production resample_mask_to_target).
  4. P1-P7 (and the arm's secondary factor) are applied to the cached
     stage-2 CT and the mask.
  5. The 107 PyRadiomics features are extracted (production extractor, D009).

Parity, per patient, before any draw: the rebuilt unperturbed mask must equal
the stage-2 mask exactly, the unperturbed canonical features must equal
features.csv, and the unperturbed PyRadiomics features must equal
features.csv. Results go to parity.csv; stage 5b refuses to gate if any
eligible patient failed.

Requires outputs/04b_calibration/calibration.json (scripts/calibrate_realistic.py).
Reads no ICC output.

Resume: per_patient/{pid}.csv is reused only if
its sidecar {pid}.json carries the current fingerprint (all src/predict
code, params.yaml, configs/default.yaml, this script, the calibration, the
patient's stage-2 CT and mask, its XML, its features.csv row and its D002
exclusions), recorded zero errors and passed parity, and the CSV holds every
planned (arm, index) exactly once. The sidecar is written last, so an
interrupted patient is recomputed.

Broken versus error (D033.1): a draw is "broken" (allowed, counted by the
5% rule in stage 5b) only when the perturbed mask is empty or below
PyRadiomics' own ROI minimums. Any other exception is an "error": the
patient fails, the script exits 1 and writes no COMPLETE.json, and stage 5b
refuses to gate. A smoke run (--limit, --pids, --max-draws) never writes
COMPLETE.json either.

Usage
-----
    python scripts/04b_realistic_perturbations.py                  # primary arm
    python scripts/04b_realistic_perturbations.py --arms all       # every arm
    python scripts/04b_realistic_perturbations.py --limit 2 --max-draws 2   # smoke

Outputs (outputs/04b_realistic/):
    per_patient/{pid}.csv    all rows for one patient (resume unit)
    per_patient/{pid}.json   fingerprint, parity, error count (written last)
    draws_{arm}.csv          aggregated, one row per (patient, draw)
    parity.csv               one row per patient
    run_header.json
    COMPLETE.json            only after a full, error-free, parity-clean run
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from predict.completion import (
    clear_marker, digest, file_sha, upstream_fingerprint, write_json_atomic, write_marker,
)
from predict.preprocess.slice_matcher import Z_MATCH_TOLERANCE_MM
from predict.provenance import code_revision

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "04b_realistic_perturbations.py"
_log = logging.getLogger("stage4b")


def run_fingerprint(cfg, calib_path: Path) -> str:
    """Stage 4b run fingerprint: stage 1-4 code and config (including this
    script) plus the calibration. Recorded in COMPLETE.json; stage 5b
    recomputes it with this function and refuses on any difference."""
    return digest({"upstream": upstream_fingerprint(cfg, extra=[SCRIPT]),
                   "calibration": file_sha(calib_path)})

CANON_RTOL, CANON_ATOL = 1e-9, 1e-9
RAD_RTOL, RAD_ATOL = 1e-7, 1e-9


def _load_script_module(rel_path: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else "missing"


# ───────────────────────── worker ─────────────────────────


_state: dict = {}


def _init_worker(calib_path: str) -> None:
    from predict.config import load_config
    from predict.features.radiomics import create_extractor
    from predict.io.spacing import load_spacing_metadata
    from predict.stability.realistic import Calibration
    cfg = load_config()
    preproc = cfg.paths.outputs / "02_preprocessed"
    _state.update(
        cfg=cfg,
        preproc=preproc,
        spacing=tuple(float(s) for s in load_spacing_metadata(preproc / "spacing.json")),
        extractor=create_extractor(),
        calib=Calibration.from_json(Path(calib_path)),
    )


def _max_rel(a: dict, b: dict, keys, rtol: float, atol: float) -> tuple[float, bool, list[str]]:
    worst, bad = 0.0, []
    for k in keys:
        x, y = a.get(k, np.nan), b.get(k, np.nan)
        if (x is None or (isinstance(x, float) and np.isnan(x))) and (
                y is None or (isinstance(y, float) and np.isnan(y))):
            continue
        x, y = float(x), float(y)
        d = abs(x - y) / max(abs(y), 1e-12)
        worst = max(worst, d)
        if not np.isclose(x, y, rtol=rtol, atol=atol):
            bad.append(k)
    return worst, not bad, bad


def _extract_draw(ct, mask, spacing, extractor, pid: str) -> tuple[str, str, dict]:
    """D033.1: ``("broken", reason, {})`` only when the perturbed mask is
    empty or at/below PyRadiomics' own ROI minimums (decided before extraction,
    from the locked params); otherwise extract. Anything the extractor raises
    propagates: it is a software or input error, not a broken draw."""
    from predict.features.radiomics import extract_pyradiomics, roi_too_small
    if int(np.count_nonzero(mask)) == 0:
        return "broken", "empty mask", {}
    why = roi_too_small(mask, extractor)
    if why is not None:
        return "broken", why, {}
    return "ok", "", extract_pyradiomics(ct, mask, spacing, extractor, pid=pid)


def _patient_fingerprint(run_fp: str, pid: str, preproc: Path, xml_dir: Path,
                         baseline_row: dict, excluded: set) -> str:
    """Everything a patient's draws depend on (see the module docstring)."""
    return digest({
        "run": run_fp,
        "ct": file_sha(preproc / f"{pid}_ct.npy"),
        "mask": file_sha(preproc / f"{pid}_mask.npy"),
        "xml": file_sha(xml_dir / f"{pid}.xml"),
        "baseline_row": {k: repr(v) for k, v in sorted(baseline_row.items())},
        "excluded": sorted(excluded),
    })


def load_run_inputs(cfg) -> tuple[pd.DataFrame, list[str], dict[str, set]]:
    """features.csv rows, the baseline columns and the D002 exclusions,
    exactly as 4b feeds its workers. Stage 5b uses the same function."""
    from predict.features.feature_schema import feature_names
    outputs = cfg.paths.outputs
    feats = pd.read_csv(outputs / "03_features" / "features.csv", dtype={"pid": str}).set_index("pid")
    keep_cols = list(feature_names()) + [c for c in feats.columns if c.startswith("original_")]
    trips = pd.read_csv(outputs / "02_preprocessed" / "xml_roundtrip.csv", dtype={"pid": str})
    fails = trips[(~trips["passes"].astype(bool)) & (trips["matched_via"] != "dirty")]
    excluded_by_pid: dict[str, set] = {}
    for r in fails.itertuples():
        excluded_by_pid.setdefault(str(r.pid), set()).add((int(r.image_index), int(r.roi_idx_in_slice)))
    return feats, keep_cols, excluded_by_pid


def baseline_row(feats: pd.DataFrame, keep_cols: list[str], pid: str) -> dict:
    return {k: feats.at[pid, k] for k in keep_cols if k in feats.columns}


def patient_fingerprints(cfg, calib_path: Path, pids: list[str], n_jobs: int = 1) -> dict[str, str]:
    """Every patient's input fingerprint (stage-2 CT and mask, XML,
    features.csv row, exclusions, run fingerprint), computed with the same
    function the workers use. Stage 5b recomputes these to bind COMPLETE.json
    to the patient inputs the draws were made from."""
    from concurrent.futures import ThreadPoolExecutor
    run_fp = run_fingerprint(cfg, calib_path)
    feats, keep_cols, excluded = load_run_inputs(cfg)
    preproc = cfg.paths.outputs / "02_preprocessed"

    def one(pid: str) -> tuple[str, str]:
        return pid, _patient_fingerprint(run_fp, pid, preproc, cfg.paths.data_xml,
                                         baseline_row(feats, keep_cols, pid),
                                         excluded.get(pid, set()))

    with ThreadPoolExecutor(max_workers=max(1, n_jobs)) as ex:
        return dict(ex.map(one, pids))


def combine_patient_fingerprints(fps: dict[str, str]) -> str:
    return digest(sorted(fps.items()))


def _reusable(out_csv: Path, meta_json: Path, fingerprint: str,
              want: set) -> dict | None:
    """The stored parity record if the cached patient can be reused, else None."""
    if not (out_csv.exists() and meta_json.exists()):
        return None
    try:
        meta = json.loads(meta_json.read_text(encoding="utf-8"))
        have = pd.read_csv(out_csv, usecols=["arm", "index", "status"])
    except Exception:  # noqa: BLE001  (unreadable cache = recompute)
        return None
    if meta.get("fingerprint") != fingerprint or meta.get("n_error") != 0:
        return None
    if not meta.get("parity", {}).get("ok"):
        return None
    if (have["status"] == "error").any():
        return None
    keys = list(zip(have["arm"].astype(str), have["index"].astype(int)))
    if len(keys) != len(set(keys)) or not want <= set(keys):
        return None
    return meta["parity"]


def _process_patient(pid: str, plan: list[tuple[str, int]], excluded: set,
                     baseline_row: dict, out_dir: str, run_fp: str) -> dict:
    from predict.features.canonical import compute_canonical_features, grouping_from_config
    from predict.features.feature_schema import feature_names
    from predict.features.radiomics import extract_pyradiomics
    from predict.io import load_patient_dicom, load_patient_metadata, parse_calcium_xml
    from predict.preprocess.mask_builder import build_3d_mask, mask_to_sitk
    from predict.preprocess.resampling import resample_mask_to_target
    from predict.stability.realistic import (
        apply_draw_to_volumes, baseline_draw, jitter_parse_result, sample_draw,
    )

    t0 = time.perf_counter()
    cfg, spacing, extractor, calib = _state["cfg"], _state["spacing"], _state["extractor"], _state["calib"]
    grouping = grouping_from_config(cfg)
    out_csv = Path(out_dir) / "per_patient" / f"{pid}.csv"
    meta_json = out_csv.with_suffix(".json")
    want = {(a, i) for a, i in plan}
    parity = {"pid": pid}
    try:
        fingerprint = _patient_fingerprint(run_fp, pid, _state["preproc"], cfg.paths.data_xml,
                                           baseline_row, excluded)
        stored = _reusable(out_csv, meta_json, fingerprint, want)
        if stored is not None:
            return {"pid": pid, "skipped": True, "parity": stored, "fingerprint": fingerprint}

        data_root = cfg.paths.data_raw.parent
        loaded = load_patient_dicom(pid, data_root)
        meta = load_patient_metadata(pid, data_root)   # the metadata stage 3 used
        parse = parse_calcium_xml(pid, cfg.paths.data_xml)
        ct = np.load(_state["preproc"] / f"{pid}_ct.npy")
        mask_stage2 = np.load(_state["preproc"] / f"{pid}_mask.npy")
        tol = Z_MATCH_TOLERANCE_MM
        native_sp = float(meta.pixel_spacing[0])
        sigma0 = calib.sigma0[pid]
        corr = calib.noise_corr(loaded.kernel)

        def build_mask(pr):
            m, _ = build_3d_mask(pr, loaded, tolerance_mm=tol, excluded_roi_ids=excluded)
            ms = resample_mask_to_target(mask_to_sitk(m, loaded.ct_sitk), loaded.ct_sitk, spacing)
            return sitk.GetArrayFromImage(ms).astype(np.uint8)

        # ── parity ──────────────────────────────────────────────
        base_mask = build_mask(parse)
        parity["mask_equal"] = bool(base_mask.shape == mask_stage2.shape
                                    and np.array_equal(base_mask, mask_stage2))
        base_canon, _, _ = compute_canonical_features(parse, meta, excluded, **grouping)
        parity["canon_max_rel"], parity["canon_ok"], bad_c = _max_rel(
            base_canon, baseline_row, feature_names(), CANON_RTOL, CANON_ATOL)
        base_rad = extract_pyradiomics(ct, mask_stage2, spacing, extractor, pid=f"{pid}:baseline")
        rad_keys = [k for k in baseline_row if k.startswith("original_")]
        parity["rad_max_rel"], parity["rad_ok"], bad_r = _max_rel(
            base_rad, baseline_row, rad_keys, RAD_RTOL, RAD_ATOL)
        parity["bad_features"] = ";".join((bad_c + bad_r)[:10])
        parity["ok"] = parity["mask_equal"] and parity["canon_ok"] and parity["rad_ok"]

        rows = [{"pid": pid, **baseline_draw().as_record(), "status": "ok", "error": "",
                 "mask_voxels": int(mask_stage2.sum()), "n_vanished": 0,
                 **base_canon, **base_rad}]

        # ── draws ───────────────────────────────────────────────
        for arm, idx in plan:
            draw = sample_draw(pid, arm, idx, calib, native_sp)
            rec = {"pid": pid, **draw.as_record()}
            try:
                pj = jitter_parse_result(parse, draw.m1_rms_px, draw.m1_seed, excluded)
                canon, _, _ = compute_canonical_features(pj, meta, excluded, **grouping)
                m0 = base_mask if pj is parse else build_mask(pj)
                vr = apply_draw_to_volumes(
                    ct, m0, spacing, draw,
                    sigma0=sigma0, corr_sigma_vox=corr,
                    background_hu=cfg.stability.background_fill_hu,
                    clip_min=cfg.hu.clip_min, clip_max=cfg.hu.clip_max,
                )
                rec.update(mask_voxels=int(vr.mask.sum()), n_vanished=vr.n_vanished, **canon)
                status, why, rad = _extract_draw(vr.ct, vr.mask, spacing, extractor,
                                                 pid=f"{pid}:{arm}:{idx}")
                rec.update(rad)
                rec.update(status=status, error=why[:200])
            except Exception as exc:  # noqa: BLE001  (recorded, then fails the patient)
                rec.update(status="error", error=f"{type(exc).__name__}: {exc}"[:300])
            rows.append(rec)

        parity["runtime_sec"] = round(time.perf_counter() - t0, 1)
        parity["n_broken"] = sum(1 for r in rows if r["status"] == "broken")
        parity["n_error"] = sum(1 for r in rows if r["status"] == "error")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_csv.with_suffix(".csv.tmp")
        pd.DataFrame(rows).to_csv(tmp, index=False)
        tmp.replace(out_csv)
        # Sidecar last: its presence means the CSV above is complete.
        write_json_atomic(meta_json, {
            "fingerprint": fingerprint, "parity": parity, "n_error": parity["n_error"],
            "n_broken": parity["n_broken"], "revision": code_revision(REPO_ROOT),
            "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        return {"pid": pid, "skipped": False, "parity": parity, "fingerprint": fingerprint}
    except Exception as exc:  # noqa: BLE001
        parity.update(ok=False, n_error=1, error=f"{type(exc).__name__}: {exc}"[:300])
        return {"pid": pid, "skipped": False, "parity": parity}


def _task(args):
    return _process_patient(*args)


# ───────────────────────── main ─────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="primary",
                    help="primary | secondary | attribution | all | comma list (e.g. S1,attr_M1)")
    ap.add_argument("--limit", type=int, default=None, help="first N eligible patients (smoke)")
    ap.add_argument("--pids", default=None, help="comma list of pids (smoke)")
    ap.add_argument("--max-draws", type=int, default=None, help="cap draws per arm (smoke)")
    ap.add_argument("--n-workers", type=int, default=None,
                    help="default: predict.config.default_n_jobs(cap=16)")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from predict.config import default_n_jobs, load_config
    from predict.features.feature_schema import feature_names
    from predict.stability.realistic import arm_draw_plan, expand_arms

    cfg = load_config()
    outputs = cfg.paths.outputs
    out_dir = args.out_dir or outputs / "04b_realistic"
    calib_path = outputs / "04b_calibration" / "calibration.json"
    if not calib_path.exists():
        raise SystemExit(f"{calib_path} missing: run scripts/calibrate_realistic.py first")

    arms = expand_arms(args.arms)
    plan = arm_draw_plan(arms)
    if args.max_draws is not None:
        plan = [(a, i) for a, i in plan if i < args.max_draws]

    stage4 = _load_script_module("scripts/04_perturbations.py", "_prod_stage4")
    features_csv = outputs / "03_features" / "features.csv"
    pids = stage4._eligible_pids(features_csv)
    if args.pids:
        wanted = [p.strip() for p in args.pids.split(",") if p.strip()]
        pids = [p for p in pids if p in set(wanted)]
    if args.limit:
        pids = pids[: args.limit]

    feats, keep_cols, excluded_by_pid = load_run_inputs(cfg)

    n_workers = args.n_workers or default_n_jobs(cap=16)
    _log.info("D033 stage 4b: %d patients x %d draws (arms %s), workers %d",
              len(pids), len(plan), ",".join(arms), n_workers)

    out_dir.mkdir(parents=True, exist_ok=True)
    clear_marker(out_dir)   # set again only if this run finishes cleanly
    smoke = bool(args.limit or args.pids or args.max_draws is not None)
    run_fp = run_fingerprint(cfg, calib_path)
    header = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": code_revision(REPO_ROOT),
        "d033_doc_sha": _sha(REPO_ROOT / "decisions" / "D033-realistic-perturbation-set.md"),
        "calibration_sha": _sha(calib_path),
        "arms": arms, "n_draws": len(plan), "n_patients": len(pids),
        "run_fingerprint": run_fp, "smoke": smoke,
    }
    (out_dir / "run_header.json").write_text(json.dumps(header, indent=2) + "\n", encoding="utf-8")

    tasks = [(p, plan, excluded_by_pid.get(p, set()),
              baseline_row(feats, keep_cols, p), str(out_dir), run_fp)
             for p in pids]
    results = []
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                             initargs=(str(calib_path),)) as pool:
        futs = [pool.submit(_task, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            if i % 10 == 0 or i == len(futs):
                _log.info("patients done %d/%d", i, len(futs))

    # Every patient of this run, fresh or reused (reused ones carry their
    # stored parity record), so parity.csv describes exactly this run.
    new_parity = pd.DataFrame([r["parity"] for r in results])
    par_path = out_dir / "parity.csv"
    new_parity.to_csv(par_path, index=False)
    n_reused = sum(1 for r in results if r.get("skipped"))

    # Aggregate per-patient files into one CSV per arm.
    frames = []
    for p in pids:
        f = out_dir / "per_patient" / f"{p}.csv"
        if f.exists():
            frames.append(pd.read_csv(f, dtype={"pid": str}))
    if frames:
        allrows = pd.concat(frames, ignore_index=True)
        for arm, g in allrows.groupby("arm"):
            g.to_csv(out_dir / f"draws_{arm}.csv", index=False)

    ok_col = new_parity["ok"].fillna(False).astype(bool) if len(new_parity) else pd.Series(dtype=bool)
    n_fail = int((~ok_col).sum())
    n_err = (int(pd.to_numeric(new_parity["n_error"], errors="coerce").fillna(1).sum())
             if "n_error" in new_parity.columns else len(new_parity))
    _log.info("parity: %d/%d patients ok; %d reused from cache; %d draw errors",
              len(new_parity) - n_fail, len(new_parity), n_reused, n_err)
    if n_fail or n_err:
        _log.error("stage 4b FAILED: %d patients failed parity or raised, %d draw errors; "
                   "see %s and per_patient/*.csv. No COMPLETE.json; stage 5b will refuse.",
                   n_fail, n_err, par_path)
        return 1
    if smoke:
        _log.warning("smoke run: no COMPLETE.json written; stage 5b will refuse this output.")
        return 0
    fps = {r["pid"]: r["fingerprint"] for r in results if "fingerprint" in r}
    if set(fps) != set(pids):
        _log.error("stage 4b FAILED: %d patients have no input fingerprint.", len(set(pids) - set(fps)))
        return 1
    write_marker(out_dir, {
        "patients_fingerprint": combine_patient_fingerprints(fps),
        "features_csv_sha": file_sha(features_csv),
        "calibration_sha": file_sha(calib_path),
        "run_fingerprint": run_fp,
        "arms": arms,
        "n_patients": len(pids),
        "revision": code_revision(REPO_ROOT),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
