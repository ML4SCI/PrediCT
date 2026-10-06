#!/usr/bin/env python
"""D033 calibration (C1, C1b, C2, C3). Runs BEFORE any D033 ICC.

Reads only preprocessed CT arrays, DICOM headers, the manifest (kernel) and the
eligible-cohort list (D015, from the radiomics_status column). It never reads a
feature value or an ICC output: the constants it writes must not depend on
which features pass.

  C1   sigma0 per patient: Immerkaer noise sd in the soft-tissue band
  C1b  per kernel: median lag-1 noise autocorrelation rho1 -> sigma_n (voxels)
  C2   sigma_delta = sqrt(|sigma_n(Qr36d/2)^2 - sigma_n(I30f/3)^2|) (voxels)
  C3   native in-plane spacing of every eligible patient

Usage
-----
    python scripts/calibrate_realistic.py

Outputs (outputs/04b_calibration/):
    calibration.json          consumed by scripts/04b_realistic_perturbations.py
    calibration_patients.csv  per-patient sigma0, rho1, kernel, native spacing
    calibration_summary.txt
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
_log = logging.getLogger("calibrate_realistic")


def _load_script_module(rel_path: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def _patient(pid: str, preproc: str, data_root: str) -> dict:
    from predict.io import load_patient_metadata
    from predict.stability.realistic import immerkaer_sigma, noise_lag1_correlation
    try:
        ct = np.load(Path(preproc) / f"{pid}_ct.npy")
        meta = load_patient_metadata(pid, Path(data_root))
        return {
            "pid": pid,
            "sigma0": immerkaer_sigma(ct),
            "rho1": noise_lag1_correlation(ct),
            "native_spacing_mm": float(meta.pixel_spacing[0]),
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {"pid": pid, "sigma0": np.nan, "rho1": np.nan,
                "native_spacing_mm": np.nan, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/04b_calibration"))
    ap.add_argument("--n-jobs", type=int, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from predict.config import default_n_jobs, load_config
    from predict.stability.realistic import P5_MIN_SIGMA_DELTA_VOX, sigma_from_lag1

    cfg = load_config()
    outputs = cfg.paths.outputs
    stage4 = _load_script_module("scripts/04_perturbations.py", "_prod_stage4")
    pids = stage4._eligible_pids(outputs / "03_features" / "features.csv")
    manifest = pd.read_csv(outputs / "01_manifest" / "manifest.csv", dtype={"pid": str})
    kernel_of = dict(zip(manifest["pid"].astype(str), manifest["kernel"].astype(str)))
    main_kernels = [str(k) for k in cfg.raw["reduce"]["combat_kernel_groups"]]
    n_jobs = args.n_jobs or default_n_jobs(cap=16)
    _log.info("eligible N=%d, main kernels %s, n_jobs %d", len(pids), main_kernels, n_jobs)

    rows = []
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futs = [pool.submit(_patient, p, str(outputs / "02_preprocessed"),
                            str(cfg.paths.data_raw.parent)) for p in pids]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 50 == 0 or i == len(futs):
                _log.info("patients done %d/%d", i, len(futs))
    df = pd.DataFrame(rows)
    df["kernel"] = df["pid"].map(kernel_of).fillna("unknown")
    bad = df[df["error"] != ""]
    if len(bad):
        raise RuntimeError(f"{len(bad)} patients failed calibration, e.g. "
                           f"{bad.head(3).to_dict('records')}")
    if df["sigma0"].isna().any():
        raise RuntimeError(f"sigma0 undefined for pids {df.loc[df['sigma0'].isna(), 'pid'].tolist()}")

    rho_by_kernel = df.groupby("kernel")["rho1"].median()
    sigma_n = {k: sigma_from_lag1(float(r)) for k, r in rho_by_kernel.items()}
    sigma_n_default = sigma_from_lag1(float(df["rho1"].median()))
    missing = [k for k in main_kernels if k not in sigma_n]
    if missing:
        raise RuntimeError(f"main kernels absent from the eligible cohort: {missing}")
    sigma_delta = float(np.sqrt(abs(sigma_n[main_kernels[0]] ** 2 - sigma_n[main_kernels[1]] ** 2)))

    calib = {
        "decision": "D033",
        "sigma0": {r.pid: float(r.sigma0) for r in df.itertuples()},
        "sigma_n": {k: float(v) for k, v in sigma_n.items()},
        "sigma_n_default": float(sigma_n_default),
        "sigma_delta": sigma_delta,
        "native_spacings": sorted(float(s) for s in df["native_spacing_mm"]),
        "rho1_by_kernel": {k: float(v) for k, v in rho_by_kernel.items()},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "calibration.json").write_text(json.dumps(calib, indent=2) + "\n", encoding="utf-8")
    df.to_csv(args.out_dir / "calibration_patients.csv", index=False)

    L = ["=" * 78, "D033 CALIBRATION (images and headers only; no features, no ICC)", "=" * 78,
         f"eligible patients {len(df)}", ""]
    L.append(f"{'kernel':<14}{'n':>5}{'sigma0 med [IQR] HU':>26}{'rho1 med':>10}{'sigma_n vox':>13}")
    for k, g in df.groupby("kernel"):
        L.append(f"{k:<14}{len(g):>5}{g['sigma0'].median():>12.1f} "
                 f"[{g['sigma0'].quantile(.25):5.1f}-{g['sigma0'].quantile(.75):5.1f}]"
                 f"{rho_by_kernel[k]:>10.3f}{sigma_n[k]:>13.3f}")
    L += ["",
          f"sigma_n default (pooled) {sigma_n_default:.3f} vox",
          f"C2 sigma_delta = {sigma_delta:.3f} vox  -> P5 "
          + ("ACTIVE" if sigma_delta >= P5_MIN_SIGMA_DELTA_VOX else
             f"DROPPED (< {P5_MIN_SIGMA_DELTA_VOX} vox, as D033 pre-specifies)"),
          f"C3 native spacing: min {df['native_spacing_mm'].min():.3f}, median "
          f"{df['native_spacing_mm'].median():.3f}, max {df['native_spacing_mm'].max():.3f} mm",
          "=" * 78]
    report = "\n".join(L)
    (args.out_dir / "calibration_summary.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
