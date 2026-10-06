#!/usr/bin/env python
"""Per-patient pixel-spacing audit for the COCA cohort.

Reads ``outputs/01_manifest/manifest.csv`` and, for every included patient,
peeks one DICOM header (``stop_before_pixels=True``, ~10x faster than
loading the full volume) to pull native ``PixelSpacing`` (x, y) and
``SliceThickness`` (z). Writes:

  outputs/_diagnostics/pixel_spacing_per_patient.csv
    pid, kernel, scanner_model, spacing_x_mm, spacing_y_mm,
    slice_thickness_mm, in_plane_mm (mean of x,y), n_slices

  outputs/_diagnostics/pixel_spacing_summary.json
    cohort min / p25 / median / p75 / max for each spacing axis

  outputs/_diagnostics/pixel_spacing_histogram.png
    3-panel histogram: in-plane spacing, slice thickness, n_slices

Usage::

    python scripts/diagnose_pixel_spacing.py [--config configs/default.yaml]
                                              [--n-jobs 16]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from predict.config import default_n_jobs, load_config
from predict.io.dicom_loader import load_patient_metadata


def _peek(pid: str, data_root: Path) -> dict:
    try:
        meta = load_patient_metadata(pid, data_root)
        sx, sy, _ = meta.pixel_spacing
        return {
            "pid": pid,
            "scanner_model": meta.scanner_model,
            "kernel": meta.kernel,
            "spacing_x_mm": float(sx),
            "spacing_y_mm": float(sy),
            "slice_thickness_mm": float(meta.slice_thickness),
            "in_plane_mm": float((sx + sy) / 2.0),
            "n_slices": int(meta.n_slices),
            "error": "",
        }
    except Exception as e:
        return {
            "pid": pid, "scanner_model": "", "kernel": "",
            "spacing_x_mm": np.nan, "spacing_y_mm": np.nan,
            "slice_thickness_mm": np.nan, "in_plane_mm": np.nan,
            "n_slices": 0, "error": str(e),
        }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--n-jobs", type=int, default=default_n_jobs())
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("pixspace")

    cfg = load_config(args.config)
    data_root = cfg.paths.data_raw.parent
    manifest_path = cfg.paths.outputs / "01_manifest" / "manifest.csv"
    if not manifest_path.exists():
        log.error("missing %s; run scripts/01_discover.py first", manifest_path)
        return 1

    manifest = pd.read_csv(manifest_path, dtype={"pid": str})
    pids = manifest["pid"].tolist()
    log.info("peeking pixel spacing for %d patients (n_jobs=%d)...",
             len(pids), args.n_jobs)

    rows = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_peek)(pid, data_root) for pid in pids
    )
    df = pd.DataFrame(rows)

    out_dir = cfg.paths.outputs / "_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "pixel_spacing_per_patient.csv"
    df.to_csv(csv_path, index=False)
    log.info("wrote %s (%d rows; %d errors)",
             csv_path, len(df), int((df["error"] != "").sum()))

    ok = df[df["error"] == ""].copy()
    pct = lambda s, q: float(np.percentile(s.dropna(), q))
    summary = {
        "n_patients": int(len(ok)),
        "n_errors": int(len(df) - len(ok)),
        "in_plane_mm": {
            "min": float(ok["in_plane_mm"].min()),
            "p25": pct(ok["in_plane_mm"], 25),
            "median": pct(ok["in_plane_mm"], 50),
            "p75": pct(ok["in_plane_mm"], 75),
            "max": float(ok["in_plane_mm"].max()),
        },
        "slice_thickness_mm": {
            "min": float(ok["slice_thickness_mm"].min()),
            "p25": pct(ok["slice_thickness_mm"], 25),
            "median": pct(ok["slice_thickness_mm"], 50),
            "p75": pct(ok["slice_thickness_mm"], 75),
            "max": float(ok["slice_thickness_mm"].max()),
        },
        "n_slices": {
            "min": int(ok["n_slices"].min()),
            "p25": pct(ok["n_slices"], 25),
            "median": pct(ok["n_slices"], 50),
            "p75": pct(ok["n_slices"], 75),
            "max": int(ok["n_slices"].max()),
        },
        "by_kernel_in_plane_median": (
            ok.groupby("kernel")["in_plane_mm"].median().round(4).to_dict()
        ),
    }
    json_path = out_dir / "pixel_spacing_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    log.info("wrote %s", json_path)
    log.info("in-plane mm  : min=%.3f p25=%.3f med=%.3f p75=%.3f max=%.3f",
             *[summary["in_plane_mm"][k]
               for k in ("min", "p25", "median", "p75", "max")])
    log.info("thickness mm : min=%.3f p25=%.3f med=%.3f p75=%.3f max=%.3f",
             *[summary["slice_thickness_mm"][k]
               for k in ("min", "p25", "median", "p75", "max")])
    log.info("n_slices     : min=%d med=%.0f max=%d",
             summary["n_slices"]["min"], summary["n_slices"]["median"],
             summary["n_slices"]["max"])

    # Histograms
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        ax_in, ax_th, ax_n = axes

        ax_in.hist(ok["in_plane_mm"], bins=40, color="#4C72B0",
                   edgecolor="white")
        ax_in.axvline(summary["in_plane_mm"]["median"], c="black",
                      ls="--", lw=1)
        ax_in.set_xlabel("in-plane pixel spacing (mm)")
        ax_in.set_ylabel("patients")
        ax_in.set_title(f"In-plane spacing\nmedian = "
                        f"{summary['in_plane_mm']['median']:.3f} mm")

        ax_th.hist(ok["slice_thickness_mm"], bins=30, color="#55A868",
                   edgecolor="white")
        ax_th.axvline(summary["slice_thickness_mm"]["median"], c="black",
                      ls="--", lw=1)
        ax_th.set_xlabel("slice thickness (mm)")
        ax_th.set_title(f"Slice thickness\nmedian = "
                        f"{summary['slice_thickness_mm']['median']:.3f} mm")

        ax_n.hist(ok["n_slices"], bins=40, color="#C44E52",
                  edgecolor="white")
        ax_n.axvline(summary["n_slices"]["median"], c="black", ls="--", lw=1)
        ax_n.set_xlabel("n_slices per patient")
        ax_n.set_title(f"Slice count\nmedian = "
                       f"{summary['n_slices']['median']:.0f}")

        fig.suptitle(f"Per-patient native acquisition geometry "
                     f"(N = {len(ok)})", fontweight="bold")
        fig.tight_layout()
        png_path = out_dir / "pixel_spacing_histogram.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        log.info("wrote %s", png_path)
    except Exception as e:
        log.warning("matplotlib failed (%s); CSV + JSON still written", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
