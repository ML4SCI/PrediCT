#!/usr/bin/env python
"""Sensitivity probe for the lesion-grouping rule (D007).

Picks 10 patients spanning the Agatston quartiles and reports the lesion
count at every combination of ``(max_inplane_mm ∈ {3, 5, 8}) × (max_slice_gap
∈ {1, 2})``. If the default (5 mm, gap=1) sits in a stable region (counts
within ±10% of neighbouring settings on most patients), D007 stands.

Outputs ``outputs/03_features/lesion_grouping_probe.csv``.

Run after stage 3 has been executed (needs ``features.csv`` to stratify).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from predict.config import load_config
from predict.features.lesion_ccl import group_rois_into_lesions
from predict.io import load_patient_metadata, parse_calcium_xml


THRESHOLDS_MM = (3.0, 5.0, 8.0)
GAPS = (1, 2)
N_PATIENTS = 10


def _select_patients(features_csv: Path) -> list[str]:
    df = pd.read_csv(features_csv, dtype={"pid": str})
    df = df.sort_values("agatston_total").reset_index(drop=True)
    # 10 evenly-spaced indices across the Agatston ranking.
    idx = [int(round(i * (len(df) - 1) / (N_PATIENTS - 1))) for i in range(N_PATIENTS)]
    return [df.iloc[i]["pid"] for i in idx]


def _count_per_setting(pid: str, data_root: Path, xml_dir: Path) -> dict:
    """Return dict of per-patient lesion counts at each (mm, gap) setting."""
    meta = load_patient_metadata(pid, data_root)
    pr = parse_calcium_xml(pid, xml_dir)
    row: dict = {"pid": pid}
    for mm in THRESHOLDS_MM:
        for gap in GAPS:
            lesions = group_rois_into_lesions(
                pr,
                slice_positions=meta.slice_positions,
                pixel_spacing_xy=(meta.pixel_spacing[0], meta.pixel_spacing[1]),
                slice_thickness_mm=meta.slice_thickness,
                max_inplane_mm=mm,
                max_slice_gap=gap,
            )
            row[f"n_{int(mm)}mm_gap{gap}"] = sum(len(v) for v in lesions.values())
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log = logging.getLogger("probe")

    cfg = load_config(args.config)
    out_dir = cfg.paths.outputs / "03_features"
    features_csv = out_dir / "features.csv"
    pids = _select_patients(features_csv)
    log.info("Probing %d patients across Agatston quartiles: %s", len(pids), pids)

    rows = [_count_per_setting(pid, cfg.paths.data_raw.parent, cfg.paths.data_xml)
            for pid in pids]
    df = pd.DataFrame(rows)

    # Compute % change vs the default column (5mm, gap=1).
    base_col = "n_5mm_gap1"
    pct_cols = []
    for col in df.columns:
        if col in ("pid", base_col):
            continue
        pct_col = col + "_pct_vs_base"
        df[pct_col] = ((df[col] - df[base_col]) / df[base_col].replace(0, 1) * 100).round(1)
        pct_cols.append(pct_col)

    out_path = out_dir / "lesion_grouping_probe.csv"
    df.to_csv(out_path, index=False)
    log.info("Wrote %s", out_path)
    print("\n" + df.to_string(index=False))
    print("\nAbsolute max deviation from (5mm, gap=1) across neighbour settings:")
    print(df[pct_cols].abs().max().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
