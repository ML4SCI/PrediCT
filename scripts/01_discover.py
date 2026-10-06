#!/usr/bin/env python
"""Stage 1: patient discovery.

Builds the cohort manifest from ``paths.data_raw`` and ``paths.data_xml``,
applies the config-driven exclusions (D004) and the slice geometry check
(D039), and writes ``outputs/01_manifest/manifest.csv`` plus
``exclusions.csv`` (one row per excluded pid, with the reason).

Usage::

    python scripts/01_discover.py [--config configs/default.yaml]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from predict.config import load_config
from predict.io import discover_patients, manifest_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to config YAML (default: configs/default.yaml)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("discover")

    cfg = load_config(args.config)
    raw_dir, xml_dir = cfg.paths.data_raw, cfg.paths.data_xml
    out_dir = cfg.paths.outputs / "01_manifest"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("raw_dir = %s", raw_dir)
    log.info("xml_dir = %s", xml_dir)
    log.info("exclude_pids = %s", cfg.cohort.exclude_pids)
    log.info("exclude_ge_scanners = %s", cfg.cohort.exclude_ge_scanners)

    result = discover_patients(
        raw_dir,
        xml_dir,
        exclude_pids=cfg.cohort.exclude_pids,
        exclude_ge_scanners=cfg.cohort.exclude_ge_scanners,
    )

    df = pd.DataFrame(manifest_rows(result))
    out_path = out_dir / "manifest.csv"
    df.to_csv(out_path, index=False)
    log.info("Wrote %s (%d patients)", out_path, len(df))

    # Audit trail of exclusions, useful in the paper.
    audit_path = out_dir / "exclusions.csv"
    audit_rows = []
    for pid in result.excluded_no_dicom:
        audit_rows.append({"pid": pid, "reason": "no_dicom", "detail": ""})
    for pid in result.excluded_no_xml:
        audit_rows.append({"pid": pid, "reason": "no_xml", "detail": ""})
    for pid in result.excluded_by_config:
        audit_rows.append({"pid": pid, "reason": "config_excluded", "detail": ""})
    for pid in result.excluded_ge:
        audit_rows.append({"pid": pid, "reason": "ge_scanner", "detail": ""})
    for pid, why in result.excluded_geometry:
        audit_rows.append({"pid": pid, "reason": "irregular_geometry", "detail": why})
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    log.info("Wrote %s (%d exclusions)", audit_path, len(audit_rows))

    # Kernel breakdown for sanity.
    if not df.empty:
        log.info("Kernel breakdown:\n%s", df["kernel"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
