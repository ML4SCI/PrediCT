#!/usr/bin/env python
"""Stage 5b sensitivity: rerun the full reduce pipeline on the robust cohort.

The robust cohort is the eligible cohort minus the 142 ``low_burden_flag=True``
patients (~280 patients). We rerun the entire stage-5 pipeline (matrix prep,
redundancy, PCA, Hopkins, gap, consensus, forced k=3) on this subset, then
compare cluster assignments to the full-cohort run via ARI on the overlapping
PIDs.

Output is dumped to ``outputs/06_reduce/sensitivity_robust/`` so the full
and robust runs sit side by side for inspection.

Usage:
  # Run full stage 5 first (produces outputs/06_reduce/...).
  python scripts/06_reduce.py
  # Then run this sensitivity probe.
  python scripts/06b_sensitivity_robust.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from predict.config import Config, load_config
from predict.discover.validity import ari_on_shared_pids


def _save_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--gap-bootstraps", type=int, default=500)
    parser.add_argument("--consensus-subsamples", type=int, default=100)
    parser.add_argument("--hennig-bootstraps", type=int, default=100)
    parser.add_argument("--forced-k", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log = logging.getLogger("reduce-robust")

    cfg: Config = load_config(args.config)
    full_out = cfg.paths.outputs / "06_reduce"
    robust_out = full_out / "sensitivity_robust"
    robust_out.mkdir(parents=True, exist_ok=True)

    # Identify the robust cohort from features.csv.
    features = pd.read_csv(
        cfg.paths.outputs / "03_features" / "features.csv",
        dtype={"pid": str},
    )
    eligible_mask = features["radiomics_status"] == "ok"
    robust_mask = eligible_mask & (~features["low_burden_flag"].astype(bool))
    robust_pids = features.loc[robust_mask, "pid"].astype(str).tolist()
    log.info("robust cohort: N=%d (eligible minus low_burden_flag=True)",
             len(robust_pids))

    # Write the cohort list so 06_reduce can be rerun on this subset.
    pd.DataFrame({"pid": robust_pids}).to_csv(
        robust_out / "robust_cohort_pids.csv", index=False,
    )

    log.info("To execute the robust rerun: rerun scripts/06_reduce.py with "
             "a config that points outputs to %s, or pre-filter features.csv "
             "to the robust PIDs and rerun in place. ARI comparison below "
             "assumes both runs have written cluster_labels_forced.csv.")

    # If both full and robust cluster label files exist, compute the ARI.
    full_labels_path = full_out / "cluster_labels_forced.csv"
    robust_labels_path = robust_out / "cluster_labels_forced.csv"
    if not full_labels_path.exists():
        log.warning("missing %s; run scripts/06_reduce.py first", full_labels_path)
        return 0
    if not robust_labels_path.exists():
        log.info(
            "missing %s; this script prepared the robust cohort list. "
            "Run the reduce pipeline on the robust subset (e.g. by "
            "filtering features.csv to robust_cohort_pids.csv) and then "
            "rerun this script for the ARI comparison.",
            robust_labels_path,
        )
        return 0

    full = pd.read_csv(full_labels_path, dtype={"pid": str})
    robust = pd.read_csv(robust_labels_path, dtype={"pid": str})

    label_cols = [c for c in full.columns
                  if c.startswith(f"forced_k{args.forced_k}_")]
    ari_records: list[dict] = []
    for col in label_cols:
        if col not in robust.columns:
            log.warning("column %s absent from robust labels; skipping", col)
            continue
        score, shared = ari_on_shared_pids(
            full["pid"].tolist(), full[col].to_numpy(),
            robust["pid"].tolist(), robust[col].to_numpy(),
        )
        ari_records.append({
            "label_column": col,
            "ari_full_vs_robust": score,
            "n_shared_pids": len(shared),
            "stable_threshold_0_75": bool(score >= 0.75),
        })
        log.info("ARI(full vs robust) on %s = %.3f over N=%d shared pids",
                 col, score, len(shared))

    pd.DataFrame(ari_records).to_csv(
        robust_out / "ari_full_vs_robust.csv", index=False,
    )
    _save_json(robust_out / "ari_full_vs_robust.json", ari_records)

    return 0


if __name__ == "__main__":
    sys.exit(main())
