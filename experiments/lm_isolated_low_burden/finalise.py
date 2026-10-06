#!/usr/bin/env python
"""Bundle outputs from run.py into a human-readable report.txt.

Reads all the JSON + CSV outputs and prints a structured summary that
mirrors the findings.md narrative.

Usage:
    python experiments/lm_isolated_low_burden/finalise.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from predict.config import load_config
from predict.stage9 import stage9_root


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _report_text(out_dir: Path) -> str:
    summary = _load_json(out_dir / "summary.json") or {}
    fisher = _load_json(out_dir / "fisher_test.json") or {}
    cs = _load_json(out_dir / "cross_stratum.json") or {}
    dprof = _load_json(out_dir / "density_profile.json") or {}
    coverlap = _load_json(out_dir / "cluster_overlap.json") or {}

    disp_path = out_dir / "displaced_patients.csv"
    disp_df = pd.read_csv(disp_path) if disp_path.exists() else pd.DataFrame()

    lines: list[str] = []
    lines.append("Isolated LM calcification at low total Agatston burden")
    lines.append("=" * 60)
    lines.append("")

    # ── Headline numbers ──
    lines.append("HEADLINE")
    lines.append(f"  Cohort N                       = {summary.get('n_cohort')}")
    lines.append(f"  Low-burden tertile N           = {summary.get('n_low_tertile')}")
    lines.append(f"  Displaced subgroup N           = {summary.get('n_displaced')}")
    lines.append(f"  Displaced LM-positive          = {summary.get('displaced_lm_count')}")
    lines.append(f"  Displaced LM rate              = {summary.get('displaced_lm_rate', 0)*100:.1f}%")
    lo = summary.get('wilson95_lo')
    hi = summary.get('wilson95_hi')
    if lo is not None and hi is not None:
        lines.append(f"  Wilson 95% CI                  = ({lo*100:.1f}%, {hi*100:.1f}%)")
    lines.append("")

    # ── P2 Fisher test ──
    lines.append("P2: Fisher exact (displaced vs non-displaced LM rate)")
    lines.append(f"  Contingency table              = {fisher.get('table')}")
    lines.append(f"  Displaced LM rate              = {fisher.get('displaced_lm_rate', 0)*100:.1f}%")
    lines.append(f"  Non-displaced LM rate          = {fisher.get('non_displaced_lm_rate', 0)*100:.1f}%")
    lines.append(f"  Odds ratio                     = {fisher.get('fisher_odds_ratio')}")
    lines.append(f"  Fisher p (one-sided greater)   = {fisher.get('fisher_p_one_sided_greater')}")
    lines.append(f"  P2 PASS                        = {fisher.get('passes')}")
    lines.append("")

    # ── P3 cross-stratum ──
    lines.append("P3: cross-stratum replication")
    for kern, info in (cs.get("per_stratum") or {}).items():
        lines.append(f"  [{kern}]")
        lines.append(f"    n_low                       = {info.get('n_low')}")
        lines.append(f"    n_displaced                 = {info.get('n_displaced')}")
        lines.append(f"    LM rate (displaced)         = {info.get('displaced_lm_rate', 0)*100:.1f}%")
        lines.append(f"    LM rate (non-displaced)     = {info.get('non_displaced_lm_rate', 0)*100:.1f}%")
        lines.append(f"    Wilson 95% CI               = "
                     f"({info.get('displaced_wilson95_lo', 0)*100:.1f}%, "
                     f"{info.get('displaced_wilson95_hi', 0)*100:.1f}%)")
        lines.append(f"    Stratum PASS                = {info.get('passes')}")
    lines.append(f"  Overall P3 PASS                = {cs.get('passes')}")
    lines.append("")

    # ── P4 density ──
    lines.append("P4: density profile of displaced patients' LM lesions")
    lines.append(f"  n                              = {dprof.get('n')}")
    lines.append(f"  Median max-HU                  = {dprof.get('median_max_hu_lm')}")
    lines.append(f"  Min / max max-HU               = "
                 f"{dprof.get('min_max_hu_lm')} / {dprof.get('max_max_hu_lm')}")
    lines.append(f"  Tier breakdown                 = {dprof.get('tier_breakdown')}")
    lines.append(f"  Soft (W1/W2)                   = {dprof.get('n_soft_w1_w2')}")
    lines.append(f"  Dense (W3/W4)                  = {dprof.get('n_dense_w3_w4')}")
    lines.append(f"  Framing                        = {dprof.get('framing')}")
    lines.append("")

    # ── P5 cluster overlap ──
    lad_ids = coverlap.get("lad_dominant_clusters_used", "?")
    lines.append(f"P5 - overlap with LAD-phenotype clusters {lad_ids} (discovered by signature)")
    if coverlap.get("evaluable") is False:
        lines.append(f"  NOT EVALUABLE: {coverlap.get('reason')}")
        lines.append("")
        coverlap = {}
    lines.append(f"  Total LM lesions               = {coverlap.get('lesion_count')}")
    lines.append(f"  In LAD-phenotype clusters      = {coverlap.get('n_in_lad_dominant_clusters')}")
    frac = coverlap.get("overlap_fraction")
    lines.append(f"  Overlap fraction               = "
                 f"{'n/a' if frac is None else f'{frac * 100:.1f}%'}")
    lines.append(f"  Threshold for distinctness     = "
                 f"{coverlap.get('overlap_threshold', 0)*100:.0f}%")
    lines.append(f"  Cluster breakdown              = {coverlap.get('cluster_breakdown')}")
    lines.append(f"  Framing                        = {coverlap.get('framing')}")
    lines.append(f"  P5 distinctness PASS           = {coverlap.get('passes_distinctness')}")
    lines.append("")

    # ── Per-patient table ──
    if not disp_df.empty:
        lines.append("Per-patient displaced subgroup")
        keep_cols = [c for c in (
            "pid", "kernel", "pc1", "pc2",
            "agatston_total", "agatston_lm", "max_hu_lm",
            "lesion_count_lm", "n_calcified_arteries",
        ) if c in disp_df.columns]
        lines.append(disp_df[keep_cols].sort_values("agatston_lm").to_string(index=False))
        lines.append("")

    # ── Overall verdict ──
    lines.append("OVERALL VERDICT")
    lines.append(f"  All criteria pass (P2 + P3 + P5)  = "
                 f"{summary.get('overall_passes_all_criteria')}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = (stage9_root(cfg) / "lm_isolated_low_burden")
    if not out_dir.exists():
        print(f"missing {out_dir}; nothing to finalise.")
        return 2

    text = _report_text(out_dir)
    (out_dir / "report.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
