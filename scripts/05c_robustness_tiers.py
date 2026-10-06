#!/usr/bin/env python
"""Stage 5c, D034 robustness tiers (exploratory) on top of the D033 gate.

No new extraction. Reads the stage-5b reports and the stage-4b primary draws:

  1. Grades every feature on the nested ladder in predict.stability.tiers
     (T1 primary -> T2 lower CI -> T3 + S1 z-shift -> T4 + all measurable
     arms -> T5 + ICC >= 0.90).
  2. Check: recomputes ICC(3,1) absolute agreement on the same 20-draw
     matrices and reports how far it is from ICC(1,1) and whether any
     feature would change pass status. Random draws have no systematic
     rater effect, so the two should nearly coincide.
  3. Writes, per tier, icc_report.csv + gated_features.csv in the format
     stage 6 reads, so any tier can be fed downstream.

Usage
-----
    python scripts/05c_robustness_tiers.py

Outputs (outputs/05c_robustness_tiers/):
    tiers.csv                    one row per feature: criteria flags, tier, stops_at
    icc3_check.csv               ICC(1,1) vs ICC(3,1) per feature
    <tier>/icc_report.csv, <tier>/gated_features.csv
    summary.txt
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
_log = logging.getLogger("stage5c")


def _load_script_module(rel_path: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from predict.config import load_config
    from predict.stability.icc import icc_3_1_absolute
    from predict.stability.realistic import N_PRIMARY_DRAWS
    from predict.stability.tiers import TIERS, assign_tiers

    cfg = load_config()
    outputs = cfg.paths.outputs
    thr = float(cfg.stability.icc_threshold)
    gate_dir = outputs / "05b_icc_realistic"
    draw_dir = outputs / "04b_realistic"
    out_dir = args.out_dir or outputs / "05c_robustness_tiers"
    if (gate_dir / "PARTIAL_SMOKE_DO_NOT_USE").exists():
        raise SystemExit("05b output is a partial smoke run; rerun stage 5b on the full cohort")

    stage5b = _load_script_module("scripts/05b_icc_gate_realistic.py", "_stage5b")
    stage4 = _load_script_module("scripts/04_perturbations.py", "_prod_stage4")

    primary = pd.read_csv(gate_dir / "icc_report_full.csv").set_index("feature")
    arms = pd.read_csv(gate_dir / "arms.csv")
    arm_pass = arms.pivot(index="feature", columns="arm", values="passes_gate").astype(bool)
    tiers = assign_tiers(primary, arm_pass, thr)
    tiers.insert(0, "family", primary["family"])

    # ── ICC(3,1) check on the same matrices (same patients as 5b) ─────────
    eligible = stage4._eligible_pids(outputs / "03_features" / "features.csv")
    parity = pd.read_csv(draw_dir / "parity.csv", dtype={"pid": str})
    ok = set(parity.loc[parity["ok"].astype(bool), "pid"])
    draws = pd.read_csv(draw_dir / "draws_primary.csv", dtype={"pid": str})
    present = set(draws["pid"])
    pids = [p for p in eligible if p in ok and p in present]
    if len(pids) != len(eligible):
        raise SystemExit(f"{len(eligible) - len(pids)} eligible pids missing or failed parity")
    rows = []
    for f in primary.index:
        m = stage5b._matrix(draws, f, pids, N_PRIMARY_DRAWS)
        icc3 = icc_3_1_absolute(m)
        icc1 = float(primary.at[f, "icc"])
        rows.append({"feature": f, "icc_1_1": icc1, "icc_3_1": icc3, "diff": icc3 - icc1,
                     "pass_1_1": bool(primary.at[f, "passes_gate"]),
                     "pass_3_1_same_rules": bool(primary.at[f, "passes_gate"]) if
                     primary.at[f, "icc_source"] == "invariant_by_construction" else
                     bool(np.isfinite(icc3) and icc3 >= thr
                          and primary.at[f, "excluded_frac"] <= stage5b.MAX_EXCLUDED_FRAC)})
    icc3 = pd.DataFrame(rows)
    flips = icc3[icc3["pass_1_1"] != icc3["pass_3_1_same_rules"]]

    # ── outputs ────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    tiers.to_csv(out_dir / "tiers.csv", index_label="feature")
    icc3.to_csv(out_dir / "icc3_check.csv", index=False)
    compat = primary.reset_index()[["feature", "icc", "icc_source", "n_subjects"]].copy()
    compat["icc"] = compat["icc"].round(6)
    compat["n_raters"] = N_PRIMARY_DRAWS
    for name in TIERS:
        member = tiers[name]
        d = out_dir / name
        d.mkdir(exist_ok=True)
        c = compat.assign(passes_gate=compat["feature"].map(member).astype(bool))
        c[["feature", "icc", "icc_source", "n_subjects", "n_raters", "passes_gate"]].to_csv(
            d / "icc_report.csv", index=False)
        pd.DataFrame({"feature": member.index[member].tolist()}).to_csv(
            d / "gated_features.csv", index=False)

    L = ["=" * 84, "D034 ROBUSTNESS TIERS (exploratory; built on the D033 gate)", "=" * 84,
         f"features {len(tiers)}; threshold {thr}; patients {len(pids)}", ""]
    fams = sorted(tiers["family"].unique(), key=lambda s: (s != "canonical", s))
    L.append(f"{'tier':<16}{'total':>7}" + "".join(f"{f[:10]:>12}" for f in fams))
    for name in TIERS:
        m = tiers[name]
        L.append(f"{name:<16}{int(m.sum()):>7}"
                 + "".join(f"{int(m[tiers.family == f].sum()):>7}/{int((tiers.family == f).sum()):<4}" for f in fams))
    L.append("")
    L.append("Where each feature stops (first rung failed):")
    for reason, g in tiers[tiers["stops_at"] != ""].groupby("stops_at", sort=False):
        L.append(f"  {reason:<26}{len(g):>4}   {', '.join(s.replace('original_', '') for s in g.index[:12])}"
                 + (" ..." if len(g) > 12 else ""))
    L.append("")
    fin = icc3.dropna(subset=["icc_3_1"])
    L.append(f"ICC(3,1) check: max |ICC(3,1) - ICC(1,1)| = {fin['diff'].abs().max():.4f} "
             f"(median {fin['diff'].abs().median():.5f}); pass-status flips: {len(flips)}"
             + (f" {flips['feature'].tolist()}" if len(flips) else ""))
    L.append("=" * 84)
    text = "\n".join(L)
    (out_dir / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
