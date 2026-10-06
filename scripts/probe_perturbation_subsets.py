#!/usr/bin/env python
"""Sensitivity probe: does the ICC gate depend on which perturbations are in it?

Motivation
----------
D014 locks 14 perturbations (rotations +/-5, +/-10 deg; translations +/-2,
+/-5 mm in x and y; noise sigma 5, 10 HU) with the mask held fixed. Only 20 of
107 PyRadiomics features pass ICC >= 0.75. The concern: median per-vessel ROI
area is ~5 mm^2 (~2.5 mm across), so a 5 mm shift moves most lesions entirely
out of the fixed mask. If +/-5 mm dominates the gate, it may be testing
"survives total signal loss" rather than "reliable under plausible error".

This script answers that empirically. It is a SENSITIVITY PROBE. It does not
change D014 or the primary gate. Changing the primary would need a new decision
whose rationale does not depend on what this probe shows.

What it does
------------
Recomputes ICC(3,1) absolute agreement from the perturbation CSVs already on
disk (outputs/04_perturbations/*.csv). Nothing is re-extracted. For each
perturbation subset x cohort it reports the pass count, the per-family
breakdown, and exactly which features enter or leave the survivor set
relative to the full 14.

Subsets are built from PerturbationSpec fields (kind, magnitude), not by
string-matching names, so they cannot silently drift from D014.

Two cohorts: all eligible patients (D015), and low_burden_flag == False
(larger masks). If the +/-5 mm losses are driven by small lesions sliding out
of the mask, they should shrink in the larger-mask cohort. That is the
mechanism test.

Safety / parity
---------------
Before reporting anything, the full-14 x full-cohort ICCs are compared to
outputs/05_icc/icc_report.csv. The report stores round(icc, 6), so the probe
applies the same rounding and requires EXACT equality on every feature. If
any differs the script stops: a probe that cannot reproduce production is not
measuring the same thing. Writes only to --out-dir.

Usage
-----
    python scripts/probe_perturbation_subsets.py

Outputs (into --out-dir, default outputs/probe_perturbation_subsets/):
    icc_by_subset.csv       one row per (cohort, subset, feature)
    survivor_diff.csv       features gained / lost vs the full 14
    subset_summary.txt      human-readable report
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
_log = logging.getLogger("probe_perturbation_subsets")

# icc_report.csv stores ICC as round(icc, 6) (05_icc_gate.py:194).
REPORT_DECIMALS = 6


def _load_script_module(rel_path: str, alias: str):
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


# Production loaders from the ICC gate. Imported, never copied.
_gate = _load_script_module("scripts/05_icc_gate.py", "_prod_icc_gate")


# ───────────────────────── subset definitions ─────────────────────────
# Each predicate takes a PerturbationSpec and returns True to KEEP it.

def _is_trans(s, mm=None) -> bool:
    if s.kind != "translate":
        return False
    if mm is None:
        return True
    return max(abs(s.tx_mm), abs(s.ty_mm)) == float(mm)


SUBSETS: dict[str, tuple[str, Callable]] = {
    "all_14":          ("D014 primary (reference)",
                        lambda s: True),
    "drop_trans_5mm":  ("drop the four +/-5 mm translations",
                        lambda s: not _is_trans(s, 5)),
    "drop_all_trans":  ("rotations + noise only",
                        lambda s: not _is_trans(s)),
    "trans_2mm_only":  ("only the four +/-2 mm translations",
                        lambda s: _is_trans(s, 2)),
    "trans_5mm_only":  ("only the four +/-5 mm translations",
                        lambda s: _is_trans(s, 5)),
    "trans_only":      ("all eight translations",
                        lambda s: _is_trans(s)),
}


def _family(feature: str) -> str:
    # original_<family>_<name>
    parts = feature.split("_", 2)
    return parts[1] if len(parts) > 2 else "unknown"


# ───────────────────────── core ─────────────────────────


def icc_for_subset(features: list[str], baseline: pd.DataFrame,
                   pert_dfs: dict[str, pd.DataFrame],
                   keep: list[str]) -> dict[str, tuple[float, int, int]]:
    from predict.stability.icc import build_reliability_matrix, icc_3_1_absolute

    sub = {k: pert_dfs[k] for k in keep}
    out: dict[str, tuple[float, int, int]] = {}
    for feat in features:
        matrix, _pids, _raters = build_reliability_matrix(
            feat, baseline, sub, pid_col="pid",
        )
        if matrix.size == 0:
            out[feat] = (float("nan"), 0, 0)
            continue
        n_complete = int(np.sum(~np.isnan(matrix).any(axis=1)))
        out[feat] = (float(icc_3_1_absolute(matrix)), n_complete,
                     int(matrix.shape[1]))
    return out


def parity_check(icc_full: dict[str, tuple[float, int, int]],
                 icc_report: Path) -> None:
    if not icc_report.exists():
        raise FileNotFoundError(
            f"{icc_report} missing; cannot verify the probe reproduces "
            f"production. Run scripts/05_icc_gate.py first.")
    rep = pd.read_csv(icc_report)
    rep = rep[rep["icc_source"] == "empirical"].set_index("feature")["icc"]

    # 05_icc_gate.py:194 writes `round(r.icc, 6)`, so the report holds
    # 6-decimal values while this probe computes full precision. Comparing
    # raw values would flag pure rounding (max 5e-7) as a mismatch. Instead
    # apply the SAME rounding and require EXACT equality: stricter than any
    # tolerance, and it fails on a genuine 1e-6 discrepancy.
    mismatches: list[tuple[str, float, float]] = []
    worst_raw, n_checked = 0.0, 0
    for feat, (icc, _, _) in icc_full.items():
        if feat not in rep.index:
            raise KeyError(f"{feat} absent from {icc_report}")
        ref = float(rep.loc[feat])
        if np.isnan(icc) and np.isnan(ref):
            continue
        n_checked += 1
        worst_raw = max(worst_raw, abs(icc - ref))
        if round(icc, REPORT_DECIMALS) != ref:
            mismatches.append((feat, icc, ref))
    if mismatches:
        f, a, b = mismatches[0]
        raise SystemExit(
            f"PARITY FAILED on {len(mismatches)} feature(s), e.g. {f}: probe "
            f"{a:.9f} rounds to {round(a, REPORT_DECIMALS)}, report has {b}. "
            f"The probe is not measuring what the gate measured; refusing "
            f"to report subset results.")
    _log.info("parity OK: %d features match icc_report.csv exactly at %d "
              "decimals (raw max |delta| %.2e, pure rounding)",
              n_checked, REPORT_DECIMALS, worst_raw)


def render(summary: pd.DataFrame, diff: pd.DataFrame,
           fam: pd.DataFrame, threshold: float) -> str:
    L: list[str] = []
    add = L.append
    add("=" * 80)
    add("ICC GATE SENSITIVITY TO THE PERTURBATION SET  (sensitivity probe, not primary)")
    add("=" * 80)
    add(f"Threshold ICC >= {threshold}. Parity with icc_report.csv verified first.")
    add("")

    add("-" * 80)
    add("1. PASS COUNT BY SUBSET  (of 107 PyRadiomics features)")
    add("-" * 80)
    add(f"{'subset':<16} {'raters':>6} {'all eligible':>13} "
        f"{'low_burden=F':>13}   description")
    for name, (desc, _) in SUBSETS.items():
        r = summary[summary["subset"] == name]
        a = r[r["cohort"] == "all_eligible"]
        b = r[r["cohort"] == "low_burden_false"]
        add(f"{name:<16} {int(a['n_raters'].iloc[0]):>6} "
            f"{int(a['n_pass'].iloc[0]):>13} {int(b['n_pass'].iloc[0]):>13}   {desc}")
    add("")
    add("Read: if drop_trans_5mm passes far more than all_14, the +/-5 mm")
    add("translations are the binding constraint. If trans_2mm_only is also")
    add("harsh, the 'shift >= lesion size' argument indicts +/-2 mm too.")
    add("If losses shrink in low_burden=F, small masks drive them.")
    add("")

    add("-" * 80)
    add("2. PER-FAMILY PASS COUNT  (all eligible)")
    add("-" * 80)
    fams = sorted(fam["family"].unique())
    add(f"{'subset':<16} " + " ".join(f"{f:>10}" for f in fams))
    for name in SUBSETS:
        row = fam[(fam["subset"] == name) & (fam["cohort"] == "all_eligible")]
        cnt = {r["family"]: f"{int(r['n_pass'])}/{int(r['n_total'])}"
               for _, r in row.iterrows()}
        add(f"{name:<16} " + " ".join(f"{cnt.get(f, '-'):>10}" for f in fams))
    add("")
    add("Sanity: shape must pass 14/14 in EVERY subset (mask is fixed). If not,")
    add("the perturbation pipeline has a bug.")
    add("")

    add("-" * 80)
    add("3. SURVIVOR SET CHANGES vs all_14  (all eligible)")
    add("-" * 80)
    for name in SUBSETS:
        if name == "all_14":
            continue
        d = diff[(diff["subset"] == name) & (diff["cohort"] == "all_eligible")]
        gained = d[d["change"] == "gained"]["feature"].tolist()
        lost = d[d["change"] == "lost"]["feature"].tolist()
        add(f"{name}: +{len(gained)} gained, -{len(lost)} lost")
        for f in gained:
            add(f"    + {f}")
        for f in lost:
            add(f"    - {f}")
    add("")
    add("=" * 80)
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("outputs/probe_perturbation_subsets"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    from predict.config import load_config
    from predict.stability.perturbations import enumerate_perturbations

    cfg = load_config()
    threshold = float(cfg.stability.icc_threshold)
    outputs = cfg.paths.outputs

    baseline_all = _gate._load_baseline(outputs / "03_features" / "features.csv")
    specs = enumerate_perturbations(cfg)
    names = [s.name for s in specs]
    pert_dfs = _gate._load_perturbation_csvs(
        outputs / "04_perturbations", names,
        set(baseline_all["pid"].astype(str)),
    )
    features = _gate._identify_empirical_features(baseline_all, pert_dfs)
    _log.info("eligible N=%d, perturbations=%d, empirical features=%d",
              len(baseline_all), len(pert_dfs), len(features))

    if "low_burden_flag" not in baseline_all.columns:
        raise KeyError("features.csv missing low_burden_flag")
    lb = baseline_all["low_burden_flag"].astype(str).str.lower().isin(
        ("true", "1", "1.0"))
    cohorts = {
        "all_eligible": baseline_all,
        "low_burden_false": baseline_all[~lb],
    }
    _log.info("low_burden_false N=%d", len(cohorts["low_burden_false"]))

    long_rows: list[dict] = []
    survivors: dict[tuple[str, str], set[str]] = {}

    for cname, base in cohorts.items():
        for sname, (_desc, pred) in SUBSETS.items():
            keep = [s.name for s in specs if pred(s)]
            res = icc_for_subset(features, base, pert_dfs, keep)
            if cname == "all_eligible" and sname == "all_14":
                parity_check(res, outputs / "05_icc" / "icc_report.csv")
            passed = set()
            for feat, (icc, n, k) in res.items():
                ok = (not np.isnan(icc)) and icc >= threshold
                if ok:
                    passed.add(feat)
                long_rows.append({
                    "cohort": cname, "subset": sname, "feature": feat,
                    "family": _family(feat), "icc": icc,
                    "n_subjects": n, "n_raters": k, "passes": ok,
                })
            survivors[(cname, sname)] = passed
            _log.info("%-17s %-15s raters=%2d  pass=%3d",
                      cname, sname, len(keep) + 1, len(passed))

    long = pd.DataFrame(long_rows)

    summary = (long.groupby(["cohort", "subset"])
               .agg(n_pass=("passes", "sum"), n_raters=("n_raters", "max"))
               .reset_index())
    fam = (long.groupby(["cohort", "subset", "family"])
           .agg(n_pass=("passes", "sum"), n_total=("passes", "size"))
           .reset_index())

    diff_rows: list[dict] = []
    for cname in cohorts:
        ref = survivors[(cname, "all_14")]
        for sname in SUBSETS:
            if sname == "all_14":
                continue
            cur = survivors[(cname, sname)]
            for f in sorted(cur - ref):
                diff_rows.append({"cohort": cname, "subset": sname,
                                  "feature": f, "change": "gained"})
            for f in sorted(ref - cur):
                diff_rows.append({"cohort": cname, "subset": sname,
                                  "feature": f, "change": "lost"})
    diff = pd.DataFrame(diff_rows, columns=["cohort", "subset", "feature", "change"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    long.to_csv(args.out_dir / "icc_by_subset.csv", index=False)
    diff.to_csv(args.out_dir / "survivor_diff.csv", index=False)
    report = render(summary, diff, fam, threshold)
    (args.out_dir / "subset_summary.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    _log.info("wrote %s", args.out_dir)
    _log.info("outputs/05_icc was NOT modified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
