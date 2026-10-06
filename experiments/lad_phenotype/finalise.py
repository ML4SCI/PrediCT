#!/usr/bin/env python
"""Step 5 of the LAD-phenotype experiment.

Reads the full-cohort + per-stratum outputs and:
  - Bundles cross-stratum replication into one JSON.
  - Writes a human-readable report.txt summary.

Pre-registered replication PASS criterion (per plan.md):
  Each verdict must hold (PASS) in BOTH strata independently, OR each
  stratum's effect-size estimate must be within +/-0.10 of the
  full-cohort estimate.

Run AFTER:
  python experiments/lad_phenotype/run.py
  python experiments/lad_phenotype/run.py --stratum Qr36d/2
  python experiments/lad_phenotype/run.py --stratum I30f/3
  python experiments/lad_phenotype/matched.py
  python experiments/lad_phenotype/matched.py --stratum Qr36d/2
  python experiments/lad_phenotype/matched.py --stratum I30f/3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from predict.config import load_config
from predict.stage9 import stage9_root

REPLICATION_DELTA_TOL = 0.10
STRATA = ("Qr36d_2", "I30f_3")

_log = logging.getLogger("lad_phenotype.finalise")


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8",
    )


def _gather_one(d: Path) -> dict:
    """One cohort's results. The matched comparison is read only where the
    matching step reports success; a comparison next to a terminated match
    is a stale or inconsistent result and stops the finaliser (D043)."""
    sig = _load_json(d / "lad_cluster_signature.json")
    md = _load_json(d / "matched_diagnostics.json")
    if sig is None or md is None:
        raise SystemExit(f"hard stop: {d} lacks lad_cluster_signature.json or "
                         "matched_diagnostics.json; run run.py and matched.py for it")
    mc = _load_json(d / "matched_comparison_summary.json")
    if (md.get("verdict") == "match_ok") != (mc is not None):
        raise SystemExit(f"hard stop: {d}: matching verdict {md.get('verdict')!r} but a matched "
                         f"comparison is {'present' if mc is not None else 'missing'}")
    return {
        "signature": sig,
        "axial": _load_json(d / "axial_summary.json"),
        "carrier": _load_json(d / "carrier_summary.json"),
        "match_diagnostics": md,
        "match_comparison": mc,
    }


def _gather(out_root: Path) -> dict:
    base = {"full": _gather_one(out_root)}
    for s in STRATA:
        base[s] = _gather_one(out_root / f"stratified_{s}")
    return base


def _status(per_stratum_ok: dict, not_evaluable: str | None = None) -> str:
    if not_evaluable:
        return "not_evaluable"
    if any(v is False for v in per_stratum_ok.values()):
        return "does_not_replicate"
    if all(v is True for v in per_stratum_ok.values()):
        return "replicates"
    return "not_evaluable"


def _compare_replication(full: dict, strata: dict) -> dict:
    """Cross-stratum replication per plan.md P5, per component (signature,
    axial, matched) with status replicates / does_not_replicate /
    not_evaluable. ``overall_replicates`` is True only when every component
    replicates, False when any does not, and None otherwise; it is never
    True for missing components (the old code returned True on no results)."""
    out: dict = {}
    full_ids = set(full["signature"].get("lad_dominant_cluster_ids", []))
    no_sig = None if full_ids else "no LAD-dominant cluster in the full cohort"

    # Signature: same cluster ids, each stratum computed on its own lesions.
    sig_ok = {s: set(strata[s]["signature"].get("lad_dominant_cluster_ids", [])) == full_ids
              for s in STRATA}
    out["signature_replication"] = {
        "full_ids": sorted(full_ids),
        "per_stratum": {s: strata[s]["signature"].get("lad_dominant_cluster_ids", []) for s in STRATA},
        "status": _status(sig_ok, no_sig),
    }

    # Axial verdict (plan P5): the stratum reaches the FULL cohort's verdict
    # (pass or fail), or its effect (median rel-z difference) is within
    # +/-0.10 of the full-cohort effect. Judging the stratum's own pass
    # instead would report a reversed verdict as replicated and a preserved
    # negative verdict as not replicated (D043).
    ax_ok: dict = {}
    per_stratum: dict = {}
    full_ax = full.get("axial")
    full_pass = bool(full_ax.get("passes")) if full_ax else None
    full_diff = float(full_ax.get("median_diff")) if full_ax else float("nan")
    for s in STRATA:
        ax = strata[s].get("axial")
        if ax is None or full_ax is None:   # no LAD-dominant cluster: not computed
            per_stratum[s], ax_ok[s] = None, None
            continue
        diff = float(ax.get("median_diff"))
        within = abs(diff - full_diff) <= REPLICATION_DELTA_TOL
        same = bool(ax.get("passes")) == full_pass
        per_stratum[s] = {"passes": bool(ax.get("passes")), "same_verdict_as_full": same,
                          "median_diff": diff, "within_tol": bool(within)}
        ax_ok[s] = same or bool(within)
    out["axial_replication"] = {
        "full_passes": full_pass,
        "full_median_diff": full_diff,
        "per_stratum": per_stratum,
        "status": _status(ax_ok, no_sig),
    }

    # Matched comparison: evaluable only where matching succeeded; the
    # stratum must reach the full cohort's verdict. The summary carries no
    # single effect-size estimate (three tests, different statistics), so
    # the plan's tolerance alternative does not apply here.
    full_md = full["match_diagnostics"]["verdict"]
    full_mc = full.get("match_comparison")
    mc_ok: dict = {}
    per_stratum = {}
    for s in STRATA:
        verdict = strata[s]["match_diagnostics"]["verdict"]
        mc = strata[s].get("match_comparison")
        per_stratum[s] = {"verdict": verdict,
                          "overall_pass": bool(mc["overall_pass"]) if mc else None}
        mc_ok[s] = (bool(mc["overall_pass"]) == bool(full_mc["overall_pass"])
                    if mc is not None and full_mc is not None else None)
    out["matched_replication"] = {
        "full_verdict": full_md,
        "full_overall_pass": bool(full["match_comparison"]["overall_pass"]) if full.get("match_comparison") else None,
        "per_stratum": per_stratum,
        "status": _status(mc_ok, None if full_md == "match_ok"
                          else f"full-cohort matching {full_md} (pre-registered termination)"),
    }

    statuses = [v["status"] for v in out.values()]
    out["overall_replicates"] = (False if "does_not_replicate" in statuses
                                 else True if all(st == "replicates" for st in statuses)
                                 else None)
    out["not_evaluable_components"] = [k for k, v in out.items()
                                       if isinstance(v, dict) and v.get("status") == "not_evaluable"]
    return out


def _report_text(bundle: dict, replication: dict) -> str:
    lines: list[str] = []
    lines.append("LAD-phenotype experiment - final report")
    lines.append("=" * 60)
    lines.append("")
    full = bundle.get("full", {})

    sig = full.get("signature")
    if sig is None:
        lines.append("STEP 1: signature discovery NOT RUN")
    else:
        ids = sig.get("lad_dominant_cluster_ids", [])
        lines.append(f"STEP 1: LAD-dominant cluster ids (pre-reg) = {ids}")
        if not ids:
            lines.append("  Termination: no cluster matches pre-registered "
                          "signature. Experiment terminates per plan.md.")
            return "\n".join(lines)

    ax = full.get("axial")
    if ax:
        lines.append("")
        lines.append("STEP 2: within-LAD axial localisation")
        lines.append(f"  rel-z median (LAD-cluster lesions) = "
                     f"{ax.get('median_in_cluster')}")
        lines.append(f"  rel-z median (other LAD lesions)   = "
                     f"{ax.get('median_other')}")
        lines.append(f"  median diff = {ax.get('median_diff')}")
        lines.append(f"  Mann-Whitney p (one-sided, less) = "
                     f"{ax.get('mw_p_one_sided_less')}")
        lines.append(f"  PASSES: {ax.get('passes')}")

    cs = full.get("carrier")
    if cs:
        lines.append("")
        lines.append("STEP 3: carrier patient signature")
        lines.append(f"  directional confirmations: "
                     f"{cs.get('n_passing_directional')}"
                     f"/{cs.get('n_total_directional')}")
        lines.append(f"  overall PASSES: {cs.get('overall_passes')}")

    md = full.get("match_diagnostics")
    if md:
        lines.append("")
        lines.append("STEP 4: burden-propensity match diagnostics")
        lines.append(f"  verdict: {md.get('verdict')}")
        lines.append(f"  yield = {md.get('match_yield')}")
        lines.append(f"  SMD pre = {md.get('smd_pre')}, "
                     f"post = {md.get('smd_post')}")

    mc = full.get("match_comparison")
    if mc:
        lines.append("")
        lines.append("STEP 4 (cont.): post-match tests")
        lines.append(f"  {mc.get('n_passing')}/{mc.get('n_total')} pass")
        lines.append(f"  interpretation: {mc.get('interpretation')}")

    lines.append("")
    lines.append("STEP 5: cross-stratum replication")
    lines.append(f"  overall replicates: {replication.get('overall_replicates')} "
                 f"(not evaluable: {replication.get('not_evaluable_components')})")
    for key, sub in replication.items():
        if isinstance(sub, dict) and "status" in sub:
            lines.append(f"  {key}: {sub['status']}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = load_config(args.config)
    out_root = stage9_root(cfg) / "lad_phenotype"
    if not out_root.exists():
        _log.error("missing %s; nothing to finalise.", out_root)
        return 2

    bundle = _gather(out_root)
    strata = {s: bundle.get(s, {}) for s in STRATA}
    replication = _compare_replication(bundle.get("full", {}), strata)
    _save_json(out_root / "cross_stratum_replication.json", replication)

    report = _report_text(bundle, replication)
    (out_root / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
