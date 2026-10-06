#!/usr/bin/env python
"""Step 4 of the LAD-phenotype experiment: burden-propensity matching.

Reads ``outputs/exploratory/lad_phenotype/carrier_profile.csv`` and
``axial_within_lad.csv`` plus stage-3 features. Builds a 1:k caliper
match on log(agatston_total + 1). Reports post-match SMD diagnostic
and the three pre-registered post-match tests.

Termination per plan.md: if SMD >= 0.1 or fewer than 50% of cases
match, the script writes the failure and exits non-zero. Caliper is
NOT relaxed.

Usage:
    python experiments/lad_phenotype/matched.py
    python experiments/lad_phenotype/matched.py --stratum Qr36d/2
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from predict.analyse import cliffs_delta, mannwhitney_u_pval
from predict.config import load_config
from predict.stage9 import stage9_root

sys.path.insert(0, str(Path(__file__).resolve().parent))
from propensity import (  # noqa: E402
    caliper_match,
    match_yield,
)


# ─── Pre-registered thresholds (locked in plan.md) ───

MATCH_CALIPER_SD = 0.2
MATCH_K = 3
MATCH_SMD_MAX = 0.1
MATCH_YIELD_MIN = 0.5

POST_MATCH_LAD_SHARE_CLIFFS_MIN = 0.20
POST_MATCH_PROXIMAL_PROP_P_MAX = 0.05
POST_MATCH_MAX_HU_LAD_CLIFFS_MIN = 0.15


_log = logging.getLogger("lad_phenotype.matched")


def _save_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--stratum", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = load_config(args.config)
    out_dir = stage9_root(cfg) / "lad_phenotype"
    if args.stratum:
        slug = args.stratum.replace("/", "_")
        out_dir = out_dir / f"stratified_{slug}"

    # No result of an earlier run may survive this one: a terminated match
    # must not leave an earlier comparison behind (D043).
    for stale in ("matched_pairs.csv", "matched_comparison.csv",
                  "matched_comparison_summary.json", "matched_diagnostics.json"):
        (out_dir / stale).unlink(missing_ok=True)

    sig_path = out_dir / "lad_cluster_signature.json"
    if not sig_path.exists():
        raise SystemExit(f"hard stop: {sig_path} missing; run experiments/lad_phenotype/run.py first")
    if not json.loads(sig_path.read_text(encoding="utf-8"))["lad_dominant_cluster_ids"]:
        # Step 1 terminated (no LAD-dominant cluster): nothing to match.
        _save_json(out_dir / "matched_diagnostics.json", {"verdict": "no_signature"})
        _log.warning("no LAD-dominant cluster (step 1 terminated); matching not run")
        return 0
    carrier_path = out_dir / "carrier_profile.csv"
    if not carrier_path.exists():
        raise SystemExit(f"hard stop: {carrier_path} missing although step 1 found clusters")
    carrier_table = pd.read_csv(carrier_path, dtype={"pid": str})

    # Build matching covariate: log(agatston_total + 1).
    carrier_table["match_var"] = np.log(
        carrier_table["agatston_total"].astype(float).clip(lower=0) + 1.0
    )

    cases_df = carrier_table[carrier_table["is_carrier"]]
    controls_df = carrier_table[~carrier_table["is_carrier"]]

    if cases_df.empty:
        _log.error("zero carriers; cannot match.")
        _save_json(out_dir / "matched_diagnostics.json", {
            "verdict": "no_carriers", "n_cases_in": 0,
        })
        # A pre-registered termination is an outcome, not a failure: exit 0 so
        # pipeline runners continue; the verdict is in matched_diagnostics.json.
        return 0

    cases = pd.Series(
        cases_df["match_var"].to_numpy(),
        index=cases_df["pid"].astype(str),
    )
    controls = pd.Series(
        controls_df["match_var"].to_numpy(),
        index=controls_df["pid"].astype(str),
    )

    result = caliper_match(
        cases, controls,
        caliper_sd=MATCH_CALIPER_SD, k=MATCH_K, random_state=42,
    )
    yield_ = match_yield(result)
    _log.info(
        "matching: cases_in=%d controls_in=%d cases_matched=%d "
        "yield=%.2f smd_pre=%.3f smd_post=%.3f caliper=%.3f",
        result.n_cases_in, result.n_controls_in, result.n_cases_matched,
        yield_, result.smd_pre, result.smd_post, result.caliper,
    )

    diagnostics = {
        "n_cases_in": result.n_cases_in,
        "n_controls_in": result.n_controls_in,
        "n_cases_matched": result.n_cases_matched,
        "n_controls_used": result.n_controls_used,
        "match_yield": yield_,
        "smd_pre": result.smd_pre,
        "smd_post": result.smd_post,   # matched cases once vs matched controls (as compared)
        "smd_post_pair_weighted": result.smd_post_pair_weighted,   # pre-2026-10-05 definition
        "caliper_sd": MATCH_CALIPER_SD,
        "caliper_absolute": result.caliper,
        "k": MATCH_K,
        "thresholds": {
            "smd_post_max": MATCH_SMD_MAX,
            "yield_min": MATCH_YIELD_MIN,
        },
    }

    # Termination check.
    smd_ok = abs(result.smd_post) < MATCH_SMD_MAX
    yield_ok = yield_ >= MATCH_YIELD_MIN
    if not (smd_ok and yield_ok):
        diagnostics["verdict"] = "match_infeasible"
        diagnostics["smd_ok"] = bool(smd_ok)
        diagnostics["yield_ok"] = bool(yield_ok)
        _save_json(out_dir / "matched_diagnostics.json", diagnostics)
        _log.warning("match infeasible at pre-registered caliper. "
                     "Per plan.md, experiment terminates without relaxing.")
        return 0   # pre-registered termination (see the no_carriers branch)
    diagnostics["verdict"] = "match_ok"
    result.pairs.to_csv(out_dir / "matched_pairs.csv", index=False)
    _save_json(out_dir / "matched_diagnostics.json", diagnostics)

    # ── Post-match tests (P4) ────────────────────────────────
    matched_case_pids = set(result.case_pids_matched)
    matched_control_pids = set(result.pairs["control_pid"].astype(str))

    # Restrict carrier_table to the matched set.
    case_rows = carrier_table[
        carrier_table["pid"].astype(str).isin(matched_case_pids)
    ].copy()
    ctrl_rows = carrier_table[
        carrier_table["pid"].astype(str).isin(matched_control_pids)
    ].copy()

    comparison_rows: list[dict] = []

    # Test 1: lad share of total burden, cases > controls
    case_rows["lad_share"] = (
        case_rows.get("agatston_lad", 0).fillna(0)
        / (case_rows["agatston_total"].clip(lower=0) + 1e-6)
    )
    ctrl_rows["lad_share"] = (
        ctrl_rows.get("agatston_lad", 0).fillna(0)
        / (ctrl_rows["agatston_total"].clip(lower=0) + 1e-6)
    )
    delta_share = float(cliffs_delta(
        case_rows["lad_share"].dropna().to_numpy(),
        ctrl_rows["lad_share"].dropna().to_numpy(),
    ))
    mw_p_share = float(mannwhitney_u_pval(
        case_rows["lad_share"].dropna().to_numpy(),
        ctrl_rows["lad_share"].dropna().to_numpy(),
        alternative="greater",
    ))
    comparison_rows.append({
        "test": "lad_share_higher_in_cases",
        "cliffs_delta": delta_share,
        "mw_p": mw_p_share,
        "threshold_cliffs_delta_min": POST_MATCH_LAD_SHARE_CLIFFS_MIN,
        "passes": bool(delta_share >= POST_MATCH_LAD_SHARE_CLIFFS_MIN),
    })

    # Test 2: proximal-proportion chi-square on LAD lesions
    axial_path = out_dir / "axial_within_lad.csv"
    if axial_path.exists():
        axial = pd.read_csv(axial_path, dtype={"pid": str})
        axial["proximal"] = axial["relative_z_within_LAD"] < 0.5
        case_prox = axial.loc[
            axial["pid"].isin(matched_case_pids), "proximal",
        ]
        ctrl_prox = axial.loc[
            axial["pid"].isin(matched_control_pids), "proximal",
        ]
        if len(case_prox) >= 5 and len(ctrl_prox) >= 5:
            table = np.array([
                [int(case_prox.sum()), int((~case_prox).sum())],
                [int(ctrl_prox.sum()), int((~ctrl_prox).sum())],
            ])
            chi2_p = float(stats.chi2_contingency(table)[1])
            case_prox_frac = float(case_prox.mean())
            ctrl_prox_frac = float(ctrl_prox.mean())
            comparison_rows.append({
                "test": "proximal_proportion_higher_in_cases",
                "case_proximal_fraction": case_prox_frac,
                "control_proximal_fraction": ctrl_prox_frac,
                "chi2_p": chi2_p,
                "threshold_p_max": POST_MATCH_PROXIMAL_PROP_P_MAX,
                "passes": bool(
                    case_prox_frac > ctrl_prox_frac
                    and chi2_p < POST_MATCH_PROXIMAL_PROP_P_MAX
                ),
            })
        else:
            comparison_rows.append({
                "test": "proximal_proportion_higher_in_cases",
                "passes": False, "fail_reason": "insufficient_lad_lesions",
            })

    # Test 3: max_hu_lad cases > controls
    if "max_hu_lad" in case_rows.columns:
        a = case_rows["max_hu_lad"].dropna().to_numpy()
        b = ctrl_rows["max_hu_lad"].dropna().to_numpy()
        if a.size >= 5 and b.size >= 5:
            delta_hu = float(cliffs_delta(a, b))
            mw_p_hu = float(mannwhitney_u_pval(a, b, alternative="greater"))
            comparison_rows.append({
                "test": "max_hu_lad_higher_in_cases",
                "cliffs_delta": delta_hu,
                "mw_p": mw_p_hu,
                "threshold_cliffs_delta_min": POST_MATCH_MAX_HU_LAD_CLIFFS_MIN,
                "passes": bool(
                    delta_hu >= POST_MATCH_MAX_HU_LAD_CLIFFS_MIN
                ),
            })

    df = pd.DataFrame(comparison_rows)
    df.to_csv(out_dir / "matched_comparison.csv", index=False)
    n_pass = int(df["passes"].sum())
    overall = n_pass >= 2
    summary = {
        "n_passing": n_pass,
        "n_total": int(len(df)),
        "overall_pass": overall,
        "interpretation": (
            "LAD-bias survives burden matching"
            if overall
            else "LAD-bias is burden-confounded"
        ),
    }
    _save_json(out_dir / "matched_comparison_summary.json", summary)
    _log.info("post-match: %d/%d tests pass; overall = %s",
              n_pass, len(df), summary["interpretation"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
