#!/usr/bin/env python
"""D038 (experimental): taxometric test of dimensional vs categorical structure.

Rules and interpretation are fixed in decisions/D038-taxometric-continuum-test.md
before this was run.

1. Indicators, equal representation across the six D022 blocks (burden,
   hu_statistics, density_tier, spatial, texture, shape): within each block,
   among its representatives with at least MIN_DISTINCT distinct values in
   the prepared matrix, rank by redundancy-cluster size (descending), then
   ICC (descending), then name, and take the top PER_BLOCK. A block with
   fewer eligible representatives contributes what it has.
2. Controls (must pass before the real data are read): planted groups via
   the D036 helpers in the full representative matrix, then the same
   indicator subset. Negative control: gaussian background, delta 0.
   Positive control: gaussian background, k_true 2. Also reported: real
   background floors and k_true 3.
3. Real data: CCFI with N_COMP_REAL comparison datasets per model.

Usage:
    python scripts/probe_taxometric.py --root outputs
    python scripts/probe_taxometric.py --root outputs --smoke
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from predict.config import default_n_jobs
from predict.discover.detectability import assign_groups, plant, planting_directions, zscore
from predict.discover.taxometric import ccfi, verdict

REPO_ROOT = Path(__file__).resolve().parents[1]
BLOCKS = ("burden", "hu_statistics", "density_tier", "spatial", "texture", "shape")
PER_BLOCK, MIN_DISTINCT = 2, 10
N_COMP_REAL, N_COMP_CONTROL = 100, 50
CTRL_K = (2, 3)
CTRL_DELTAS = (0.0, 4.0, 8.0, 12.0, 16.0)
CTRL_PROPS = ("balanced", "imbalanced")
CTRL_BG = ("gaussian", "real")
CTRL_REPS = 5
RATE_MIN = 0.80


def select_indicators(red: Path, prep: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    asg = pd.read_csv(red / "multi_block_assignments.csv")
    reps = set(pd.read_csv(red / "representative_features.csv").iloc[:, 0])
    rep_rows = asg[asg.feature.isin(reps) & asg.is_representative.astype(bool)].copy()
    rep_rows["n_distinct"] = [prep[f].round(9).nunique() for f in rep_rows.feature]
    rep_rows["eligible"] = rep_rows.n_distinct >= MIN_DISTINCT
    chosen = []
    for b in BLOCKS:
        g = rep_rows[(rep_rows.block == b) & rep_rows.eligible].sort_values(
            ["cluster_size", "icc", "feature"], ascending=[False, False, True])
        chosen += g.feature.head(PER_BLOCK).tolist()
    rep_rows["chosen"] = rep_rows.feature.isin(chosen)
    return chosen, rep_rows.sort_values(["block", "chosen"], ascending=[True, False])


def control_run(X0, cov, ind_idx, k, delta, prop, bg, rep, n_comp):
    seed = [k, int(delta * 10), CTRL_PROPS.index(prop), CTRL_BG.index(bg), rep, 38]
    rng = np.random.default_rng(seed)
    n, d = X0.shape
    Xb = X0 if bg == "real" else rng.multivariate_normal(np.zeros(d), cov, size=n)
    labels = assign_groups(rng, n, k, prop)
    U = planting_directions(rng, d, k - 1)
    Xp = zscore(plant(Xb, labels, delta, U, cov))[:, ind_idx]
    r = ccfi(Xp, n_comparison=n_comp, seed=int(rng.integers(0, 2**31 - 1)))
    return {"k_true": k, "delta": delta, "proportions": prop, "background": bg, "rep": rep,
            **{f"ccfi_{p}": v for p, v in r.ccfi.items()}, "mean_ccfi": r.mean_ccfi,
            "verdict": r.verdict, "base_rate": r.base_rate}


def main() -> int:
    from joblib import Parallel, delayed
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=default_n_jobs())
    args = ap.parse_args()

    root = REPO_ROOT / args.root
    red = root / "06_reduce"
    out = root / ("taxometric_smoke" if args.smoke else "taxometric")
    out.mkdir(parents=True, exist_ok=True)
    prep = pd.read_csv(red / "prepared_matrix.csv", dtype={"pid": str})
    reps = pd.read_csv(red / "representative_features.csv").iloc[:, 0].tolist()
    chosen, audit = select_indicators(red, prep)
    audit.to_csv(out / "indicator_selection.csv", index=False)
    X0 = zscore(prep[reps].to_numpy(float))
    cov = np.cov(X0, rowvar=False)
    ind_idx = [reps.index(f) for f in chosen]

    n_real, n_ctrl, reps_ctrl, deltas = N_COMP_REAL, N_COMP_CONTROL, CTRL_REPS, CTRL_DELTAS
    ks = CTRL_K
    if args.smoke:
        n_real, n_ctrl, reps_ctrl, deltas, ks = 10, 5, 1, (0.0, 16.0), (2,)
    (out / "run_config.json").write_text(json.dumps({
        "root": args.root, "indicators": chosen, "per_block": PER_BLOCK, "min_distinct": MIN_DISTINCT,
        "n_comp_real": n_real, "n_comp_control": n_ctrl, "control_reps": reps_ctrl,
        "control_deltas": list(deltas), "control_k": list(ks), "rate_min": RATE_MIN,
    }, indent=2), encoding="utf-8")

    t0 = time.time()
    grid = list(itertools.product(ks, deltas, CTRL_PROPS, CTRL_BG, range(reps_ctrl)))
    rows = Parallel(n_jobs=args.n_jobs)(
        delayed(control_run)(X0, cov, ind_idx, k, dl, pr, bg, r, n_ctrl) for k, dl, pr, bg, r in grid)
    ctrl = pd.DataFrame(rows)
    ctrl.to_csv(out / "controls.csv", index=False)
    print(f"controls: {len(ctrl)} runs in {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    real = ccfi(prep[chosen].to_numpy(float), n_comparison=n_real, seed=38)
    print(f"real data: {time.time() - t0:.0f}s", flush=True)

    neg = ctrl[(ctrl.background == "gaussian") & (ctrl.delta == 0)]
    neg_rate = float((neg.verdict == "dimensional").mean())
    neg_pass = neg_rate >= RATE_MIN

    def floor(bg, k, prop):
        g = ctrl[(ctrl.background == bg) & (ctrl.k_true == k) & (ctrl.proportions == prop) & (ctrl.delta > 0)]
        rates = g.groupby("delta")["verdict"].apply(lambda v: float((v == "categorical").mean()))
        hit = rates[rates >= RATE_MIN]
        return (float(hit.index.min()) if len(hit) else np.nan), rates

    pos_floor, _ = floor("gaussian", 2, "balanced")
    pos_pass = not np.isnan(pos_floor)
    usable = neg_pass and pos_pass

    L = ["=" * 96, f"D038 TAXOMETRIC TEST (experimental)  root={args.root}", "=" * 96,
         f"indicators ({len(chosen)}): " + ", ".join(f.replace("original_", "") for f in chosen)]
    for b in BLOCKS:
        g = audit[(audit.block == b) & audit.chosen]
        L.append(f"  {b:<14} {', '.join(g.feature.str.replace('original_', '')) or '(none eligible)'}")
    L.append(f"\nnegative control (gaussian, delta 0): dimensional in {neg_rate:.2f} of runs -> "
             f"{'PASS' if neg_pass else 'FAIL'}")
    L.append("categorical-call rate by delta (and floor = smallest delta with rate >= "
             f"{RATE_MIN}):")
    for bg, k, prop in itertools.product(CTRL_BG, ks, CTRL_PROPS):
        fl, rates = floor(bg, k, prop)
        L.append(f"  {bg:<8} k={k} {prop:<10} floor {('none' if np.isnan(fl) else f'{fl:g}'):<5} "
                 + "  ".join(f"d{dl:g}:{r:.1f}" for dl, r in rates.items()))
    L.append(f"positive control (gaussian, k 2, balanced): floor "
             f"{'none' if np.isnan(pos_floor) else f'{pos_floor:g}'} -> {'PASS' if pos_pass else 'FAIL'}")
    mc = ctrl.groupby(["background", "k_true", "delta"])["mean_ccfi"].median().unstack("delta")
    L.append("\nmedian mean-CCFI (rows background x k, columns delta):")
    L.append(mc.round(3).to_string())
    L.append(f"\nREAL DATA: CCFI " + ", ".join(f"{p} {v:.3f}" for p, v in real.ccfi.items())
             + f"; mean {real.mean_ccfi:.3f} -> {real.verdict} (categorical-model base rate "
             f"{real.base_rate:.2f}, {real.n_comparison} comparison datasets per model)")
    if usable:
        L.append("RULE: controls pass -> the real-data verdict is interpretable "
                 "(see D038 interpretation table).")
    else:
        L.append("RULE: a control FAILED -> the real-data verdict is NOT interpretable; the test is "
                 "not usable on this geometry.")
    L.append("=" * 96)
    text = "\n".join(L)
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    json.dump({"ccfi": real.ccfi, "mean_ccfi": real.mean_ccfi, "verdict": real.verdict,
               "base_rate": real.base_rate, "controls_usable": usable,
               "negative_control_rate": neg_rate, "positive_control_floor": pos_floor},
              open(out / "real_result.json", "w"), indent=2, default=float)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
