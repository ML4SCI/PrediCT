#!/usr/bin/env python
"""Quantify gap-statistic dispersion and discriminating power.

Motivation
----------
Figure 2's original caption claimed the gap curves "rise monotonically
through k_max across all 9 combinations". The figure disproves it: the GMM
column is visibly non-monotonic (dip at k=2 on full; spike at k=6 then a
drop at k=7 on burden-residualised; local peak at k=3 on spatial-only), and
both spatial-only k-means and Ward dip around k=4-5.

The claim that actually survives is stronger anyway, but it needs numbers:

  1. No curve rises to a knee and then PLATEAUS   (the real elbow test).
  2. Where curves are non-monotonic, the wobble sits inside the bootstrap
     error bars, i.e. the dips are noise rather than structure.
  3. Adjacent-k gap differences are largely within 1 SE, i.e. the gap
     statistic cannot discriminate k on this data at all.

Point 3 is the load-bearing one. "The method designed to find k cannot
find one" is a better continuum argument than "the curve slopes upward".

This script computes all three from the gap curves already on disk. It
reads only ``gap_statistic.json`` and writes a report; it re-runs nothing
and re-fits nothing, so it is cheap and safe to run repeatedly.

Usage
-----
    python scripts/probe_gap_dispersion.py                       # full cohort
    python scripts/probe_gap_dispersion.py --cohort-dir outputs/06_reduce/stratified_Qr36d_2/
    python scripts/probe_gap_dispersion.py --all-cohorts         # all three

Outputs (into --cohort-dir):
    gap_dispersion.csv    one row per (feature_space, algorithm)
    gap_dispersion.txt    human-readable report
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

_log = logging.getLogger("probe_gap_dispersion")

DEFAULT_COHORTS = (
    ("full", Path("outputs/06_reduce")),
    ("Qr36d/2", Path("outputs/06_reduce/stratified_Qr36d_2")),
    ("I30f/3", Path("outputs/06_reduce/stratified_I30f_3")),
)


# ───────────────────────── curve diagnostics ─────────────────────────


def strictly_monotonic(gap: np.ndarray) -> bool:
    """True if every step up the k range increases the gap."""
    return bool(np.all(np.diff(gap) > 0))


def n_decreases(gap: np.ndarray) -> int:
    """How many adjacent-k steps go DOWN."""
    return int(np.sum(np.diff(gap) < 0))


def max_decrease(gap: np.ndarray) -> float:
    """Largest single downward step (0.0 if monotonic)."""
    d = np.diff(gap)
    neg = d[d < 0]
    return float(-neg.min()) if neg.size else 0.0


def frac_steps_within_1se(gap: np.ndarray, sk: np.ndarray) -> float:
    """Fraction of adjacent-k steps smaller than the SE at that step.

    This is the discriminating-power measure. If most consecutive k values
    differ by less than the bootstrap noise, the gap statistic is not
    distinguishing between them and no k can be claimed.

    The comparison uses the SE at the LATER k of each pair, matching how
    the 1-SE selection rule consumes sk.
    """
    if gap.size < 2:
        return float("nan")
    steps = np.abs(np.diff(gap))
    se = sk[1:]
    return float(np.mean(steps <= se))


def wobbles_within_error(gap: np.ndarray, sk: np.ndarray) -> bool:
    """True if every DOWNWARD step is smaller than the SE at that point.

    This is the "the dips are noise" test. A non-monotonic curve whose
    every dip is inside the error bars is not evidence of structure.
    """
    d = np.diff(gap)
    se = sk[1:]
    down = d < 0
    if not np.any(down):
        return True
    return bool(np.all(np.abs(d[down]) <= se[down]))


def plateau_detected(gap: np.ndarray, sk: np.ndarray,
                     min_tail: int = 3,
                     rise_se_mult: float = 2.0) -> tuple[bool, float, int]:
    """True if the curve knees and then flattens: the actual elbow test.

    Returns ``(detected, tail_rise_in_se_units, knee_k_index)``.

    An elbow in the Tibshirani sense is a rise followed by a plateau, so
    both halves must be demonstrated:

      * **flat tail** -- the CUMULATIVE rise from the knee to k_max is
        within 1 SE at the knee. Not "each step is within 1 SE": steps
        accumulate, so three consecutive 0.015 steps against SE 0.02 pass
        individually while climbing 0.045 together, which is more than
        2 SE. Testing per-step flags curves that are still genuinely
        rising and reports plateaus on data that mostly has none.
      * **real rise before it** -- the climb from k=1 to the knee exceeds
        ``rise_se_mult`` SE, so a curve that was flat throughout is not
        reported as an elbow.

    The tail must be at least ``min_tail`` points. Among qualifying knees
    the earliest is returned, since that is the elbow a reader would draw.
    """
    n = gap.size
    if n < min_tail + 2:
        return False, float("nan"), -1

    best: tuple[bool, float, int] = (False, float("nan"), -1)
    for cut in range(1, n - min_tail):
        se_here = sk[cut]
        if se_here <= 0:
            continue
        tail_rise = float(gap[-1] - gap[cut])
        head_rise = float(gap[cut] - gap[0])
        if tail_rise <= se_here and head_rise > rise_se_mult * se_here:
            return True, tail_rise / se_here, int(cut)
    # not an elbow; still report how flat the final tail is, for context
    se_tail = sk[n - min_tail - 1]
    tail_rise = float(gap[-1] - gap[n - min_tail - 1])
    if se_tail > 0:
        best = (False, tail_rise / se_tail, -1)
    return best


# ───────────────────────── per-cohort driver ─────────────────────────


def analyse_cohort(cohort_dir: Path, cohort_name: str) -> pd.DataFrame:
    path = cohort_dir / "gap_statistic.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    records = json.loads(path.read_text(encoding="utf-8"))

    rows: list[dict] = []
    for rec in records:
        k = np.asarray(rec["k_range"], dtype=int)
        gap = np.asarray(rec["gap_values"], dtype=float)
        sk = np.asarray(rec["sk_values"], dtype=float)
        sel = int(rec["selected_k"])
        _pl = plateau_detected(gap, sk)

        rows.append({
            "cohort": cohort_name,
            "feature_space": rec["feature_space"],
            "algorithm": rec["algorithm"],
            "selected_k": sel,
            "k_max": int(k.max()),
            "selected_at_boundary": bool(sel >= int(k.max()) - 2),
            "n_bootstrap": int(rec.get("n_bootstrap", -1)),
            # dispersion
            "sk_median": float(np.median(sk)),
            "sk_mean": float(np.mean(sk)),
            "sk_max": float(np.max(sk)),
            # shape
            "strictly_monotonic": strictly_monotonic(gap),
            "n_decreases": n_decreases(gap),
            "max_decrease": max_decrease(gap),
            # the load-bearing numbers
            "frac_steps_within_1se": frac_steps_within_1se(gap, sk),
            "dips_within_error": wobbles_within_error(gap, sk),
            "plateau_detected": _pl[0],
            "tail_rise_se_units": _pl[1],
            "knee_k": int(k[_pl[2]]) if _pl[2] >= 0 else -1,
            # context
            "gap_at_selected": float(gap[list(k).index(sel)]),
            "gap_max": float(gap.max()),
            "gap_range": float(gap.max() - gap.min()),
        })
    return pd.DataFrame(rows)


def render_report(df: pd.DataFrame) -> str:
    L: list[str] = []
    add = L.append
    add("=" * 78)
    add("GAP-STATISTIC DISPERSION AND DISCRIMINATING POWER")
    add("=" * 78)
    add("")
    add("Reads gap curves already on disk. Nothing is re-fitted.")
    add("")

    add("-" * 78)
    add("1. DISPERSION BY ALGORITHM  (is GMM really noisier?)")
    add("-" * 78)
    by_algo = df.groupby("algorithm").agg(
        sk_median=("sk_median", "median"),
        sk_max=("sk_max", "max"),
        n_runs=("sk_median", "size"),
    ).sort_values("sk_median")
    base = by_algo["sk_median"].min()
    add(f"{'algorithm':<10} {'median SE':>10} {'max SE':>9} {'x quietest':>11} {'runs':>5}")
    for algo, r in by_algo.iterrows():
        ratio = r["sk_median"] / base if base > 0 else float("nan")
        add(f"{algo:<10} {r['sk_median']:>10.4f} {r['sk_max']:>9.4f} "
            f"{ratio:>10.2f}x {int(r['n_runs']):>5}")
    add("")
    add("If GMM's median SE is a large multiple of k-means/Ward, that single")
    add("number justifies treating GMM as the high-variance estimator and")
    add("removes any suspicion it was downweighted for giving a disliked k.")
    add("")

    add("-" * 78)
    add("2. THE ELBOW TEST  (rise, then plateau?)")
    add("-" * 78)
    n_plateau = int(df["plateau_detected"].sum())
    add("Elbow = cumulative rise from the knee to k_max within 1 SE,")
    add("AND a climb of more than 2 SE before the knee. Cumulative, not")
    add("per-step: steps accumulate and a per-step test over-calls.")
    add("")
    add(f"Combinations with a detected plateau: {n_plateau} of {len(df)}")
    if n_plateau == 0:
        add("  -> NO elbow anywhere. This is the claim Figure 2 should make.")
    else:
        add("  -> a plateau was detected. Inspect these:")
        for _, r in df[df["plateau_detected"]].iterrows():
            add(f"     {r['cohort']:<8} {r['feature_space']:<20} "
                f"{r['algorithm']:<8} knee at k={int(r['knee_k'])}, "
                f"tail rise {r['tail_rise_se_units']:.2f} SE")
    add("")
    add("Tail flatness for every run (rise over the last 3 steps, in SE):")
    add(f"{'cohort':<8} {'space':<20} {'algo':<8} {'tail rise (SE)':>15}")
    for _, r in df.sort_values("tail_rise_se_units").iterrows():
        add(f"{r['cohort']:<8} {r['feature_space']:<20} {r['algorithm']:<8} "
            f"{r['tail_rise_se_units']:>15.2f}")
    add("")
    add("Large values mean the curve is STILL CLIMBING at k_max, which is")
    add("the continuum signature. Values near 0 mean it has gone flat.")
    add("")

    add("-" * 78)
    add("3. DISCRIMINATING POWER  (can the gap statistic tell k apart?)")
    add("-" * 78)
    add(f"{'cohort':<8} {'space':<20} {'algo':<8} {'sel_k':>5} "
        f"{'mono':>5} {'dips':>5} {'dip<SE':>7} {'steps<1SE':>10}")
    for _, r in df.sort_values(
        ["cohort", "feature_space", "algorithm"]
    ).iterrows():
        add(f"{r['cohort']:<8} {r['feature_space']:<20} {r['algorithm']:<8} "
            f"{r['selected_k']:>5} "
            f"{'Y' if r['strictly_monotonic'] else 'n':>5} "
            f"{r['n_decreases']:>5} "
            f"{'Y' if r['dips_within_error'] else 'NO':>7} "
            f"{r['frac_steps_within_1se']:>9.0%}")
    add("")
    overall = df["frac_steps_within_1se"].median()
    add(f"Median fraction of adjacent-k steps within 1 SE: {overall:.0%}")
    add("")
    add("Read this as: across the k range, that share of consecutive k values")
    add("are statistically indistinguishable from each other. The higher it")
    add("is, the less the gap statistic can support ANY particular k.")
    add("")

    n_mono = int(df["strictly_monotonic"].sum())
    add("-" * 78)
    add("4. MONOTONICITY  (the claim that must be dropped)")
    add("-" * 78)
    add(f"Strictly monotonic curves: {n_mono} of {len(df)}")
    non_mono = df[~df["strictly_monotonic"]]
    if len(non_mono):
        add("Non-monotonic:")
        for _, r in non_mono.iterrows():
            safe = "dips inside error bars" if r["dips_within_error"] \
                else "DIP EXCEEDS ERROR BARS"
            add(f"  {r['cohort']:<8} {r['feature_space']:<20} "
                f"{r['algorithm']:<8} {r['n_decreases']} dip(s), "
                f"largest {r['max_decrease']:.4f}  [{safe}]")
    add("")
    n_bad = int((~df["dips_within_error"]).sum())
    if n_bad:
        add(f"WARNING: {n_bad} run(s) have a dip LARGER than the local SE.")
        add("Those wobbles cannot be dismissed as bootstrap noise and need")
        add("explaining in the text rather than glossing.")
    else:
        add("Every downward step in every curve is within its local SE, so")
        add("all non-monotonicity is attributable to bootstrap noise.")
    add("")
    add("=" * 78)
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort-dir", type=Path, default=Path("outputs/06_reduce"))
    ap.add_argument("--cohort-name", type=str, default="full")
    ap.add_argument("--all-cohorts", action="store_true",
                    help="run over full + both kernel strata")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where to write reports (default: --cohort-dir)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    targets = list(DEFAULT_COHORTS) if args.all_cohorts else [
        (args.cohort_name, args.cohort_dir)
    ]

    frames: list[pd.DataFrame] = []
    for name, d in targets:
        if not (d / "gap_statistic.json").exists():
            _log.warning("skipping %s: no gap_statistic.json in %s", name, d)
            continue
        _log.info("reading %s", d / "gap_statistic.json")
        frames.append(analyse_cohort(d, name))

    if not frames:
        _log.error("no gap_statistic.json found in any target")
        return 2

    df = pd.concat(frames, ignore_index=True)
    out_dir = args.out_dir or args.cohort_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "gap_dispersion.csv"
    txt_path = out_dir / "gap_dispersion.txt"
    df.to_csv(csv_path, index=False)
    report = render_report(df)
    txt_path.write_text(report + "\n", encoding="utf-8")

    print(report)
    _log.info("wrote %s", csv_path)
    _log.info("wrote %s", txt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
