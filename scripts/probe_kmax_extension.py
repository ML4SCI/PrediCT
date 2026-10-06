#!/usr/bin/env python
"""Is the gap-statistic 'elbow' real, or an artifact of where k_max sits?

The question
------------
``probe_gap_dispersion.py`` reported 6 plateaus across 27 runs at k=1..12.
Four of the six knees landed at **k=9**, which is exactly the latest knee
that detector can report on a 12-point range (``min_tail=3`` reserves the
final three points as the tail, so the last testable cut is index
``n - min_tail - 1`` = 8, i.e. k=9).

Detections piling up at the edge of an instrument's range usually indict
the instrument. This script settles it by re-running the gap statistic on
a longer k range and asking where the knees go:

  * knee **migrates** with the ceiling (k=9 -> k=17)  ->  ARTIFACT.
    It was never an elbow, just the curve running out of search range.
  * knee **stays put** (k=9 with an 11-point tail)    ->  REAL.
    Something genuinely flattens there and needs explaining.

Safety
------
This script NEVER writes into the cohort directory. ``07_discover.py``
overwrites ``gap_statistic.json`` in place, which would destroy the
production k=1..12 curves. Everything here goes to ``--out-dir``.

The seam loading and the three feature-space constructions are imported
from ``scripts/07_discover.py`` rather than reimplemented, so this probe
cannot silently drift from what production actually clustered.

Usage
-----
    # fast smoke first (low bootstraps, ~1 min/cohort)
    python scripts/probe_kmax_extension.py --gap-bootstraps 50

    # real run, all three cohorts (~20 min total)
    python scripts/probe_kmax_extension.py --all-cohorts

    # push further out if k=20 is still ambiguous
    python scripts/probe_kmax_extension.py --k-max 30

Outputs (into --out-dir, default outputs/probe_kmax/):
    <cohort>/gap_statistic_kmax<N>.json   full extended curves
    knee_migration.csv                    per-run knee comparison + verdict
    knee_migration.txt                    human-readable report
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
_log = logging.getLogger("probe_kmax_extension")

DEFAULT_COHORTS = (
    ("full", Path("outputs/06_reduce")),
    ("Qr36d_2", Path("outputs/06_reduce/stratified_Qr36d_2")),
    ("I30f_3", Path("outputs/06_reduce/stratified_I30f_3")),
)


def _load_script_module(rel_path: str, alias: str):
    """Import a scripts/NN_name.py file whose name is not a valid identifier."""
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


# Production seam loader + feature-space definitions, and the plateau
# detector. Imported, never copied.
_disc = _load_script_module("scripts/07_discover.py", "_prod_discover")
_disp = _load_script_module("scripts/probe_gap_dispersion.py", "_prod_dispersion")

plateau_detected = _disp.plateau_detected
ALGORITHMS = _disc.ALGORITHMS
SPATIAL_FEATURES_AFTER_D017 = _disc.SPATIAL_FEATURES_AFTER_D017


def ceiling_knee_k(k_range: list[int], min_tail: int = 3) -> int:
    """The latest knee ``plateau_detected`` can report on this k range.

    Mirrors its loop bound: ``for cut in range(1, n - min_tail)``, so the
    last testable cut index is ``n - min_tail - 1``.
    """
    idx = len(k_range) - min_tail - 1
    return int(k_range[idx]) if idx >= 1 else -1


# ───────────────────────── feature spaces ─────────────────────────


def build_feature_spaces(cohort_dir: Path, random_state: int, cfg):
    """Reproduce 07_discover.py's three feature spaces exactly."""
    from predict.discover.cluster_discovery import burden_residualise
    from predict.reduce.pca import fit_pca

    prep_df, X_full, pid_order, cohort_meta = _disc._load_seam(cohort_dir)

    spaces: dict[str, np.ndarray] = {"full": X_full}

    if "agatston_total" in cohort_meta.columns:
        burden = cohort_meta["agatston_total"].to_numpy(dtype=float)
        spaces["burden_residualised"] = burden_residualise(
            X_full, burden, log_transform=True,
        )
    else:
        _log.warning("no agatston_total; skipping burden_residualised")

    spatial_cols = [c for c in SPATIAL_FEATURES_AFTER_D017
                    if c in prep_df.columns]
    if len(spatial_cols) >= 2:
        spatial_pca = fit_pca(prep_df, spatial_cols,
                              cumvar_threshold=cfg.reduce.pca_cumvar,
                              random_state=random_state)
        if spatial_pca.pid_order != pid_order:
            spaces["spatial_only"] = pd.DataFrame(
                spatial_pca.scores, index=spatial_pca.pid_order,
            ).loc[pid_order].to_numpy(dtype=float)
        else:
            spaces["spatial_only"] = spatial_pca.scores
        _log.info("spatial_only: %d features -> %d PCs",
                  len(spatial_cols), spaces["spatial_only"].shape[1])
    else:
        _log.warning("too few spatial features; skipping spatial_only")

    return spaces


# ───────────────────────── extended gap run ─────────────────────────


def run_extended(cohort_name: str, cohort_dir: Path, out_dir: Path,
                 k_max: int, n_bootstrap: int, random_state: int,
                 n_jobs: int, cfg) -> list[dict]:
    from predict.discover.cluster_discovery import gap_statistic

    k_range = tuple(range(1, k_max + 1))
    spaces = build_feature_spaces(cohort_dir, random_state, cfg)

    records: list[dict] = []
    for label, X in spaces.items():
        for algo in ALGORITHMS:
            t0 = time.perf_counter()
            res = gap_statistic(
                X, algorithm=algo, k_range=k_range,
                n_bootstrap=n_bootstrap, random_state=random_state,
                n_jobs=n_jobs,
            )
            dt = time.perf_counter() - t0
            _log.info("  %-20s %-7s selected_k=%2d  (%.0fs)",
                      label, algo, res.selected_k, dt)
            records.append({
                "feature_space": label,
                "algorithm": algo,
                "k_range": list(res.k_range),
                "gap_values": res.gap_values.tolist(),
                "sk_values": res.sk_values.tolist(),
                "selected_k": int(res.selected_k),
                "n_bootstrap": int(res.n_bootstrap),
                "random_state": int(res.random_state),
            })

    dest = out_dir / cohort_name
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"gap_statistic_kmax{k_max}.json"
    path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    _log.info("wrote %s", path)
    return records


# ───────────────────────── comparison ─────────────────────────


def compare(cohort_name: str, baseline_records: list[dict],
            extended_records: list[dict],
            selected_k_only: bool = False) -> pd.DataFrame:
    """Match runs across the two k ranges and classify each knee."""
    base = {(r["feature_space"], r["algorithm"]): r for r in baseline_records}
    rows: list[dict] = []

    for ext in extended_records:
        key = (ext["feature_space"], ext["algorithm"])
        b = base.get(key)
        if b is None:
            _log.warning("no baseline run for %s; skipping", key)
            continue

        # The knee test thresholds on sk, and sk scales ~1/sqrt(n_bootstrap).
        # Comparing a 50-bootstrap extended run against a 500-bootstrap
        # baseline changes the error bars by ~3.2x, which moves the knee
        # verdict for reasons that have nothing to do with k_max. Refuse to
        # report a knee comparison across mismatched bootstrap counts.
        b_boot = int(b.get("n_bootstrap", -1))
        e_boot = int(ext.get("n_bootstrap", -1))
        if b_boot != e_boot and not selected_k_only:
            raise ValueError(
                f"bootstrap mismatch for {key}: baseline n_bootstrap="
                f"{b_boot}, extended n_bootstrap={e_boot}. The knee test "
                f"thresholds on sk, which scales ~1/sqrt(n_bootstrap), so "
                f"this comparison would attribute a bootstrap effect to "
                f"k_max. Re-run with --gap-bootstraps {b_boot} to match the "
                f"baseline. (A smoke run at low bootstraps is still useful "
                f"for timing and for the selected_k trend, which is not "
                f"sk-thresholded: pass --selected-k-only to skip the knee "
                f"comparison.)"
            )

        b_k, e_k = list(b["k_range"]), list(ext["k_range"])
        b_det, _, b_idx = plateau_detected(
            np.asarray(b["gap_values"], float), np.asarray(b["sk_values"], float))
        e_det, e_tail, e_idx = plateau_detected(
            np.asarray(ext["gap_values"], float),
            np.asarray(ext["sk_values"], float))

        b_knee = int(b_k[b_idx]) if b_idx >= 0 else -1
        e_knee = int(e_k[e_idx]) if e_idx >= 0 else -1
        b_ceil, e_ceil = ceiling_knee_k(b_k), ceiling_knee_k(e_k)

        if selected_k_only and b_boot != e_boot:
            verdict = "knee comparison SKIPPED (bootstrap mismatch)"
        elif not b_det and not e_det:
            verdict = "no elbow in either"
        elif b_det and not e_det:
            verdict = "ARTIFACT (vanished when k_max grew)"
        elif not b_det and e_det:
            # No baseline knee to compare against, so the ceiling-tracking
            # test above cannot fire. But a knee that appears only at the
            # longer range AND lands exactly on that range's ceiling is the
            # same boundary artifact, just without a baseline counterpart.
            if e_knee == e_ceil:
                verdict = f"ARTIFACT (appeared at ceiling k={e_knee})"
            else:
                verdict = f"appeared only at larger k_max (k={e_knee}) - inspect"
        elif b_knee == b_ceil and e_knee == e_ceil:
            verdict = "ARTIFACT (knee tracked the ceiling)"
        elif b_knee == e_knee:
            verdict = "REAL (knee held position)"
        else:
            verdict = f"MOVED {b_knee} -> {e_knee} (not to ceiling)"

        rows.append({
            "cohort": cohort_name,
            "feature_space": ext["feature_space"],
            "algorithm": ext["algorithm"],
            "k_max_baseline": max(b_k),
            "k_max_extended": max(e_k),
            "plateau_baseline": b_det,
            "plateau_extended": e_det,
            "knee_baseline": b_knee,
            "knee_extended": e_knee,
            "ceiling_baseline": b_ceil,
            "ceiling_extended": e_ceil,
            "knee_at_ceiling_baseline": b_knee == b_ceil,
            "knee_at_ceiling_extended": e_knee == e_ceil,
            "tail_rise_se_extended": e_tail,
            "selected_k_baseline": int(b["selected_k"]),
            "selected_k_extended": int(ext["selected_k"]),
            "verdict": verdict,
        })
    return pd.DataFrame(rows)


def render(df: pd.DataFrame) -> str:
    L: list[str] = []
    add = L.append
    add("=" * 82)
    add("KNEE MIGRATION UNDER EXTENDED k_max")
    add("=" * 82)
    add("")
    add("A knee that tracks the detector ceiling was never an elbow.")
    add("A knee that holds position while the tail grows is real.")
    add("")

    add(f"{'cohort':<9} {'space':<20} {'algo':<7} {'knee':>10} "
        f"{'ceiling':>10}  verdict")
    add("-" * 82)
    for _, r in df.sort_values(
        ["cohort", "feature_space", "algorithm"]
    ).iterrows():
        knee = f"{r['knee_baseline']}->{r['knee_extended']}"
        ceil = f"{r['ceiling_baseline']}->{r['ceiling_extended']}"
        add(f"{r['cohort']:<9} {r['feature_space']:<20} {r['algorithm']:<7} "
            f"{knee:>10} {ceil:>10}  {r['verdict']}")
    add("")

    add("-" * 82)
    add("SUMMARY")
    add("-" * 82)
    n = len(df)
    art = int(df["verdict"].str.startswith("ARTIFACT").sum())
    real = int(df["verdict"].str.startswith("REAL").sum())
    none = int((df["verdict"] == "no elbow in either").sum())
    other = n - art - real - none
    add(f"  no elbow in either k range : {none:>3} of {n}")
    add(f"  ARTIFACT                   : {art:>3} of {n}")
    add(f"  REAL                       : {real:>3} of {n}")
    add(f"  other / inspect            : {other:>3} of {n}")
    add("")
    if real == 0 and art > 0:
        add("  -> Every detected plateau was a boundary artifact. The elbow")
        add("     question closes: no combination shows a genuine elbow.")
    elif real > 0:
        add("  -> Some knees held position. These are real and need an")
        add("     explanation in the text, not a dismissal:")
        for _, r in df[df["verdict"].str.startswith("REAL")].iterrows():
            add(f"       {r['cohort']:<9} {r['feature_space']:<20} "
                f"{r['algorithm']:<7} knee k={r['knee_extended']}, "
                f"tail rise {r['tail_rise_se_extended']:.2f} SE")
    add("")

    add("selected_k stability (a different question, reported for context):")
    moved = df[df["selected_k_baseline"] != df["selected_k_extended"]]
    add(f"  selected_k changed in {len(moved)} of {n} runs when k_max grew.")
    if len(moved):
        for _, r in moved.iterrows():
            add(f"    {r['cohort']:<9} {r['feature_space']:<20} "
                f"{r['algorithm']:<7} {r['selected_k_baseline']:>3} -> "
                f"{r['selected_k_extended']:>3}")
        add("  A selected_k that keeps climbing with k_max is the continuum")
        add("  signature: no k in ANY search range is sufficient.")
    add("")
    add("=" * 82)
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort-dir", type=Path, default=Path("outputs/06_reduce"))
    ap.add_argument("--cohort-name", type=str, default="full")
    ap.add_argument("--all-cohorts", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/probe_kmax"))
    ap.add_argument("--k-max", type=int, default=20)
    ap.add_argument("--gap-bootstraps", type=int, default=500,
                    help="match production (500). Use 50 for a smoke run.")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=None,
                    help="default: scheduler allocation via default_n_jobs()")
    ap.add_argument("--selected-k-only", action="store_true",
                    help="skip the knee comparison and report only the "
                         "selected_k trend. Use with a low --gap-bootstraps "
                         "smoke run: selected_k is not sk-thresholded the way "
                         "the knee test is, so its trend survives a bootstrap "
                         "mismatch that would invalidate the knee verdicts.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    from predict.config import default_n_jobs, load_config
    cfg = load_config()
    n_jobs = args.n_jobs if args.n_jobs is not None else default_n_jobs()
    _log.info("k_max=%d  bootstraps=%d  seed=%d  n_jobs=%d",
              args.k_max, args.gap_bootstraps, args.random_state, n_jobs)

    targets = list(DEFAULT_COHORTS) if args.all_cohorts else [
        (args.cohort_name, args.cohort_dir)
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []

    for name, cdir in targets:
        baseline_path = cdir / "gap_statistic.json"
        if not baseline_path.exists():
            _log.warning("skipping %s: no baseline %s", name, baseline_path)
            continue
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        _log.info("=== %s (baseline k_max=%d) ===",
                  name, max(baseline[0]["k_range"]))

        extended = run_extended(
            name, cdir, args.out_dir, args.k_max, args.gap_bootstraps,
            args.random_state, n_jobs, cfg,
        )
        frames.append(compare(name, baseline, extended,
                              selected_k_only=args.selected_k_only))

    if not frames:
        _log.error("nothing to compare")
        return 2

    df = pd.concat(frames, ignore_index=True)
    csv_path = args.out_dir / "knee_migration.csv"
    txt_path = args.out_dir / "knee_migration.txt"
    df.to_csv(csv_path, index=False)
    report = render(df)
    txt_path.write_text(report + "\n", encoding="utf-8")

    print(report)
    _log.info("wrote %s", csv_path)
    _log.info("wrote %s", txt_path)
    _log.info("production gap_statistic.json files were NOT modified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
