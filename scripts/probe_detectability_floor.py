#!/usr/bin/env python
"""D036 detectability floor: would the pipeline have found clusters if they existed?

Plants k groups at a known separation into the gated representative-feature
matrix of a root (default outputs), then runs the unchanged D020 PCA and
D021 discovery steps and records whether the planted structure is recovered.
Rules and the interpretation are fixed in decisions/D036-detectability-floor.md
before this was run.

Grid (one replicate = one fresh planting):
  k_true      2, 3, 4
  delta       0, 1, 2, 3, 4, 6   (background SDs between group means)
  direction   random | burden_orthogonal (orthogonal to real PC1)
  proportions balanced | imbalanced (1, 1/2, 1/4 ...)
  background  real (the 440 x reps matrix itself) | gaussian (N(0, real cov))
  replicates  10
Per replicate: re-z-score, PCA at the config cumvar, Hopkins, gap (kmeans
and ward, k 1..12, B bootstraps), forced-k_true kmeans: ARI vs planted
labels, Hennig (100 bootstraps).

Calibration: the cell (k=3, random, balanced, real) is also run with B=500
(D021's value) to check the B=100 decisions.

Resumable: one CSV per cell under <root>/detectability/cells/.

D036.1 amendment (--amendment; decisions/D036.1-detectability-amendment.md,
fixed before running): delta grid extended to 8, 10, 12, 16; adds the D021
GMM (full covariance) arm: GMM gap and forced-k GMM recovery; reports the
separation also in PC1-SD units and as the planted share of total variance;
deltas <= 6 reuse the D036 seeds, so kmeans / ward there must reproduce the
first run exactly. Output: <root>/detectability_d0361/.

Usage:
    python scripts/probe_detectability_floor.py --root outputs
    python scripts/probe_detectability_floor.py --root outputs --smoke
    python scripts/probe_detectability_floor.py --root outputs --amendment
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

from predict.config import default_n_jobs, load_config
from predict.discover.cluster_discovery import fit_cluster, gap_statistic
from predict.discover.clusterability import hopkins_statistic
from predict.discover.detectability import (
    assign_groups, plant, planting_directions, zscore,
)
from predict.discover.validity import hennig_clusterboot
from predict.reduce.pca import fit_pca

REPO_ROOT = Path(__file__).resolve().parents[1]
K_TRUE = (2, 3, 4)
DELTAS = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0)
DELTAS_D0361 = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0)
DIRECTIONS = ("random", "burden_orthogonal")
PROPORTIONS = ("balanced", "imbalanced")
BACKGROUNDS = ("real", "gaussian")
N_REPS = 10
K_RANGE = tuple(range(1, 13))
ARI_MIN, HENNIG_MIN, RATE_MIN = 0.80, 0.75, 0.80
CALIB_CELL = (3, "random", "balanced", "real")


def one_replicate(X0, cov, pc1, reps, cumvar, k, delta, direction, prop, bg,
                  rep, n_boot, hennig_boot, seed_tag, with_gmm=False, lam1=None):
    from sklearn.metrics import adjusted_rand_score
    seed = [k, DIRECTIONS.index(direction), PROPORTIONS.index(prop),
            BACKGROUNDS.index(bg), int(delta * 10), rep, seed_tag]
    rng = np.random.default_rng(seed)
    n, d = X0.shape
    Xb = X0 if bg == "real" else rng.multivariate_normal(np.zeros(d), cov, size=n)
    labels = assign_groups(rng, n, k, prop)
    U = planting_directions(rng, d, k - 1, avoid=pc1 if direction == "burden_orthogonal" else None)
    Xp = zscore(plant(Xb, labels, delta, U, cov))
    sd_along = np.sqrt(np.einsum("ij,jk,ki->i", U.T, cov, U))
    grand = Xp.mean(axis=0)
    between = sum(int((labels == g).sum()) * float(((Xp[labels == g].mean(axis=0) - grand) ** 2).sum())
                  for g in range(k))
    df = pd.DataFrame(Xp, columns=reps)
    df.insert(0, "pid", [f"s{i}" for i in range(n)])
    pca = fit_pca(df, reps, cumvar_threshold=cumvar, random_state=42)
    Z = pca.scores
    rs = int(np.random.default_rng(seed + [1]).integers(1, 2**31 - 1))
    row = {"k_true": k, "delta": delta, "direction": direction, "proportions": prop,
           "background": bg, "rep": rep, "n_boot": n_boot, "n_pcs": int(Z.shape[1]),
           "hopkins": float(hopkins_statistic(Z, sample_size=max(2, int(0.1 * n)), random_state=rs)),
           "sep_pc1_units": float(delta * sd_along.mean() / np.sqrt(lam1)) if lam1 else np.nan,
           "between_var_share": float(between / ((Xp - grand) ** 2).sum())}
    for algo in ("kmeans", "ward") + (("gmm",) if with_gmm else ()):
        g = gap_statistic(Z, algorithm=algo, k_range=K_RANGE, n_bootstrap=n_boot,
                          random_state=rs % 100_000, n_jobs=1)
        row[f"gap_k_{algo}"] = int(g.selected_k)
        row[f"gap_at_ktrue_{algo}"] = float(g.gap_values[K_RANGE.index(k)])
    lab = fit_cluster(Z, k=k, algorithm="kmeans", random_state=42)
    row["ari_forced"] = float(adjusted_rand_score(labels, lab))
    hen = hennig_clusterboot(Z, lab, k=k, algorithm="kmeans", n_bootstrap=hennig_boot,
                             threshold=HENNIG_MIN, random_state=42, n_jobs=1)
    row["hennig_min_jaccard"] = float(np.min(hen.jaccard_median))
    row["detected_primary"] = row["gap_k_kmeans"] == k
    row["detected_ward"] = row["gap_k_ward"] == k
    row["recovered"] = row["ari_forced"] >= ARI_MIN and row["hennig_min_jaccard"] >= HENNIG_MIN
    if with_gmm:
        lab_g = fit_cluster(Z, k=k, algorithm="gmm", random_state=42)
        row["ari_forced_gmm"] = float(adjusted_rand_score(labels, lab_g))
        hen_g = hennig_clusterboot(Z, lab_g, k=k, algorithm="gmm", n_bootstrap=hennig_boot,
                                   threshold=HENNIG_MIN, random_state=42, n_jobs=1)
        row["hennig_min_jaccard_gmm"] = float(np.min(hen_g.jaccard_median))
        row["detected_gmm"] = row["gap_k_gmm"] == k
        row["recovered_gmm"] = (row["ari_forced_gmm"] >= ARI_MIN
                                and row["hennig_min_jaccard_gmm"] >= HENNIG_MIN)
    return row


def floor_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rates = (df.groupby(["background", "direction", "proportions", "k_true", "delta"])[col]
             .mean().rename("rate").reset_index())
    out = []
    for key, g in rates.groupby(["background", "direction", "proportions", "k_true"]):
        g = g.sort_values("delta")
        hit = g[(g.delta > 0) & (g.rate >= RATE_MIN)]
        out.append(dict(zip(["background", "direction", "proportions", "k_true"], key),
                        floor_delta=float(hit.delta.min()) if len(hit) else np.nan,
                        **{f"rate_d{dl:g}": float(r) for dl, r in zip(g.delta, g.rate)}))
    return pd.DataFrame(out)


def main() -> int:
    from joblib import Parallel, delayed
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs")
    ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument("--hennig-boot", type=int, default=100)
    ap.add_argument("--smoke", action="store_true", help="tiny grid, B=20, 2 reps; separate dir")
    ap.add_argument("--amendment", action="store_true",
                    help="D036.1: deltas to 16, GMM arm, PC1-SD units; separate dir")
    ap.add_argument("--n-jobs", type=int, default=default_n_jobs())
    args = ap.parse_args()

    root = REPO_ROOT / args.root
    red = root / "06_reduce"
    cfg = load_config(root / "config.yaml" if (root / "config.yaml").exists() else None)
    prep = pd.read_csv(red / "prepared_matrix.csv", dtype={"pid": str})
    reps = pd.read_csv(red / "representative_features.csv").iloc[:, 0].tolist()
    X0 = zscore(prep[reps].to_numpy(float))
    cov = np.cov(X0, rowvar=False)
    real_pca = fit_pca(prep, reps, cumvar_threshold=cfg.reduce.pca_cumvar, random_state=42)
    pc1 = real_pca.components[0]
    lam1 = float(np.linalg.eigvalsh(cov)[-1])

    grid = dict(k=K_TRUE, deltas=DELTAS, n_reps=N_REPS, n_boot=args.n_boot,
                hennig=args.hennig_boot)
    if args.amendment:
        grid["deltas"] = DELTAS_D0361
    if args.smoke:
        grid = dict(k=(3,), deltas=(0.0, 4.0, 16.0) if args.amendment else (0.0, 4.0),
                    n_reps=2, n_boot=20, hennig=20)
    name = "detectability_d0361" if args.amendment else "detectability"
    out = root / (name + ("_smoke" if args.smoke else ""))
    cells_dir = out / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps({
        "root": args.root, "n_patients": int(X0.shape[0]), "n_reps_features": len(reps),
        "pca_cumvar": cfg.reduce.pca_cumvar, "real_n_pcs": int(real_pca.n_retain),
        **{k: list(v) if isinstance(v, tuple) else v for k, v in grid.items()},
        "directions": DIRECTIONS, "proportions": PROPORTIONS, "backgrounds": BACKGROUNDS,
        "rules": {"ari_min": ARI_MIN, "hennig_min": HENNIG_MIN, "rate_min": RATE_MIN},
        "amendment": bool(args.amendment), "real_lambda1": lam1,
    }, indent=2), encoding="utf-8")

    cells = [(k, dr, pr, bg, grid["n_boot"], 36)
             for k, dr, pr, bg in itertools.product(grid["k"], DIRECTIONS, PROPORTIONS, BACKGROUNDS)]
    if not args.smoke and not args.amendment:
        cells.append((*CALIB_CELL, 500, 36))   # same plantings, only B differs
    for k, dr, pr, bg, nb, tag in cells:
        f = cells_dir / f"k{k}_{dr}_{pr}_{bg}_B{nb}.csv"
        if f.exists():
            continue
        t0 = time.time()
        rows = Parallel(n_jobs=args.n_jobs)(
            delayed(one_replicate)(X0, cov, pc1, reps, cfg.reduce.pca_cumvar, k, dl, dr, pr, bg,
                                   r, nb, grid["hennig"], tag, args.amendment, lam1)
            for dl in grid["deltas"] for r in range(grid["n_reps"]))
        pd.DataFrame(rows).to_csv(f, index=False)
        print(f"cell {f.stem}: {len(rows)} runs in {time.time() - t0:.0f}s", flush=True)

    allrows = pd.concat([pd.read_csv(f) for f in sorted(cells_dir.glob("*.csv"))], ignore_index=True)
    allrows.to_csv(out / "all_runs.csv", index=False)
    main_runs = allrows[allrows.n_boot == grid["n_boot"]]
    cols = ["detected_primary", "detected_ward", "recovered"]
    if args.amendment:
        cols += ["detected_gmm", "recovered_gmm"]
    tabs = {c: floor_table(main_runs, c) for c in cols}
    for c, t in tabs.items():
        t.to_csv(out / f"floor_{c}.csv", index=False)

    L = ["=" * 100, f"D036 DETECTABILITY FLOOR  root={args.root}  "
         f"(N={X0.shape[0]}, {len(reps)} reps, real PCs {real_pca.n_retain}, B={grid['n_boot']})",
         "=" * 100]
    null = main_runs[main_runs.delta == 0]
    for bg, g in null.groupby("background"):
        L.append(f"delta=0 ({bg}): gap k kmeans median {g.gap_k_kmeans.median():.0f} "
                 f"[{g.gap_k_kmeans.min()}-{g.gap_k_kmeans.max()}], ward median {g.gap_k_ward.median():.0f}, "
                 f"Hopkins median {g.hopkins.median():.3f}")
    for c, t in tabs.items():
        L.append(f"\nfloor (smallest delta with rate >= {RATE_MIN}) for {c}:")
        L.append("  background direction         proportions  k   floor   " +
                 "  ".join(f"d{dl:g}" for dl in grid["deltas"]))
        for r in t.itertuples():
            rates = "  ".join(f"{getattr(r, f'rate_d{dl:g}'):.1f}" for dl in grid["deltas"])
            fl = "none" if np.isnan(r.floor_delta) else f"{r.floor_delta:g}"
            L.append(f"  {r.background:<10} {r.direction:<18} {r.proportions:<11} {r.k_true}   {fl:<6}  {rates}")
    if args.amendment:
        su = main_runs.groupby("delta")[["sep_pc1_units", "between_var_share"]].median()
        L.append("\nseparation units (median over runs): delta -> PC1-SD units, planted variance share")
        L.append("  " + ", ".join(f"{dl:g}: {r.sep_pc1_units:.2f} / {r.between_var_share:.3f}"
                                  for dl, r in su.iterrows()))
        rb = main_runs[(main_runs.direction == "random") & (main_runs.proportions == "balanced")]
        L.append("\ngap-selected k, median [min-max] (random, balanced): "
                 "kmeans | ward | gmm, and forced-k ARI kmeans | gmm")
        for (bg, kt), g in rb.groupby(["background", "k_true"]):
            L.append(f"  {bg} k_true={kt}:")
            for dl, h in g.groupby("delta"):
                L.append("    d{:<4g} ".format(dl) + " | ".join(
                    f"{h[c].median():.0f} [{h[c].min()}-{h[c].max()}]"
                    for c in ("gap_k_kmeans", "gap_k_ward", "gap_k_gmm"))
                    + f"   ARI {h.ari_forced.median():.2f} | {h.ari_forced_gmm.median():.2f}")
        first = root / "detectability" / "all_runs.csv"
        if first.exists() and not args.smoke:
            keys = ["k_true", "delta", "direction", "proportions", "background", "rep"]
            f0 = pd.read_csv(first)
            m = main_runs.merge(f0[f0.n_boot == grid["n_boot"]], on=keys, suffixes=("", "_d036"))
            same = ((m.gap_k_kmeans == m.gap_k_kmeans_d036) & (m.gap_k_ward == m.gap_k_ward_d036)
                    & np.isclose(m.ari_forced, m.ari_forced_d036))
            L.append(f"\nreproducibility vs D036 first run (deltas <= 6, same seeds): "
                     f"{int(same.sum())}/{len(m)} identical")
    cal = allrows[allrows.n_boot == 500]
    if len(cal):
        m = cal.merge(main_runs, on=["k_true", "delta", "direction", "proportions", "background", "rep"],
                      suffixes=("_b500", "_b100"))
        agree = (m.gap_k_kmeans_b500 == m.gap_k_kmeans_b100).mean()
        dagree = (m.detected_primary_b500 == m.detected_primary_b100).mean()
        L.append(f"\ncalibration B=500 vs B={grid['n_boot']} on {CALIB_CELL}: selected k agree "
                 f"{agree:.2f}, detection agree {dagree:.2f} (n={len(m)})")
    L.append("=" * 100)
    text = "\n".join(L)
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
