#!/usr/bin/env python
"""Trace a planted signal through the D036 instrument, step by step.

The D036 run found no recovery even at delta = 6 on the gaussian background,
where two groups six background SDs apart should be trivially separable. This
checks, for k=2, random direction, balanced, each background and delta in
{0, 2, 4, 6}, one replicate with the exact D036 seeding:

  1. covariance spectrum of the real matrix (is one direction dominant?)
  2. separation of the planted labels along the planting axis u before and
     after re-z-scoring (should equal delta)
  3. which retained PC carries the planted labels (max |point-biserial r|)
     and how much of the PC space's variance the planted axis holds
  4. kmeans k=2 ARI on (a) the 1-D oracle projection onto u, (b) the D020
     PCs, (c) the PCs without PC1; plus kmeans SSE reduction of the planted
     split vs the kmeans split
  5. the D036 recovered flag pieces (ARI, Hennig) for the same replicate

Usage: OMP_NUM_THREADS=1 python scripts/diagnose_detectability.py --root outputs
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from predict.config import load_config
from predict.discover.cluster_discovery import fit_cluster, within_cluster_dispersion
from predict.discover.detectability import assign_groups, plant, planting_directions, zscore
from predict.reduce.pca import fit_pca

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sep(x: np.ndarray, lab: np.ndarray) -> float:
    a, b = x[lab == 0], x[lab == 1]
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float(abs(a.mean() - b.mean()) / pooled)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs")
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("_d036", REPO_ROOT / "scripts" / "probe_detectability_floor.py")
    d036 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d036)

    root = REPO_ROOT / args.root
    red = root / "06_reduce"
    cfg = load_config(root / "config.yaml" if (root / "config.yaml").exists() else None)
    prep = pd.read_csv(red / "prepared_matrix.csv", dtype={"pid": str})
    reps = pd.read_csv(red / "representative_features.csv").iloc[:, 0].tolist()
    X0 = zscore(prep[reps].to_numpy(float))
    cov = np.cov(X0, rowvar=False)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    print(f"real matrix {X0.shape}; cov eigenvalues top 6: {np.round(ev[:6], 2).tolist()}; "
          f"trace {ev.sum():.1f}; PC1 share {ev[0] / ev.sum():.2f}")
    k, direction, prop = 2, "random", "balanced"
    for bg in ("gaussian", "real"):
        for delta in (0.0, 2.0, 4.0, 6.0):
            seed = [k, 0, 0, d036.BACKGROUNDS.index(bg), int(delta * 10), 0, 36]
            rng = np.random.default_rng(seed)
            n, d = X0.shape
            Xb = X0 if bg == "real" else rng.multivariate_normal(np.zeros(d), cov, size=n)
            lab = assign_groups(rng, n, k, prop)
            U = planting_directions(rng, d, 1)
            u = U[:, 0]
            Xraw = plant(Xb, lab, delta, U, cov)
            Xp = zscore(Xraw)
            df = pd.DataFrame(Xp, columns=reps)
            df.insert(0, "pid", [f"s{i}" for i in range(n)])
            pca = fit_pca(df, reps, cumvar_threshold=cfg.reduce.pca_cumvar, random_state=42)
            Z = pca.scores
            r = [abs(np.corrcoef(Z[:, j], lab)[0, 1]) for j in range(Z.shape[1])]
            j_best = int(np.argmax(r))
            proj_u = Xp @ u
            lab_oracle = fit_cluster(proj_u[:, None], k=2, algorithm="kmeans", random_state=42)
            lab_pc = fit_cluster(Z, k=2, algorithm="kmeans", random_state=42)
            lab_nopc1 = fit_cluster(Z[:, 1:], k=2, algorithm="kmeans", random_state=42)
            W1 = within_cluster_dispersion(Z, np.zeros(n, int))
            print(f"[{bg:8s} delta {delta:.0f}] sd_along {np.sqrt(u @ cov @ u):.2f}; "
                  f"sep along u raw {_sep(Xraw @ u, lab):.2f} / after z {_sep(proj_u, lab):.2f}; "
                  f"PCs {Z.shape[1]}, best PC{j_best + 1} |r| {r[j_best]:.2f}; "
                  f"ARI oracle {adjusted_rand_score(lab, lab_oracle):.2f}, PCs {adjusted_rand_score(lab, lab_pc):.2f}, "
                  f"PCs-no-PC1 {adjusted_rand_score(lab, lab_nopc1):.2f}; "
                  f"SSE drop planted {1 - within_cluster_dispersion(Z, lab) / W1:.3f} vs kmeans "
                  f"{1 - within_cluster_dispersion(Z, lab_pc) / W1:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
