#!/usr/bin/env python
"""Root cause of the D038 unit-test failure: is it skew, or a bug?

Runs CCFI on synthetic cases with both dimensional comparison models
(Ruscio-style normal copula vs the D038 skewed-factor model), 3 seeds each:
  normal-latent continuum, skewed-latent continuum (lognormal sigma 0.4 /
  0.6 / 0.9), and a two-group taxon (separation 3 SD, base rate 0.3).
If the copula model reads the normal continuum dimensional but the skewed
ones categorical, and the skewed-factor model reads all continua
dimensional and the taxon categorical, the cause is the copula's handling
of a skewed latent dimension, not an implementation bug.

Usage: OMP_NUM_THREADS=1 python scripts/diagnose_taxometric.py
"""
from __future__ import annotations

import numpy as np

from predict.discover.taxometric import ccfi, latent_moments


def continuum(sigma, n=600, d=6, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal(n) if sigma == 0 else rng.lognormal(0.0, sigma, n)
    f = (f - f.mean()) / f.std()
    return 0.75 * f[:, None] + np.sqrt(1 - 0.75 ** 2) * rng.standard_normal((n, d))


def taxon(n=600, d=6, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.random(n) < 0.3
    return rng.standard_normal((n, d)) + 3.0 * g[:, None]


def main() -> int:
    cases = [("continuum normal", lambda s: continuum(0, seed=s))]
    cases += [(f"continuum lognormal s={sg}", lambda s, sg=sg: continuum(sg, seed=s)) for sg in (0.4, 0.6, 0.9)]
    cases += [("taxon 3 SD, base rate .3", lambda s: taxon(seed=s))]
    print(f"{'case':<28} {'model':<14} mean CCFI per seed            verdicts")
    tally = {"continuum": [0, 0], "taxon": [0, 0]}
    for name, make in cases:
        est = [latent_moments(make(s))[2] for s in range(3)]
        print(f"{name:<28} estimated latent skewness per seed: " + "  ".join(f"{e:.2f}" for e in est))
        for model in ("copula", "skewed_factor"):
            rs = [ccfi(make(s), n_comparison=20, seed=s, dimensional_model=model) for s in range(3)]
            print(f"{name:<28} {model:<14} " + "  ".join(f"{r.mean_ccfi:.3f}" for r in rs)
                  + "        " + ", ".join(r.verdict for r in rs), flush=True)
            if model == "skewed_factor":
                kind = "taxon" if name.startswith("taxon") else "continuum"
                want = "categorical" if kind == "taxon" else "dimensional"
                tally[kind][0] += sum(r.verdict == want for r in rs)
                tally[kind][1] += len(rs)
    c, t = tally["continuum"], tally["taxon"]
    ok = c[0] >= c[1] - 1 and t[0] == t[1]
    print()
    print(f"STOPPING RULE (D038 revision 2, fixed before this run): skewed_factor continua "
          f"dimensional {c[0]}/{c[1]} (need >= {c[1] - 1}), taxon categorical {t[0]}/{t[1]} "
          f"(need {t[1]}) -> {'ACCEPT' if ok else 'REJECT: taxometric test not usable'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
