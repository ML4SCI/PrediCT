#!/usr/bin/env python
"""Do C8 carriers have more disease OUTSIDE their C8 lesions?

The carrier vs non-carrier Agatston contrast (Cliff's delta .93) is partly
mechanical: one C8 lesion (~250 mm3, ~830 HU) adds a large Agatston by
itself. lesions.csv has no per-lesion Agatston, so this compares carriers'
NON-C8 lesions with all lesions of non-carriers, per patient:

  n_other        number of lesions
  vol_other      total volume (mm3)
  mass_other     sum of volume x mean_hu_weighted (a density-weighted proxy)
  vessels_other  number of distinct vessels with a lesion

Rule fixed before running (primary k only): C8 marks heavier disease beyond
itself iff carriers exceed non-carriers on BOTH n_other and vol_other with
one-sided Mann-Whitney p < 0.05 AND Cliff's delta >= 0.33 (Romano medium).
The same metrics including the C8 lesions are reported to show how much of
the raw contrast is mechanical. Every k of the sweep is reported
descriptively.

C8 identity per k comes from k_sensitivity.py (same refit, same signature
rule, predict.figures._lesion_classes).

Usage:
    python experiments/lesion_morphology/c8_other_lesions.py [--config CONFIG]
Outputs: <outputs>/exploratory/lesion_morphology/c8_other_lesions/
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from predict.analyse.profiles import cliffs_delta, mannwhitney_u_pval
from predict.config import load_config
from predict.stage9 import stage9_root
from predict.discover.cluster_discovery import fit_cluster

HERE = Path(__file__).resolve().parent
METRICS = ("n_other", "vol_other", "mass_other", "vessels_other")
P_MAX, DELTA_MIN = 0.05, 0.33


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_lm_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"_lm_{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


def per_patient(lesions: pd.DataFrame, pids: list[str]) -> pd.DataFrame:
    g = lesions.assign(mass=lesions["volume_mm3"] * lesions["mean_hu_weighted"]).groupby("pid")
    out = pd.DataFrame({"n_other": g.size(), "vol_other": g["volume_mm3"].sum(),
                        "mass_other": g["mass"].sum(), "vessels_other": g["vessel"].nunique()})
    return out.reindex(pids).fillna(0.0)


def compare(df: pd.DataFrame, carrier: pd.Series) -> list[dict]:
    rows = []
    for m in METRICS:
        a, b = df.loc[carrier, m].to_numpy(float), df.loc[~carrier, m].to_numpy(float)
        rows.append({"metric": m, "carrier_median": float(np.median(a)),
                     "noncarrier_median": float(np.median(b)),
                     "mwu_p_one_sided": float(mannwhitney_u_pval(a, b, alternative="greater")),
                     "cliffs_delta": float(cliffs_delta(a, b))})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--k-values", type=int, nargs="+", default=[7, 8, 9, 10, 11, 12])
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    ks = _load("k_sensitivity")
    vessel_chi = ks._finalise_module().vessel_chi_square_per_cluster
    cfg = load_config(args.config)
    in_dir = stage9_root(cfg) / "lesion_morphology"
    out_dir = in_dir / "c8_other_lesions"
    out_dir.mkdir(parents=True, exist_ok=True)

    morph = pd.read_csv(in_dir / "lesion_features.csv", dtype={"pid": str})
    raw = pd.read_csv(cfg.paths.outputs / "03_features" / "lesions.csv", dtype={"pid": str})
    lesions = morph[ks.KEYS].merge(raw, on=ks.KEYS, how="left", validate="one_to_one")
    X = morph[[c for c in morph.columns if c not in ks.KEYS]].to_numpy(float)
    pids = pd.read_csv(cfg.paths.outputs / "06_reduce" / "cohort_metadata.csv",
                       dtype={"pid": str})["pid"].tolist()
    stored = pd.read_csv(in_dir / "lesion_cluster_labels.csv", dtype={"pid": str})
    k_primary = int([c for c in stored.columns if c.startswith("cluster_kmeans_k")][0].rsplit("k", 1)[1])

    rows = []
    for k in args.k_values:
        labels = fit_cluster(X, k=k, algorithm="kmeans", random_state=args.random_state)
        cid, _, _ = ks._c8(lesions, labels, "full", vessel_chi)
        if cid is None:
            rows.append({"k": k, "c8_found": False})
            continue
        in_c8 = labels == cid
        carrier = pd.Series(pids, index=pids).isin(set(lesions.loc[in_c8, "pid"]))
        only_c8 = int((per_patient(lesions[~in_c8], pids).loc[carrier, "n_other"] == 0).sum())
        for scope, df in (("other_lesions", per_patient(lesions[~in_c8], pids)),
                          ("all_lesions", per_patient(lesions, pids))):
            for r in compare(df, carrier):
                rows.append({"k": k, "is_primary": k == k_primary, "c8_found": True,
                             "scope": scope, "n_carriers": int(carrier.sum()),
                             "carriers_with_only_c8": only_c8, **r})

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "comparison.csv", index=False)

    prim = res[(res.k == k_primary) & (res.scope == "other_lesions")].set_index("metric")
    passes = {m: bool(prim.loc[m, "mwu_p_one_sided"] < P_MAX and prim.loc[m, "cliffs_delta"] >= DELTA_MIN)
              for m in ("n_other", "vol_other")} if len(prim) else {}
    verdict = bool(passes) and all(passes.values())

    L = ["=" * 96, f"C8 CARRIERS: DISEASE OUTSIDE THE C8 LESIONS (primary k={k_primary})", "=" * 96,
         " k  scope          metric          carrier  non-carrier   p(one-sided)  delta"]
    for r in res[res.c8_found].itertuples():
        L.append(f"{r.k:2d}{'*' if r.is_primary else ' '} {r.scope:<14} {r.metric:<14} "
                 f"{r.carrier_median:8.1f}  {r.noncarrier_median:10.1f}   {r.mwu_p_one_sided:11.3g}  "
                 f"{r.cliffs_delta:5.2f}")
    if len(prim):
        p0 = res[(res.k == k_primary)].iloc[0]
        L.append(f"\nprimary k={k_primary}: {int(p0.n_carriers)} carriers, "
                 f"{int(p0.carriers_with_only_c8)} have no lesion outside C8")
    L.append(f"RULE (fixed before running): n_other AND vol_other with p < {P_MAX} and delta >= {DELTA_MIN} "
             f"-> {passes} => {'SUPPORTED' if verdict else 'NOT SUPPORTED'}")
    L.append("=" * 96)
    text = "\n".join(L)
    (out_dir / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
