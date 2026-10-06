#!/usr/bin/env python
"""Does C8 carry information beyond total burden? Burden-matched comparison.

Each C8 carrier (primary k) is matched to the nearest non-carrier on
log1p(Agatston total), exactly within the same reconstruction kernel, with
replacement, caliper 0.2 SD of log1p(Agatston). Outcomes use the NON-C8
lesions only, so the C8 lesion itself never enters an outcome.

At equal total Agatston a carrier's burden is partly the C8 plaque, so its
other lesions are fewer / smaller by construction. Count and volume are
therefore reported descriptively, not judged. The primary outcomes are ones
the amount of other disease does not mechanically drive:

  P1 density_other    median mean_hu_weighted of the patient's non-C8 lesions
  P2 rca_share_other  RCA share of the patient's non-C8 lesion volume

Rule fixed before running: C8 carries information beyond burden iff at least
one primary outcome differs between matched carriers and controls with
Wilcoxon signed-rank p < 0.025 (Bonferroni over 2) AND |Cliff's delta| >=
0.33. If fewer than 40 carriers are matched, the verdict is INCONCLUSIVE.
Secondary, descriptive: n_other, vol_other, vessels_other, lm_any_other,
max_hu_other. With replacement, reused controls make the signed-rank test
approximate (reported: number of distinct controls).

Usage:
    python experiments/lesion_morphology/c8_burden_matched.py [--config CONFIG]
Outputs: <outputs>/exploratory/lesion_morphology/c8_burden_matched/
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from predict.analyse.profiles import cliffs_delta
from predict.config import load_config
from predict.stage9 import stage9_root
from predict.discover.cluster_discovery import fit_cluster

HERE = Path(__file__).resolve().parent
PRIMARY = ("density_other", "rca_share_other")
SECONDARY = ("n_other", "vol_other", "vessels_other", "lm_any_other", "max_hu_other")
P_MAX, DELTA_MIN, MIN_MATCHED, CALIPER_SD = 0.025, 0.33, 40, 0.2


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_lm_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"_lm_{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


def patient_outcomes(other: pd.DataFrame, pids: list[str]) -> pd.DataFrame:
    g = other.groupby("pid")
    rca_vol = other[other["vessel"] == "RCA"].groupby("pid")["volume_mm3"].sum()
    out = pd.DataFrame({
        "density_other": g["mean_hu_weighted"].median(),
        "rca_share_other": rca_vol.reindex(g.size().index).fillna(0.0) / g["volume_mm3"].sum(),
        "n_other": g.size().astype(float),
        "vol_other": g["volume_mm3"].sum(),
        "vessels_other": g["vessel"].nunique().astype(float),
        "lm_any_other": g["vessel"].apply(lambda s: float((s == "LM").any())),
        "max_hu_other": g["max_hu"].max(),
    }).reindex(pids)
    # No non-C8 lesion: counts are 0; density / share / max HU stay undefined.
    for c in ("n_other", "vol_other", "vessels_other", "lm_any_other"):
        out[c] = out[c].fillna(0.0)
    return out


def match(meta: pd.DataFrame, carriers: set[str]) -> tuple[pd.DataFrame, float]:
    x = np.log1p(meta["agatston_total"].astype(float))
    caliper = CALIPER_SD * float(x.std(ddof=1))
    rows = []
    for pid in sorted(carriers):
        pool = meta.index[(~meta.index.isin(carriers)) & (meta["kernel"] == meta.at[pid, "kernel"])]
        if len(pool) == 0:
            continue
        d = (x.loc[pool] - x.loc[pid]).abs()
        best = d.idxmin()
        if d.loc[best] <= caliper:
            rows.append({"carrier": pid, "control": best, "kernel": meta.at[pid, "kernel"],
                         "log_agatston_carrier": float(x.loc[pid]),
                         "log_agatston_control": float(x.loc[best]),
                         "abs_diff": float(d.loc[best])})
    return pd.DataFrame(rows), caliper


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    ks = _load("k_sensitivity")
    vessel_chi = ks._finalise_module().vessel_chi_square_per_cluster
    cfg = load_config(args.config)
    in_dir = stage9_root(cfg) / "lesion_morphology"
    out_dir = in_dir / "c8_burden_matched"
    out_dir.mkdir(parents=True, exist_ok=True)

    morph = pd.read_csv(in_dir / "lesion_features.csv", dtype={"pid": str})
    raw = pd.read_csv(cfg.paths.outputs / "03_features" / "lesions.csv", dtype={"pid": str})
    lesions = morph[ks.KEYS].merge(raw, on=ks.KEYS, how="left", validate="one_to_one")
    X = morph[[c for c in morph.columns if c not in ks.KEYS]].to_numpy(float)
    meta = pd.read_csv(cfg.paths.outputs / "06_reduce" / "cohort_metadata.csv",
                       dtype={"pid": str}).set_index("pid")
    meta["kernel"] = meta["kernel"].astype(str)
    pids = meta.index.tolist()
    stored = pd.read_csv(in_dir / "lesion_cluster_labels.csv", dtype={"pid": str})
    k = int([c for c in stored.columns if c.startswith("cluster_kmeans_k")][0].rsplit("k", 1)[1])

    labels = fit_cluster(X, k=k, algorithm="kmeans", random_state=args.random_state)
    cid, _, _ = ks._c8(lesions, labels, "full", vessel_chi)
    if cid is None:
        raise SystemExit(f"no C8-like cluster at k={k}")
    in_c8 = labels == cid
    carriers = set(lesions.loc[in_c8, "pid"]) & set(pids)
    outc = patient_outcomes(lesions[~in_c8], pids)
    pairs, caliper = match(meta, carriers)
    pairs.to_csv(out_dir / "pairs.csv", index=False)

    n_matched = len(pairs)
    smd_before = None
    x = np.log1p(meta["agatston_total"].astype(float))
    car, non = x.loc[list(carriers)], x.loc[~x.index.isin(carriers)]
    pooled = np.sqrt((car.var(ddof=1) + non.var(ddof=1)) / 2)
    smd_before = float((car.mean() - non.mean()) / pooled)
    smd_after = (float((pairs.log_agatston_carrier.mean() - pairs.log_agatston_control.mean()) / pooled)
                 if n_matched else float("nan"))

    rows = []
    for m in PRIMARY + SECONDARY:
        a = outc.loc[pairs["carrier"], m].to_numpy(float) if n_matched else np.array([])
        b = outc.loc[pairs["control"], m].to_numpy(float) if n_matched else np.array([])
        ok = ~(np.isnan(a) | np.isnan(b))
        a, b = a[ok], b[ok]
        diff = a - b
        p = float(wilcoxon(diff).pvalue) if len(diff) >= 10 and np.any(diff != 0) else float("nan")
        rows.append({"outcome": m, "role": "primary" if m in PRIMARY else "secondary",
                     "n_pairs": int(ok.sum()), "carrier_median": float(np.median(a)) if len(a) else np.nan,
                     "control_median": float(np.median(b)) if len(b) else np.nan,
                     "median_pair_diff": float(np.median(diff)) if len(diff) else np.nan,
                     "wilcoxon_p": p,
                     "cliffs_delta": float(cliffs_delta(a, b)) if len(a) else np.nan})
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "comparison.csv", index=False)

    prim = res[res.role == "primary"]
    hits = {r.outcome: bool(r.wilcoxon_p < P_MAX and abs(r.cliffs_delta) >= DELTA_MIN)
            for r in prim.itertuples()}
    if n_matched < MIN_MATCHED:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "SUPPORTED" if any(hits.values()) else "NOT SUPPORTED"

    L = ["=" * 96, f"C8 BEYOND BURDEN: burden-matched comparison (k={k}, C8 = C{cid})", "=" * 96,
         f"carriers in cohort {len(carriers)}; matched {n_matched} (exact kernel, caliper "
         f"{caliper:.3f} on log1p Agatston = {CALIPER_SD} SD); distinct controls "
         f"{pairs['control'].nunique() if n_matched else 0}",
         f"log1p Agatston SMD before {smd_before:.2f}, after {smd_after:.3f}",
         "", " outcome           role       n   carrier   control   pair diff   p(Wilcoxon)  delta"]
    for r in res.itertuples():
        L.append(f" {r.outcome:<17} {r.role:<9} {r.n_pairs:3d} {r.carrier_median:9.2f} "
                 f"{r.control_median:9.2f} {r.median_pair_diff:10.2f}   {r.wilcoxon_p:11.3g}  "
                 f"{r.cliffs_delta:5.2f}")
    L.append(f"\nRULE (fixed before running): a primary outcome with p < {P_MAX} and |delta| >= "
             f"{DELTA_MIN}; >= {MIN_MATCHED} matched -> {hits} => {verdict}")
    L.append("Secondary rows: n_other / vol_other are expected LOWER in carriers at equal "
             "Agatston (the C8 plaque is part of their burden).")
    L.append("=" * 96)
    text = "\n".join(L)
    (out_dir / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
