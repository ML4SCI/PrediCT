#!/usr/bin/env python
"""Sensitivity of the lesion-level findings to the k-means cluster count.

The primary lesion partition stays at run.py's gap-selected k. This sweeps k
(default 7..12) on the same z-scored morphology matrix and seed, and reports
per k:

  * broad-class counts (soft / moderate / dense) by the locked signature rule
  * the C8-like cluster by the locked signature rule (or none), its medians,
    vessel obs/exp, Hennig Jaccard, lesion-set Jaccard against the primary
    C8, and carrier burden vs non-carriers (one-sided MWU, Cliff's delta)
  * per-kernel-stratum refit at the same k: is a C8-like cluster found?

Rule fixed before the sweep was run: the C8 phenotype is called robust to k
only if, at EVERY k in the sweep, a cluster matches the signature in the full
cohort AND in both kernel strata. The primary k is not changed by this.

Signature rules: predict.figures._lesion_classes (shared with finalise.py
and the figures).

Usage:
    python experiments/lesion_morphology/k_sensitivity.py [--config CONFIG]
Outputs: <outputs>/exploratory/lesion_morphology/k_sensitivity/
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from predict.analyse.profiles import cliffs_delta, mannwhitney_u_pval
from predict.config import default_n_jobs, load_config
from predict.stage9 import stage9_root
from predict.discover.cluster_discovery import fit_cluster
from predict.discover.validity import hennig_clusterboot
from predict.figures._lesion_classes import discover_broad_classes, discover_c8_like_cluster

HERE = Path(__file__).resolve().parent
KEYS = ["pid", "vessel", "lesion_idx"]


def _finalise_module():
    spec = importlib.util.spec_from_file_location("_lm_finalise", HERE / "finalise.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_lm_finalise"] = mod
    spec.loader.exec_module(mod)
    return mod


def _profiles(lesions: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    return (lesions.assign(cluster=labels)
            .groupby("cluster")
            .agg(n_lesions=("volume_mm3", "size"),
                 volume_mm3_median=("volume_mm3", "median"),
                 max_hu_median=("max_hu", "median"),
                 n_rois_median=("n_rois", "median"))
            .reset_index())


def _c8(lesions: pd.DataFrame, labels: np.ndarray, label: str, vessel_chi):
    prof = _profiles(lesions, labels)
    vc = vessel_chi(lesions, labels, cohort_label=label)
    cid = discover_c8_like_cluster(prof, vc)
    row = None
    if cid is not None:
        row = prof.merge(vc[["cluster", "rca_obs_over_exp", "lm_obs_over_exp", "cramers_v"]],
                         on="cluster").set_index("cluster").loc[cid]
    return cid, row, prof


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--k-values", type=int, nargs="+", default=[7, 8, 9, 10, 11, 12])
    ap.add_argument("--hennig-bootstraps", type=int, default=100)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=default_n_jobs())
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log = logging.getLogger("k_sensitivity")

    cfg = load_config(args.config)
    in_dir = stage9_root(cfg) / "lesion_morphology"
    out_dir = in_dir / "k_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    vessel_chi = _finalise_module().vessel_chi_square_per_cluster

    morph = pd.read_csv(in_dir / "lesion_features.csv", dtype={"pid": str})
    raw = pd.read_csv(cfg.paths.outputs / "03_features" / "lesions.csv", dtype={"pid": str})
    lesions = morph[KEYS].merge(raw, on=KEYS, how="left", validate="one_to_one")
    feat_cols = [c for c in morph.columns if c not in KEYS]
    X = morph[feat_cols].to_numpy(float)
    meta = pd.read_csv(cfg.paths.outputs / "06_reduce" / "cohort_metadata.csv",
                       dtype={"pid": str}).set_index("pid")
    agatston = meta["agatston_total"].astype(float)
    stored = pd.read_csv(in_dir / "lesion_cluster_labels.csv", dtype={"pid": str})
    primary_col = [c for c in stored.columns if c.startswith("cluster_kmeans_k")][0]
    k_primary = int(primary_col.rsplit("k", 1)[1])
    primary = morph[KEYS].merge(stored[KEYS + [primary_col]], on=KEYS, how="left")[primary_col].to_numpy()
    p_cid, _, _ = _c8(lesions, primary, "full", vessel_chi)
    primary_c8 = set(np.flatnonzero(primary == p_cid)) if p_cid is not None else set()
    log.info("primary k=%d, primary C8-like cluster C%s (%d lesions)", k_primary, p_cid, len(primary_c8))

    rows, strata_rows = [], []
    for k in args.k_values:
        labels = fit_cluster(X, k=k, algorithm="kmeans", random_state=args.random_state)
        cid, c8row, prof = _c8(lesions, labels, "full", vessel_chi)
        broad = discover_broad_classes(prof)
        rec = {"k": k, "is_primary": k == k_primary,
               "n_soft": sum(v == "soft_microspots" for v in broad.values()),
               "n_moderate": sum(v == "moderate_nodules" for v in broad.values()),
               "n_dense": sum(v == "dense_plaques" for v in broad.values()),
               "c8_found": cid is not None}
        if k == k_primary:
            from sklearn.metrics import adjusted_rand_score
            rec["ari_vs_stored_primary"] = float(adjusted_rand_score(labels, primary))
        if cid is not None:
            members = np.flatnonzero(labels == cid)
            carriers = set(lesions.loc[members, "pid"])
            c_b = agatston.loc[agatston.index.intersection(carriers)].dropna().to_numpy()
            n_b = agatston.loc[~agatston.index.isin(carriers)].dropna().to_numpy()
            hen = hennig_clusterboot(X, labels, k=k, algorithm="kmeans",
                                     n_bootstrap=args.hennig_bootstraps, threshold=0.75,
                                     random_state=args.random_state, n_jobs=args.n_jobs)
            jac = dict(zip(hen.cluster_ids, hen.jaccard_median))
            inter = len(primary_c8 & set(members))
            rec.update({
                "c8_cluster": int(cid), "c8_n_lesions": int(len(members)),
                "c8_vol_median": float(c8row["volume_mm3_median"]),
                "c8_max_hu_median": float(c8row["max_hu_median"]),
                "c8_n_rois_median": float(c8row["n_rois_median"]),
                "c8_rca_obs_over_exp": float(c8row["rca_obs_over_exp"]),
                "c8_lm_obs_over_exp": float(c8row["lm_obs_over_exp"]),
                "c8_cramers_v": float(c8row["cramers_v"]),
                "c8_hennig_jaccard_median": float(jac.get(cid, np.nan)),
                "n_stable_clusters": int(sum(v >= 0.75 for v in jac.values())),
                "c8_jaccard_vs_primary_c8":
                    inter / len(primary_c8 | set(members)) if primary_c8 else np.nan,
                "c8_n_carriers": len(carriers),
                "carrier_agatston_median": float(np.median(c_b)),
                "noncarrier_agatston_median": float(np.median(n_b)),
                "carrier_mwu_p_one_sided": float(mannwhitney_u_pval(c_b, n_b, alternative="greater")),
                "carrier_cliffs_delta": float(cliffs_delta(c_b, n_b)),
            })
        for stratum in ("Qr36d/2", "I30f/3"):
            pids = set(meta.index[meta["kernel"].astype(str) == stratum])
            m = lesions["pid"].isin(pids).to_numpy()
            ls = fit_cluster(X[m], k=k, algorithm="kmeans", random_state=args.random_state)
            sid, srow, _ = _c8(lesions[m].reset_index(drop=True), ls, stratum, vessel_chi)
            strata_rows.append({
                "k": k, "stratum": stratum, "n_lesions": int(m.sum()), "c8_found": sid is not None,
                **({} if sid is None else {
                    "c8_n_lesions": int(srow["n_lesions"]),
                    "c8_vol_median": float(srow["volume_mm3_median"]),
                    "c8_max_hu_median": float(srow["max_hu_median"]),
                    "c8_n_rois_median": float(srow["n_rois_median"]),
                    "c8_rca_obs_over_exp": float(srow["rca_obs_over_exp"]),
                    "c8_lm_obs_over_exp": float(srow["lm_obs_over_exp"])})})
        rows.append(rec)
        log.info("k=%d done: C8 %s", k, "found" if cid is not None else "NOT found")

    summ, strata = pd.DataFrame(rows), pd.DataFrame(strata_rows)
    summ.to_csv(out_dir / "summary.csv", index=False)
    strata.to_csv(out_dir / "strata.csv", index=False)

    found_all = bool(summ["c8_found"].all())
    strata_all = bool(strata["c8_found"].all())
    L = ["=" * 100, f"LESION k SENSITIVITY (primary k={k_primary}; sweep {args.k_values})", "=" * 100,
         " k  soft/mod/dense  C8  n_les  vol   maxHU  rois  RCA   LM  Hennig  Jac>k8  carriers  "
         "Agatston carrier/non  delta"]
    for r in summ.itertuples():
        mark = "*" if r.is_primary else " "
        if r.c8_found:
            L.append(f"{r.k:2d}{mark} {r.n_soft:4d}/{r.n_moderate}/{r.n_dense:<5d}  yes {r.c8_n_lesions:5d} "
                     f"{r.c8_vol_median:5.0f} {r.c8_max_hu_median:6.0f} {r.c8_n_rois_median:5.0f} "
                     f"{r.c8_rca_obs_over_exp:4.2f} {r.c8_lm_obs_over_exp:4.2f} {r.c8_hennig_jaccard_median:6.2f} "
                     f"{r.c8_jaccard_vs_primary_c8:7.2f} {r.c8_n_carriers:8d}  "
                     f"{r.carrier_agatston_median:7.0f}/{r.noncarrier_agatston_median:<6.0f}    "
                     f"{r.carrier_cliffs_delta:5.2f}")
        else:
            L.append(f"{r.k:2d}{mark} {r.n_soft:4d}/{r.n_moderate}/{r.n_dense:<5d}  NO")
    L.append("\nper-stratum C8-like found:")
    for k, g in strata.groupby("k"):
        L.append(f"  k={k:2d}: " + ", ".join(
            f"{r.stratum} {'yes' if r.c8_found else 'NO'}"
            + (f" (vol {r.c8_vol_median:.0f}, HU {r.c8_max_hu_median:.0f}, RCA {r.c8_rca_obs_over_exp:.2f})"
               if r.c8_found else "") for r in g.itertuples()))
    L.append(f"\nRULE (fixed before running): C8 robust to k iff found at every k in full cohort "
             f"AND both strata -> full {found_all}, strata {strata_all} => "
             f"{'ROBUST' if found_all and strata_all else 'NOT ROBUST'}")
    L.append("=" * 100)
    text = "\n".join(L)
    (out_dir / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
