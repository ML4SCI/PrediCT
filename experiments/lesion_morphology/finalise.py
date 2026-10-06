#!/usr/bin/env python
"""Stage-7-equivalent rigour applied to the lesion-morphology exploration.

Closes the 8-item plan handed over after the k=12 rerun at 0.5 mm. The
primary partition k is whatever run.py's gap vote selects, so NO cluster id
is hardcoded: broad classes and the C8-like cluster are discovered from each
run's cluster medians by the locked signature rules in
predict.figures._lesion_classes (the same rules the figures use). "C8" below means the C8-like cluster, whatever its label.

  1. Lock the 3 broad class names (density-based: soft_microspots /
     moderate_nodules / dense_plaques), assigned per cluster by signature.
     Primary clusters stay C0..C{k-1}.
  2. C8 deep dive: per-patient burden distribution + Mann-Whitney; within
     RCA z-position relative to each carrier's RCA z-range ((z - min z) /
     range, 0 = lowest lesion, +z superior). This is the superior-inferior
     coordinate; it does not establish proximal / distal position along the
     curving RCA (D044).
  3. Jonckheere-Terpstra trend test for each cluster fraction across
     burden tertiles. Formal monotone-trend p-value.
  4. Patient mixture GMM k=4 phenotype claim (Finding 5 candidate):
     fit on CLR-transformed mixtures + Hennig stability + per-cluster
     mean-fraction profile + crosstab against stage-6 focal/diffuse +
     burden tertile + kernel.
  5. Secondary anatomical biases: every vessel-biased cluster (table).
  6. Why GMM separates by density not size: Spearman rho of cluster_id
     against max_hu vs against volume_mm3.
  7. (Folded into 4: the mixture GMM k=4 IS the new patient phenotype claim.)
  8. Sensitivity: does the C8-like signature replicate when the partition
     is refit on each kernel stratum separately?

NOT part of the production pipeline. Outputs land in
``outputs/exploratory/lesion_morphology/finalise/``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

from predict.analyse.profiles import (
    cliffs_delta,
    determine_focal_diffuse_mapping,
    mannwhitney_u_pval,
)
from predict.config import default_n_jobs, load_config
from predict.stage9 import stage9_root
from predict.discover.cluster_discovery import fit_cluster
from predict.discover.clusterability import assess_clusterability
from predict.discover.validity import hennig_clusterboot
from predict.figures._lesion_classes import (
    discover_broad_classes,
    discover_c8_like_cluster,
)


# ─────────────────────── helpers ───────────────────────


def _git_hash(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root), capture_output=True, text=True, check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _file_sha(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _save_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n",
                    encoding="utf-8")


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ─────────────────────── Jonckheere-Terpstra ───────────────────────


def jonckheere_terpstra(groups: list[np.ndarray]) -> dict:
    """JT trend test for ordered alternatives across k >= 2 groups.

    Groups must be passed in ASCENDING ordinal order (e.g. low, mid, high).
    Tests H0: distributions are equal vs H1: F_1 <= F_2 <= ... <= F_k
    with at least one strict inequality (i.e., monotone trend).

    Returns:
        J            : JT statistic (with mid-rank treatment of ties)
        E_J, var_J   : null mean and variance (Lehmann 1975)
        Z            : standardised statistic
        p_two_sided  : two-sided normal-approximation p-value
        direction    : "increasing" if Z>0, "decreasing" if Z<0, "flat" if Z=0
    """
    arrays = [np.asarray(g, dtype=float) for g in groups]
    arrays = [a[~np.isnan(a)] for a in arrays]
    k = len(arrays)
    n_per = [len(a) for a in arrays]
    n_total = sum(n_per)
    if n_total == 0 or k < 2:
        return {
            "J": float("nan"), "E_J": float("nan"), "var_J": float("nan"),
            "Z": float("nan"), "p_two_sided": float("nan"),
            "direction": "undefined",
        }
    J = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            a, b = arrays[i], arrays[j]
            cmp = a[:, None] < b[None, :]
            ties = a[:, None] == b[None, :]
            J += float(cmp.sum() + 0.5 * ties.sum())
    E_J = (n_total ** 2 - sum(ni * ni for ni in n_per)) / 4.0
    var_J = (
        n_total ** 2 * (2 * n_total + 3)
        - sum(ni * ni * (2 * ni + 3) for ni in n_per)
    ) / 72.0
    if var_J <= 0:
        Z = float("nan")
        p = float("nan")
    else:
        Z = (J - E_J) / np.sqrt(var_J)
        p = 2.0 * (1.0 - stats.norm.cdf(abs(Z)))
    direction = "increasing" if Z > 0 else "decreasing" if Z < 0 else "flat"
    return {
        "J": float(J), "E_J": float(E_J), "var_J": float(var_J),
        "Z": float(Z), "p_two_sided": float(p), "direction": direction,
    }


# ─────────────────────── CLR (lifted from analyse.py) ───────────────────────


def clr_transform(fractions: np.ndarray,
                  pseudocount: float | None = None) -> np.ndarray:
    F = np.asarray(fractions, dtype=float).copy()
    if F.ndim != 2:
        raise ValueError("clr_transform expects 2D input")
    n_cat = F.shape[1]
    if pseudocount is None:
        pseudocount = 1.0 / (2.0 * n_cat)
    F[F < pseudocount] = pseudocount
    F = F / F.sum(axis=1, keepdims=True)
    log_F = np.log(F)
    return log_F - log_F.mean(axis=1, keepdims=True)


# ─────────────────────── vessel chi-square (lifted) ───────────────────────


def vessel_chi_square_per_cluster(
    lesions: pd.DataFrame,
    labels: np.ndarray,
    cohort_label: str,
) -> pd.DataFrame:
    vessels = ["LAD", "RCA", "LCx", "LM"]
    cohort_counts = lesions["vessel"].value_counts().reindex(vessels, fill_value=0)
    cohort_total = int(cohort_counts.sum())
    cohort_p = cohort_counts / cohort_total
    rows = []
    for cid in sorted(np.unique(labels)):
        rows_in = lesions[labels == cid]
        n_in = int(len(rows_in))
        obs = rows_in["vessel"].value_counts().reindex(vessels, fill_value=0)
        exp = n_in * cohort_p
        mask = exp > 0
        if mask.sum() < 2:
            chi2_stat, chi2_p, V = float("nan"), float("nan"), float("nan")
        else:
            chi2_stat, chi2_p = stats.chisquare(
                obs[mask].to_numpy(), exp[mask].to_numpy(),
            )
            V = float(np.sqrt(chi2_stat / (n_in * (int(mask.sum()) - 1))))
        row = {
            "cohort": cohort_label,
            "cluster": int(cid),
            "n_lesions": n_in,
            "chi2_p": float(chi2_p),
            "cramers_v": V,
            "vessel_biased": (
                bool(chi2_p < 0.001 and V >= 0.20)
                if not np.isnan(chi2_p) else False
            ),
        }
        for v in vessels:
            row[f"{v.lower()}_obs_over_exp"] = (
                float(obs[v] / exp[v]) if exp[v] > 0 else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────── main ───────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--n-jobs", type=int, default=default_n_jobs())
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--mixture-k", type=int, default=4,
                        help="patient-mixture GMM k for Finding-5 phenotype probe")
    parser.add_argument("--hennig-bootstraps", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("finalise")

    cfg = load_config(args.config)
    in_dir = stage9_root(cfg) / "lesion_morphology"
    out_dir = _ensure_dir(in_dir / "finalise")
    repo_root = cfg.paths.outputs.parent

    # ── Load existing seam ────────────────────────────────────────
    lesions_csv = cfg.paths.outputs / "03_features" / "lesions.csv"
    labels_csv = in_dir / "lesion_cluster_labels.csv"
    morph_csv = in_dir / "lesion_features.csv"
    profiles_csv = in_dir / "cluster_profiles.csv"
    vessel_chi_csv = in_dir / "cluster_vessel_chi_square.csv"
    cohort_meta_csv = cfg.paths.outputs / "06_reduce" / "cohort_metadata.csv"
    spatial_csv = cfg.paths.outputs / "06_reduce" / "cluster_labels_spatial_k2.csv"
    features_csv = cfg.paths.outputs / "03_features" / "features.csv"

    for p in (lesions_csv, labels_csv, morph_csv, profiles_csv, vessel_chi_csv,
              cohort_meta_csv, spatial_csv, features_csv):
        if not p.exists():
            log.error("missing %s", p)
            return 2

    lesions = pd.read_csv(lesions_csv, dtype={"pid": str})
    labels_df = pd.read_csv(labels_csv, dtype={"pid": str})
    morph_df = pd.read_csv(morph_csv, dtype={"pid": str})
    profiles = pd.read_csv(profiles_csv)
    vessel_chi = pd.read_csv(vessel_chi_csv)
    meta = pd.read_csv(cohort_meta_csv, dtype={"pid": str})
    spatial = pd.read_csv(spatial_csv, dtype={"pid": str})
    features = pd.read_csv(features_csv, dtype={"pid": str})

    primary_col = [c for c in labels_df.columns
                   if c.startswith("cluster_kmeans_k")][0]
    k_primary = int(labels_df[primary_col].max() + 1)
    lesions = lesions.merge(
        labels_df[["pid", "vessel", "lesion_idx", primary_col]],
        on=["pid", "vessel", "lesion_idx"], how="left",
    )
    lesions[primary_col] = lesions[primary_col].astype(int)
    primary_labels = lesions[primary_col].to_numpy()
    log.info("loaded %d lesions partitioned at k=%d (column %s)",
             len(lesions), k_primary, primary_col)

    # Patient-level helpers
    meta_indexed = meta.set_index("pid")
    spatial_indexed = spatial.set_index("pid")["spatial_only_gmm_k2"].astype(int)
    n_calc = features.set_index("pid")["n_calcified_arteries"]
    # focal/diffuse mapping (D023). Delegates to the canonical production
    # helper rather than reimplementing the rule here.
    #
    # The previous local implementation was `min(cm, key=cm.get)` paired with
    # `max(...)`. `n_calcified_arteries` is an integer in {1,2,3,4}, so the two
    # arms sharing a median is an ordinary outcome, not an edge case. On a tie
    # both calls return the SAME key, the dict literal
    # {focal_cid: "focal", diffuse_cid: "diffuse"} collapses to one entry, and
    # the other cluster silently mapped to NaN. The canonical helper validates
    # the 2-cluster precondition and RAISES on identical medians, which is the
    # correct behaviour: if the arms are indistinguishable on the mapping
    # variable, the focal/diffuse label has no basis and the run should stop.
    shared_pids = spatial_indexed.index.intersection(n_calc.dropna().index)
    if len(shared_pids) == 0:
        raise ValueError(
            "focal/diffuse mapping: no pids shared between the spatial-k2 "
            "labels and n_calcified_arteries"
        )
    mapping = determine_focal_diffuse_mapping(
        pd.DataFrame(
            {"n_calcified_arteries": n_calc.loc[shared_pids]},
            index=shared_pids,
        ),
        spatial_indexed.loc[shared_pids],
    )
    focal_cid = next(c for c, lab in mapping.items() if lab == "focal")
    diffuse_cid = next(c for c, lab in mapping.items() if lab == "diffuse")
    focal_diffuse = spatial_indexed.map(mapping)
    log.info("focal/diffuse mapping: cluster %d=focal, %d=diffuse "
             "(n=%d pids used for the mapping)",
             focal_cid, diffuse_cid, len(shared_pids))

    agatston = meta_indexed["agatston_total"].astype(float)
    kernel = meta_indexed["kernel"].astype(str)

    # ── 1. Broad class per primary cluster, by signature ──────────
    primary_to_broad = discover_broad_classes(profiles)
    primary_to_broad_df = pd.DataFrame(
        [(cid, primary_to_broad.get(cid, "unassigned"))
         for cid in range(k_primary)],
        columns=["primary_cluster", "broad_class"],
    )
    primary_to_broad_df.to_csv(out_dir / "broad_class_mapping.csv", index=False)
    log.info("broad classes by signature: %s", primary_to_broad)

    # ── 2. C8 deep dive ────────────────────────────────────────────
    log.info("--- 2. C8 deep dive ---")
    c8_id = discover_c8_like_cluster(profiles, vessel_chi)
    if c8_id is None:
        raise SystemExit(
            "no cluster matches the C8-like signature in this run "
            "(predict.figures._lesion_classes); the C8 deep dive has no subject"
        )
    log.info("C8-like cluster by signature: C%d (of k=%d)", c8_id, k_primary)
    c8_lesions = lesions[primary_labels == c8_id]
    c8_patients = c8_lesions["pid"].unique()
    non_c8_patients = [p for p in meta["pid"] if p not in set(c8_patients)]

    c8_burden = agatston.loc[agatston.index.intersection(c8_patients)].dropna()
    non_c8_burden = agatston.loc[
        agatston.index.intersection(non_c8_patients)
    ].dropna()
    c8_burden_mw_p = mannwhitney_u_pval(c8_burden.to_numpy(),
                                         non_c8_burden.to_numpy(),
                                         alternative="greater")
    c8_burden_delta = cliffs_delta(c8_burden.to_numpy(), non_c8_burden.to_numpy())

    c8_deep = {
        "c8_like_cluster_id": int(c8_id),
        "k_primary": int(k_primary),
        "n_c8_lesions": int(len(c8_lesions)),
        "n_c8_patients": int(len(c8_patients)),
        "c8_patient_burden_median": float(np.median(c8_burden)),
        "non_c8_patient_burden_median": float(np.median(non_c8_burden)),
        "mannwhitney_p_one_sided_c8_higher": float(c8_burden_mw_p),
        "cliffs_delta_burden_c8_vs_non": float(c8_burden_delta),
    }

    # Per-patient relative z-position of C8 RCA lesions within RCA z-range.
    # For each C8-containing patient, compute (z - rca_min_z) / (rca_max_z -
    # rca_min_z) for both C8 and other RCA lesions. Compare distributions.
    rca = lesions[lesions["vessel"] == "RCA"]
    rel_z_records = []
    for pid in c8_patients:
        rca_in_p = rca[rca["pid"] == pid]
        if len(rca_in_p) < 2:
            continue
        zmin = rca_in_p["centroid_z_mm"].min()
        zmax = rca_in_p["centroid_z_mm"].max()
        if zmax == zmin:
            continue
        for _, row in rca_in_p.iterrows():
            rel = (row["centroid_z_mm"] - zmin) / (zmax - zmin)
            rel_z_records.append({
                "pid": pid,
                "in_c8": bool(row[primary_col] == c8_id),
                "relative_z_within_RCA": float(rel),
            })
    rel_z_df = pd.DataFrame(rel_z_records)
    rel_z_df.to_csv(out_dir / "c8_rca_relative_z.csv", index=False)
    if len(rel_z_df) > 0:
        c8_rel_z = rel_z_df.loc[rel_z_df["in_c8"], "relative_z_within_RCA"].to_numpy()
        other_rel_z = rel_z_df.loc[~rel_z_df["in_c8"], "relative_z_within_RCA"].to_numpy()
        if c8_rel_z.size > 0 and other_rel_z.size > 0:
            c8_deep["c8_RCA_relative_z_median"] = float(np.median(c8_rel_z))
            c8_deep["other_RCA_relative_z_median"] = float(np.median(other_rel_z))
            c8_deep["c8_vs_other_RCA_z_mw_p_two_sided"] = float(
                mannwhitney_u_pval(c8_rel_z, other_rel_z, alternative="two-sided")
            )
            c8_deep["c8_vs_other_RCA_z_cliffs_delta"] = float(
                cliffs_delta(c8_rel_z, other_rel_z)
            )
    _save_json(out_dir / "c8_deep_dive.json", c8_deep)
    log.info("C8 patients (N=%d) burden median = %.1f vs non-C8 (N=%d) median = "
             "%.1f; Mann-Whitney p=%.3g; Cliff's delta=%.3f",
             c8_deep["n_c8_patients"], c8_deep["c8_patient_burden_median"],
             len(non_c8_patients), c8_deep["non_c8_patient_burden_median"],
             c8_deep["mannwhitney_p_one_sided_c8_higher"],
             c8_deep["cliffs_delta_burden_c8_vs_non"])

    # ── 3. Jonckheere-Terpstra trend per cluster ──────────────────
    log.info("--- 3. Jonckheere-Terpstra trend test per cluster ---")
    # Per-patient cluster fractions
    counts = (lesions.groupby(["pid", primary_col]).size()
              .unstack(fill_value=0))
    for c in range(k_primary):
        if c not in counts.columns:
            counts[c] = 0
    counts = counts.sort_index(axis=1)
    fractions = counts.div(counts.sum(axis=1), axis=0)
    fractions.columns = [f"frac_C{c}" for c in counts.columns]

    common = fractions.index.intersection(agatston.index)
    tertile = pd.qcut(agatston.loc[common], q=3,
                      labels=["low", "mid", "high"], duplicates="drop")

    jt_rows = []
    for c in range(k_primary):
        col = f"frac_C{c}"
        if col not in fractions.columns:
            continue
        vals = fractions.loc[common, col]
        groups = [vals[tertile == t].to_numpy() for t in ("low", "mid", "high")]
        jt = jonckheere_terpstra(groups)
        jt_rows.append({
            "cluster": c,
            "broad_class": primary_to_broad.get(c, "unassigned"),
            "median_low": float(np.median(groups[0])),
            "median_mid": float(np.median(groups[1])),
            "median_high": float(np.median(groups[2])),
            "JT_Z": jt["Z"],
            "JT_p_two_sided": jt["p_two_sided"],
            "direction": jt["direction"],
            "monotone_trend_significant":
                bool(not np.isnan(jt["p_two_sided"]) and jt["p_two_sided"] < 0.001),
        })
    jt_df = pd.DataFrame(jt_rows).sort_values("JT_p_two_sided")
    jt_df.to_csv(out_dir / "jonckheere_terpstra_trends.csv", index=False)
    log.info("JT trend tests:\n%s", jt_df.to_string(index=False))

    # ── 4. Patient mixture GMM k=4 (Finding 5 candidate) ──────────
    log.info("--- 4. Patient mixture GMM k=%d ---", args.mixture_k)
    clr = clr_transform(fractions.loc[common].to_numpy())
    log.info("CLR mixture shape: %s", clr.shape)

    mix_labels = fit_cluster(clr, k=args.mixture_k, algorithm="gmm",
                             random_state=args.random_state)
    log.info("mixture cluster sizes: %s",
             dict(zip(*np.unique(mix_labels, return_counts=True))))

    # Hennig stability
    hen = hennig_clusterboot(
        clr, mix_labels, k=args.mixture_k, algorithm="gmm",
        n_bootstrap=args.hennig_bootstraps,
        threshold=0.75, random_state=args.random_state, n_jobs=args.n_jobs,
    )
    hen_records = []
    for cid, med, mn in zip(hen.cluster_ids, hen.jaccard_median, hen.jaccard_mean):
        hen_records.append({
            "mix_cluster": int(cid),
            "jaccard_median": float(med),
            "jaccard_mean": float(mn),
            "stable": bool(med >= 0.75),
        })
    _save_json(out_dir / "mixture_k4_hennig.json", hen_records)
    n_stable = sum(1 for r in hen_records if r["stable"])
    log.info("mixture k=%d Hennig: %d of %d clusters stable",
             args.mixture_k, n_stable, len(hen_records))

    # Per-mix-cluster profile (mean fractions over the k_primary clusters)
    mix_pids = list(common)
    mix_assign = pd.Series(mix_labels, index=mix_pids, name="mix_cluster")
    mix_profile = (
        fractions.loc[common].assign(mix_cluster=mix_assign)
        .groupby("mix_cluster").mean()
    )
    mix_profile["n_patients"] = mix_assign.value_counts().sort_index()
    mix_profile["agatston_median"] = (
        agatston.loc[common].groupby(mix_assign).median()
    )
    mix_profile["agatston_iqr_lo"] = (
        agatston.loc[common].groupby(mix_assign).quantile(0.25)
    )
    mix_profile["agatston_iqr_hi"] = (
        agatston.loc[common].groupby(mix_assign).quantile(0.75)
    )
    # Dominant lesion-cluster + dominant broad class per mix cluster
    dominant_primary = {}
    dominant_broad = {}
    for cid in sorted(mix_assign.unique()):
        pf = mix_profile.loc[cid, [f"frac_C{c}" for c in range(k_primary)]]
        dominant_primary[cid] = int(pf.idxmax().replace("frac_C", ""))
        # Average fraction within each broad class
        broad_frac = {}
        for c in range(k_primary):
            bc = primary_to_broad.get(c, "unassigned")
            broad_frac[bc] = broad_frac.get(bc, 0.0) + float(pf[f"frac_C{c}"])
        dominant_broad[cid] = max(broad_frac, key=broad_frac.get)
    mix_profile["dominant_primary_cluster"] = [
        dominant_primary[c] for c in mix_profile.index
    ]
    mix_profile["dominant_broad_class"] = [
        dominant_broad[c] for c in mix_profile.index
    ]
    mix_profile.to_csv(out_dir / "mixture_k4_profile.csv")

    # Crosstabs with stage-6 spatial labels + burden tertile + kernel
    fd_common = mix_assign.index.intersection(focal_diffuse.index)
    fd_ct = pd.crosstab(focal_diffuse.loc[fd_common],
                        mix_assign.loc[fd_common])
    fd_ct.index.name = "stage6_focal_diffuse"
    burden_ct = pd.crosstab(tertile.loc[common.intersection(tertile.index)],
                            mix_assign.loc[common.intersection(tertile.index)])
    burden_ct.index.name = "burden_tertile"
    k_common = mix_assign.index.intersection(kernel.index)
    kernel_ct = pd.crosstab(kernel.loc[k_common], mix_assign.loc[k_common])
    kernel_ct.index.name = "kernel"
    with open(out_dir / "mixture_k4_crosstabs.csv", "w", encoding="utf-8") as f:
        for name, df in (("focal_diffuse_vs_mix_cluster", fd_ct),
                         ("burden_tertile_vs_mix_cluster", burden_ct),
                         ("kernel_vs_mix_cluster", kernel_ct)):
            f.write(f"# {name}\n")
            df.to_csv(f)
            f.write("\n")

    # Save the patient mixture-cluster assignment for downstream
    mix_assign_df = mix_assign.reset_index()
    mix_assign_df.columns = ["pid", "mixture_cluster"]
    mix_assign_df.to_csv(out_dir / "patient_mixture_cluster_assignment.csv",
                        index=False)
    log.info("mixture cluster profile (rows=mix_cluster, cols=primary cluster "
             "fractions + dominant cluster + agatston median):\n%s",
             mix_profile[
                 [f"frac_C{c}" for c in range(k_primary)]
                 + ["n_patients", "agatston_median",
                    "dominant_primary_cluster", "dominant_broad_class"]
             ].to_string())

    # ── 5. Secondary anatomical biases (just report) ──────────────
    # Every vessel-biased cluster from analyse.py's cluster_vessel_chi_square
    # (no hardcoded cluster ids), annotated with its broad class.
    biased = vessel_chi[vessel_chi["vessel_biased"]].assign(
        broad_class=lambda d: d["cluster"].map(primary_to_broad),
        is_c8_like=lambda d: d["cluster"] == c8_id,
    )
    biased.to_csv(out_dir / "vessel_biased_clusters_full_cohort.csv", index=False)
    log.info("vessel-biased clusters (full cohort): %d total", len(biased))

    # ── 6. Why GMM separates by density (rho cluster_id vs feature) ──
    log.info("--- 6. GMM-k=3 separates by density not size: check ---")
    # Compute GMM k=3 labels (deterministic)
    X = morph_df[[c for c in morph_df.columns
                  if c not in ("pid", "vessel", "lesion_idx")]].to_numpy(float)
    gmm3 = fit_cluster(X, k=3, algorithm="gmm",
                       random_state=args.random_state)
    # Spearman rho between gmm3 cluster id and raw feature (orderable)
    # but cluster id is nominal, so use a different probe:
    # for each (raw feature, cluster) pair, compute per-cluster median
    # then Spearman rho across the 3 cluster medians.
    rho_records = []
    for feat in ("volume_mm3", "total_area_mm2", "mean_hu_weighted", "max_hu",
                 "n_rois"):
        if feat not in lesions.columns:
            continue
        cluster_medians = (
            lesions.assign(gmm3=gmm3)
            .groupby("gmm3")[feat].median()
            .reset_index()
            .sort_values("gmm3")
        )
        rho_records.append({
            "feature": feat,
            "cluster_0_median": float(cluster_medians.iloc[0][feat]),
            "cluster_1_median": float(cluster_medians.iloc[1][feat]),
            "cluster_2_median": float(cluster_medians.iloc[2][feat]),
            "range_log10":
                float(np.log10(
                    cluster_medians[feat].max() / max(cluster_medians[feat].min(), 1e-9)
                )),
        })
    # Component names by the same signature rule as the primary clusters
    # (GMM component order is arbitrary, so names are never hardcoded).
    gmm3_profile = (
        lesions.assign(gmm3=gmm3).groupby("gmm3")[["max_hu", "volume_mm3"]]
        .median().reset_index()
        .rename(columns={"gmm3": "cluster", "max_hu": "max_hu_median",
                         "volume_mm3": "volume_mm3_median"})
    )
    gmm3_names = discover_broad_classes(gmm3_profile)
    for rec in rho_records:
        for i in range(3):
            rec[f"cluster_{i}_broad_class"] = gmm3_names.get(i, "unassigned")
    pd.DataFrame(rho_records).to_csv(
        out_dir / "gmm3_density_vs_size_separation.csv", index=False,
    )
    log.info("GMM k=3 component names by signature: %s", gmm3_names)
    log.info("GMM-k=3 per-cluster medians (size vs density):\n%s",
             pd.DataFrame(rho_records).to_string(index=False))

    # ── 8. Per-stratum replication (the big test) ─────────────────
    log.info("--- 8. Per-stratum replication of the 3-class taxonomy + C8 ---")
    stratum_results = {}
    for stratum_name, stratum_filter in (
        ("Qr36d_2", lambda k: k == "Qr36d/2"),
        ("I30f_3",  lambda k: k == "I30f/3"),
    ):
        pids_in_stratum = meta.loc[
            meta["kernel"].apply(stratum_filter), "pid"
        ].tolist()
        lesions_in = lesions[lesions["pid"].isin(pids_in_stratum)].copy()
        morph_in = morph_df[morph_df["pid"].isin(pids_in_stratum)].copy()
        log.info("[%s] n_lesions=%d, n_patients=%d",
                 stratum_name, len(lesions_in), len(set(lesions_in["pid"])))
        if len(lesions_in) < 100:
            log.warning("[%s] too few lesions; skipping", stratum_name)
            continue

        # Refit at k_primary within the stratum, on the same morphology features
        Xs = morph_in[[c for c in morph_in.columns
                       if c not in ("pid", "vessel", "lesion_idx")]
                      ].to_numpy(float)
        labels_s = fit_cluster(Xs, k=k_primary, algorithm="kmeans",
                               random_state=args.random_state)
        morph_in = morph_in.assign(cluster=labels_s)
        lesions_in = lesions_in.merge(
            morph_in[["pid", "vessel", "lesion_idx", "cluster"]],
            on=["pid", "vessel", "lesion_idx"], how="left",
        )

        # Vessel chi-square per cluster within stratum
        vc = vessel_chi_square_per_cluster(
            lesions_in, lesions_in["cluster"].to_numpy(),
            cohort_label=stratum_name,
        )
        vc_csv_path = out_dir / f"stratum_{stratum_name}_vessel_chi_square.csv"
        vc.to_csv(vc_csv_path, index=False)

        # C8-like cluster within the stratum by the same locked signature
        # rule as the full cohort (predict.figures._lesion_classes).
        per_cluster_med = (
            lesions_in.groupby("cluster")[["volume_mm3", "max_hu", "n_rois"]]
            .median().reset_index()
        )
        prof_s = per_cluster_med.rename(columns={
            "volume_mm3": "volume_mm3_median", "max_hu": "max_hu_median",
            "n_rois": "n_rois_median"})
        c8_s = discover_c8_like_cluster(prof_s, vc)
        vc_with_med = vc.merge(per_cluster_med, on="cluster", how="left")
        vc_with_med["c8_signature_match"] = vc_with_med["cluster"] == c8_s
        vc_with_med.to_csv(out_dir / f"stratum_{stratum_name}_c8_like.csv",
                           index=False)
        top = (vc_with_med[vc_with_med["cluster"] == c8_s].iloc[0]
               if c8_s is not None else None)
        stratum_results[stratum_name] = {
            "c8_like_found": top is not None,
            "n_lesions": int(len(lesions_in)),
            "n_patients": int(lesions_in["pid"].nunique()),
            "n_clusters_vessel_biased": int(vc["vessel_biased"].sum()),
            "c8_like_cluster_id":
                int(top["cluster"]) if top is not None else None,
            "c8_like_n_lesions":
                int(top["n_lesions"]) if top is not None else None,
            "c8_like_rca_obs_over_exp":
                float(top["rca_obs_over_exp"]) if top is not None else None,
            "c8_like_lm_obs_over_exp":
                float(top["lm_obs_over_exp"]) if top is not None else None,
            "c8_like_cramers_v":
                float(top["cramers_v"]) if top is not None else None,
            "c8_like_volume_median":
                float(top["volume_mm3"]) if top is not None else None,
            "c8_like_max_hu_median":
                float(top["max_hu"]) if top is not None else None,
            "c8_like_n_rois_median":
                float(top["n_rois"]) if top is not None else None,
            "c8_like_chi2_p":
                float(top["chi2_p"]) if top is not None else None,
        }
        log.info("[%s] top C8-like cluster: id=%s, n=%s, RCA obs/exp=%.2f, "
                 "LM obs/exp=%.2f, V=%.3f, vol=%.1f, max_hu=%.1f",
                 stratum_name,
                 stratum_results[stratum_name]["c8_like_cluster_id"],
                 stratum_results[stratum_name]["c8_like_n_lesions"],
                 stratum_results[stratum_name]["c8_like_rca_obs_over_exp"] or 0,
                 stratum_results[stratum_name]["c8_like_lm_obs_over_exp"] or 0,
                 stratum_results[stratum_name]["c8_like_cramers_v"] or 0,
                 stratum_results[stratum_name]["c8_like_volume_median"] or 0,
                 stratum_results[stratum_name]["c8_like_max_hu_median"] or 0)
    _save_json(out_dir / "per_stratum_c8_replication.json", stratum_results)

    # ── Run header ────────────────────────────────────────────────
    header = {
        "experiment": "lesion_morphology_finalise",
        "scope": "exploratory; not part of production pipeline",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_hash(repo_root),
        "args": vars(args),
        "k_primary": k_primary,
        "n_lesions": int(len(lesions)),
        "input_sha": {
            "lesion_cluster_labels": _file_sha(labels_csv),
            "lesions_csv": _file_sha(lesions_csv),
            "cohort_metadata_csv": _file_sha(cohort_meta_csv),
            "spatial_labels_csv": _file_sha(spatial_csv),
            "features_csv": _file_sha(features_csv),
        },
        "c8_deep_dive": c8_deep,
        "mixture_k4_hennig_n_stable": n_stable,
        "mixture_k4_hennig_n_total": len(hen_records),
        "per_stratum_c8_replication": stratum_results,
        "c8_like_cluster_id": int(c8_id),
        "primary_to_broad": primary_to_broad,
        "gmm3_broad_class_names": gmm3_names,
    }
    for mod in ("numpy", "pandas", "scipy", "sklearn"):
        try:
            m = __import__(mod)
            header[f"{mod}_version"] = getattr(m, "__version__", "unknown")
        except Exception:
            header[f"{mod}_version"] = "n/a"
    _save_json(out_dir / "run_header.json", header)

    print()
    print("=" * 80)
    print(f"finalise.py complete. outputs in {out_dir}")
    print(f"  C8 patient burden median = {c8_deep['c8_patient_burden_median']:.1f} "
          f"(vs non-C8 = {c8_deep['non_c8_patient_burden_median']:.1f}); "
          f"MW p = {c8_deep['mannwhitney_p_one_sided_c8_higher']:.3g}")
    print(f"  mixture k={args.mixture_k} Hennig: {n_stable} of {len(hen_records)} stable")
    for s, r in stratum_results.items():
        print(f"  [{s}] C8-like cluster: id={r.get('c8_like_cluster_id')}, "
              f"RCA obs/exp={r.get('c8_like_rca_obs_over_exp')}, "
              f"LM obs/exp={r.get('c8_like_lm_obs_over_exp')}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
