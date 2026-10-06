#!/usr/bin/env python
"""Exploratory: lesion-level morphology clustering.

Self-contained experiment. NOT part of the production pipeline. Does not
produce any seam file consumed by stages 1-8. Reuses the production-tested
helpers from predict.discover for methodological consistency with the
patient-level Hopkins + gap + Hennig analysis.

Pipeline (single-script, ~12 sections):

  1. Load lesions.csv (3179 rows) + patient-level seam files
  2. Build the 6-feature morphology matrix (log volume, log area,
     mean_hu_weighted, max_hu, log n_rois, log(volume/area)) + z-score
  3. Hopkins clusterability on the morphology subspace
  4. Gap statistic (k=1..8, kmeans + ward + gmm, 200 bootstraps)
  5. Fit clusters at gap-selected k (or --k override)
  6. Profile each cluster (raw medians, n, vessel composition)
  7. Hennig clusterboot stability (100 bootstraps unless --no-hennig)
  8. Patient-level aggregation: per-patient cluster fractions
  9. Cross-tab: patient-level focal/diffuse (stage 6) vs dominant
     lesion cluster
 10. Cross-tab: patient-level burden tertile vs dominant lesion cluster
 11. Cross-tab: kernel vs lesion cluster (does any cluster track scanner?)
 12. Write outputs + human-readable report.txt

Outputs land in outputs/exploratory/lesion_morphology/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from predict.config import default_n_jobs, load_config
from predict.stage9 import stage9_root
from predict.discover.cluster_discovery import (
    fit_cluster,
    gap_statistic,
)
from predict.discover.clusterability import assess_clusterability
from predict.discover.validity import hennig_clusterboot


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


def _save_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ─────────────────────── feature engineering ───────────────────────


MORPHOLOGY_FEATURES_RAW = (
    "volume_mm3", "total_area_mm2", "mean_hu_weighted", "max_hu", "n_rois",
)


def build_morphology_matrix(lesions: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return a (lesions x 4 features) z-scored morphology matrix (D037).

    Features:
      log1p_volume, mean_hu_weighted, max_hu, log1p_n_rois

    D037 removed two features that are deterministic functions of the others:
    lesion volume is total_area x slice thickness (lesion_ccl.py), so
    log1p_area duplicated log1p_volume, and the volume / area ratio was the
    constant slice thickness (3.0 mm for all 3201 lesions), which z-scoring
    turned into floating-point noise.

    Centroid coordinates and vessel id are EXCLUDED (they are anatomical,
    not morphological). They are retained on the input frame for post-hoc
    stratification.
    """
    df = lesions.copy()
    df["log1p_volume"] = np.log1p(df["volume_mm3"])
    df["log1p_n_rois"] = np.log1p(df["n_rois"])
    feature_cols = ["log1p_volume", "mean_hu_weighted", "max_hu", "log1p_n_rois"]
    # z-score each column (independent of other lesions in the same patient).
    # A near-constant column is an error, never silently scaled: dividing
    # rounding noise by a tiny SD is how the ratio feature entered the
    # clustering before D037.
    z = df[feature_cols].copy()
    for c in feature_cols:
        mu = z[c].mean()
        sigma = z[c].std(ddof=1)
        if not sigma > 1e-9 * max(1.0, abs(mu)):
            raise ValueError(f"morphology feature {c!r} is (near-)constant: sd {sigma:.3g}")
        z[c] = (z[c] - mu) / sigma
    return z, feature_cols


# ─────────────────────── patient-level seam ───────────────────────


def _load_patient_seam(
    config_outputs: Path,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return (spatial_labels, agatston_total, kernel, low_burden_flag).

    spatial_labels: stage 6 full-cohort focal/diffuse mapping if available.
    Falls back to raw GMM k=2 cluster ids if cluster_labels_spatial_k2.csv
    is the only thing present.

    Returns Series indexed by pid (str).
    """
    reduce_root = config_outputs / "06_reduce"
    spatial_csv = reduce_root / "cluster_labels_spatial_k2.csv"
    meta_csv = reduce_root / "cohort_metadata.csv"
    features_csv = config_outputs / "03_features" / "features.csv"

    spatial = pd.read_csv(spatial_csv, dtype={"pid": str})
    spatial_labels = spatial.set_index("pid")["spatial_only_gmm_k2"].astype(int)

    meta = pd.read_csv(meta_csv, dtype={"pid": str})
    agatston = meta.set_index("pid")["agatston_total"].astype(float)
    kernel = meta.set_index("pid")["kernel"].astype(str)
    low_burden = meta.set_index("pid")["low_burden_flag"]

    # Compute focal/diffuse using the same rule as profiles.py: lower
    # median n_calcified_arteries = focal.
    feats = pd.read_csv(features_csv, dtype={"pid": str}).set_index("pid")
    if "n_calcified_arteries" in feats.columns and len(spatial_labels.unique()) == 2:
        cluster_medians = {}
        for cid in sorted(spatial_labels.unique()):
            pids_in = spatial_labels[spatial_labels == cid].index
            vals = feats.loc[feats.index.intersection(pids_in),
                             "n_calcified_arteries"].dropna()
            cluster_medians[cid] = float(np.median(vals))
        focal_cid = min(cluster_medians, key=cluster_medians.get)
        focal_diffuse = spatial_labels.map(
            {focal_cid: "focal",
             [c for c in spatial_labels.unique() if c != focal_cid][0]: "diffuse"}
        )
    else:
        focal_diffuse = spatial_labels.astype(str)

    return focal_diffuse, agatston, kernel, low_burden


# ─────────────────────── main ───────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--k-range", type=int, nargs="+",
                        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                        help="1..12, the same range as the patient-level analysis")
    parser.add_argument("--gap-bootstraps", type=int, default=200,
                        help="lower than the production 500 since lesion N=3179 "
                             "and we are in an exploratory mode")
    parser.add_argument("--k", type=int, default=None,
                        help="force k for the cluster fit and Hennig; "
                             "default uses the gap-selected k")
    parser.add_argument("--hennig-bootstraps", type=int, default=100)
    parser.add_argument("--no-hennig", action="store_true",
                        help="skip Hennig bootstrap stability (faster smoke)")
    parser.add_argument("--n-jobs", type=int, default=default_n_jobs())
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("lesion_morph")

    cfg = load_config(args.config)
    out_dir = _ensure_dir(stage9_root(cfg) / "lesion_morphology")
    repo_root = cfg.paths.outputs.parent

    # Run header
    header = {
        "experiment": "lesion_morphology",
        "scope": "exploratory; not part of production pipeline",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_hash(repo_root),
        "python_version": sys.version.split()[0],
        "args": vars(args),
        "input_sha": {
            "lesions_csv": _file_sha(cfg.paths.outputs / "03_features" / "lesions.csv"),
            "features_csv": _file_sha(cfg.paths.outputs / "03_features" / "features.csv"),
            "spatial_labels_csv": _file_sha(
                cfg.paths.outputs / "06_reduce" / "cluster_labels_spatial_k2.csv"
            ),
            "cohort_metadata_csv": _file_sha(
                cfg.paths.outputs / "06_reduce" / "cohort_metadata.csv"
            ),
        },
    }
    for mod in ("numpy", "pandas", "scipy", "sklearn"):
        try:
            m = __import__(mod)
            header[f"{mod}_version"] = getattr(m, "__version__", "unknown")
        except Exception:
            header[f"{mod}_version"] = "n/a"

    # ── 1. Load lesions + patient seam ───────────────────────────
    lesions_csv = cfg.paths.outputs / "03_features" / "lesions.csv"
    if not lesions_csv.exists():
        log.error("missing %s", lesions_csv)
        return 2
    lesions = pd.read_csv(lesions_csv, dtype={"pid": str})
    # Drop any rows with missing morphology fields (rare)
    for c in MORPHOLOGY_FEATURES_RAW:
        if c not in lesions.columns:
            log.error("lesions.csv missing column %s", c)
            return 2
    before = len(lesions)
    lesions = lesions.dropna(subset=list(MORPHOLOGY_FEATURES_RAW)).reset_index(drop=True)
    log.info("loaded %d lesions (%d after dropping NaN morphology rows)",
             before, len(lesions))

    spatial_labels, agatston, kernel, low_burden = _load_patient_seam(cfg.paths.outputs)
    log.info("loaded patient seam: %d spatial labels, %d agatston records",
             len(spatial_labels), len(agatston))

    # ── 2. Morphology matrix ─────────────────────────────────────
    z_df, feature_cols = build_morphology_matrix(lesions)
    log.info("morphology matrix shape: %s (features: %s)",
             z_df.shape, feature_cols)
    pd.concat([lesions[["pid", "vessel", "lesion_idx"]], z_df], axis=1).to_csv(
        out_dir / "lesion_features.csv", index=False,
    )

    X = z_df[feature_cols].to_numpy(dtype=float)

    # ── 3. Hopkins ───────────────────────────────────────────────
    log.info("Hopkins clusterability...")
    hopkins = assess_clusterability(
        X, sample_frac=cfg.raw["hopkins"]["sample_frac"],
        threshold=cfg.raw["hopkins"]["cluster_tendency_threshold"],
        ambiguous_band=tuple(cfg.raw["hopkins"]["ambiguous_band"]),
        random_state=args.random_state,
    )
    _save_json(out_dir / "hopkins.json", hopkins.to_dict())
    log.info("Hopkins H=%.3f (%s)", hopkins.H, hopkins.verdict)

    # ── 4. Gap statistic ─────────────────────────────────────────
    log.info("gap statistic across kmeans/ward/gmm...")
    k_range = tuple(int(k) for k in args.k_range)
    gap_records: list[dict] = []
    for algo in ("kmeans", "ward", "gmm"):
        t0 = time.perf_counter()
        result = gap_statistic(
            X, algorithm=algo, k_range=k_range,
            n_bootstrap=args.gap_bootstraps,
            random_state=args.random_state, n_jobs=args.n_jobs,
        )
        log.info("gap[%s] selected k=%d (%.1fs)",
                 algo, result.selected_k, time.perf_counter() - t0)
        gap_records.append({
            "algorithm": algo,
            "k_range": list(result.k_range),
            "gap_values": result.gap_values.tolist(),
            "sk_values": result.sk_values.tolist(),
            "selected_k": int(result.selected_k),
            "n_bootstrap": int(result.n_bootstrap),
        })
    _save_json(out_dir / "gap_statistic.json", gap_records)

    # Selected k for the rest of the analysis
    if args.k is not None:
        k_final = int(args.k)
        log.info("using user-provided k=%d", k_final)
    else:
        # Use the modal (most common) selected k across algorithms;
        # tiebreak: smallest k.
        votes = {}
        for r in gap_records:
            votes[r["selected_k"]] = votes.get(r["selected_k"], 0) + 1
        k_final = min(
            (k for k, v in votes.items() if v == max(votes.values()))
        )
        log.info("using gap-modal k=%d (votes: %s)", k_final, votes)

    # ── 5. Cluster fit at k_final ────────────────────────────────
    log.info("fitting clusters at k=%d on three algorithms...", k_final)
    labels_per_algo: dict[str, np.ndarray] = {}
    for algo in ("kmeans", "ward", "gmm"):
        labels_per_algo[algo] = fit_cluster(
            X, k=k_final, algorithm=algo, random_state=args.random_state,
        )

    # Write the per-algorithm labels alongside the lesion identifier.
    labels_df = lesions[["pid", "vessel", "lesion_idx"]].copy()
    for algo, lab in labels_per_algo.items():
        labels_df[f"cluster_{algo}_k{k_final}"] = lab
    labels_df.to_csv(out_dir / "lesion_cluster_labels.csv", index=False)

    # Primary algorithm for downstream cross-tabs: kmeans (most familiar).
    primary_labels = labels_per_algo["kmeans"]
    lesions = lesions.assign(lesion_cluster=primary_labels)

    # ── 6. Cluster profiles ──────────────────────────────────────
    log.info("computing per-cluster profiles...")
    profile_rows: list[dict] = []
    for cid in sorted(np.unique(primary_labels)):
        rows_in = lesions[primary_labels == cid]
        n = int(len(rows_in))
        row = {"cluster": int(cid), "n_lesions": n,
               "fraction_of_total": round(n / len(lesions), 4)}
        for c in MORPHOLOGY_FEATURES_RAW:
            vals = rows_in[c].dropna()
            row[f"{c}_median"] = float(np.median(vals)) if len(vals) else float("nan")
            row[f"{c}_iqr_lo"] = float(np.percentile(vals, 25)) if len(vals) else float("nan")
            row[f"{c}_iqr_hi"] = float(np.percentile(vals, 75)) if len(vals) else float("nan")
        profile_rows.append(row)
    profile_df = pd.DataFrame(profile_rows)
    profile_df.to_csv(out_dir / "cluster_profiles.csv", index=False)

    # Per-cluster vessel composition (post-hoc anatomical stratification)
    vessel_ct = pd.crosstab(primary_labels, lesions["vessel"])
    vessel_ct.index.name = "cluster"
    vessel_ct.to_csv(out_dir / "cluster_vessel_distribution.csv")

    # ── 7. Hennig stability ──────────────────────────────────────
    if not args.no_hennig:
        log.info("Hennig clusterboot stability (%d bootstraps, k=%d)...",
                 args.hennig_bootstraps, k_final)
        hen = hennig_clusterboot(
            X, primary_labels, k=k_final, algorithm="kmeans",
            n_bootstrap=args.hennig_bootstraps,
            threshold=0.75, random_state=args.random_state,
            n_jobs=args.n_jobs,
        )
        hen_records = []
        for cid, med, mean_ in zip(hen.cluster_ids, hen.jaccard_median, hen.jaccard_mean):
            hen_records.append({
                "cluster": int(cid),
                "jaccard_median": float(med),
                "jaccard_mean": float(mean_),
                "stable": bool(med >= 0.75),
            })
        _save_json(out_dir / "hennig_stability.json", hen_records)
        for r in hen_records:
            log.info("Hennig cluster %d: median Jaccard=%.3f (%s)",
                     r["cluster"], r["jaccard_median"],
                     "STABLE" if r["stable"] else "UNSTABLE")
    else:
        hen_records = []
        log.info("--no-hennig set; skipping stability")

    # ── 8. Patient-level aggregation ─────────────────────────────
    log.info("aggregating to per-patient cluster fractions...")
    # Per-patient counts of lesions per cluster
    patient_counts = (
        lesions.groupby(["pid", "lesion_cluster"]).size().unstack(fill_value=0)
    )
    # Per-patient fractions
    patient_fractions = patient_counts.div(patient_counts.sum(axis=1), axis=0)
    patient_fractions.columns = [f"frac_cluster_{c}" for c in patient_fractions.columns]
    # Dominant cluster per patient (argmax of counts; tie -> lowest cid)
    dominant = patient_counts.idxmax(axis=1).rename("dominant_lesion_cluster")
    n_lesions_per_patient = patient_counts.sum(axis=1).rename("n_lesions")

    patient_signature = pd.concat(
        [n_lesions_per_patient, dominant, patient_fractions], axis=1
    ).reset_index()
    patient_signature.to_csv(out_dir / "patient_lesion_signatures.csv", index=False)

    # ── 9-11. Cross-tabs with patient-level seam ─────────────────
    log.info("cross-tabulating patient-level signatures against stages 6 + 7 seams...")
    ps = patient_signature.set_index("pid")
    crosstabs: dict[str, pd.DataFrame] = {}

    # focal / diffuse (stage 6 mapping)
    common = ps.index.intersection(spatial_labels.index)
    if len(common) > 0:
        ct = pd.crosstab(
            spatial_labels.loc[common],
            ps.loc[common, "dominant_lesion_cluster"],
        )
        ct.index.name = "patient_phenotype"
        crosstabs["focal_diffuse_x_dominant_lesion_cluster"] = ct

    # Agatston tertile
    common_a = ps.index.intersection(agatston.index)
    if len(common_a) > 0:
        tertile = pd.qcut(agatston.loc[common_a], q=3,
                          labels=["low", "mid", "high"], duplicates="drop")
        ct = pd.crosstab(tertile, ps.loc[common_a, "dominant_lesion_cluster"])
        ct.index.name = "burden_tertile"
        crosstabs["burden_tertile_x_dominant_lesion_cluster"] = ct

    # Kernel (does any cluster track scanner?)
    common_k = ps.index.intersection(kernel.index)
    if len(common_k) > 0:
        ct = pd.crosstab(kernel.loc[common_k], ps.loc[common_k, "dominant_lesion_cluster"])
        ct.index.name = "kernel"
        crosstabs["kernel_x_dominant_lesion_cluster"] = ct

    # Write all crosstabs into one CSV for easy reading
    if crosstabs:
        with open(out_dir / "patient_phenotype_crosstab.csv", "w", encoding="utf-8") as f:
            for name, df in crosstabs.items():
                f.write(f"# {name}\n")
                df.to_csv(f)
                f.write("\n")
        for name, ct in crosstabs.items():
            log.info("crosstab[%s]:\n%s", name, ct.to_string())

    # ── 12. Write report.txt ─────────────────────────────────────
    report_lines: list[str] = []

    def w(s: str = "") -> None:
        report_lines.append(s)

    w("=" * 96)
    w("Lesion-level morphology clustering: exploratory report")
    w("=" * 96)
    w()
    w(f"n_lesions             : {len(lesions)}")
    w(f"n_patients (with lesions): {lesions['pid'].nunique()}")
    w(f"morphology features    : {feature_cols}")
    w()
    w(f"Hopkins H = {hopkins.H:.3f}  ({hopkins.verdict})")
    w()
    w("Gap statistic selected k:")
    for r in gap_records:
        gap_at = float(r['gap_values'][r['k_range'].index(r['selected_k'])])
        sk_at = float(r['sk_values'][r['k_range'].index(r['selected_k'])])
        w(f"  {r['algorithm']:<6} k={r['selected_k']}  (gap={gap_at:.3f}, sk={sk_at:.3f})")
    w()
    w(f"Chosen k for downstream = {k_final}")
    w()
    w("Cluster profiles (raw medians):")
    cols = ["cluster", "n_lesions", "fraction_of_total"]
    cols += [f"{c}_median" for c in MORPHOLOGY_FEATURES_RAW]
    w(profile_df[cols].to_string(index=False))
    w()
    w("Per-cluster vessel composition:")
    w(vessel_ct.to_string())
    w()
    if hen_records:
        w("Hennig stability (kmeans):")
        for r in hen_records:
            mark = "STABLE" if r["stable"] else "UNSTABLE"
            w(f"  cluster {r['cluster']}: median Jaccard = {r['jaccard_median']:.3f}  ({mark})")
        w()
    w("Patient-level cluster signatures (head):")
    w(patient_signature.head(10).to_string(index=False))
    w()
    for name, ct in crosstabs.items():
        w(f"Crosstab: {name}")
        w(ct.to_string())
        w()

    w("Headline interpretation:")
    w("  Compare Hopkins, gap, Hennig against the patient-level values")
    w("  from stages 6 and 7 (Hopkins H=0.717, gap monotonic to k=12,")
    w("  Hennig 0.85-0.92 on the spatial k=2 partition):")
    w("    - If lesion Hopkins ~ 0.5 and gap monotonic to k_max:")
    w("        lesion morphology IS a continuum too (consistent with")
    w("        patient-level Finding 1 at a finer resolution).")
    w("    - If lesion Hopkins > 0.65 and gap shows a clean elbow at")
    w("        some k* with all-stable Hennig medians >= 0.75:")
    w("        discrete morphology phenotypes EXIST. Then the patient-")
    w("        level focal/diffuse split may correspond to characteristic")
    w("        mixtures of lesion types (check the focal_diffuse x dominant_")
    w("        lesion_cluster crosstab).")
    w()
    (out_dir / "report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # Run header (finalised)
    header["lesion_count"] = int(len(lesions))
    header["patient_count_with_lesions"] = int(lesions["pid"].nunique())
    header["hopkins_H"] = float(hopkins.H)
    header["hopkins_verdict"] = hopkins.verdict
    header["gap_selected_k_per_algorithm"] = {
        r["algorithm"]: r["selected_k"] for r in gap_records
    }
    header["k_final"] = int(k_final)
    if hen_records:
        header["hennig_stable_clusters"] = sum(1 for r in hen_records if r["stable"])
        header["hennig_total_clusters"] = len(hen_records)
    _save_json(out_dir / "run_header.json", header)

    log.info("experiment complete. outputs in %s", out_dir)
    print("\n" + "=" * 80)
    print(f"Quick read:   Hopkins H={hopkins.H:.3f}   k_final={k_final}")
    print(f"              outputs at {out_dir}")
    print(f"              see report.txt for the full human-readable summary")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
