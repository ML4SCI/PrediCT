#!/usr/bin/env python
"""Systematic analysis of the LM-isolated low-burden displaced subgroup.

Reads stage-3 features, stage-5 PCA seam, and the lesion-experiment
cluster labels. Identifies the displaced subgroup per plan.md P1,
tests the LM-enrichment hypothesis per P2, replicates across kernel
strata per P3, characterises lesion density per P4, and checks lesion-
cluster overlap with the LAD-experiment clusters per P5.

Writes outputs to ``outputs/exploratory/lm_isolated_low_burden/``.
No production files are modified.

Usage:
    python experiments/lm_isolated_low_burden/run.py
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
from statsmodels.stats.proportion import proportion_confint

from predict.config import load_config
from predict.stage9 import stage9_root


# ─── Pre-registered thresholds (locked in plan.md) ───

P1_PC1_THRESHOLD = 0.0
P1_PC2_THRESHOLD = 2.5

P2_FISHER_P_MAX = 0.001
P2_LM_RATE_MIN = 0.50

P3_STRATUM_LM_RATE_MIN = 0.50
P3_STRATUM_NONDISP_LM_RATE_MAX = 0.50

P5_OVERLAP_THRESHOLD = 0.20

# The LAD-dominant clusters come from the CURRENT run's LAD-phenotype
# signature (k-means labels are arbitrary across fits). There is no
# fallback to fixed cluster ids: a missing, corrupt or empty signature
# stops the analysis or makes P5 not evaluable (D043).


def _discover_lad_dominant_clusters(stage9_dir: Path) -> tuple[int, ...]:
    """LAD-dominant cluster ids from this run's LAD signature.

    A missing or malformed signature stops the run. An empty signature
    (no cluster matched the pre-registered criteria) returns ``()``, and
    P5 is then not evaluable.
    """
    sig_path = stage9_dir / "lad_phenotype" / "lad_cluster_signature.json"
    if not sig_path.exists():
        raise SystemExit(f"hard stop: {sig_path} missing; run the LAD phenotype experiment first")
    sig = json.loads(sig_path.read_text(encoding="utf-8"))
    if "lad_dominant_cluster_ids" not in sig or not isinstance(sig["lad_dominant_cluster_ids"], list):
        raise SystemExit(f"hard stop: {sig_path} has no lad_dominant_cluster_ids list")
    return tuple(int(i) for i in sig["lad_dominant_cluster_ids"])

# Agatston tier bin edges per IBSI calcium convention.
HU_TIER_EDGES = (130, 200, 300, 400, float("inf"))
HU_TIER_LABELS = ("W1", "W2", "W3", "W4")


_log = logging.getLogger("lm_isolated.run")


# ─── reproducibility breadcrumbs ───


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
    path.write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8",
    )


# ─── core analysis helpers ───


def classify_hu_tier(max_hu: float) -> str | None:
    """Map a max_hu value to its Agatston weight tier (W1-W4)."""
    if max_hu is None or np.isnan(max_hu):
        return None
    if max_hu < HU_TIER_EDGES[0]:
        return None  # below the IBSI calcium floor
    for i, edge in enumerate(HU_TIER_EDGES[1:]):
        if max_hu < edge:
            return HU_TIER_LABELS[i]
    return HU_TIER_LABELS[-1]


def identify_displaced(
    pca_scores: np.ndarray,
    pid_order: list,
    agatston: pd.Series,
    kernel: pd.Series,
) -> pd.DataFrame:
    """Return a DataFrame of every cohort patient with their PC scores,
    Agatston tertile, kernel, and a boolean 'displaced' flag per plan.md
    P1. The 'displaced' flag is True only for low-tertile patients that
    satisfy PC1 > 0 OR PC2 > 2.5.
    """
    df = pd.DataFrame({
        "pid": [str(p) for p in pid_order],
        "pc1": pca_scores[:, 0].astype(float),
        "pc2": pca_scores[:, 1].astype(float),
    })
    df["agatston_total"] = df["pid"].map(agatston).astype(float)
    df["kernel"] = df["pid"].map(kernel).astype(str)
    df["tertile"] = pd.qcut(
        df["agatston_total"], q=3,
        labels=["low", "mid", "high"], duplicates="drop",
    ).astype(str)
    df["displaced"] = (
        (df["tertile"] == "low")
        & ((df["pc1"] > P1_PC1_THRESHOLD) | (df["pc2"] > P1_PC2_THRESHOLD))
    )
    return df


def attach_lm_features(
    displaced_df: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join the displaced-flag dataframe with per-patient LM
    feature columns: agatston_lm, max_hu_lm, mean_hu_lm, lesion_count_lm,
    n_calcified_arteries, lesion_count_total.
    """
    cols_to_take = [
        "agatston_lm", "max_hu_lm", "mean_hu_lm",
        "lesion_count_lm", "n_calcified_arteries",
        "lesion_count_total", "lesion_count_lad",
        "lesion_count_rca", "lesion_count_lcx",
        "agatston_lad", "agatston_rca", "agatston_lcx",
        "gini_lesion_volume",
    ]
    feat = features.set_index(features["pid"].astype(str))
    out = displaced_df.copy()
    for col in cols_to_take:
        if col in feat.columns:
            out[col] = out["pid"].map(feat[col])
    out["has_lm"] = (out["lesion_count_lm"].fillna(0) > 0)
    out["is_multivessel"] = (out["n_calcified_arteries"].fillna(0) >= 3)
    return out


def fisher_displaced_vs_nondisplaced_lm(
    full_df: pd.DataFrame,
) -> dict:
    """Run the P2 Fisher exact test on the displaced-vs-non-displaced
    LM contingency within the low-burden tertile."""
    low = full_df[full_df["tertile"] == "low"]
    disp = low[low["displaced"]]
    nondisp = low[~low["displaced"]]
    a = int(disp["has_lm"].sum())
    b = int((~disp["has_lm"]).sum())
    c = int(nondisp["has_lm"].sum())
    d = int((~nondisp["has_lm"]).sum())
    table = [[a, b], [c, d]]
    if a + b == 0 or c + d == 0:
        return {
            "table": table, "n_displaced": a + b,
            "n_non_displaced": c + d,
            "displaced_lm_rate": float("nan"),
            "non_displaced_lm_rate": float("nan"),
            "fisher_odds_ratio": float("nan"),
            "fisher_p_one_sided_greater": float("nan"),
            "passes": False,
            "fail_reason": "empty contingency",
        }
    odds, p = stats.fisher_exact(table, alternative="greater")
    disp_rate = a / (a + b)
    nondisp_rate = c / (c + d)
    return {
        "table": table,
        "n_displaced": a + b,
        "n_non_displaced": c + d,
        "displaced_lm_rate": float(disp_rate),
        "non_displaced_lm_rate": float(nondisp_rate),
        "fisher_odds_ratio": float(odds),
        "fisher_p_one_sided_greater": float(p),
        "p_threshold": P2_FISHER_P_MAX,
        "lm_rate_threshold": P2_LM_RATE_MIN,
        "passes": bool(
            p < P2_FISHER_P_MAX and disp_rate >= P2_LM_RATE_MIN
        ),
    }


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided Wilson 95% CI on the binomial proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
    return (float(lo), float(hi))


def cross_stratum_replication(full_df: pd.DataFrame) -> dict:
    """Per-stratum displaced count + LM rate + Wilson CI."""
    out: dict = {"per_stratum": {}, "passes": True}
    for kern, grp in full_df.groupby("kernel"):
        low = grp[grp["tertile"] == "low"]
        disp = low[low["displaced"]]
        nondisp = low[~low["displaced"]]
        n_disp = int(len(disp))
        n_disp_lm = int(disp["has_lm"].sum())
        n_nondisp = int(len(nondisp))
        n_nondisp_lm = int(nondisp["has_lm"].sum())
        disp_rate = (n_disp_lm / n_disp) if n_disp else float("nan")
        nondisp_rate = (n_nondisp_lm / n_nondisp) if n_nondisp else float("nan")
        lo, hi = wilson_ci(n_disp_lm, n_disp) if n_disp else (
            float("nan"), float("nan")
        )
        stratum_passes = (
            n_disp > 0
            and disp_rate >= P3_STRATUM_LM_RATE_MIN
            and nondisp_rate < P3_STRATUM_NONDISP_LM_RATE_MAX
        )
        out["per_stratum"][str(kern)] = {
            "n_low": int(len(low)),
            "n_displaced": n_disp,
            "n_displaced_lm_pos": n_disp_lm,
            "displaced_lm_rate": disp_rate,
            "n_non_displaced": n_nondisp,
            "non_displaced_lm_rate": nondisp_rate,
            "displaced_wilson95_lo": lo,
            "displaced_wilson95_hi": hi,
            "passes": bool(stratum_passes),
        }
        if not stratum_passes:
            out["passes"] = False
    return out


def density_profile(displaced_with_features: pd.DataFrame) -> dict:
    """Return descriptive density statistics + tier breakdown."""
    disp = displaced_with_features[displaced_with_features["displaced"]].copy()
    if disp.empty:
        return {"n": 0, "passes": False, "fail_reason": "no displaced patients"}
    disp["max_hu_tier"] = disp["max_hu_lm"].apply(classify_hu_tier)
    hu = disp["max_hu_lm"].dropna()
    out = {
        "n": int(len(disp)),
        "median_max_hu_lm": float(hu.median()),
        "min_max_hu_lm": float(hu.min()),
        "max_max_hu_lm": float(hu.max()),
        "tier_breakdown": disp["max_hu_tier"].value_counts(dropna=False).to_dict(),
        "n_soft_w1_w2": int(disp["max_hu_tier"].isin(["W1", "W2"]).sum()),
        "n_dense_w3_w4": int(disp["max_hu_tier"].isin(["W3", "W4"]).sum()),
    }
    out["framing"] = (
        "advanced isolated LM disease (W3/W4 dominant)"
        if out["n_dense_w3_w4"] > out["n_soft_w1_w2"]
        else "early-stage LM disease (W1/W2 dominant)"
    )
    return out


def cluster_overlap_with_lad(
    displaced_pids: list,
    lesion_labels: pd.DataFrame,
    lad_dominant_clusters: tuple[int, ...],
) -> dict:
    """For every LM lesion belonging to a displaced patient, look up its
    cluster id. Report breakdown and the LAD-cluster overlap fraction
    per plan.md P5.

    ``lad_dominant_clusters`` comes from the CURRENT run's LAD signature
    (cluster labels are arbitrary across k-means reruns). If it is empty
    (no LAD-dominant cluster this run), P5 is not evaluable and
    ``passes_distinctness`` is None.
    """
    if len(lad_dominant_clusters) == 0:
        return {
            "evaluable": False,
            "reason": "no LAD-dominant cluster in this run's LAD signature",
            "lad_dominant_clusters_used": [],
            "passes_distinctness": None,
        }
    primary_col = next(
        (c for c in lesion_labels.columns
         if c.startswith("cluster_kmeans_k")),
        None,
    )
    if primary_col is None:
        return {
            "passes": False, "fail_reason": "no cluster_kmeans_k column",
        }
    sub = lesion_labels[
        lesion_labels["pid"].astype(str).isin([str(p) for p in displaced_pids])
        & (lesion_labels["vessel"] == "LM")
    ].copy()
    sub[primary_col] = sub[primary_col].astype("Int64")
    sub = sub.dropna(subset=[primary_col])
    if sub.empty:
        return {
            "passes": False,
            "fail_reason": "no LM lesions found for displaced patients",
            "lesion_count": 0,
        }
    cluster_breakdown = sub[primary_col].astype(int).value_counts().to_dict()
    in_lad_clusters = sub[primary_col].astype(int).isin(lad_dominant_clusters)
    n_in_lad = int(in_lad_clusters.sum())
    n_total = int(len(sub))
    overlap_frac = n_in_lad / n_total if n_total else 0.0
    out = {
        "evaluable": True,
        "lesion_count": n_total,
        "cluster_breakdown": {int(k): int(v) for k, v in cluster_breakdown.items()},
        "lad_dominant_clusters_used": list(lad_dominant_clusters),
        "n_in_lad_dominant_clusters": n_in_lad,
        # legacy key for backwards compat with finalise.py + older reports
        "n_in_lad_clusters_10_11": n_in_lad,
        "overlap_fraction": float(overlap_frac),
        "overlap_threshold": P5_OVERLAP_THRESHOLD,
        "framing": (
            "low-burden expression of LAD/LM phenotype"
            if overlap_frac >= P5_OVERLAP_THRESHOLD
            else "biologically distinct from LAD/LM phenotype"
        ),
        "passes_distinctness": bool(overlap_frac < P5_OVERLAP_THRESHOLD),
        "per_lesion": sub[["pid", "vessel", "lesion_idx", primary_col]]
                      .rename(columns={primary_col: "cluster"})
                      .to_dict(orient="records"),
    }
    return out


# ─── orchestration ───


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    cfg = load_config(args.config)
    out_dir = (stage9_root(cfg) / "lm_isolated_low_burden")
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_root = cfg.paths.outputs.parent

    # Inputs
    pca_npy = cfg.paths.outputs / "06_reduce" / "pca_scores.npy"
    pca_pid_csv = cfg.paths.outputs / "06_reduce" / "pca_scores_pid_order.csv"
    meta_csv = cfg.paths.outputs / "06_reduce" / "cohort_metadata.csv"
    features_csv = cfg.paths.outputs / "03_features" / "features.csv"
    lesion_labels_csv = (stage9_root(cfg) / "lesion_morphology" / "lesion_cluster_labels.csv")
    for p in (pca_npy, pca_pid_csv, meta_csv, features_csv, lesion_labels_csv):
        if not p.exists():
            _log.error("missing input: %s", p)
            return 2

    pca_scores = np.load(pca_npy)
    pid_order = pd.read_csv(pca_pid_csv, dtype={"pid": str})["pid"].tolist()
    meta = pd.read_csv(meta_csv, dtype={"pid": str}).set_index("pid")
    features = pd.read_csv(features_csv, dtype={"pid": str})
    lesion_labels = pd.read_csv(lesion_labels_csv, dtype={"pid": str})

    # Step 1: identify displaced
    full_df = identify_displaced(
        pca_scores, pid_order,
        meta["agatston_total"], meta["kernel"],
    )
    full_with_lm = attach_lm_features(full_df, features)
    displaced_pids = full_with_lm.loc[full_with_lm["displaced"], "pid"].tolist()
    _log.info("Step 1: %d displaced low-burden patients identified",
              len(displaced_pids))

    full_with_lm.to_csv(out_dir / "full_cohort_displacement.csv", index=False)
    full_with_lm[full_with_lm["displaced"]].to_csv(
        out_dir / "displaced_patients.csv", index=False,
    )

    # Step 2: Fisher test
    fisher_result = fisher_displaced_vs_nondisplaced_lm(full_with_lm)
    _save_json(out_dir / "fisher_test.json", fisher_result)
    _log.info(
        "Step 2 P2: displaced LM rate = %.2f, non-displaced = %.2f, "
        "Fisher p = %.2e, passes = %s",
        fisher_result.get("displaced_lm_rate", float("nan")),
        fisher_result.get("non_displaced_lm_rate", float("nan")),
        fisher_result.get("fisher_p_one_sided_greater", float("nan")),
        fisher_result.get("passes"),
    )

    # Step 3: cross-stratum replication
    cs = cross_stratum_replication(full_with_lm)
    _save_json(out_dir / "cross_stratum.json", cs)
    for kern, info in cs["per_stratum"].items():
        _log.info(
            "Step 3 P3: stratum %s -> n_displaced=%d, LM rate=%.2f, "
            "non-disp LM rate=%.2f, passes=%s",
            kern, info["n_displaced"], info["displaced_lm_rate"],
            info["non_displaced_lm_rate"], info["passes"],
        )
    _log.info("Step 3 overall replication passes: %s", cs["passes"])

    # Step 4: density profile
    dprof = density_profile(full_with_lm)
    _save_json(out_dir / "density_profile.json", dprof)
    _log.info(
        "Step 4 P4: median max-HU = %.1f, W1/W2 = %d, W3/W4 = %d, framing = %s",
        dprof.get("median_max_hu_lm", float("nan")),
        dprof.get("n_soft_w1_w2", 0),
        dprof.get("n_dense_w3_w4", 0),
        dprof.get("framing", "n/a"),
    )

    # Step 5: cluster overlap (with LAD-dominant clusters discovered at
    # runtime; k-means labels are arbitrary across reruns)
    lad_dominant = _discover_lad_dominant_clusters(stage9_root(cfg))
    _log.info("Step 5: LAD-dominant clusters discovered at runtime = %s",
              lad_dominant)
    coverlap = cluster_overlap_with_lad(
        displaced_pids, lesion_labels,
        lad_dominant_clusters=lad_dominant,
    )
    _save_json(out_dir / "cluster_overlap.json", coverlap)
    _log.info(
        "Step 5 P5: %d LM lesions, %d in LAD-dominant clusters %s "
        "(%.0f%%), framing = %s",
        coverlap.get("lesion_count", 0),
        coverlap.get("n_in_lad_dominant_clusters", 0),
        list(lad_dominant),
        100 * coverlap.get("overlap_fraction", 0.0),
        coverlap.get("framing", "n/a"),
    )

    # Wilson CIs at the top-level summary
    n_disp = fisher_result.get("n_displaced", 0)
    n_lm_disp = sum(1 for pid in displaced_pids
                     if full_with_lm.loc[full_with_lm.pid == pid, "has_lm"].iloc[0])
    lo, hi = wilson_ci(n_lm_disp, n_disp) if n_disp else (float("nan"), float("nan"))

    summary = {
        "n_cohort": int(len(full_with_lm)),
        "n_low_tertile": int((full_with_lm["tertile"] == "low").sum()),
        "n_displaced": int(n_disp),
        "displaced_lm_count": int(n_lm_disp),
        "displaced_lm_rate": float(n_lm_disp / n_disp) if n_disp else float("nan"),
        "wilson95_lo": lo,
        "wilson95_hi": hi,
        "fisher_p": fisher_result.get("fisher_p_one_sided_greater"),
        "fisher_passes_P2": fisher_result.get("passes"),
        "cross_stratum_passes_P3": cs["passes"],
        "density_framing_P4": dprof.get("framing"),
        "cluster_distinctness_passes_P5": coverlap.get("passes_distinctness"),
        # None when P5 is not evaluable (no LAD-dominant cluster this run).
        "overall_passes_all_criteria": (
            None if coverlap.get("passes_distinctness") is None
            else bool(fisher_result.get("passes") and cs["passes"]
                      and coverlap["passes_distinctness"])
        ),
    }
    _save_json(out_dir / "summary.json", summary)
    _log.info("Overall PASS: %s", summary["overall_passes_all_criteria"])

    # Run header with seam SHAs + plan.md SHA
    plan_md = Path(__file__).resolve().parent / "plan.md"
    header = {
        "experiment": "lm_isolated_low_burden",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_hash(repo_root),
        "plan_md_sha": _file_sha(plan_md),
        "python_version": sys.version.split()[0],
        "args": vars(args),
        "seam_shas": {
            "pca_scores.npy": _file_sha(pca_npy),
            "cohort_metadata.csv": _file_sha(meta_csv),
            "features.csv": _file_sha(features_csv),
            "lesion_cluster_labels.csv": _file_sha(lesion_labels_csv),
        },
        "thresholds": {
            "P1_pc1": P1_PC1_THRESHOLD,
            "P1_pc2": P1_PC2_THRESHOLD,
            "P2_fisher_p_max": P2_FISHER_P_MAX,
            "P2_lm_rate_min": P2_LM_RATE_MIN,
            "P3_stratum_lm_rate_min": P3_STRATUM_LM_RATE_MIN,
            "P5_overlap_threshold": P5_OVERLAP_THRESHOLD,
        },
    }
    for mod in ("numpy", "pandas", "scipy", "statsmodels"):
        try:
            m = __import__(mod)
            header[f"{mod}_version"] = getattr(m, "__version__", "unknown")
        except Exception:
            header[f"{mod}_version"] = "n/a"
    _save_json(out_dir / "run_header.json", header)

    _log.info("done. outputs in %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
