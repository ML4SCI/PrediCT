#!/usr/bin/env python
"""Step 1-3 of the LAD-phenotype experiment.

Reads stage-3 + lesion-morphology outputs and:

  Step 1 (P1 in plan.md): identifies which lesion clusters match the
         pre-registered LAD-dominant signature.
  Step 2 (P2): within-LAD axial localisation; one-sided Mann-Whitney
         test that LAD-cluster lesions sit proximally.
  Step 3 (P3): carrier patient signature with Cliff's delta and
         FDR-corrected Mann-Whitney across the directional bundle.

Writes outputs to ``outputs/exploratory/lad_phenotype/``. No production
files are modified.

Usage:
    python experiments/lad_phenotype/run.py
    python experiments/lad_phenotype/run.py --stratum Qr36d/2
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

from predict.analyse import cliffs_delta, mannwhitney_u_pval
from predict.config import load_config
from predict.stage9 import stage9_root
from statsmodels.stats.multitest import multipletests


# ─── Pre-registered thresholds (locked in plan.md, do not edit lightly) ───

P1_LAD_OBS_OVER_EXP_MIN = 1.30
P1_RCA_OBS_OVER_EXP_MAX = 1.00
P1_MAX_HU_MIN = 400.0
P1_VOLUME_MM3_MIN = 50.0

P2_REL_Z_MEDIAN_DIFF_MIN = 0.10
P2_MW_P_MAX = 0.01

P3_CONFIRM_FRAC_MIN = 3 / 5  # at least 3 of 5 directional confirmations


_log = logging.getLogger("lad_phenotype.run")


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


# ─── Step 1 inputs: per-cluster tables ───

KEYS = ["pid", "vessel", "lesion_idx"]


def _lesion_analyse_module():
    """experiments/lesion_morphology/analyse.py (producer of
    cluster_vessel_chi_square.csv), loaded by path."""
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "lesion_morphology" / "analyse.py"
    spec = importlib.util.spec_from_file_location("_lesion_analyse", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cluster_tables(clustered: pd.DataFrame, primary_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-cluster medians (as lesion_morphology/run.py writes
    cluster_profiles.csv) and vessel obs/exp (lesion_morphology/analyse.py's
    function) for the given clustered lesions with their shared-partition
    labels. Stratum runs use these on the stratum's own lesions, so the
    stratum signature is not a copy of the full-cohort one (D043)."""
    labels = clustered[primary_col].to_numpy()
    rows = []
    for cid in sorted(np.unique(labels)):
        part = clustered[labels == cid]
        row = {"cluster": int(cid), "n_lesions": int(len(part))}
        for c in ("max_hu", "volume_mm3"):
            vals = part[c].dropna()
            row[f"{c}_median"] = float(np.median(vals)) if len(vals) else float("nan")
        rows.append(row)
    vessel = _lesion_analyse_module().per_cluster_vessel_chi_square(
        clustered.reset_index(drop=True), labels)
    return pd.DataFrame(rows), vessel


def _require_same(stored: pd.DataFrame, recomputed: pd.DataFrame, cols: list[str], what: str) -> None:
    a = stored.set_index("cluster")[cols].sort_index()
    b = recomputed.set_index("cluster")[cols].sort_index()
    if not (a.index.equals(b.index) and np.allclose(a.to_numpy(float), b.to_numpy(float),
                                                    rtol=1e-9, atol=1e-9, equal_nan=True)):
        raise SystemExit(f"hard stop: recomputed {what} differ from the stored full-cohort "
                         "table; the stratum recompute would not use the same definitions")


# ─── Step 1: signature discovery ───


def discover_lad_dominant_clusters(
    profiles: pd.DataFrame,
    vessel_chi: pd.DataFrame,
) -> list[int]:
    """Return cluster ids matching all four pre-registered criteria."""
    required_p = {"cluster", "max_hu_median", "volume_mm3_median"}
    required_v = {"cluster", "lad_obs_over_exp", "rca_obs_over_exp"}
    if missing := required_p - set(profiles.columns):
        raise ValueError(f"profiles missing columns: {missing}")
    if missing := required_v - set(vessel_chi.columns):
        raise ValueError(f"vessel_chi missing columns: {missing}")

    merged = profiles.merge(vessel_chi[list(required_v)],
                             on="cluster", how="left")
    qualified = merged[
        (merged["lad_obs_over_exp"] > P1_LAD_OBS_OVER_EXP_MIN)
        & (merged["rca_obs_over_exp"] < P1_RCA_OBS_OVER_EXP_MAX)
        & (merged["max_hu_median"] > P1_MAX_HU_MIN)
        & (merged["volume_mm3_median"] > P1_VOLUME_MM3_MIN)
    ]
    return sorted(int(c) for c in qualified["cluster"].tolist())


# ─── Step 2: within-LAD axial localisation ───


def compute_within_lad_relative_z(
    lesions: pd.DataFrame,
    lad_cluster_ids: list[int],
    primary_col: str,
) -> pd.DataFrame:
    """Per-patient relative-z of every LAD lesion.

    Convention (matches plan.md): rel_z=0 is most PROXIMAL (top of the
    LAD, closest to the LM origin at the heart base), rel_z=1 is most
    DISTAL (apex). In standard DICOM patient coordinates, +z is
    SUPERIOR (toward the head). The heart base sits superiorly, the
    apex inferiorly, so within the LAD: proximal = HIGH z, distal =
    LOW z. The formula inverts the raw min-max scaling accordingly:

        rel_z = (max_z - centroid_z) / (max_z - min_z)

    A previous implementation used (centroid_z - min_z) / range, which
    flipped the axis (rel_z=0 -> apex). That was a coordinate-direction
    bug, not a hypothesis change; the fix re-aligns code with plan.md.

    Patients with <2 LAD lesions are skipped (no range to normalise).
    """
    lad = lesions[lesions["vessel"] == "LAD"].copy()
    out_rows: list[dict] = []
    for pid, grp in lad.groupby("pid"):
        if len(grp) < 2:
            continue
        z_min = grp["centroid_z_mm"].min()
        z_max = grp["centroid_z_mm"].max()
        if z_max == z_min:
            continue
        for _, row in grp.iterrows():
            rel_z = (z_max - row["centroid_z_mm"]) / (z_max - z_min)
            out_rows.append({
                "pid": str(pid),
                "vessel": "LAD",
                "lesion_idx": int(row["lesion_idx"]),
                "centroid_z_mm": float(row["centroid_z_mm"]),
                "relative_z_within_LAD": float(rel_z),
                "cluster": int(row[primary_col]) if pd.notna(row[primary_col])
                            else -1,
                "in_lad_dominant_cluster": bool(
                    int(row[primary_col]) in lad_cluster_ids
                ) if pd.notna(row[primary_col]) else False,
            })
    return pd.DataFrame(out_rows)


def evaluate_axial_hypothesis(rel_z_df: pd.DataFrame) -> dict:
    """One-sided Mann-Whitney: LAD-cluster lesions are proximal."""
    in_cluster = rel_z_df.loc[
        rel_z_df["in_lad_dominant_cluster"], "relative_z_within_LAD",
    ].to_numpy()
    other = rel_z_df.loc[
        ~rel_z_df["in_lad_dominant_cluster"], "relative_z_within_LAD",
    ].to_numpy()
    if in_cluster.size == 0 or other.size == 0:
        return {
            "n_in_cluster": int(in_cluster.size),
            "n_other": int(other.size),
            "median_in_cluster": float("nan"),
            "median_other": float("nan"),
            "median_diff": float("nan"),
            "mw_p_one_sided_less": float("nan"),
            "passes": False,
            "fail_reason": "empty subset",
        }
    median_in = float(np.median(in_cluster))
    median_other = float(np.median(other))
    median_diff = median_other - median_in  # positive = LAD-cluster more proximal
    mw_p = float(stats.mannwhitneyu(
        in_cluster, other, alternative="less",
    ).pvalue)
    passes = bool(
        median_diff >= P2_REL_Z_MEDIAN_DIFF_MIN and mw_p < P2_MW_P_MAX
    )
    return {
        "n_in_cluster": int(in_cluster.size),
        "n_other": int(other.size),
        "median_in_cluster": median_in,
        "median_other": median_other,
        "median_diff": median_diff,
        "mw_p_one_sided_less": mw_p,
        "threshold_median_diff_min": P2_REL_Z_MEDIAN_DIFF_MIN,
        "threshold_mw_p_max": P2_MW_P_MAX,
        "passes": passes,
    }


# ─── Step 3: carrier signature ───


CARRIER_DIRECTIONAL_BUNDLE = (
    # (feature_name, direction "greater" / "less" for carriers vs non-carriers)
    ("agatston_total", "greater"),
    ("agatston_lad", "greater"),
    ("n_calcified_arteries", "greater"),
    ("rca_share", "less"),       # share of total burden in RCA - LAD-cluster should be lower
    ("agatston_lm", "less"),     # LM-sparing pattern
)


def carrier_flag_per_pid(
    lesion_labels: pd.DataFrame,
    primary_col: str,
    lad_cluster_ids: list[int],
) -> pd.Series:
    """Boolean Series indexed by pid (index name = 'pid'): True if
    patient carries any lesion in the LAD-dominant clusters."""
    mask = lesion_labels[primary_col].isin(lad_cluster_ids)
    carriers = lesion_labels.loc[mask, "pid"].astype(str).unique()
    all_pids = lesion_labels["pid"].astype(str).unique()
    s = pd.Series(
        [pid in set(carriers) for pid in all_pids],
        index=all_pids, name="is_carrier",
    )
    s.index.name = "pid"  # ensure index name survives downstream joins
    return s


def build_carrier_feature_table(
    features: pd.DataFrame,
    carrier: pd.Series,
) -> pd.DataFrame:
    """Aggregate per-pid features needed for the carrier bundle.

    Returns a dataframe with explicit 'pid' column (no relying on
    pandas to preserve the index name through joins).
    """
    df = features.copy()
    df["pid"] = df["pid"].astype(str)
    df = df.set_index("pid")
    df.index.name = "pid"
    carrier = carrier.copy()
    carrier.index = carrier.index.astype(str)
    carrier.index.name = "pid"
    df = df.join(carrier.rename("is_carrier"), how="inner")
    df["agatston_total_safe"] = df["agatston_total"].fillna(0).clip(lower=0)
    df["rca_share"] = (
        df.get("agatston_rca", pd.Series(0, index=df.index)).fillna(0)
        / (df["agatston_total_safe"] + 1e-6)
    )
    out = df.reset_index()
    if "pid" not in out.columns:
        # defensive: if reset_index produced 'index' instead of 'pid', fix it
        out = out.rename(columns={"index": "pid"})
    return out


def evaluate_carrier_bundle(carrier_table: pd.DataFrame) -> dict:
    """Cliff's delta + MW p across the bundle. FDR-BH across rows."""
    out_rows: list[dict] = []
    carrier_mask = carrier_table["is_carrier"].astype(bool).to_numpy()
    for feature_name, direction in CARRIER_DIRECTIONAL_BUNDLE:
        if feature_name not in carrier_table.columns:
            out_rows.append({
                "feature": feature_name,
                "missing": True,
                "passes": False,
            })
            continue
        a = carrier_table.loc[carrier_mask, feature_name].astype(float).dropna()
        b = carrier_table.loc[~carrier_mask, feature_name].astype(float).dropna()
        if a.size == 0 or b.size == 0:
            out_rows.append({
                "feature": feature_name,
                "missing": False,
                "n_carriers": int(a.size), "n_other": int(b.size),
                "passes": False,
            })
            continue
        delta = float(cliffs_delta(a.to_numpy(), b.to_numpy()))
        if direction == "greater":
            mw_p = float(mannwhitney_u_pval(a.to_numpy(), b.to_numpy(),
                                              alternative="greater"))
            direction_confirmed = delta > 0
        else:
            mw_p = float(mannwhitney_u_pval(a.to_numpy(), b.to_numpy(),
                                              alternative="less"))
            direction_confirmed = delta < 0
        out_rows.append({
            "feature": feature_name,
            "missing": False,
            "n_carriers": int(a.size),
            "n_other": int(b.size),
            "median_carrier": float(np.median(a)),
            "median_other": float(np.median(b)),
            "cliffs_delta": delta,
            "mw_p_one_sided": mw_p,
            "direction_predicted": direction,
            "direction_confirmed": direction_confirmed,
        })
    df = pd.DataFrame(out_rows)
    if not df.empty and "mw_p_one_sided" in df.columns:
        p_vals = df["mw_p_one_sided"].fillna(1.0).to_numpy()
        # BH-FDR across the directional bundle.
        _, q_vals, _, _ = multipletests(p_vals, alpha=0.05, method="fdr_bh")
        df["fdr_bh_q"] = q_vals
        df["passes"] = (df["direction_confirmed"]
                         & (df["fdr_bh_q"] < 0.05))
    else:
        df["fdr_bh_q"] = float("nan")
        df["passes"] = False
    n_pass = int(df["passes"].sum())
    n_total = int(len(df))
    return {
        "n_passing_directional": n_pass,
        "n_total_directional": n_total,
        "frac_passing": n_pass / n_total if n_total else 0.0,
        "frac_passing_threshold": P3_CONFIRM_FRAC_MIN,
        "overall_passes": (n_pass / n_total) >= P3_CONFIRM_FRAC_MIN
                          if n_total else False,
        "per_feature": df.to_dict(orient="records"),
    }


# ─── orchestration ───


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--stratum", type=str, default=None,
        help="If set, restrict the cohort to a kernel stratum "
              "(e.g. 'Qr36d/2' or 'I30f/3').",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = load_config(args.config)
    repo_root = cfg.paths.outputs.parent
    out_dir = stage9_root(cfg) / "lad_phenotype"
    if args.stratum:
        slug = args.stratum.replace("/", "_")
        out_dir = out_dir / f"stratified_{slug}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # No result of an earlier run may survive this one (D043).
    for stale in ("axial_within_lad.csv", "axial_summary.json", "carrier_profile.csv",
                  "carrier_summary.json", "lad_cluster_signature.json",
                  "stratum_cluster_profiles.csv", "stratum_vessel_chi_square.csv"):
        (out_dir / stale).unlink(missing_ok=True)

    # Inputs.
    lesion_root = (stage9_root(cfg) / "lesion_morphology")
    profiles_path = lesion_root / "cluster_profiles.csv"
    vessel_chi_path = lesion_root / "cluster_vessel_chi_square.csv"
    labels_path = lesion_root / "lesion_cluster_labels.csv"
    lesions_path = cfg.paths.outputs / "03_features" / "lesions.csv"
    features_path = cfg.paths.outputs / "03_features" / "features.csv"
    cohort_meta_path = cfg.paths.outputs / "06_reduce" / "cohort_metadata.csv"

    for p in (profiles_path, vessel_chi_path, labels_path,
              lesions_path, features_path, cohort_meta_path):
        if not p.exists():
            _log.error("missing input: %s", p)
            return 2

    profiles = pd.read_csv(profiles_path)
    vessel_chi = pd.read_csv(vessel_chi_path)
    lesion_labels = pd.read_csv(labels_path, dtype={"pid": str})
    lesions = pd.read_csv(lesions_path, dtype={"pid": str})
    features = pd.read_csv(features_path, dtype={"pid": str})
    cohort_meta = pd.read_csv(cohort_meta_path, dtype={"pid": str})

    # Stratify if requested.
    if args.stratum:
        keep_pids = set(cohort_meta.loc[
            cohort_meta["kernel"] == args.stratum, "pid",
        ].astype(str))
        lesion_labels = lesion_labels[
            lesion_labels["pid"].astype(str).isin(keep_pids)
        ]
        lesions = lesions[lesions["pid"].astype(str).isin(keep_pids)]
        features = features[features["pid"].astype(str).isin(keep_pids)]
        _log.info("stratum=%s -> %d pids", args.stratum, len(keep_pids))

    # Step 1 inputs: the full cohort uses the stored lesion-morphology tables
    # (checked against a recompute); a stratum recomputes them on its own
    # lesions with the shared partition.
    primary_col = next(
        c for c in lesion_labels.columns if c.startswith("cluster_kmeans_k")
    )
    clustered = lesions.merge(lesion_labels[KEYS + [primary_col]], on=KEYS,
                              how="inner", validate="one_to_one")
    rec_profiles, rec_vessel = cluster_tables(clustered, primary_col)
    if args.stratum:
        profiles, vessel_chi = rec_profiles, rec_vessel
        profiles.to_csv(out_dir / "stratum_cluster_profiles.csv", index=False)
        vessel_chi.to_csv(out_dir / "stratum_vessel_chi_square.csv", index=False)
    else:
        _require_same(profiles, rec_profiles, ["max_hu_median", "volume_mm3_median"],
                      "cluster profiles")
        _require_same(vessel_chi, rec_vessel, ["lad_obs_over_exp", "rca_obs_over_exp"],
                      "vessel obs/exp")

    # Step 1.
    lad_clusters = discover_lad_dominant_clusters(profiles, vessel_chi)
    _log.info("Step 1: LAD-dominant clusters (pre-reg signature) = %s",
              lad_clusters)
    _save_json(out_dir / "lad_cluster_signature.json", {
        "lad_dominant_cluster_ids": lad_clusters,
        "tables": ("recomputed on this stratum's lesions (shared partition)"
                   if args.stratum else "stored full-cohort lesion-morphology tables"),
        "thresholds": {
            "lad_obs_over_exp_min": P1_LAD_OBS_OVER_EXP_MIN,
            "rca_obs_over_exp_max": P1_RCA_OBS_OVER_EXP_MAX,
            "max_hu_min": P1_MAX_HU_MIN,
            "volume_mm3_min": P1_VOLUME_MM3_MIN,
        },
        "passes": len(lad_clusters) > 0,
    })

    if not lad_clusters:
        _log.warning("No LAD-dominant cluster matches pre-registered "
                     "signature. Experiment terminates per plan.md.")
        # Still write a stub run_header so finalise.py can detect this.
        _save_json(out_dir / "run_header.json", _build_run_header(
            repo_root, cfg, args, profiles_path, vessel_chi_path,
            labels_path, lesions_path, features_path,
        ))
        return 0

    # Merge cluster labels onto lesions.
    merged_lesions = lesions.merge(
        lesion_labels[["pid", "vessel", "lesion_idx", primary_col]],
        on=["pid", "vessel", "lesion_idx"], how="left",
    )

    # Step 2.
    rel_z_df = compute_within_lad_relative_z(
        merged_lesions, lad_clusters, primary_col,
    )
    rel_z_df.to_csv(out_dir / "axial_within_lad.csv", index=False)
    axial_summary = evaluate_axial_hypothesis(rel_z_df)
    _save_json(out_dir / "axial_summary.json", axial_summary)
    _log.info("Step 2: rel-z median (LAD-cluster) = %.3f, "
              "rel-z median (other LAD) = %.3f, MW p = %.2e, passes = %s",
              axial_summary.get("median_in_cluster"),
              axial_summary.get("median_other"),
              axial_summary.get("mw_p_one_sided_less"),
              axial_summary.get("passes"))

    # Step 3.
    carrier = carrier_flag_per_pid(
        merged_lesions, primary_col, lad_clusters,
    )
    carrier_table = build_carrier_feature_table(features, carrier)
    carrier_table.to_csv(out_dir / "carrier_profile.csv", index=False)
    carrier_summary = evaluate_carrier_bundle(carrier_table)
    _save_json(out_dir / "carrier_summary.json", carrier_summary)
    _log.info("Step 3: directional confirmations = %d/%d, "
              "overall passes = %s",
              carrier_summary["n_passing_directional"],
              carrier_summary["n_total_directional"],
              carrier_summary["overall_passes"])

    # Run header.
    _save_json(out_dir / "run_header.json", _build_run_header(
        repo_root, cfg, args, profiles_path, vessel_chi_path,
        labels_path, lesions_path, features_path,
    ))
    _log.info("done. outputs in %s", out_dir)
    return 0


def _build_run_header(
    repo_root: Path, cfg, args, *seam_paths: Path,
) -> dict:
    plan_path = Path(__file__).resolve().parent / "plan.md"
    info: dict = {
        "experiment": "lad_phenotype",
        "step": "run.py (Steps 1-3)",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_hash(repo_root),
        "plan_md_sha": _file_sha(plan_path),
        "python_version": sys.version.split()[0],
        "args": vars(args),
        "seam_shas": {p.name: _file_sha(p) for p in seam_paths},
        "pre_registered_thresholds": {
            "P1_lad_obs_over_exp_min": P1_LAD_OBS_OVER_EXP_MIN,
            "P1_rca_obs_over_exp_max": P1_RCA_OBS_OVER_EXP_MAX,
            "P1_max_hu_min": P1_MAX_HU_MIN,
            "P1_volume_mm3_min": P1_VOLUME_MM3_MIN,
            "P2_rel_z_median_diff_min": P2_REL_Z_MEDIAN_DIFF_MIN,
            "P2_mw_p_max": P2_MW_P_MAX,
            "P3_confirm_frac_min": P3_CONFIRM_FRAC_MIN,
        },
    }
    for mod in ("numpy", "pandas", "scipy"):
        try:
            m = __import__(mod)
            info[f"{mod}_version"] = getattr(m, "__version__", "unknown")
        except Exception:
            info[f"{mod}_version"] = "n/a"
    return info


if __name__ == "__main__":
    sys.exit(main())
