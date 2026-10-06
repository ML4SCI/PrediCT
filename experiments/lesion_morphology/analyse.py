#!/usr/bin/env python
"""Post-hoc analyses on the lesion-morphology clustering results.

Consumes the outputs of ``run.py``. Does NOT re-cluster; reads
``lesion_cluster_labels.csv`` and the seam files. Numeric cluster labels
only; clinical naming is not attempted (cluster ids are arbitrary across
fits).

Pipeline (9 sections):

  1. Numeric labels + per-cluster medians
  2. Per-cluster vessel composition + chi-square against COHORT base
     rates (not uniform) + Cramer's V
  3. Patient mixture matrix (patients x cluster fractions)
  4. CLR (centred log-ratio) transform of the mixture matrix
     (Aitchison compositional data analysis; standard practice for
     fractions that sum to 1)
  5. Hopkins + gap + Hennig on the CLR mixture vectors:
        does a discrete patient mixture-phenotype exist?
  6. Burden-stratified mixture composition + per-cluster Spearman rho
     against agatston_total (cross-sectional)
  7. k=7 sensitivity refit + ARI vs the primary partition
  8. GMM k=3 forced + ARI vs the signature {soft, moderate, dense} grouping
  9. Write a generated findings outline (findings.md in the output folder)

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
from scipy import stats
from sklearn.metrics import adjusted_rand_score

from predict.config import default_n_jobs, load_config
from predict.stage9 import stage9_root
from predict.discover.cluster_discovery import fit_cluster, gap_statistic
from predict.discover.clusterability import assess_clusterability
from predict.discover.validity import hennig_clusterboot
from predict.figures._lesion_classes import discover_broad_classes


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


# ─────────────────────── compositional data: CLR ───────────────────────


def clr_transform(
    fractions: np.ndarray,
    pseudocount: float | None = None,
) -> np.ndarray:
    """Centred log-ratio transform for compositional data (Aitchison 1986).

    ``fractions``: (n_patients, n_categories) matrix of cluster fractions
    summing to 1 per row. Zeros are replaced with a small pseudocount.
    Default pseudocount = 1 / (2 * n_categories) per Aitchison's
    recommendation. The transform produces unconstrained coordinates
    suitable for Euclidean-distance methods (k-means, Hopkins, gap).

    Returns array of the same shape, columns sum to 0 per row.
    """
    F = np.asarray(fractions, dtype=float).copy()
    if F.ndim != 2:
        raise ValueError("clr_transform expects 2D input")
    n_categories = F.shape[1]
    if pseudocount is None:
        pseudocount = 1.0 / (2.0 * n_categories)
    F[F < pseudocount] = pseudocount
    # Renormalise so each row still sums to 1 after pseudocount injection
    F = F / F.sum(axis=1, keepdims=True)
    log_F = np.log(F)
    return log_F - log_F.mean(axis=1, keepdims=True)


# ─────────────────────── vessel chi-square ───────────────────────


def per_cluster_vessel_chi_square(
    lesions: pd.DataFrame,
    primary_labels: np.ndarray,
) -> pd.DataFrame:
    """For each cluster, chi-square test of its vessel composition against
    the COHORT-level vessel base rate. Cramer's V for effect size.

    Cohort base rate is the empirical distribution of vessels over ALL
    lesions in the cohort (NOT a uniform 25 / 25 / 25 / 25 baseline).
    """
    vessels = ["LAD", "RCA", "LCx", "LM"]
    cohort_counts = lesions["vessel"].value_counts().reindex(vessels, fill_value=0)
    cohort_total = int(cohort_counts.sum())
    cohort_p = cohort_counts / cohort_total
    rows: list[dict] = []
    for cid in sorted(np.unique(primary_labels)):
        rows_in = lesions[primary_labels == cid]
        n_in = int(len(rows_in))
        obs = rows_in["vessel"].value_counts().reindex(vessels, fill_value=0)
        exp = n_in * cohort_p
        # Drop categories with expected count = 0 to keep chi-square valid
        mask = exp > 0
        if mask.sum() < 2:
            chi2_stat = float("nan")
            chi2_p = float("nan")
            cramers_v = float("nan")
        else:
            chi2_stat, chi2_p = stats.chisquare(
                obs[mask].to_numpy(), exp[mask].to_numpy(),
            )
            # Cramer's V for a goodness-of-fit chi-square
            # (1 sample, k categories): V = sqrt(chi2 / (n * (k - 1)))
            k = int(mask.sum())
            cramers_v = float(np.sqrt(chi2_stat / (n_in * (k - 1))))
        row = {
            "cluster": int(cid),
            "n_lesions": n_in,
            "lad_obs": int(obs["LAD"]), "rca_obs": int(obs["RCA"]),
            "lcx_obs": int(obs["LCx"]), "lm_obs": int(obs["LM"]),
            "lad_exp": float(exp["LAD"]), "rca_exp": float(exp["RCA"]),
            "lcx_exp": float(exp["LCx"]), "lm_exp": float(exp["LM"]),
            "chi2": float(chi2_stat),
            "chi2_p": float(chi2_p),
            "cramers_v": cramers_v,
            "vessel_biased": bool(chi2_p < 0.001 and cramers_v >= 0.20)
                              if not np.isnan(chi2_p) else False,
        }
        # Per-vessel observed/expected ratio (a quick interpretive column)
        for v in vessels:
            row[f"{v.lower()}_obs_over_exp"] = (
                float(obs[v] / exp[v]) if exp[v] > 0 else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────── mixture matrix ───────────────────────


def build_patient_mixture_matrix(
    lesions: pd.DataFrame,
    cluster_col: str,
    n_clusters: int,
) -> pd.DataFrame:
    """Return a (n_patients x n_clusters) matrix of per-patient cluster
    fractions, indexed by pid."""
    counts = (
        lesions.groupby(["pid", cluster_col]).size().unstack(fill_value=0)
    )
    # Ensure every cluster has a column even if it has 0 lesions
    for c in range(n_clusters):
        if c not in counts.columns:
            counts[c] = 0
    counts = counts.sort_index(axis=1)
    fractions = counts.div(counts.sum(axis=1), axis=0)
    fractions.columns = [f"frac_C{c}" for c in counts.columns]
    return fractions


# ─────────────────────── main ───────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--primary-algorithm", default="kmeans",
                        choices=("kmeans", "ward", "gmm"))
    parser.add_argument("--gap-bootstraps", type=int, default=200)
    parser.add_argument("--hennig-bootstraps", type=int, default=100)
    parser.add_argument("--no-hennig", action="store_true")
    parser.add_argument("--k-range-mixture", type=int, nargs="+",
                        default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--n-jobs", type=int, default=default_n_jobs())
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("analyse_lesion_morph")

    cfg = load_config(args.config)
    in_dir = stage9_root(cfg) / "lesion_morphology"
    out_dir = in_dir  # write into the same dir
    if not in_dir.exists():
        log.error("missing %s; run experiments/lesion_morphology/run.py first",
                  in_dir)
        return 2

    # ── 1. Load run.py outputs + raw lesions ─────────────────────
    lesions_csv = cfg.paths.outputs / "03_features" / "lesions.csv"
    labels_csv = in_dir / "lesion_cluster_labels.csv"
    profiles_csv = in_dir / "cluster_profiles.csv"
    for p in (lesions_csv, labels_csv, profiles_csv):
        if not p.exists():
            log.error("missing %s", p)
            return 2

    lesions = pd.read_csv(lesions_csv, dtype={"pid": str})
    labels_df = pd.read_csv(labels_csv, dtype={"pid": str})
    cluster_col = f"cluster_{args.primary_algorithm}_k8"
    if cluster_col not in labels_df.columns:
        # try without _k8 suffix in case run.py used a different k
        candidates = [c for c in labels_df.columns
                      if c.startswith(f"cluster_{args.primary_algorithm}_k")]
        if not candidates:
            log.error("no %s_kN column in lesion_cluster_labels.csv",
                      args.primary_algorithm)
            return 2
        cluster_col = candidates[0]
    log.info("using primary cluster column: %s", cluster_col)
    lesions = lesions.merge(
        labels_df[["pid", "vessel", "lesion_idx", cluster_col]],
        on=["pid", "vessel", "lesion_idx"], how="left",
    )
    lesions[cluster_col] = lesions[cluster_col].astype(int)
    primary_labels = lesions[cluster_col].to_numpy()
    k_primary = int(lesions[cluster_col].max() + 1)
    log.info("primary partition: k=%d, n_lesions=%d",
             k_primary, len(lesions))

    # Header info
    profiles = pd.read_csv(profiles_csv)
    header = {
        "experiment": "lesion_morphology_analyse",
        "scope": "exploratory; not part of production pipeline",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_hash(cfg.paths.outputs.parent),
        "python_version": sys.version.split()[0],
        "args": vars(args),
        "primary_partition": cluster_col,
        "k_primary": k_primary,
        "input_sha": {
            "lesion_cluster_labels": _file_sha(labels_csv),
            "lesions_csv": _file_sha(lesions_csv),
        },
    }

    # ── 2. Vessel composition + chi-square (item 2) ──────────────
    log.info("computing per-cluster vessel composition + chi-square...")
    vessel_table = per_cluster_vessel_chi_square(lesions, primary_labels)
    vessel_table.to_csv(out_dir / "cluster_vessel_chi_square.csv", index=False)
    biased = vessel_table[vessel_table["vessel_biased"]]
    log.info("%d of %d clusters are vessel-biased (chi2 p<0.001 AND "
             "Cramer's V >=0.20)", len(biased), len(vessel_table))
    for _, row in biased.iterrows():
        ratio = f"LAD/RCA={row['lad_obs_over_exp']:.2f}/{row['rca_obs_over_exp']:.2f}"
        log.info("  C%d (n=%d): chi2_p=%.2g, V=%.3f, %s",
                 int(row["cluster"]), int(row["n_lesions"]),
                 row["chi2_p"], row["cramers_v"], ratio)

    # ── 3. Patient mixture matrix (item 3 setup) ──────────────────
    log.info("building patient mixture matrix...")
    mixture = build_patient_mixture_matrix(lesions, cluster_col, k_primary)
    log.info("mixture matrix shape: %s", mixture.shape)

    # ── 4. CLR transform (item 3 transform) ─────────────────────
    log.info("CLR-transforming mixture matrix for compositional analysis...")
    clr_mix = clr_transform(mixture.to_numpy())
    clr_df = pd.DataFrame(
        clr_mix, index=mixture.index,
        columns=[f"clr_C{c}" for c in range(k_primary)],
    )
    pd.concat([mixture, clr_df], axis=1).to_csv(
        out_dir / "patient_mixture_clr.csv",
    )

    # ── 5. Hopkins + gap + Hennig on CLR vectors (item 3 result) ─
    log.info("Hopkins on CLR mixture...")
    hopkins_mix = assess_clusterability(
        clr_mix,
        sample_frac=cfg.raw["hopkins"]["sample_frac"],
        threshold=cfg.raw["hopkins"]["cluster_tendency_threshold"],
        ambiguous_band=tuple(cfg.raw["hopkins"]["ambiguous_band"]),
        random_state=args.random_state,
    )
    log.info("Hopkins on CLR mixture: H=%.3f (%s)",
             hopkins_mix.H, hopkins_mix.verdict)
    _save_json(out_dir / "mixture_hopkins.json", hopkins_mix.to_dict())

    log.info("gap statistic on CLR mixture (k=%s)...",
             tuple(args.k_range_mixture))
    mixture_gap: list[dict] = []
    for algo in ("kmeans", "ward", "gmm"):
        result = gap_statistic(
            clr_mix, algorithm=algo,
            k_range=tuple(int(k) for k in args.k_range_mixture),
            n_bootstrap=args.gap_bootstraps,
            random_state=args.random_state, n_jobs=args.n_jobs,
        )
        mixture_gap.append({
            "algorithm": algo,
            "k_range": list(result.k_range),
            "gap_values": result.gap_values.tolist(),
            "sk_values": result.sk_values.tolist(),
            "selected_k": int(result.selected_k),
            "n_bootstrap": int(result.n_bootstrap),
        })
        log.info("  gap[%s] on CLR mixture: selected k=%d",
                 algo, result.selected_k)
    _save_json(out_dir / "mixture_gap_statistic.json", mixture_gap)

    if not args.no_hennig:
        # Hennig stability of the gap-modal k on CLR mixture
        votes: dict[int, int] = {}
        for r in mixture_gap:
            votes[r["selected_k"]] = votes.get(r["selected_k"], 0) + 1
        k_mix = min(k for k, v in votes.items() if v == max(votes.values()))
        if k_mix >= 2:
            mix_labels = fit_cluster(clr_mix, k=k_mix, algorithm="kmeans",
                                     random_state=args.random_state)
            log.info("Hennig on CLR mixture at k=%d (kmeans)...", k_mix)
            hen_mix = hennig_clusterboot(
                clr_mix, mix_labels, k=k_mix, algorithm="kmeans",
                n_bootstrap=args.hennig_bootstraps,
                threshold=0.75, random_state=args.random_state,
                n_jobs=args.n_jobs,
            )
            hen_mix_records = [
                {"cluster": int(cid),
                 "jaccard_median": float(med),
                 "jaccard_mean": float(mn),
                 "stable": bool(med >= 0.75)}
                for cid, med, mn in zip(hen_mix.cluster_ids,
                                        hen_mix.jaccard_median,
                                        hen_mix.jaccard_mean)
            ]
            _save_json(out_dir / "mixture_hennig.json", hen_mix_records)
            for r in hen_mix_records:
                log.info("  mixture cluster %d: median Jaccard=%.3f (%s)",
                         r["cluster"], r["jaccard_median"],
                         "STABLE" if r["stable"] else "UNSTABLE")
        else:
            log.info("mixture gap selected k=1; skipping Hennig")

    # ── 6. Burden-stratified mixture composition (item 4) ────────
    log.info("burden-stratified mixture composition + per-cluster vs "
             "agatston Spearman...")
    meta_csv = cfg.paths.outputs / "06_reduce" / "cohort_metadata.csv"
    if meta_csv.exists():
        meta = pd.read_csv(meta_csv, dtype={"pid": str})
        agatston = meta.set_index("pid")["agatston_total"].astype(float)
        common = mixture.index.intersection(agatston.index)
        mix_common = mixture.loc[common]
        ag_common = agatston.loc[common]
        tertile = pd.qcut(ag_common, q=3,
                          labels=["low", "mid", "high"], duplicates="drop")
        per_tertile_means = mix_common.groupby(tertile).mean()
        per_tertile_means.to_csv(out_dir / "mixture_by_burden_tertile.csv")
        log.info("mean cluster fraction per burden tertile:")
        log.info("\n%s", per_tertile_means.round(3).to_string())

        # Per-cluster Spearman rho vs raw agatston
        spearman_rows = []
        for c in mixture.columns:
            rho, p = stats.spearmanr(mix_common[c], ag_common)
            spearman_rows.append({
                "cluster_fraction": c,
                "spearman_rho_vs_agatston": float(rho),
                "p_value": float(p),
            })
        pd.DataFrame(spearman_rows).to_csv(
            out_dir / "mixture_burden_spearman.csv", index=False,
        )
    else:
        log.warning("cohort_metadata.csv missing; skipping burden analysis")

    # ── 7. k=7 sensitivity refit (item 5) ────────────────────────
    log.info("k=7 sensitivity refit on lesion morphology...")
    morphology_csv = in_dir / "lesion_features.csv"
    if morphology_csv.exists():
        morph_df = pd.read_csv(morphology_csv, dtype={"pid": str})
        feature_cols = [c for c in morph_df.columns
                        if c not in ("pid", "vessel", "lesion_idx")]
        X = morph_df[feature_cols].to_numpy(dtype=float)
        # Refit at k=7
        labels_k7 = fit_cluster(X, k=7, algorithm=args.primary_algorithm,
                                random_state=args.random_state)
        labels_k7_df = morph_df[["pid", "vessel", "lesion_idx"]].copy()
        labels_k7_df["cluster_k7"] = labels_k7
        labels_k7_df.to_csv(out_dir / "lesion_cluster_labels_k7.csv",
                            index=False)
        ari_k8_vs_k7 = float(adjusted_rand_score(primary_labels, labels_k7))
        log.info("ARI between k=%d and k=7 partition: %.3f",
                 k_primary, ari_k8_vs_k7)
        # Where do cluster-4 lesions go at k=7?
        if 4 in np.unique(primary_labels):
            c4_to_k7 = pd.crosstab(
                primary_labels[primary_labels == 4],
                labels_k7[primary_labels == 4],
            )
            c4_to_k7.to_csv(out_dir / "cluster4_to_k7_migration.csv")
            log.info("cluster 4 -> k=7 migration:\n%s",
                     c4_to_k7.to_string())
        # Hennig on k=7 partition
        if not args.no_hennig:
            hen_k7 = hennig_clusterboot(
                X, labels_k7, k=7, algorithm=args.primary_algorithm,
                n_bootstrap=args.hennig_bootstraps,
                threshold=0.75, random_state=args.random_state,
                n_jobs=args.n_jobs,
            )
            hen_k7_records = [
                {"cluster": int(cid),
                 "jaccard_median": float(med),
                 "jaccard_mean": float(mn),
                 "stable": bool(med >= 0.75)}
                for cid, med, mn in zip(hen_k7.cluster_ids,
                                        hen_k7.jaccard_median,
                                        hen_k7.jaccard_mean)
            ]
            _save_json(out_dir / "k7_hennig_stability.json", hen_k7_records)
            log.info("k=7 Hennig stability: %d of %d clusters stable",
                     sum(1 for r in hen_k7_records if r["stable"]),
                     len(hen_k7_records))
    else:
        log.warning("lesion_features.csv missing; skipping k=7 sensitivity")

    # ── 8. GMM k=3 forced + ARI vs the signature grouping (item 6) ─
    # The grouping of primary clusters into {soft, moderate, dense} comes
    # from each run's cluster medians via the locked signature rule
    # (predict.figures._lesion_classes.discover_broad_classes), not from a
    # table of cluster ids: ids are arbitrary across reruns and k changes
    # with the grid (k=12 at 0.5 mm, k=8 at 0.375 mm).
    log.info("GMM k=3 forced + ARI vs signature {soft, moderate, dense} grouping...")
    if morphology_csv.exists():
        labels_gmm3 = fit_cluster(X, k=3, algorithm="gmm",
                                  random_state=args.random_state)
        signature_grouping = discover_broad_classes(profiles)
        broad_codes = {n: i for i, n in enumerate(
            sorted(set(signature_grouping.values()))
        )}
        hyp_groupings = np.array([
            broad_codes[signature_grouping[int(c)]] for c in primary_labels
        ])
        ari_gmm3_vs_hyp = float(
            adjusted_rand_score(hyp_groupings, labels_gmm3)
        )
        log.info("ARI between GMM k=3 and signature {soft, moderate, "
                 "dense} grouping: %.3f", ari_gmm3_vs_hyp)
        # Crosstab GMM k=3 vs the 8-cluster primary
        gmm3_vs_primary = pd.crosstab(labels_gmm3, primary_labels)
        gmm3_vs_primary.index.name = "gmm_k3_cluster"
        gmm3_vs_primary.columns.name = f"primary_k{k_primary}_cluster"
        gmm3_vs_primary.to_csv(out_dir / "gmm_k3_vs_primary.csv")
        log.info("GMM k=3 vs primary k=%d crosstab:\n%s",
                 k_primary, gmm3_vs_primary.to_string())
        # Reverse-engineer: which primary clusters dominate each GMM k=3 cluster?
        majority_primary = {}
        for gmm3_id in sorted(np.unique(labels_gmm3)):
            cols = gmm3_vs_primary.loc[gmm3_id]
            majority_primary[int(gmm3_id)] = int(cols.idxmax())
        _save_json(out_dir / "gmm_k3_summary.json", {
            "ari_vs_signature_grouping": ari_gmm3_vs_hyp,
            "signature_grouping": signature_grouping,
            "grouping_rule": "predict.figures._lesion_classes.discover_broad_classes",
            "broad_class_codes": broad_codes,
            "majority_primary_cluster_per_gmm3": majority_primary,
        })

    # ── 9. Write structured findings.md outline ──────────────────
    findings_path = out_dir / "findings.md"
    lines: list[str] = []
    lines.append("# Lesion morphology experiment: findings outline\n")
    lines.append("Generated automatically from `analyse.py` from the "
                 "run + analyse outputs. The interpreted findings are in "
                 "`experiments/lesion_morphology/findings.md`.\n\n")
    lines.append("## Cluster identity (numeric, unnamed pending sensitivity)\n")
    lines.append(f"Per-cluster descriptions are this run's medians (k={k_primary}); "
                 "broad class by the locked signature rule "
                 "(predict.figures._lesion_classes). Cluster ids are arbitrary "
                 "across reruns. Clinical names are deferred.\n")
    broad = discover_broad_classes(profiles)
    for r in profiles.sort_values("cluster").itertuples():
        lines.append(f"- C{int(r.cluster)}: max_hu ~{r.max_hu_median:.0f}, "
                     f"vol ~{r.volume_mm3_median:.0f} mm3, "
                     f"n_rois ~{r.n_rois_median:.0f}, n={int(r.n_lesions)} "
                     f"({broad[int(r.cluster)]})\n")
    lines.append("\n## Vessel-biased clusters\n")
    biased_rows = vessel_table[vessel_table["vessel_biased"]]
    if len(biased_rows) == 0:
        lines.append("No cluster is significantly vessel-biased against the "
                     "cohort-level vessel base rate (chi2 p<0.001 AND "
                     "Cramer's V >= 0.20).\n")
    else:
        for _, row in biased_rows.iterrows():
            lines.append(
                f"- **C{int(row['cluster'])}** (n={int(row['n_lesions'])}): "
                f"chi2 p={row['chi2_p']:.2g}, Cramer's V={row['cramers_v']:.3f}; "
                f"obs/exp LAD={row['lad_obs_over_exp']:.2f}, "
                f"RCA={row['rca_obs_over_exp']:.2f}, "
                f"LCx={row['lcx_obs_over_exp']:.2f}, "
                f"LM={row['lm_obs_over_exp']:.2f}\n"
            )
    lines.append("\n## Patient-level mixture clusterability\n")
    lines.append(
        f"Hopkins on CLR-transformed mixture vectors: "
        f"H = {hopkins_mix.H:.3f} ({hopkins_mix.verdict})\n"
    )
    lines.append("Gap-selected k on CLR mixture:\n")
    for r in mixture_gap:
        lines.append(f"- {r['algorithm']}: k={r['selected_k']}\n")
    lines.append("\n## Sensitivity: k=7 vs the primary k\n")
    lines.append(f"ARI between the primary partition (k={k_primary}) and "
                 f"the k=7 refit: see `lesion_cluster_labels_k7.csv` and "
                 f"`k7_hennig_stability.json`.\n")
    lines.append("\n## GMM k=3 hierarchical interpretation\n")
    lines.append("If `ari_vs_signature_grouping` in `gmm_k3_summary.json` "
                 "is >= 0.85, the signature {soft, moderate, dense} grouping is "
                 "empirically supported. See the crosstab in "
                 "`gmm_k3_vs_primary.csv`.\n")
    findings_path.write_text("".join(lines), encoding="utf-8")
    log.info("findings.md written: %s", findings_path)

    # Run header (finalised)
    header["mixture_hopkins_H"] = float(hopkins_mix.H)
    header["mixture_hopkins_verdict"] = hopkins_mix.verdict
    header["mixture_gap_selected_k"] = {
        r["algorithm"]: r["selected_k"] for r in mixture_gap
    }
    _save_json(out_dir / "run_header_analyse.json", header)

    print()
    print("=" * 80)
    print(f"Lesion morph post-hoc analysis complete.")
    print(f"  primary partition: {cluster_col}")
    print(f"  mixture Hopkins H = {hopkins_mix.H:.3f}")
    print(f"  outputs at {out_dir}")
    print(f"  read findings.md for the structured interpretation outline")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
