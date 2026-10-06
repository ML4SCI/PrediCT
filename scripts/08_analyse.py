#!/usr/bin/env python
"""Stage 7 orchestrator (D023, D024, D025, D026, D027, D028).

Wires the 7 analyse-stage modules together over the three production
cohorts (full + Qr36d/2 + I30f/3) plus the low-burden robust sensitivity
subset.

Pipeline (in order):
  0. Label balance check (D023) on every spatial_k2 partition
  1. Focal / diffuse mapping (D023 rule: lower median n_calcified_arteries
     = focal) on each cohort's spatial labels
  2. Biological sanity check (D023): focal median max_hu >= 0.9 * diffuse
  3. Per-cohort cluster profiles + FDR-BH (D023) on the 41-feature bundle:
     28 cross-cohort-intersection representatives + 13 spatial PCA inputs
  4. Top-5 signature features per (cohort x partition x cluster)
  5. Burden orthogonality (D024): Mann-Whitney + Levene + Cliff's delta
     on agatston_total between focal and diffuse
  6. Burden-stratified spatial replication (D024 part 2)
  7. Directional hypothesis test (D025) per cohort + primary + secondary
     + overall verdict
  8. Monotonicity classification (D026) on the 28 robust features
  9. Cross-cohort feature consistency (D027 three-rule criterion)
 10. Partition-level ARI on shared pids (D027 complementary check)
 11. Main paper table (D028: 15 rows)
 12. Low-burden sensitivity (D028, low_burden_flag=False subset of the full
     cohort, full-cohort spatial labels): D024, D025 primary rule,
     signatures, the 5-row table and an explicit sensitivity verdict
 13. run_header_analyse.json with SHA of every seam file consumed

Before anything is computed, every cohort's stage 6 record and the raw
features.csv (against the hash stage 5 recorded) are verified. Outputs are
written to outputs/07_analyse.partial/ and published to outputs/07_analyse/
with COMPLETE_stage7.json only after a complete run; a failed run leaves the
last complete result in place.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from predict.provenance import code_revision
from predict.completion import files_manifest, write_marker
from predict.publish import fresh_partial, publish
from predict.analyse.sensitivity import ROBUST_LABEL, low_burden_sensitivity
from predict.analyse.stage7 import (
    COHORT_DIRS,
    STAGE7_MARKER,
    cohort_dir,
    features_csv_path,
    stage7_fingerprint,
    stage7_inputs,
)
from predict.analyse.cross_cohort import (
    consistency_table,
    partition_ari_table,
    robust_discriminator_count_summary,
)
from predict.analyse.derived_features import augment_raw_with_derived
from predict.analyse.hypotheses import (
    DIRECTIONAL_HYPOTHESES,
    directional_hypotheses_table,
    overall_verdict,
    primary_pass,
    secondary_pass,
)
from predict.analyse.monotonicity import (
    classification_summary,
    compute_monotonicity,
)
from predict.analyse.orthogonality import (
    assess_burden_orthogonality,
    burden_stratified_pass_verdict,
    burden_stratified_spatial_replication,
)
from predict.analyse.paper_table import (
    build_paper_table,
    build_robust_sensitivity_table,
)
from predict.analyse.profiles import (
    apply_fdr_bh,
    assert_biological_sanity,
    assert_label_balance,
    compute_cluster_profile,
    determine_focal_diffuse_mapping,
)
from predict.analyse.signatures import top_n_signatures
from predict.config import Config, load_config


# The 13 spatial-distribution features used as RAW inputs to the spatial PCA
# (matches scripts/07_discover.py SPATIAL_FEATURES_AFTER_D017).
SPATIAL_FEATURES_AFTER_D017: tuple[str, ...] = (
    "lesion_count_lad", "lesion_count_rca", "lesion_count_lcx", "lesion_count_lm",
    "lesion_count_total",
    "n_calcified_arteries",
    "gini_lesion_volume",
    "dist_from_top_max", "dist_from_top_mean",
    "center_of_mass_z",
    "inter_lesion_dist_mean_lad", "inter_lesion_dist_max_lad",
    "first_to_last_dist_lad",
)


# ─────────────────────── reproducibility ───────────────────────


def _file_sha(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _save_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


# ─────────────────────── seam loading ───────────────────────


def _load_cohort_seam(reduce_root: Path, cohort: str, cfg: Config) -> dict:
    """Read the stage 5 + stage 6 seam files for one cohort. main() has
    already verified the cohort's stage 6 record (stage7_inputs)."""
    cdir = cohort_dir(reduce_root, cohort)
    paths = {
        "cohort_metadata": cdir / "cohort_metadata.csv",
        "spatial_labels": cdir / "cluster_labels_spatial_k2.csv",
        "forced_labels": cdir / "cluster_labels_forced.csv",
        "representatives": cdir / "representative_features.csv",
        "multi_block": cdir / "multi_block_assignments.csv",
        "validity": cdir / "validity_checks.csv",
    }
    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(
                f"missing seam file for cohort '{cohort}': {p}"
            )

    cohort_meta = pd.read_csv(paths["cohort_metadata"], dtype={"pid": str})
    spatial_df = pd.read_csv(paths["spatial_labels"], dtype={"pid": str})
    spatial_labels = (
        spatial_df.set_index("pid")["spatial_only_gmm_k2"].astype(int)
    )
    representatives = pd.read_csv(paths["representatives"])["feature"].tolist()
    multi_block = pd.read_csv(paths["multi_block"])
    block_lookup = dict(zip(multi_block["feature"], multi_block["block"]))
    validity = pd.read_csv(paths["validity"])
    return {
        "cohort_metadata": cohort_meta,
        "spatial_labels_raw": spatial_labels,
        "representatives": representatives,
        "block_lookup": block_lookup,
        "validity": validity,
        "seam_sha": {k: _file_sha(p) for k, p in paths.items()},
    }


def _load_raw_features(features_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(features_csv, dtype={"pid": str})
    return df.set_index("pid")


def _hennig_lookup_from_validity(
    validity: pd.DataFrame, cohort: str,
) -> dict:
    """Build {(cohort, partition, cluster): jaccard} from a cohort's
    validity_checks.csv. Stage 6 writes Hennig median Jaccard per cluster
    for the forced k=3 (full feature space) and the spatial-only k=2
    bundles.
    """
    out: dict = {}
    hennig_rows = validity[validity["test"] == "hennig_clusterboot"].copy()
    for _, row in hennig_rows.iterrows():
        feature_space = row.get("feature_space")
        cluster_id = int(row["cluster_id"])
        jaccard = float(row["jaccard_median"])
        if pd.isna(feature_space) or feature_space == "full":
            # Forced k=3 on full PC space; not used by the paper table
            # (which uses qcut tertiles for burden_k3), but kept for
            # completeness.
            continue
        if feature_space == "spatial_only":
            # cluster_id 0 / 1 -- map to focal / diffuse downstream
            out[(cohort, "spatial_k2_raw", cluster_id)] = jaccard
    return out


def _remap_spatial_hennig(
    raw_hennig: dict, focal_map: dict[int, str], cohort: str,
) -> dict:
    """Translate spatial_k2_raw cluster_id Hennig values to (cohort,
    spatial_k2, focal/diffuse)."""
    out: dict = {}
    for (c, partition, cluster_id), jaccard in raw_hennig.items():
        if c != cohort or partition != "spatial_k2_raw":
            continue
        cluster_name = focal_map.get(cluster_id)
        if cluster_name is None:
            continue
        out[(cohort, "spatial_k2", cluster_name)] = jaccard
    return out


# ─────────────────────── main ───────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--include-robust-sensitivity", dest="include_robust_sensitivity",
                        action="store_true", default=True,
                        help="(default) include the low-burden robust sensitivity table")
    parser.add_argument("--no-include-robust-sensitivity", dest="include_robust_sensitivity",
                        action="store_false",
                        help="skip the robust sensitivity table")
    parser.add_argument("--top-n-signatures", type=int, default=5)
    parser.add_argument("--paper-table-top-features", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log = logging.getLogger("analyse")

    cfg = load_config(args.config)
    reduce_root = cfg.paths.outputs / "06_reduce"
    target = cfg.paths.outputs / "07_analyse"
    repo_root = cfg.paths.outputs.parent

    # Stage 6 records of every cohort, and features.csv against stage 5's
    # recorded hash, before anything is computed.
    inputs = stage7_inputs(cfg)
    out_dir = fresh_partial(target)
    settings = {
        "include_robust_sensitivity": bool(args.include_robust_sensitivity),
        "top_n_signatures": int(args.top_n_signatures),
        "paper_table_top_features": int(args.paper_table_top_features),
    }

    # ── Load all cohorts ───────────────────────────────────────────
    per_cohort: dict[str, dict] = {}
    for cohort in COHORT_DIRS:
        log.info("loading seam for cohort %s", cohort)
        per_cohort[cohort] = _load_cohort_seam(reduce_root, cohort, cfg)

    raw_features = _load_raw_features(features_csv_path(cfg))
    log.info("loaded raw features: shape=%s", raw_features.shape)

    # D019 derived features (high_density_fraction, vessel_burden_gini) are
    # NOT in stage 3's features.csv (they live only in prepared_matrix.csv
    # at z-scored scale). For stage 7's raw-scale analysis we re-derive
    # them on the fly using the byte-identical formulas from
    # predict.reduce.prepare_matrix. Verified by
    # tests/analyse/test_derived_features.py against the stage-5 sources.
    n_cols_before = raw_features.shape[1]
    raw_features = augment_raw_with_derived(raw_features)
    n_added = raw_features.shape[1] - n_cols_before
    log.info("raw features augmented with %d D019 derived columns; "
             "shape now %s", n_added, raw_features.shape)

    # ── Feature list: 28 cross-cohort intersection + 13 spatial inputs ──
    rep_sets = [set(per_cohort[c]["representatives"]) for c in per_cohort]
    cross_cohort_reps = sorted(set.intersection(*rep_sets))
    log.info("cross-cohort representative intersection: %d features",
             len(cross_cohort_reps))
    feature_list = sorted(
        set(cross_cohort_reps) | set(SPATIAL_FEATURES_AFTER_D017)
    )
    log.info("stage 7 feature bundle: %d features (28 robust + 13 spatial - "
             "dedup)", len(feature_list))

    # ── Step 0+1+2: label balance, focal/diffuse mapping, biological sanity ──
    for cohort, bundle in per_cohort.items():
        # Label balance for spatial_k2
        assert_label_balance(
            bundle["spatial_labels_raw"],
            cohort=cohort, partition="spatial_k2",
        )
        # Focal/diffuse mapping
        cohort_meta = bundle["cohort_metadata"]
        cohort_pids = cohort_meta["pid"].tolist()
        raw_for_cohort = raw_features.loc[
            raw_features.index.intersection(cohort_pids)
        ]
        focal_map = determine_focal_diffuse_mapping(
            raw_for_cohort, bundle["spatial_labels_raw"],
        )
        bundle["focal_map"] = focal_map
        # Apply the mapping to produce focal/diffuse string labels
        bundle["spatial_labels"] = bundle["spatial_labels_raw"].map(focal_map)
        log.info("cohort %s focal/diffuse mapping: %s", cohort, focal_map)
        # Biological sanity check
        sanity_info = assert_biological_sanity(
            raw_for_cohort, bundle["spatial_labels"],
            focal_label="focal", diffuse_label="diffuse",
            cohort=cohort,
        )
        bundle["sanity_info"] = sanity_info
        log.info("cohort %s biological sanity passed: focal_max_hu=%.1f "
                 "diffuse_max_hu=%.1f ratio=%.3f (absolute floor 130 HU)",
                 cohort, sanity_info["focal_median"],
                 sanity_info["diffuse_median"], sanity_info["ratio"])
        if sanity_info["warning_low_ratio"]:
            log.warning(
                "cohort %s biology note: focal/diffuse max_hu ratio = "
                "%.3f < %.2f. This is calcium-positive in both clusters "
                "but unusually low-peak in focal; consistent with "
                "earlier-stage / softer-plaque biology. Gate did NOT raise.",
                cohort, sanity_info["ratio"],
                sanity_info["warning_ratio_threshold"],
            )

    # ── Step 3: per-cohort cluster profiles + FDR-BH ──────────────
    per_cohort_profiles: dict[str, pd.DataFrame] = {}
    for cohort, bundle in per_cohort.items():
        cohort_meta = bundle["cohort_metadata"]
        cohort_pids = cohort_meta["pid"].tolist()
        raw_for_cohort = raw_features.loc[
            raw_features.index.intersection(cohort_pids)
        ]
        spatial_labels = bundle["spatial_labels"]
        agatston = (
            cohort_meta.set_index("pid")
            .loc[raw_for_cohort.index, "agatston_total"]
        )
        from predict.analyse.paper_table import _agatston_tertile_labels
        burden_tertile = _agatston_tertile_labels(agatston).astype(str)
        # Spatial profile
        spatial_profile = compute_cluster_profile(
            raw_for_cohort, spatial_labels, feature_list,
            cohort=cohort, partition="spatial_k2",
        )
        burden_profile = compute_cluster_profile(
            raw_for_cohort, burden_tertile, feature_list,
            cohort=cohort, partition="burden_k3",
        )
        combined = pd.concat([spatial_profile, burden_profile],
                             ignore_index=True)
        combined = apply_fdr_bh(combined)
        per_cohort_profiles[cohort] = combined
        log.info("cohort %s profile: %d rows (%d robust discriminators)",
                 cohort, len(combined),
                 int(combined["is_robust_discriminator"].sum()))

    all_profiles = pd.concat(per_cohort_profiles.values(), ignore_index=True)
    all_profiles.to_csv(out_dir / "cluster_profiles.csv", index=False)

    # ── Step 4: top-N signatures ──────────────────────────────────
    signatures = top_n_signatures(
        all_profiles, n=args.top_n_signatures, only_robust=True,
    )
    signatures.to_csv(out_dir / "signature_features.csv", index=False)
    log.info("signature features: %d rows", len(signatures))

    # ── Step 5: burden orthogonality ──────────────────────────────
    ortho_rows = []
    for cohort, bundle in per_cohort.items():
        cohort_meta = bundle["cohort_metadata"].set_index("pid")
        spatial_labels = bundle["spatial_labels"]
        common = cohort_meta.index.intersection(spatial_labels.index)
        focal_pids = spatial_labels[spatial_labels == "focal"].index.intersection(common)
        diffuse_pids = spatial_labels[spatial_labels == "diffuse"].index.intersection(common)
        focal_agatston = cohort_meta.loc[focal_pids, "agatston_total"].to_numpy(dtype=float)
        diffuse_agatston = cohort_meta.loc[diffuse_pids, "agatston_total"].to_numpy(dtype=float)
        ortho = assess_burden_orthogonality(
            focal_agatston, diffuse_agatston, cohort=cohort,
        )
        ortho_rows.append(ortho.to_dict())
        log.info("cohort %s burden orthogonality: %s (p=%.3f, delta=%.3f, "
                 "levene p=%.3f)",
                 cohort, ortho.interpretation, ortho.mannwhitney_pval,
                 ortho.cliffs_delta_agatston, ortho.levene_pval)
    pd.DataFrame(ortho_rows).to_csv(
        out_dir / "burden_orthogonality.csv", index=False,
    )

    # ── Step 6: burden-stratified spatial replication ─────────────
    stratified_rows = []
    for cohort, bundle in per_cohort.items():
        cohort_meta = bundle["cohort_metadata"].set_index("pid")
        spatial_labels = bundle["spatial_labels"]
        raw_for_cohort = raw_features.loc[
            raw_features.index.intersection(cohort_meta.index)
        ]
        agatston = cohort_meta.loc[raw_for_cohort.index, "agatston_total"]
        stratified = burden_stratified_spatial_replication(
            raw_for_cohort, spatial_labels, agatston,
            directional_features=list(DIRECTIONAL_HYPOTHESES),
            cohort=cohort,
        )
        stratified_rows.append(stratified)
    pd.concat(stratified_rows, ignore_index=True).to_csv(
        out_dir / "burden_stratified_spatial.csv", index=False,
    )

    # ── Step 7: directional hypothesis tests ──────────────────────
    directional_rows = []
    per_cohort_directional: dict[str, pd.DataFrame] = {}
    for cohort, bundle in per_cohort.items():
        cohort_meta = bundle["cohort_metadata"]
        spatial_labels = bundle["spatial_labels"]
        cohort_pids = cohort_meta["pid"].tolist()
        raw_for_cohort = raw_features.loc[
            raw_features.index.intersection(cohort_pids)
        ]
        dh = directional_hypotheses_table(
            raw_for_cohort, spatial_labels, cohort=cohort,
        )
        per_cohort_directional[cohort] = dh
        directional_rows.append(dh)
    pd.concat(directional_rows, ignore_index=True).to_csv(
        out_dir / "directional_hypotheses.csv", index=False,
    )

    primary = primary_pass(per_cohort_directional["full"])
    secondary = secondary_pass(
        per_cohort_directional["Qr36d/2"],
        per_cohort_directional["I30f/3"],
    )
    verdict = overall_verdict(primary, secondary)
    _save_json(out_dir / "directional_verdict.json", {
        "primary": primary,
        "secondary": secondary,
        "overall_verdict": verdict,
    })
    log.info("D025 verdict: primary=%s secondary=%s overall=%s",
             primary["passes"], secondary["passes"], verdict)

    # ── Step 8: monotonicity classification ───────────────────────
    monotonicity_rows = []
    for cohort, bundle in per_cohort.items():
        cohort_meta = bundle["cohort_metadata"].set_index("pid")
        cohort_pids = cohort_meta.index.tolist()
        raw_for_cohort = raw_features.loc[
            raw_features.index.intersection(cohort_pids)
        ]
        agatston = cohort_meta.loc[raw_for_cohort.index, "agatston_total"]
        mono = compute_monotonicity(
            raw_for_cohort,
            feature_names=cross_cohort_reps,
            agatston=agatston, cohort=cohort,
            block_lookup=bundle["block_lookup"],
        )
        monotonicity_rows.append(mono)
    monotonicity_df = pd.concat(monotonicity_rows, ignore_index=True)
    monotonicity_df.to_csv(
        out_dir / "monotonicity_classification.csv", index=False,
    )
    classification_summary(monotonicity_df).to_csv(
        out_dir / "monotonicity_summary.csv", index=False,
    )

    # ── Step 9: cross-cohort 3-rule consistency ───────────────────
    consistency_df = consistency_table(per_cohort_profiles)
    consistency_df.to_csv(
        out_dir / "cross_cohort_feature_consistency.csv", index=False,
    )
    robust_discriminator_count_summary(consistency_df).to_csv(
        out_dir / "cross_cohort_robust_counts.csv", index=False,
    )
    n_robust = int(consistency_df["robust_discriminator"].sum())
    log.info("D027 robust discriminators across all 3 cohorts: %d", n_robust)

    # ── Step 10: partition ARI on shared pids ─────────────────────
    full_spatial = per_cohort["full"]["spatial_labels"]
    full_burden = (
        per_cohort["full"]["cohort_metadata"].set_index("pid")["agatston_total"]
    )
    from predict.analyse.paper_table import _agatston_tertile_labels
    full_burden_tertile = _agatston_tertile_labels(full_burden).astype(str)

    spatial_strata = {
        "Qr36d/2": per_cohort["Qr36d/2"]["spatial_labels"],
        "I30f/3": per_cohort["I30f/3"]["spatial_labels"],
    }
    burden_strata = {
        "Qr36d/2": _agatston_tertile_labels(
            per_cohort["Qr36d/2"]["cohort_metadata"].set_index("pid")["agatston_total"]
        ).astype(str),
        "I30f/3": _agatston_tertile_labels(
            per_cohort["I30f/3"]["cohort_metadata"].set_index("pid")["agatston_total"]
        ).astype(str),
    }

    ari_spatial = partition_ari_table(
        full_spatial, spatial_strata, partition="spatial_k2",
    )
    ari_burden = partition_ari_table(
        full_burden_tertile, burden_strata, partition="burden_k3",
    )
    pd.concat([ari_spatial, ari_burden], ignore_index=True).to_csv(
        out_dir / "cross_cohort_ari.csv", index=False,
    )

    # ── Step 11: main paper table ─────────────────────────────────
    hennig_lookup: dict = {}
    for cohort, bundle in per_cohort.items():
        raw_hennig = _hennig_lookup_from_validity(bundle["validity"], cohort)
        # Remap raw cluster ids to focal/diffuse using each cohort's focal_map
        focal_map_str = bundle["focal_map"]
        hennig_lookup.update(_remap_spatial_hennig(
            raw_hennig, focal_map_str, cohort,
        ))

    paper_inputs = {}
    for cohort, bundle in per_cohort.items():
        cohort_signatures = signatures[signatures["cohort"] == cohort]
        paper_inputs[cohort] = {
            "cohort_metadata": bundle["cohort_metadata"],
            "spatial_labels": bundle["spatial_labels"],
            "signatures": cohort_signatures,
        }
    paper_table = build_paper_table(paper_inputs, hennig_lookup=hennig_lookup)
    paper_table.to_csv(out_dir / "phenotype_paper_table.csv", index=False)
    log.info("paper table written: %d rows", len(paper_table))

    # ── Step 12: low-burden sensitivity (D028) ────────────────────
    robust_verdict = None
    if args.include_robust_sensitivity:
        full_bundle = per_cohort["full"]
        full_pids = full_bundle["cohort_metadata"]["pid"].tolist()
        full_ortho = next(r for r in ortho_rows if r["cohort"] == "full")
        sens = low_burden_sensitivity(
            raw_features.loc[raw_features.index.intersection(full_pids)],
            full_bundle["spatial_labels"],
            full_bundle["cohort_metadata"],
            feature_list,
            full_d024_interpretation=full_ortho["interpretation"],
            full_d025_primary_passes=primary["passes"],
            top_n=args.top_n_signatures,
        )
        sens["orthogonality"].to_csv(
            out_dir / "burden_orthogonality_robust_sensitivity.csv", index=False)
        sens["hypotheses"].to_csv(
            out_dir / "directional_hypotheses_robust_sensitivity.csv", index=False)
        sens["profiles"].to_csv(
            out_dir / "cluster_profiles_robust_sensitivity.csv", index=False)
        sens["signatures"].to_csv(
            out_dir / "signature_features_robust_sensitivity.csv", index=False)
        robust_verdict = sens["verdict"]
        _save_json(out_dir / "robust_sensitivity_verdict.json", robust_verdict)
        robust_table = build_robust_sensitivity_table(
            paper_inputs["full"], hennig_lookup=hennig_lookup,
            robust_cohort_label=ROBUST_LABEL, robust_signatures=sens["signatures"],
        )
        robust_table.to_csv(
            out_dir / "phenotype_paper_table_robust_sensitivity.csv",
            index=False,
        )
        log.info("low-burden sensitivity: N=%d, D024 %s (full %s), D025 primary %d/6 "
                 "(same as full: %s)", robust_verdict["n_patients"],
                 robust_verdict["d024_interpretation"], full_ortho["interpretation"],
                 robust_verdict["d025_primary"]["n_confirmed"],
                 robust_verdict["d025_primary_same_as_full"])

    # ── Step 13: run header ───────────────────────────────────────
    header = {
        "stage": "analyse",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": code_revision(repo_root),
        "config_yaml_sha": _file_sha(repo_root / "configs" / "default.yaml"),
        "args": vars(args),
        "n_cross_cohort_robust_features": int(len(cross_cohort_reps)),
        "n_feature_bundle": int(len(feature_list)),
        "directional_verdict": verdict,
        "n_robust_discriminators_D027": n_robust,
        "biological_sanity_per_cohort": {
            cohort: bundle["sanity_info"]
            for cohort, bundle in per_cohort.items()
        },
        "seam_sha_per_cohort": {
            cohort: bundle["seam_sha"]
            for cohort, bundle in per_cohort.items()
        },
        "inputs": inputs,
        "robust_sensitivity_verdict": robust_verdict,
    }
    for mod in ("numpy", "pandas", "scipy", "sklearn", "statsmodels"):
        try:
            m = __import__(mod)
            header[f"{mod}_version"] = getattr(m, "__version__", "unknown")
        except Exception:
            header[f"{mod}_version"] = "n/a"
    _save_json(out_dir / "run_header_analyse.json", header)

    # Inputs must still be the ones verified at the start.
    if stage7_inputs(cfg) != inputs:
        raise SystemExit("hard stop: stage 7 inputs changed during the run; nothing published")
    write_marker(out_dir, {
        "stage": 7,
        "stage7_fingerprint": stage7_fingerprint(),
        "inputs": inputs,
        "settings": settings,
        "outputs": files_manifest(out_dir, [f.name for f in out_dir.iterdir() if f.is_file()]),
        "revision": code_revision(repo_root),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, name=STAGE7_MARKER)
    publish(out_dir, target)
    log.info("stage 7 (analyse) complete. outputs in %s", target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
