#!/usr/bin/env python
"""Stage 7 results presentation report.

Reads the stage 7 outputs from ``outputs/07_analyse/`` (and, for the
synthesis, the stage 5-6 outputs they were built from) and prints:

  1. Cohort sizes + biological sanity status per cohort
  2. Burden orthogonality verdict (D024)
  3. Directional hypothesis verdict (D025) + per-hypothesis table
  4. Burden-stratified spatial replication (D024 part 2)
  5. Monotonicity classification of the cross-cohort features (D026)
  6. Cross-cohort feature consistency (D027)
  7. Phenotype paper table and the low-burden sensitivity (D028)
  8. Synthesis: every number and verdict read from the current files

Read-only. It first requires a complete, current stage 7 record
(``require_stage7_complete``), which also verifies the stage 6 records and
features.csv. Nothing is hard-coded: every number and verdict is read from
the stage 6 and 7 outputs. Gap and Hopkins results are reported
descriptively (D021.1).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from predict.analyse.stage7 import COHORT_DIRS, cohort_dir, require_stage7_complete
from predict.config import load_config

D021_1_READING = (
    "All gap-statistic results are descriptive outputs of the specified procedure. "
    "Selected k, including k = 1 or selection near the upper boundary, is not treated "
    "as evidence establishing or excluding discrete subtypes (D021.1).")


def _section(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def _subsection(title: str) -> None:
    print()
    print("-" * 80)
    print(title)
    print("-" * 80)


def _fmt_p(p: float) -> str:
    return "n/a" if pd.isna(p) else f"{p:.3g}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg.paths.outputs / "07_analyse"
    reduce_root = cfg.paths.outputs / "06_reduce"
    marker = require_stage7_complete(out_dir, cfg)

    header = json.loads((out_dir / "run_header_analyse.json").read_text())
    ortho = pd.read_csv(out_dir / "burden_orthogonality.csv")
    verdict = json.loads((out_dir / "directional_verdict.json").read_text())
    hypotheses = pd.read_csv(out_dir / "directional_hypotheses.csv")
    stratified = pd.read_csv(out_dir / "burden_stratified_spatial.csv")
    monotonicity = pd.read_csv(out_dir / "monotonicity_classification.csv")
    mono_summary = pd.read_csv(out_dir / "monotonicity_summary.csv")
    consistency = pd.read_csv(out_dir / "cross_cohort_feature_consistency.csv")
    robust_counts = pd.read_csv(out_dir / "cross_cohort_robust_counts.csv")
    ari = pd.read_csv(out_dir / "cross_cohort_ari.csv")
    paper_table = pd.read_csv(out_dir / "phenotype_paper_table.csv")
    signatures = pd.read_csv(out_dir / "signature_features.csv")
    with_sens = marker["settings"]["include_robust_sensitivity"]

    print(f"stage 7 record verified (revision {marker['revision'][:7]}, "
          f"written {marker['written_utc']})")

    # ── 1. Cohort sizes + biological sanity ──────────────────────────
    _section("1. Cohort sizes and biological sanity status")
    sizes = (paper_table[paper_table["partition"] == "spatial_k2"]
             .groupby("cohort", sort=False)["N"].sum())
    print(sizes.rename("total_N").reset_index().to_string(index=False))
    print()
    sanity = header["biological_sanity_per_cohort"]
    print(pd.DataFrame([{
        "cohort": c, "focal_max_hu_median": i["focal_median"],
        "diffuse_max_hu_median": i["diffuse_median"], "ratio": i["ratio"],
        "floor_hu": i["absolute_floor_hu"], "passes": i["passes"],
        "low_ratio_warning": i["warning_low_ratio"]} for c, i in sanity.items()]).to_string(index=False))
    n_pass = sum(bool(i["passes"]) for i in sanity.values())
    print(f"\nCalcium floor gate (focal median max HU >= floor): passed in {n_pass} of {len(sanity)} cohorts.")
    warned = [c for c, i in sanity.items() if i["warning_low_ratio"]]
    if warned:
        print(f"Low focal/diffuse max-HU ratio (warning only): {', '.join(warned)}.")

    # ── 2. Burden orthogonality (D024) ───────────────────────────────
    _section("2. Burden orthogonality of the spatial-only k=2 partition (D024)")
    cols = ["cohort", "n_focal", "n_diffuse", "focal_median_agatston", "diffuse_median_agatston",
            "mannwhitney_pval", "levene_pval", "cliffs_delta_agatston", "interpretation", "passes"]
    print(ortho[cols].to_string(index=False))
    interps = set(ortho["interpretation"])
    print()
    print(f"VERDICT: {interps.pop().upper()} in every cohort." if len(interps) == 1
          else "VERDICT: mixed across cohorts: " + ", ".join(
              f"{r.cohort} {r.interpretation}" for r in ortho.itertuples()))
    print("Cliff's delta (focal vs diffuse, Agatston): " + ", ".join(
        f"{r.cohort} {r.cliffs_delta_agatston:+.3f}" for r in ortho.itertuples()))

    # ── 3. Directional hypotheses (D025) ─────────────────────────────
    _section("3. Directional hypothesis test (D025)")
    full_n = int(sizes.get("full", 0))
    p, s = verdict["primary"], verdict["secondary"]
    print("OVERALL VERDICT:", verdict["overall_verdict"].upper())
    print(f"\n  Primary (full, N={full_n}): {p['n_confirmed']}/{p['n_total']} confirmed "
          f"(direction match and FDR p < 0.05; >= {p['min_required']} needed) -> "
          f"{'PASS' if p['passes'] else 'FAIL'}")
    print(f"  Secondary (direction only, >= {s['min_required']} of 6 in each stratum): "
          f"Qr36d/2 {s['qr36d_2_match_count']}/6, I30f/3 {s['i30f_3_match_count']}/6 -> "
          f"{'PASS' if s['passes'] else 'FAIL'}")
    _subsection("Per-hypothesis breakdown (full cohort)")
    cols = ["feature", "predicted_direction", "focal_median", "diffuse_median", "observed_sign",
            "direction_match", "fdr_bh_pval", "confirmed"]
    print(hypotheses[hypotheses["cohort"] == "full"][cols].to_string(index=False))
    _subsection("Direction-only check (Qr36d/2 and I30f/3 strata)")
    cols = ["cohort", "feature", "predicted_direction", "observed_sign", "direction_match"]
    print(hypotheses[hypotheses["cohort"] != "full"][cols].to_string(index=False))

    # ── 4. Burden-stratified spatial replication (D024 part 2) ───────
    _section("4. Burden-stratified spatial replication (D024 part 2)")
    print("Within each Agatston tertile, the focal vs diffuse direction is checked for")
    print("the directional features.")
    for cohort in COHORT_DIRS:
        sub = stratified[stratified["cohort"] == cohort]
        if len(sub) == 0:
            continue
        _subsection(f"Cohort: {cohort}")
        n_feat = sub["feature"].nunique() if "feature" in sub.columns else None
        tm = sub.groupby("tertile")["direction_match"].sum().reset_index()
        tm = tm.rename(columns={"direction_match": f"n_matches_of_{n_feat}" if n_feat else "n_matches"})
        print(tm.to_string(index=False))

    # ── 5. Monotonicity (D026) ───────────────────────────────────────
    full_mono = monotonicity[monotonicity["cohort"] == "full"]
    _section(f"5. Monotonicity classification of the {len(full_mono)} cross-cohort features (D026)")
    print(mono_summary.to_string(index=False))
    _subsection("Per-feature classification (cohort=full)")
    cols = ["feature", "block", "spearman_rho", "kendall_tau", "classification"]
    print(full_mono[cols].sort_values(["classification", "spearman_rho"],
                                      ascending=[True, False]).to_string(index=False))

    # ── 6. Cross-cohort consistency (D027) ───────────────────────────
    _section("6. Cross-cohort feature consistency (D027 three-rule criterion)")
    print("Robust cross-cohort discriminator: direction consistent in all 3 cohorts, FDR p < 0.05")
    print("in >= 2 of 3, and |Cliff's delta| >= 0.20 in all 3.\n")
    print(robust_counts.to_string(index=False))
    n_robust = int(consistency["robust_discriminator"].sum())
    print(f"\nTOTAL robust discriminators: {n_robust} of {len(consistency)} "
          f"feature x partition x cluster comparisons")
    _subsection("Partition-level ARI on shared pids")
    print(ari.to_string(index=False))
    if not ari["passes"].all():
        print("\nARI below the pass threshold for:")
        print(ari[~ari["passes"]].to_string(index=False))

    # ── 7. Paper table + low-burden sensitivity (D028) ───────────────
    _section("7. Phenotype paper table (D028)")
    cols = ["cohort", "partition", "cluster", "N", "agatston_median", "agatston_iqr_lower",
            "agatston_iqr_upper", "pct_qr36d_2", "pct_low_burden", "hennig_jaccard_median",
            "pct_zero", "pct_cat_1_99", "pct_cat_100_399", "pct_cat_ge_400"]
    print(paper_table[cols].to_string(index=False))
    _subsection("Top-3 signature features per phenotype (full cohort)")
    cols = ["partition", "cluster", "rank", "feature", "cliffs_delta", "fdr_bh_pval"]
    print(signatures[(signatures["cohort"] == "full") & (signatures["rank"] <= 3)][cols].to_string(index=False))

    rv = None
    if with_sens:
        rv = json.loads((out_dir / "robust_sensitivity_verdict.json").read_text())
        _subsection(f"Low-burden sensitivity (low_burden_flag = False, N={rv['n_patients']}; "
                    "full-cohort spatial labels kept)")
        rt = pd.read_csv(out_dir / "phenotype_paper_table_robust_sensitivity.csv")
        print(rt[["cohort", "partition", "cluster", "N", "agatston_median", "pct_low_burden",
                  "top_signature_features"]].to_string(index=False))
        ro = pd.read_csv(out_dir / "burden_orthogonality_robust_sensitivity.csv").iloc[0]
        print(f"\n  D024: {ro['interpretation']} (Cliff's delta {ro['cliffs_delta_agatston']:+.3f}, "
              f"MW p {_fmt_p(ro['mannwhitney_pval'])}); full cohort {rv['d024_full_interpretation']} "
              f"-> {'same' if rv['d024_same_as_full'] else 'DIFFERENT'}")
        rp = rv["d025_primary"]
        print(f"  D025 primary rule: {rp['n_confirmed']}/{rp['n_total']} confirmed -> "
              f"{'PASS' if rp['passes'] else 'FAIL'}; full cohort "
              f"{'PASS' if rv['d025_full_primary_passes'] else 'FAIL'} -> "
              f"{'same' if rv['d025_primary_same_as_full'] else 'DIFFERENT'}")
        print(f"  Hennig: {rv['hennig']}")
    else:
        print("\nLow-burden sensitivity was not run (--no-include-robust-sensitivity).")

    # ── 8. Synthesis ─────────────────────────────────────────────────
    _section("8. Synthesis (read from the current stage 5-7 files)")
    print("\nCluster tendency and gap statistic (stage 6):")
    for cohort in COHORT_DIRS:
        cdir = cohort_dir(reduce_root, cohort)
        h = json.loads((cdir / "hopkins.json").read_text())
        g = pd.read_csv(cdir / "gap_statistic_summary.csv")
        print(f"  {cohort:<8} Hopkins H {h['H']:.3f} (n={h['n_total']}, d={h['n_features']}); "
              f"gap-selected k {int(g['selected_k'].min())}-{int(g['selected_k'].max())} over "
              f"{len(g)} space x algorithm runs")
    print(f"  {D021_1_READING}")
    print("  Hopkins measures departure from a uniform reference; it does not distinguish")
    print("  discrete groups from a skewed or continuous distribution.")

    print("\nKernel association at the forced k (stage 6, full cohort; D021.1 policy):")
    full_dir = cohort_dir(reduce_root, "full")
    v = pd.read_csv(full_dir / "validity_checks.csv")
    for r in v[v["test"] == "kernel_chi_square"].itertuples():
        print(f"  {r.algorithm:<7} chi-square p {_fmt_p(r.pval)}; burden-adjusted p "
              f"{_fmt_p(r.p_cluster_given_burden)} -> {r.kernel_status}")
    combat = full_dir / "combat_audit.csv"
    if combat.exists():
        ca = pd.read_csv(combat)
        print(f"  ComBat: {len(ca)} harmonised columns, max post-harmonisation kernel R^2 "
              f"{ca['kernel_r2_post'].max():.5f}")

    print("\nSpatial-only k=2 partition (stage 6 Hennig, stage 7 D024 and D025):")
    for cohort in COHORT_DIRS:
        rows = paper_table[(paper_table["cohort"] == cohort) & (paper_table["partition"] == "spatial_k2")]
        jac = ", ".join(f"{r.cluster} {r.hennig_jaccard_median:.3f}" for r in rows.itertuples())
        o = ortho[ortho["cohort"] == cohort].iloc[0]
        print(f"  {cohort:<8} Hennig median Jaccard {jac}; D024 {o['interpretation']} "
              f"(Cliff's delta {o['cliffs_delta_agatston']:+.3f})")
    d025 = verdict["overall_verdict"]
    print(f"  D025 overall: {d025} (primary {p['n_confirmed']}/{p['n_total']}; strata "
          f"{s['qr36d_2_match_count']}/6 and {s['i30f_3_match_count']}/6)")
    all_confounded = (ortho["interpretation"] == "confounded").all()
    if p["passes"] and all_confounded:
        print("  The pre-registered directional predictions pass, and the two groups also differ")
        print("  strongly in burden in every cohort. Passing D025 does not establish burden")
        print("  independence; both results are reported together.")
    elif not p["passes"] and all_confounded:
        print("  The pre-registered directional predictions are not supported, and the two groups")
        print("  differ strongly in burden in every cohort.")
    else:
        print("  D024 and D025 as listed above; see sections 2 and 3.")
    if rv is not None:
        print(f"  Low-burden sensitivity: D024 {'same' if rv['d024_same_as_full'] else 'different'}, "
              f"D025 primary {'same' if rv['d025_primary_same_as_full'] else 'different'} "
              "compared with the full cohort.")
    print(f"\n  D027 robust cross-cohort discriminators: {n_robust}; read them through the D026")
    print("  monotonicity classes (section 5).")
    print("\nNothing in this report establishes or excludes discrete subtypes or a continuum")
    print("(D021.1, D036.1, D038).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
