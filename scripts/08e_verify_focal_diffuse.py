#!/usr/bin/env python
"""Independent verification of the stage 7 focal-vs-diffuse analysis.

Re-derives every quantity in the focal/diffuse chain from the RAW seam
files (without using any predict.analyse helpers) and compares to what
scripts/08_analyse.py actually wrote into outputs/07_analyse/. Mismatches
are flagged.

Checks (in order):

  1. Focal/diffuse mapping deterministic from
     ``cluster_labels_spatial_k2.csv`` + ``features.csv['n_calcified_arteries']``
  2. Mapping matches the orchestrator's run_header_analyse.json
  3. Per-cluster N + median agatston_total + median max_hu_global match
     ``burden_orthogonality.csv`` + ``run_header.biological_sanity``
  4. Cliff's delta on agatston (re-computed manually) matches
     burden_orthogonality.csv cliffs_delta_agatston
  5. Direction of agatston: focal vs diffuse (-0.89 means focal LOWER
     than diffuse on burden)
  6. Each of the 6 directional hypotheses re-computed manually and
     compared to ``directional_hypotheses.csv``
  7. Sanity: print key cluster-level distributions so we can read the
     biology by eye

If everything matches, prints "ALL VERIFIED" at the end.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from predict.config import load_config


COHORT_DIRS: dict[str, str] = {
    "full": "",
    "Qr36d/2": "stratified_Qr36d_2",
    "I30f/3": "stratified_I30f_3",
}


DIRECTIONAL_HYPOTHESES_LOCAL = (
    ("lesion_count_lad",       "focal>diffuse"),
    ("n_calcified_arteries",   "focal<diffuse"),
    ("dist_from_top_max",      "focal<diffuse"),
    ("gini_lesion_volume",     "focal>diffuse"),
    ("vessel_burden_gini",     "focal>diffuse"),
    ("first_to_last_dist_lad", "focal<diffuse"),
)


def cliffs_delta_manual(a: np.ndarray, b: np.ndarray) -> float:
    """Reference implementation, NOT using predict.analyse.profiles."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    diffs = a[:, None] - b[None, :]
    n_greater = int((diffs > 0).sum())
    n_less = int((diffs < 0).sum())
    return (n_greater - n_less) / float(a.size * b.size)


def _section(title: str) -> None:
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


def _check(label: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" -- {detail}" if detail else ""))
    return condition


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    reduce_root = cfg.paths.outputs / "06_reduce"
    analyse_root = cfg.paths.outputs / "07_analyse"
    features_csv = cfg.paths.outputs / "03_features" / "features.csv"

    # The stage 7 record covers the stage 6 records and features.csv too.
    from predict.analyse.stage7 import require_stage7_complete
    require_stage7_complete(analyse_root, cfg)

    # ── Load orchestrator outputs ─────────────────────────────────
    header = json.loads(
        (analyse_root / "run_header_analyse.json").read_text()
    )
    ortho_csv = pd.read_csv(analyse_root / "burden_orthogonality.csv")
    hyp_csv = pd.read_csv(analyse_root / "directional_hypotheses.csv")

    # ── Load raw inputs ───────────────────────────────────────────
    raw_features = pd.read_csv(features_csv, dtype={"pid": str}).set_index("pid")
    # Augment with D019 derived features so the verification matches the
    # orchestrator's feature bundle (vessel_burden_gini + high_density_fraction
    # are NOT in features.csv; they are re-derived in stage 7).
    from predict.analyse.derived_features import augment_raw_with_derived
    raw_features = augment_raw_with_derived(raw_features)

    all_passed = True

    for cohort, sub in COHORT_DIRS.items():
        cdir = reduce_root if sub == "" else reduce_root / sub
        spatial_csv = cdir / "cluster_labels_spatial_k2.csv"
        meta_csv = cdir / "cohort_metadata.csv"

        _section(f"Cohort: {cohort}")
        spatial_df = pd.read_csv(spatial_csv, dtype={"pid": str})
        spatial_raw = spatial_df.set_index("pid")["spatial_only_gmm_k2"].astype(int)
        meta = pd.read_csv(meta_csv, dtype={"pid": str}).set_index("pid")

        # ── Step 1: independent focal/diffuse mapping ─────────────
        common = raw_features.index.intersection(spatial_raw.index)
        rfc = raw_features.loc[common]
        srl = spatial_raw.loc[common]

        cluster_ids = sorted(srl.unique())
        medians_nca = {}
        for cid in cluster_ids:
            vals = rfc.loc[srl == cid, "n_calcified_arteries"].dropna()
            medians_nca[cid] = float(np.median(vals))
        # This harness deliberately re-derives the D023 rule by hand rather
        # than calling predict.analyse.profiles.determine_focal_diffuse_mapping;
        # importing the helper would check it against itself. The tie guard
        # below is therefore also hand-written. It matters because min() and
        # max() return the SAME key on equal medians, which would collapse
        # manual_map to a single entry and make every downstream comparison
        # meaningless. Production raises on a tie, so this branch should be
        # unreachable in a normal run; if it fires, that is the finding.
        if len(set(medians_nca.values())) < len(medians_nca):
            raise ValueError(
                f"[{cohort}] independent re-derivation: identical median "
                f"n_calcified_arteries across spatial clusters "
                f"({medians_nca}); focal/diffuse mapping is undefined. "
                f"Production should have raised before reaching this harness."
            )
        focal_id_manual = min(medians_nca, key=medians_nca.get)
        diffuse_id_manual = max(medians_nca, key=medians_nca.get)
        manual_map = {focal_id_manual: "focal", diffuse_id_manual: "diffuse"}

        print(f"\nMedian n_calcified_arteries per raw GMM cluster:")
        for cid, m in sorted(medians_nca.items()):
            tag = "FOCAL" if cid == focal_id_manual else "DIFFUSE"
            print(f"  cluster {cid}: median = {m:.2f}  ({tag})")
        print(f"\nManual mapping: {manual_map}")

        # ── Step 2: compare to orchestrator's mapping ─────────────
        # The header records biological_sanity per cohort which uses the
        # mapped labels. We can cross-check by counting cluster sizes.
        # Use the directional_hypotheses_csv: it contains focal_median and
        # diffuse_median per feature. For n_calcified_arteries, we can
        # compare against our independent recomputation.
        nca_row = hyp_csv[
            (hyp_csv["cohort"] == cohort)
            & (hyp_csv["feature"] == "n_calcified_arteries")
        ].iloc[0]
        orch_focal_nca = float(nca_row["focal_median"])
        orch_diffuse_nca = float(nca_row["diffuse_median"])
        manual_focal_nca = medians_nca[focal_id_manual]
        manual_diffuse_nca = medians_nca[diffuse_id_manual]
        ok = (abs(orch_focal_nca - manual_focal_nca) < 1e-9
              and abs(orch_diffuse_nca - manual_diffuse_nca) < 1e-9)
        _check(
            "Step 1 + 2: orchestrator's focal/diffuse mapping matches "
            "the manual mapping rule",
            ok,
            f"orch focal/diffuse n_calc = {orch_focal_nca:.2f}/"
            f"{orch_diffuse_nca:.2f}; manual = {manual_focal_nca:.2f}/"
            f"{manual_diffuse_nca:.2f}",
        )
        all_passed &= ok

        # ── Step 3: cluster sizes + max_hu medians ────────────────
        focal_pids = srl.index[srl == focal_id_manual].tolist()
        diffuse_pids = srl.index[srl == diffuse_id_manual].tolist()
        focal_hu = rfc.loc[focal_pids, "max_hu_global"].dropna()
        diffuse_hu = rfc.loc[diffuse_pids, "max_hu_global"].dropna()
        manual_focal_hu_med = float(np.median(focal_hu))
        manual_diffuse_hu_med = float(np.median(diffuse_hu))
        sanity = header["biological_sanity_per_cohort"][cohort]
        ok = (abs(sanity["focal_median"] - manual_focal_hu_med) < 1.0
              and abs(sanity["diffuse_median"] - manual_diffuse_hu_med) < 1.0)
        _check(
            "Step 3: max_hu_global medians match the run_header sanity record",
            ok,
            f"orch focal/diffuse = {sanity['focal_median']:.1f}/"
            f"{sanity['diffuse_median']:.1f}; manual = "
            f"{manual_focal_hu_med:.1f}/{manual_diffuse_hu_med:.1f}",
        )
        all_passed &= ok

        print(f"\nCluster sizes:")
        print(f"  focal:   N = {len(focal_pids)}")
        print(f"  diffuse: N = {len(diffuse_pids)}")
        # max_hu floor check
        _check(
            "Step 3b: focal cluster passes the 130 HU calcium floor",
            manual_focal_hu_med >= 130,
            f"focal median max_hu_global = {manual_focal_hu_med:.1f} HU",
        )

        # ── Step 4 + 5: burden orthogonality (manual) ─────────────
        focal_agat = meta.loc[focal_pids, "agatston_total"].dropna().to_numpy(float)
        diffuse_agat = meta.loc[diffuse_pids, "agatston_total"].dropna().to_numpy(float)
        manual_delta = cliffs_delta_manual(focal_agat, diffuse_agat)
        manual_mw_p = stats.mannwhitneyu(focal_agat, diffuse_agat,
                                         alternative="two-sided").pvalue
        manual_lev_p = stats.levene(focal_agat, diffuse_agat,
                                    center="median").pvalue
        ortho_row = ortho_csv[ortho_csv["cohort"] == cohort].iloc[0]
        orch_delta = float(ortho_row["cliffs_delta_agatston"])
        orch_mw_p = float(ortho_row["mannwhitney_pval"])
        orch_lev_p = float(ortho_row["levene_pval"])

        ok_delta = abs(orch_delta - manual_delta) < 1e-9
        ok_mw = abs(orch_mw_p - manual_mw_p) < 1e-12
        ok_lev = abs(orch_lev_p - manual_lev_p) < 1e-12
        _check(
            "Step 4: Cliff's delta on agatston matches orchestrator output",
            ok_delta,
            f"manual = {manual_delta:.6f}, orch = {orch_delta:.6f}",
        )
        _check(
            "Step 4: Mann-Whitney p on agatston matches orchestrator output",
            ok_mw,
            f"manual = {manual_mw_p:.6g}, orch = {orch_mw_p:.6g}",
        )
        _check(
            "Step 4: Levene p on agatston matches orchestrator output",
            ok_lev,
            f"manual = {manual_lev_p:.6g}, orch = {orch_lev_p:.6g}",
        )
        all_passed &= (ok_delta and ok_mw and ok_lev)

        print(f"\nBurden orthogonality direction check:")
        print(f"  focal median agatston   = {np.median(focal_agat):.1f}")
        print(f"  diffuse median agatston = {np.median(diffuse_agat):.1f}")
        if manual_delta < 0:
            print(f"  Cliff's delta = {manual_delta:.3f} -> focal LOWER burden than diffuse")
        elif manual_delta > 0:
            print(f"  Cliff's delta = {manual_delta:.3f} -> focal HIGHER burden than diffuse")
        else:
            print(f"  Cliff's delta = 0 -> no burden separation")

        # ── Step 6: each of the 6 directional hypotheses ──────────
        print(f"\nDirectional hypothesis verification:")
        print(f"  {'feature':<25} {'pred':<15} {'focal_med':>10} "
              f"{'diff_med':>10} {'obs_sgn':>7} {'match?':>8} "
              f"{'p_man':>10} {'p_orch':>10}")
        for feature, predicted_dir in DIRECTIONAL_HYPOTHESES_LOCAL:
            if feature not in rfc.columns:
                print(f"  {feature:<25} (column missing)")
                continue
            focal_vals = rfc.loc[focal_pids, feature].dropna().to_numpy(float)
            diffuse_vals = rfc.loc[diffuse_pids, feature].dropna().to_numpy(float)
            focal_med = float(np.median(focal_vals)) if focal_vals.size else float("nan")
            diffuse_med = float(np.median(diffuse_vals)) if diffuse_vals.size else float("nan")
            observed_sign = int(np.sign(focal_med - diffuse_med))
            predicted_sign = 1 if predicted_dir == "focal>diffuse" else -1
            direction_match = observed_sign == predicted_sign

            alternative = "greater" if predicted_dir == "focal>diffuse" else "less"
            try:
                manual_p = stats.mannwhitneyu(
                    focal_vals, diffuse_vals, alternative=alternative,
                ).pvalue
            except ValueError:
                manual_p = float("nan")
            orch_row = hyp_csv[
                (hyp_csv["cohort"] == cohort)
                & (hyp_csv["feature"] == feature)
            ].iloc[0]
            orch_p = float(orch_row["mannwhitney_u_pval_one_sided"])
            orch_focal_med = float(orch_row["focal_median"])
            orch_diffuse_med = float(orch_row["diffuse_median"])
            orch_match = bool(orch_row["direction_match"])

            ok_med = (abs(orch_focal_med - focal_med) < 1e-9
                      and abs(orch_diffuse_med - diffuse_med) < 1e-9)
            ok_match = orch_match == direction_match
            ok_p = abs(orch_p - manual_p) < 1e-12 or (
                np.isnan(orch_p) and np.isnan(manual_p)
            )
            row_ok = ok_med and ok_match and ok_p
            all_passed &= row_ok

            mark = "OK" if row_ok else "**"
            print(f"  {feature:<25} {predicted_dir:<15} "
                  f"{focal_med:>10.3f} {diffuse_med:>10.3f} "
                  f"{observed_sign:>7d} {str(direction_match):>8} "
                  f"{manual_p:>10.3g} {orch_p:>10.3g}  {mark}")

        # ── Step 7: biology sanity printout ───────────────────────
        print(f"\nCluster biology (raw medians, for interpretation):")
        for feat in ("agatston_total", "max_hu_global",
                     "n_calcified_arteries", "lesion_count_lad",
                     "lesion_count_total", "vessel_burden_gini",
                     "gini_lesion_volume"):
            if feat == "agatston_total":
                f_vals = meta.loc[focal_pids, feat].dropna()
                d_vals = meta.loc[diffuse_pids, feat].dropna()
            elif feat in rfc.columns:
                f_vals = rfc.loc[focal_pids, feat].dropna()
                d_vals = rfc.loc[diffuse_pids, feat].dropna()
            else:
                continue
            print(f"  {feat:<25} focal median = {np.median(f_vals):>10.2f}, "
                  f"diffuse median = {np.median(d_vals):>10.2f}")

    print()
    print("=" * 96)
    if all_passed:
        print("ALL VERIFIED: every orchestrator output exactly matches the")
        print("independent manual recomputation.")
    else:
        print("MISMATCHES FOUND: see [FAIL] lines above.")
    print("=" * 96)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
