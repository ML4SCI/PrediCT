#!/usr/bin/env python
"""Stage 4b, ICC computation and feature gate (D013, D016).

Reads:

  - ``outputs/03_features/features.csv``           baseline (stage 3 row per patient)
  - ``outputs/04_perturbations/{perturbation}.csv`` 14 per-perturbation CSVs

For each PyRadiomics feature, builds the 422-by-15 reliability matrix
(baseline + 14 raters) over the eligible cohort (D015) and computes ICC(3,1)
absolute agreement (D013). The 68 canonical features bypass the empirical
gate with ICC = 1.0 by construction (D016).

Writes:

  - ``outputs/05_icc/icc_report.csv``      one row per feature with columns
                                            feature, icc, icc_source,
                                            n_subjects, n_raters, passes_gate.
  - ``outputs/05_icc/gated_features.csv``  one column ``feature``, the list
                                            of features that pass D013's
                                            threshold (default 0.75).
  - ``outputs/05_icc/icc_summary.json``    aggregate counts and threshold info.

Exits non-zero if any of the following invariants are violated:
  * empirical feature count != 107
  * bypass feature count != 68
  * the union does not equal the schema (175 = 68 + 107) for non-metadata cols
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from predict.completion import file_sha, require_marker
from predict.config import Config, load_config
from predict.features.feature_schema import feature_names
from predict.stability.icc import (
    IccRecord,
    build_reliability_matrix,
    gate_features,
    icc_3_1_absolute,
    invariant_by_construction_features,
)
from predict.stability.perturbations import enumerate_perturbations
from predict.publish import fresh_partial, publish


METADATA_COLUMNS: set[str] = {
    "pid", "kernel", "scanner_model", "mask_voxels", "low_burden_flag",
    "roundtrip_quality", "category", "radiomics_status", "radiomics_reason",
}


def _load_script(name: str, alias: str):
    import importlib.util
    path = Path(__file__).resolve().parent / name
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_baseline(features_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(features_csv, dtype={"pid": str})
    if "radiomics_status" not in df.columns:
        raise KeyError(f"{features_csv} missing 'radiomics_status'.")
    eligible = df[df["radiomics_status"] == "ok"].copy()
    return eligible


def _load_perturbation_csvs(
    pert_dir: Path,
    perturbation_names: list[str],
    expected_pids: set[str],
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name in perturbation_names:
        path = pert_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing perturbation CSV: {path}. Run scripts/04_perturbations.py first."
            )
        df = pd.read_csv(path, dtype={"pid": str})
        missing = expected_pids - set(df["pid"].astype(str))
        if missing:
            raise ValueError(
                f"{path.name} missing {len(missing)} eligible pids "
                f"(first few: {sorted(missing)[:5]}). Rerun stage 4a."
            )
        # A row with an error has NaN features; the ICC would silently drop
        # that patient. Errors are failures, never missing data.
        if "error" in df.columns:
            err = df["error"].fillna("").astype(str).str.strip() != ""
            bad = sorted(set(df.loc[err, "pid"].astype(str)) & expected_pids)
            if bad:
                raise ValueError(
                    f"{path.name} has extraction errors for {len(bad)} eligible pids "
                    f"(first few: {bad[:5]}). Fix and rerun stage 4a."
                )
        out[name] = df
    return out


def _identify_empirical_features(
    baseline: pd.DataFrame,
    perturbation_dfs: dict[str, pd.DataFrame],
) -> list[str]:
    """All ``original_*`` columns present in baseline and every perturbation CSV."""
    pyr = sorted(c for c in baseline.columns if c.startswith("original_"))
    for name, df in perturbation_dfs.items():
        present = set(c for c in df.columns if c.startswith("original_"))
        missing = set(pyr) - present
        if missing:
            raise KeyError(
                f"perturbation {name!r} missing PyRadiomics columns: "
                f"{sorted(missing)[:5]} (and {max(0, len(missing) - 5)} more)"
            )
    return pyr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("icc")

    cfg: Config = load_config(args.config)
    threshold = float(cfg.stability.icc_threshold)
    publish_dir = cfg.paths.outputs / "05_icc"
    features_csv = cfg.paths.outputs / "03_features" / "features.csv"
    # Stage 4a must have finished cleanly on the current features.csv, the
    # current stage-2 folder and the current stage 1-4 code/config.
    stage4 = _load_script("04_perturbations.py", "_prod_stage4a")
    require_marker(cfg.paths.outputs / "04_perturbations", {
        "features_csv_sha": file_sha(features_csv),
        "run_fingerprint": stage4.run_fingerprint(cfg, stage4._eligible_pids(features_csv)),
    })
    out_dir = fresh_partial(publish_dir)

    # ── Load inputs ──────────────────────────────────────────────
    baseline = _load_baseline(features_csv)
    eligible_pids = set(baseline["pid"].astype(str))
    log.info("Eligible cohort (D015): N=%d", len(eligible_pids))

    specs = enumerate_perturbations(cfg)
    perturbation_names = [s.name for s in specs]
    perturbation_dfs = _load_perturbation_csvs(
        cfg.paths.outputs / "04_perturbations",
        perturbation_names,
        eligible_pids,
    )
    log.info("Perturbations loaded: %d", len(perturbation_dfs))

    # ── Build the registry split ─────────────────────────────────
    bypass = list(invariant_by_construction_features())   # 68 canonical
    empirical = _identify_empirical_features(baseline, perturbation_dfs)
    log.info("Bypass features (D016): %d | Empirical features: %d",
             len(bypass), len(empirical))

    # Invariants per D016.
    if len(bypass) != 68:
        raise SystemExit(f"D016 violation: expected 68 bypass features, got {len(bypass)}")
    if len(empirical) != 107:
        raise SystemExit(f"D016 violation: expected 107 empirical features, got {len(empirical)}")

    canonical_schema = set(feature_names())
    if set(bypass) != canonical_schema:
        diff = canonical_schema.symmetric_difference(bypass)
        raise SystemExit(f"D016 violation: bypass != schema. Symmetric diff: {sorted(diff)[:10]}")

    # ── Empirical ICC ────────────────────────────────────────────
    log.info("Computing ICC(3,1) absolute agreement on %d empirical features...",
             len(empirical))

    records: list[IccRecord] = []
    n_used_summary: list[int] = []   # rows actually used in each ICC

    for feat in empirical:
        matrix, pids, raters = build_reliability_matrix(
            feat, baseline, perturbation_dfs, pid_col="pid",
        )
        n_complete = int(np.sum(~np.isnan(matrix).any(axis=1))) if matrix.size else 0
        icc = icc_3_1_absolute(matrix) if matrix.size else float("nan")
        records.append(IccRecord(
            feature=feat,
            icc=float(icc),
            icc_source="empirical",
            n_subjects=n_complete,
            n_raters=int(matrix.shape[1]) if matrix.size else 0,
            passes_gate=False,
        ))
        n_used_summary.append(n_complete)

    # ── Bypass records ───────────────────────────────────────────
    for feat in bypass:
        records.append(IccRecord(
            feature=feat,
            icc=1.0,
            icc_source="invariant_by_construction",
            n_subjects=0,
            n_raters=0,
            passes_gate=False,
        ))

    # ── Apply D013 gate ──────────────────────────────────────────
    records, passing = gate_features(records, threshold=threshold)

    # ── Reports ──────────────────────────────────────────────────
    report_df = pd.DataFrame([
        {
            "feature": r.feature,
            "icc": round(r.icc, 6) if not np.isnan(r.icc) else "NaN",
            "icc_source": r.icc_source,
            "n_subjects": r.n_subjects,
            "n_raters": r.n_raters,
            "passes_gate": r.passes_gate,
        }
        for r in records
    ])
    # Stable order: bypass first (alphabetical), then empirical (alphabetical).
    report_df["_sort_key"] = report_df["icc_source"].map(
        {"invariant_by_construction": 0, "empirical": 1}
    )
    report_df = report_df.sort_values(["_sort_key", "feature"]).drop(columns=["_sort_key"])
    report_df.to_csv(out_dir / "icc_report.csv", index=False)

    pd.DataFrame({"feature": passing}).to_csv(out_dir / "gated_features.csv", index=False)

    # Per-source pass counts.
    by_source: dict[str, dict[str, int]] = {"invariant_by_construction": {}, "empirical": {}}
    for r in records:
        bucket = by_source[r.icc_source]
        bucket["total"] = bucket.get("total", 0) + 1
        if r.passes_gate:
            bucket["passing"] = bucket.get("passing", 0) + 1

    empirical_iccs = [r.icc for r in records
                      if r.icc_source == "empirical" and not np.isnan(r.icc)]
    summary = {
        "threshold": threshold,
        "n_features_total": len(records),
        "n_features_passing": len(passing),
        "by_source": by_source,
        "empirical_icc_stats": {
            "min": float(min(empirical_iccs)) if empirical_iccs else None,
            "median": float(np.median(empirical_iccs)) if empirical_iccs else None,
            "max": float(max(empirical_iccs)) if empirical_iccs else None,
            "mean": float(np.mean(empirical_iccs)) if empirical_iccs else None,
            "n_nan": sum(1 for r in records
                         if r.icc_source == "empirical" and np.isnan(r.icc)),
        },
        "n_subjects_per_feature": {
            "min": int(min(n_used_summary)) if n_used_summary else None,
            "median": int(np.median(n_used_summary)) if n_used_summary else None,
            "max": int(max(n_used_summary)) if n_used_summary else None,
        },
    }
    (out_dir / "icc_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )

    log.info("ICC gate complete.")
    log.info("  threshold: %.2f", threshold)
    log.info("  total features: %d", len(records))
    log.info("  passing: %d (bypass %d + empirical %d)",
             len(passing),
             by_source["invariant_by_construction"].get("passing", 0),
             by_source["empirical"].get("passing", 0))
    publish(out_dir, publish_dir)
    log.info("Outputs in %s", publish_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
