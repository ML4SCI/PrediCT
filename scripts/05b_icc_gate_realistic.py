#!/usr/bin/env python
"""Stage 5b, D033 gate: ICC(1,1) over the 20 primary composite draws.

Rules (decisions/D033-realistic-perturbation-set.md, locked before this ran):
  * ICC(1,1), one-way random, single measurement, over the 20 primary draws;
    the unperturbed baseline is not a draw and is excluded.
  * Pass: ICC >= stability.icc_threshold (0.75) AND at most 5% of the
    eligible cohort excluded for that feature (patients with any broken or
    undefined draw are excluded from that feature's ICC).
  * Canonical features with zero within-patient variance for every patient
    are tagged invariant_by_construction (D016 extended); all others are
    empirical and face the same gate.
  * Strata, secondary arms and attribution arms are reported, never gate.
  * Realism check against published Agatston variability is reported, never
    used to tune.

Hard stops: stage 4b's COMPLETE.json must exist and match the current
features.csv, calibration, stage 1-4 code/config (run fingerprint) and
every patient's stage-2 CT, mask, XML, features row and exclusions
(patients fingerprint); every eligible patient must have passed parity and have each
prescribed draw index exactly once in the primary arm and in every secondary
or attribution arm that 4b ran; no draw may have status "error" (a software
or input error, D033.1; only "broken" draws count toward the 5% rule).

--allow-partial (smoke only) skips these stops, never writes to the
production gate folder (default destination 05b_icc_realistic_smoke/) and
marks its output PARTIAL_SMOKE_DO_NOT_USE, which predict.config.gate_dir
refuses. A production run writes to 05b_icc_realistic.partial/ and replaces
05b_icc_realistic/ only when it finishes (predict.publish).

Outputs (outputs/05b_icc_realistic/):
    icc_report.csv, gated_features.csv   same format stage 6 reads from 05_icc/
    icc_report_full.csv   + CI, exclusions, fail reason
    derived_icc.csv       D040: the stage-5 transformed features
                          (has_dense_calcium, high_density_fraction,
                          vessel_burden_gini) computed per draw with stage 5's
                          own functions and gated with the same rule; stage 5
                          reads their ICC from here and admits only passing ones
    strata.csv, arms.csv, realism.csv, d014_comparison.csv, summary.txt
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
_log = logging.getLogger("stage5b")

MAX_EXCLUDED_FRAC = 0.05
DRAW_DTYPES = {"pid": str, "error": str}   # error is text or empty
PUBLISHED = {  # external rescan variability anchors (D033)
    "agatston_total": ("Detrano 2005 MESA rescan, Agatston", 20.1),
    "volume_total_mm3": ("Detrano 2005 MESA rescan, volume", 18.3),
    "mask_voxels": ("Detrano 2005 MESA rescan, volume", 18.3),
}
PUBLISHED_READER = ("Ghadri 2011 intra-observer CV, Agatston", 2.0)


def _load_script_module(rel_path: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def _family(feature: str) -> str:
    parts = feature.split("_", 2)
    return parts[1] if feature.startswith("original_") and len(parts) > 2 else "canonical"


def _matrix(draws: pd.DataFrame, feature: str, pids: list[str], n_draws: int) -> np.ndarray:
    ok = draws[["pid", "index", feature, "status"]].copy()
    vals = pd.to_numeric(ok[feature], errors="coerce")
    if not feature.startswith("original_"):
        # Canonical values are computed before extraction; only "error" rows lack them.
        vals = vals.where(ok["status"] != "error")
    else:
        vals = vals.where(ok["status"] == "ok")
    ok = ok.assign(v=vals)
    piv = ok.pivot_table(index="pid", columns="index", values="v", aggfunc="first", dropna=False)
    piv = piv.reindex(index=pids, columns=range(n_draws))
    return piv.to_numpy(dtype=float)


def _check_draws(d: pd.DataFrame, pids: list[str], n_draws: int, arm: str) -> list[str]:
    """Problems with an arm's draw table: each pid must have indices
    0..n_draws-1 exactly once, and no row may have status "error"."""
    problems = []
    errs = d.loc[d["status"] == "error", "pid"].astype(str)
    if len(errs):
        problems.append(f"{arm}: {len(errs)} error draws (pids {sorted(set(errs))[:5]})")
    want = set(range(n_draws))
    for pid in pids:
        idx = d.loc[d["pid"].astype(str) == pid, "index"].astype(int).tolist()
        if len(idx) != len(set(idx)):
            problems.append(f"{arm}: pid {pid} has duplicate draw indices")
        elif set(idx) != want:
            problems.append(f"{arm}: pid {pid} has {len(set(idx) & want)} of {n_draws} draws")
        if len(problems) >= 10:
            break
    return problems


def _gate_one(m: np.ndarray, feature: str, n_eligible: int, threshold: float) -> dict:
    from predict.stability.icc import icc_1_1
    complete = ~np.isnan(m).any(axis=1)
    n_ok = int(complete.sum())
    excl = n_eligible - n_ok
    excl_frac = excl / n_eligible if n_eligible else float("nan")
    source = "empirical"
    if not feature.startswith("original_") and n_ok and np.all(np.nanvar(m[complete], axis=1) == 0.0):
        source = "invariant_by_construction"
        icc, lo, hi = 1.0, 1.0, 1.0
    else:
        icc, lo, hi = icc_1_1(m)
    reason = ""
    if excl_frac > MAX_EXCLUDED_FRAC:
        reason = f"excluded {excl_frac:.1%} > {MAX_EXCLUDED_FRAC:.0%}"
    elif np.isnan(icc):
        reason = "icc undefined"
    elif icc < threshold:
        reason = f"icc {icc:.3f} < {threshold}"
    return {"feature": feature, "family": _family(feature), "icc": icc, "ci_low": lo, "ci_high": hi,
            "icc_source": source, "n_subjects": n_ok, "n_excluded": excl,
            "excluded_frac": excl_frac, "passes_gate": reason == "", "fail_reason": reason}


def _wcv(m: np.ndarray) -> float:
    c = m[~np.isnan(m).any(axis=1)]
    if len(c) < 2:
        return float("nan")
    mu = float(np.abs(c.mean()))
    return float(np.sqrt(c.var(axis=1, ddof=1).mean()) / mu) if mu > 0 else float("nan")


def _pairwise_rel_diff(m: np.ndarray) -> np.ndarray:
    """Per patient: mean over draw pairs of |a-b| / ((a+b)/2), in %, skipping 0/0 pairs."""
    out = []
    for row in m:
        v = row[~np.isnan(row)]
        if len(v) < 2:
            continue
        a, b = np.meshgrid(v, v, indexing="ij")
        iu = np.triu_indices(len(v), 1)
        a, b = a[iu], b[iu]
        s = (a + b) / 2.0
        keep = s != 0
        if keep.any():
            out.append(100.0 * float(np.mean(np.abs(a[keep] - b[keep]) / np.abs(s[keep]))))
    return np.asarray(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--allow-partial", action="store_true",
                    help="SMOKE ONLY: gate whatever patients are present. Output is "
                         "marked partial and must never feed stage 6.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from predict.completion import file_sha, require_marker
    from predict.config import load_config
    from predict.features.feature_schema import feature_names
    from predict.publish import fresh_partial, publish
    from predict.stability.icc import icc_1_1
    from predict.stability.realistic import (
        N_ATTRIBUTION_DRAWS, N_PRIMARY_DRAWS, PRIMARY_FACTORS, SECONDARY_FACTORS, arm_draw_plan,
    )

    cfg = load_config()
    outputs = cfg.paths.outputs
    in_dir = args.in_dir or outputs / "04b_realistic"
    production = outputs / "05b_icc_realistic"
    threshold = float(cfg.stability.icc_threshold)
    if args.allow_partial:
        target = args.out_dir or outputs / "05b_icc_realistic_smoke"
        if Path(target).resolve() == production.resolve():
            raise SystemExit("--allow-partial must not write to the production gate folder")
        out_dir = Path(target)
        out_dir.mkdir(parents=True, exist_ok=True)
        marker = None
    else:
        target = args.out_dir or production
        stage4b = _load_script_module("scripts/04b_realistic_perturbations.py", "_prod_stage4b")
        stage4a = _load_script_module("scripts/04_perturbations.py", "_prod_stage4")
        calib = outputs / "04b_calibration" / "calibration.json"
        features_csv = outputs / "03_features" / "features.csv"
        from predict.config import default_n_jobs
        # Inputs, calibration, stage 1-4 code/config AND every patient's
        # stage-2 CT, mask, XML, features row and exclusions must be what 4b
        # ran on (patient fingerprints recomputed with 4b's own function).
        _log.info("verifying stage 4b inputs (hashing patient CT, mask, XML)...")
        pfps = stage4b.patient_fingerprints(cfg, calib, stage4a._eligible_pids(features_csv),
                                            n_jobs=default_n_jobs())
        marker = require_marker(in_dir, {
            "features_csv_sha": file_sha(features_csv),
            "calibration_sha": file_sha(calib),
            "run_fingerprint": stage4b.run_fingerprint(cfg, calib),
            "patients_fingerprint": stage4b.combine_patient_fingerprints(pfps),
        })
        if "primary" not in marker.get("arms", []):
            raise SystemExit(f"hard stop: stage 4b COMPLETE.json lists no primary arm: {marker.get('arms')}")
        out_dir = fresh_partial(Path(target))

    stage4 = _load_script_module("scripts/04_perturbations.py", "_prod_stage4")
    features_csv = outputs / "03_features" / "features.csv"
    eligible = stage4._eligible_pids(features_csv)
    feats = pd.read_csv(features_csv, dtype={"pid": str}).set_index("pid")

    parity = pd.read_csv(in_dir / "parity.csv", dtype={"pid": str})
    primary = pd.read_csv(in_dir / "draws_primary.csv", dtype=DRAW_DTYPES)
    present = sorted(set(primary["pid"]), key=lambda s: (len(s), s))
    missing = sorted(set(eligible) - set(present))
    par_ok = set(parity.loc[parity["ok"].astype(bool), "pid"])
    par_bad = sorted(set(present) - par_ok)
    if (missing or par_bad) and not args.allow_partial:
        raise SystemExit(f"hard stop: {len(missing)} eligible pids missing from the primary arm "
                         f"(e.g. {missing[:5]}), {len(par_bad)} failed parity (e.g. {par_bad[:5]}). "
                         "See parity.csv.")
    pids = [p for p in eligible if p in set(present) and p in par_ok]
    n_elig = len(eligible) if not args.allow_partial else len(pids)
    problems = _check_draws(primary, pids, N_PRIMARY_DRAWS, "primary")
    if problems and not args.allow_partial:
        raise SystemExit("hard stop: " + "; ".join(problems))
    _log.info("gating %d patients (eligible %d)%s", len(pids), len(eligible),
              "  [PARTIAL, smoke only]" if args.allow_partial else "")

    radiomics = sorted(c for c in feats.columns if c.startswith("original_"))
    canonical = list(feature_names())
    if len(radiomics) != 107 or len(canonical) != 68:
        raise SystemExit(f"schema drift: {len(radiomics)} radiomics, {len(canonical)} canonical")
    all_feats = canonical + radiomics

    mats = {f: _matrix(primary, f, pids, N_PRIMARY_DRAWS) for f in all_feats}
    report = pd.DataFrame([_gate_one(mats[f], f, n_elig, threshold) for f in all_feats])
    out_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_dir / "icc_report_full.csv", index=False)
    compat = report.assign(icc=report["icc"].round(6), n_raters=N_PRIMARY_DRAWS)[
        ["feature", "icc", "icc_source", "n_subjects", "n_raters", "passes_gate"]]
    compat.to_csv(out_dir / "icc_report.csv", index=False)
    passing = report.loc[report["passes_gate"], "feature"].tolist()
    pd.DataFrame({"feature": passing}).to_csv(out_dir / "gated_features.csv", index=False)

    # ── D040: reliability of the stage-5 transformed features ──────
    from predict.reduce.prepare_matrix import STAGE5_TRANSFORMED, stage5_transformed_columns
    tf = stage5_transformed_columns(primary)
    missing_tf = [n for n in STAGE5_TRANSFORMED if n not in tf]
    if missing_tf:
        raise SystemExit(f"draws lack the inputs for transformed features {missing_tf}")
    prim_tf = primary.assign(**tf)
    derived = pd.DataFrame([_gate_one(_matrix(prim_tf, n, pids, N_PRIMARY_DRAWS), n, n_elig, threshold)
                            for n in STAGE5_TRANSFORMED]).assign(family="derived", n_raters=N_PRIMARY_DRAWS)
    derived.to_csv(out_dir / "derived_icc.csv", index=False)
    if args.allow_partial:
        (out_dir / "PARTIAL_SMOKE_DO_NOT_USE").write_text("smoke run\n", encoding="utf-8")

    # ── strata (reported only) ─────────────────────────────────
    strata_def = {}
    if "low_burden_flag" in feats.columns:
        strata_def["low_burden"] = feats["low_burden_flag"].astype(str).str.lower().isin(["true", "1", "1.0"])
    if "kernel" in feats.columns:
        strata_def["kernel"] = feats["kernel"].astype(str)
    rep = outputs / "02_preprocessed" / "preprocess_report.csv"
    if rep.exists():
        pr = pd.read_csv(rep, dtype={"pid": str}).set_index("pid")
        strata_def["metal"] = pr["metal_artifact_flag"].astype(bool)
    cal = outputs / "04b_calibration" / "calibration_patients.csv"
    if cal.exists():
        cp = pd.read_csv(cal, dtype={"pid": str}).set_index("pid")
        strata_def["spacing_tertile"] = pd.qcut(cp["native_spacing_mm"], 3, labels=["low", "mid", "high"]).astype(str)
    srows = []
    for sname, ser in strata_def.items():
        ser = ser.reindex(pids)
        for level in sorted(ser.dropna().unique(), key=str):
            idx = np.flatnonzero((ser == level).to_numpy())
            if len(idx) < 5:
                continue
            for f in all_feats:
                sub = mats[f][idx]
                icc, lo, hi = icc_1_1(sub)
                srows.append({"stratum": sname, "level": str(level), "n": len(idx), "feature": f,
                              "icc": icc, "ci_low": lo, "ci_high": hi, "wcv": _wcv(sub)})
    strata = pd.DataFrame(srows)
    strata.to_csv(out_dir / "strata.csv", index=False)

    # ── secondary and attribution arms (reported only; feed 05c tiers) ──
    # Only arms the clean 4b run covered (COMPLETE.json), each checked for its
    # prescribed draw count, never inferred from the largest index present.
    all_arms = [*SECONDARY_FACTORS, *[f"attr_{f}" for f in PRIMARY_FACTORS]]
    run_arms = all_arms if marker is None else [a for a in all_arms if a in marker["arms"]]
    arm_rows = []
    for arm in run_arms:
        path = in_dir / f"draws_{arm}.csv"
        if not path.exists():
            if marker is not None:
                raise SystemExit(f"hard stop: {path} missing although 4b ran arm {arm}")
            continue
        d = pd.read_csv(path, dtype=DRAW_DTYPES)
        n_d = len(arm_draw_plan([arm]))
        problems = _check_draws(d, pids, n_d, arm)
        if problems:
            if not args.allow_partial:
                raise SystemExit("hard stop: " + "; ".join(problems))
            _log.warning("partial arm %s: %s", arm, "; ".join(problems[:3]))
        for f in all_feats:
            r = _gate_one(_matrix(d, f, pids, n_d), f, n_elig, threshold)
            arm_rows.append({"arm": arm, **r})
    arms = pd.DataFrame(arm_rows)
    if len(arms):
        arms.to_csv(out_dir / "arms.csv", index=False)

    # ── realism check (reported only) ───────────────────────────
    real_rows = []
    for f in ("agatston_total", "volume_total_mm3", "mask_voxels"):
        m = _matrix(primary, f, pids, N_PRIMARY_DRAWS) if f != "mask_voxels" else \
            primary.pivot_table(index="pid", columns="index", values="mask_voxels").reindex(
                index=pids, columns=range(N_PRIMARY_DRAWS)).to_numpy(float)
        rd = _pairwise_rel_diff(m)
        src, pub = PUBLISHED[f]
        real_rows.append({"arm": "primary", "measure": f, "n": len(rd),
                          "mean_pct": float(rd.mean()) if len(rd) else np.nan,
                          "median_pct": float(np.median(rd)) if len(rd) else np.nan,
                          "published_pct": pub, "published_source": src,
                          "expected": "<= published (primary omits z-phase and motion)"})
    # D033 also asks for S1 (z-phase), the factor the primary arm omits.
    s1 = in_dir / "draws_S1.csv"
    if s1.exists() and "S1" in run_arms:
        d = pd.read_csv(s1, dtype=DRAW_DTYPES)
        for f in ("agatston_total", "volume_total_mm3"):
            rd = _pairwise_rel_diff(_matrix(d, f, pids, len(arm_draw_plan(["S1"]))))
            src, pub = PUBLISHED[f]
            real_rows.append({"arm": "S1", "measure": f, "n": len(rd),
                              "mean_pct": float(rd.mean()) if len(rd) else np.nan,
                              "median_pct": float(np.median(rd)) if len(rd) else np.nan,
                              "published_pct": pub, "published_source": src,
                              "expected": "closer to published than primary (adds z-phase)"})
    m1 = in_dir / "draws_attr_M1.csv"
    if m1.exists() and "attr_M1" in run_arms:
        d = pd.read_csv(m1, dtype=DRAW_DTYPES)
        rd = _pairwise_rel_diff(_matrix(d, "agatston_total", pids, N_ATTRIBUTION_DRAWS))
        real_rows.append({"arm": "attr_M1", "measure": "agatston_total", "n": len(rd),
                          "mean_pct": float(rd.mean()) if len(rd) else np.nan,
                          "median_pct": float(np.median(rd)) if len(rd) else np.nan,
                          "published_pct": PUBLISHED_READER[1], "published_source": PUBLISHED_READER[0],
                          "expected": "comparable to reader variability"})
    realism = pd.DataFrame(real_rows)
    realism.to_csv(out_dir / "realism.csv", index=False)

    # ── D014 comparison (supplementary) ─────────────────────────
    d014 = outputs / "05_icc" / "gated_features.csv"
    cmp_rows = []
    if d014.exists():
        old = set(pd.read_csv(d014)["feature"])
        new = set(passing)
        for f in sorted(old | new):
            cmp_rows.append({"feature": f, "d014": f in old, "d033": f in new})
    pd.DataFrame(cmp_rows).to_csv(out_dir / "d014_comparison.csv", index=False)

    # ── summary ─────────────────────────────────────────────────
    L = ["=" * 84, "D033 GATE: ICC(1,1) over 20 primary composite draws"
         + ("   [PARTIAL SMOKE, DO NOT USE]" if args.allow_partial else ""), "=" * 84,
         f"patients gated {len(pids)} of eligible {len(eligible)}; threshold {threshold}; "
         f"max excluded {MAX_EXCLUDED_FRAC:.0%}", ""]
    by = report.groupby("family")["passes_gate"].agg(["sum", "size"])
    L.append(f"PASS {int(report['passes_gate'].sum())} / {len(report)}   "
             f"(canonical {int(report.loc[report.family == 'canonical', 'passes_gate'].sum())}/68, "
             f"radiomics {int(report.loc[report.family != 'canonical', 'passes_gate'].sum())}/107)")
    for fam, r in by.iterrows():
        L.append(f"  {fam:<12}{int(r['sum']):>4} / {int(r['size'])}")
    inv = int((report["icc_source"] == "invariant_by_construction").sum())
    L.append(f"canonical invariant_by_construction (zero within-patient variance): {inv}")
    L.append("stage-5 transformed features (D040, reported for stage 5): " + ", ".join(
        f"{r.feature} {r.icc:.4f}{'' if r.passes_gate else ' FAIL'}" for r in derived.itertuples()))
    L.append(f"features failing on exclusions: {int(report['fail_reason'].str.startswith('excluded').sum())}")
    L.append("")
    L.append("Realism check (mean pairwise relative difference, %):")
    for r in realism.itertuples():
        L.append(f"  {r.arm:<9}{r.measure:<18}{r.mean_pct:>7.1f} (median {r.median_pct:.1f})   "
                 f"published {r.published_pct} [{r.published_source}]  expected {r.expected}")
    if len(arms):
        L.append("")
        L.append("Arms (reported only): pass count")
        for arm, g in arms.groupby("arm", sort=False):
            L.append(f"  {arm:<10}{int(g['passes_gate'].sum()):>4} / {len(g)}")
    if cmp_rows:
        c = pd.DataFrame(cmp_rows)
        L.append("")
        L.append(f"D014 (supplementary): both {int((c.d014 & c.d033).sum())}, "
                 f"D033 only {int((~c.d014 & c.d033).sum())}, D014 only {int((c.d014 & ~c.d033).sum())}")
    L.append("=" * 84)
    text = "\n".join(L)
    (out_dir / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    if not args.allow_partial:
        publish(out_dir, Path(target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
