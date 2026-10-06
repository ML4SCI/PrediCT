"""Stage 9 safeguards (D043).

Covered: the chain root (PREDICT_STAGE9_ROOT) and the read-only published
folder; begin / finish publication with COMPLETE_stage9.json; finish refuses
an incomplete chain and a matched comparison next to a terminated match;
the LAD finaliser refuses stale comparisons and never reports replication
without results; LAD balance is measured on the compared population; the
lesion stability C8 Jaccard counts lesions created by regrouping and the
baseline parity is enforced.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import predict.stage9 as stage9
from predict.config import DEFAULT_CONFIG_PATH, load_config

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    path = ROOT / rel
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod      # dataclasses with postponed annotations need it
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(path.parent))
    return mod


def _cfg(tmp_path: Path):
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["paths"] = {"data_raw": str(tmp_path / "data" / "raw"),
                    "data_xml": str(tmp_path / "data" / "xml"),
                    "outputs": str(tmp_path / "outputs")}
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p, load_config(p)


# ─────────────────────── root and publication ───────────────────────


def test_stage9_root_guards(tmp_path, monkeypatch):
    _, cfg = _cfg(tmp_path)
    monkeypatch.setenv(stage9.ROOT_ENV, "somewhere_else")
    with pytest.raises(SystemExit, match="allowed values"):
        stage9.stage9_root(cfg)
    monkeypatch.setenv(stage9.ROOT_ENV, stage9.STAGE9_PARTIAL)
    with pytest.raises(SystemExit, match="begin"):
        stage9.stage9_root(cfg)
    pub = tmp_path / "outputs" / stage9.STAGE9_DIR
    pub.mkdir(parents=True)
    (pub / stage9.STAGE9_MARKER).write_text("{}")
    monkeypatch.delenv(stage9.ROOT_ENV)
    with pytest.raises(SystemExit, match="read-only"):
        stage9.stage9_root(cfg)


def _fill_required(root: Path, lad_comparison_with_infeasible: bool = False) -> None:
    for rel in stage9.REQUIRED_OUTPUTS:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    for sub in ("", "stratified_Qr36d_2", "stratified_I30f_3"):
        d = root / "lad_phenotype" / sub if sub else root / "lad_phenotype"
        (d / "matched_diagnostics.json").write_text(json.dumps({"verdict": "match_infeasible"}))
    if lad_comparison_with_infeasible:
        (root / "lad_phenotype" / "matched_comparison_summary.json").write_text(
            json.dumps({"overall_pass": True, "interpretation": "LAD-bias survives burden matching"}))


def _run_stage9(mod, monkeypatch, cfg_path, action):
    monkeypatch.setattr(sys, "argv", ["09_stage9.py", action, "--config", str(cfg_path)])
    return mod.main()


def test_begin_finish_publishes_and_refuses_bad_chains(tmp_path, monkeypatch):
    cfg_path, cfg = _cfg(tmp_path)
    fake_inputs = {"stage7_record_sha": "x"}
    monkeypatch.setattr(stage9, "stage9_inputs", lambda cfg: fake_inputs)
    mod = _load("s09_stage9", "scripts/09_stage9.py")
    monkeypatch.setattr(mod, "stage9_inputs", lambda cfg: fake_inputs)
    partial = tmp_path / "outputs" / stage9.STAGE9_PARTIAL
    target = tmp_path / "outputs" / stage9.STAGE9_DIR

    assert _run_stage9(mod, monkeypatch, cfg_path, "begin") == 0
    with pytest.raises(SystemExit, match="incomplete"):
        _run_stage9(mod, monkeypatch, cfg_path, "finish")      # nothing produced
    _fill_required(partial, lad_comparison_with_infeasible=True)
    with pytest.raises(SystemExit, match="match_infeasible"):
        _run_stage9(mod, monkeypatch, cfg_path, "finish")      # stale comparison
    assert not target.exists()

    (partial / "lad_phenotype" / "matched_comparison_summary.json").unlink()
    assert _run_stage9(mod, monkeypatch, cfg_path, "finish") == 0
    assert not partial.exists()
    marker = stage9.require_stage9_complete(cfg)
    assert "lad_phenotype/lad_cluster_signature.json" in marker["outputs"]

    extra = target / "lesion_morphology" / "stray.csv"
    extra.write_text("x")
    with pytest.raises(SystemExit, match="changed, were added"):
        stage9.require_stage9_complete(cfg)


# ─────────────────────── LAD finaliser ───────────────────────


def _lad_cohort(d: Path, ids, verdict="match_infeasible", comparison=None, axial=None):
    d.mkdir(parents=True, exist_ok=True)
    (d / "lad_cluster_signature.json").write_text(json.dumps({"lad_dominant_cluster_ids": ids}))
    (d / "matched_diagnostics.json").write_text(json.dumps({"verdict": verdict}))
    if comparison is not None:
        (d / "matched_comparison_summary.json").write_text(json.dumps(comparison))
    if axial is not None:
        (d / "axial_summary.json").write_text(json.dumps(axial))


def test_lad_finalise_refuses_a_stale_comparison(tmp_path):
    fin = _load("s_lad_finalise", "experiments/lad_phenotype/finalise.py")
    _lad_cohort(tmp_path, [3], verdict="match_infeasible",
                comparison={"overall_pass": True, "interpretation": "LAD-bias survives burden matching"})
    with pytest.raises(SystemExit, match="match_infeasible"):
        fin._gather_one(tmp_path)


def test_lad_replication_needs_results_and_stratum_signatures(tmp_path):
    fin = _load("s_lad_finalise2", "experiments/lad_phenotype/finalise.py")
    ax = {"passes": True, "median_diff": 0.2}
    _lad_cohort(tmp_path, [], axial=None)
    for s in fin.STRATA:
        _lad_cohort(tmp_path / f"stratified_{s}", [])
    bundle = fin._gather(tmp_path)
    rep = fin._compare_replication(bundle["full"], {s: bundle[s] for s in fin.STRATA})
    assert rep["overall_replicates"] is None            # no signature: never True

    _lad_cohort(tmp_path, [3], axial=ax)
    _lad_cohort(tmp_path / "stratified_Qr36d_2", [3], axial=ax)
    _lad_cohort(tmp_path / "stratified_I30f_3", [5], axial=None)   # other signature in the stratum
    bundle = fin._gather(tmp_path)
    rep = fin._compare_replication(bundle["full"], {s: bundle[s] for s in fin.STRATA})
    assert rep["signature_replication"]["status"] == "does_not_replicate"
    assert rep["overall_replicates"] is False


@pytest.mark.parametrize("full_pass, strata_pass, expected", [
    (False, True, "does_not_replicate"),    # verdict reversed in the strata
    (False, False, "replicates"),           # negative verdict preserved
    (True, True, "replicates"),
    (True, False, "does_not_replicate"),
])
def test_lad_strata_are_judged_against_the_full_verdict(tmp_path, full_pass, strata_pass, expected):
    """Strata were judged on their own pass, so a
    reversed verdict counted as replicated and a preserved negative verdict
    as not replicated."""
    fin = _load(f"s_lad_finalise_{full_pass}_{strata_pass}", "experiments/lad_phenotype/finalise.py")
    cmp_full = {"overall_pass": full_pass, "interpretation": "x"}
    cmp_s = {"overall_pass": strata_pass, "interpretation": "x"}
    # axial effects far apart, so only the verdict can make them agree
    _lad_cohort(tmp_path, [3], verdict="match_ok", comparison=cmp_full,
                axial={"passes": full_pass, "median_diff": 0.30 if full_pass else 0.0})
    for s in fin.STRATA:
        _lad_cohort(tmp_path / f"stratified_{s}", [3], verdict="match_ok", comparison=cmp_s,
                    axial={"passes": strata_pass, "median_diff": 0.30 if strata_pass else 0.0})
    bundle = fin._gather(tmp_path)
    rep = fin._compare_replication(bundle["full"], {s: bundle[s] for s in fin.STRATA})
    assert rep["matched_replication"]["status"] == expected
    assert rep["axial_replication"]["status"] == expected


def test_lad_axial_tolerance_still_applies(tmp_path):
    """Plan P5's alternative: a stratum effect within +/-0.10 of the full
    effect replicates even if the stratum's own verdict differs."""
    fin = _load("s_lad_finalise_tol", "experiments/lad_phenotype/finalise.py")
    _lad_cohort(tmp_path, [3], axial={"passes": True, "median_diff": 0.12})
    for s in fin.STRATA:
        _lad_cohort(tmp_path / f"stratified_{s}", [3], axial={"passes": False, "median_diff": 0.05})
    bundle = fin._gather(tmp_path)
    rep = fin._compare_replication(bundle["full"], {s: bundle[s] for s in fin.STRATA})
    assert rep["axial_replication"]["status"] == "replicates"


def test_lad_balance_is_measured_on_the_compared_population():
    """With variable 1:k matching the pair-weighted SMD was
    0.000 while the compared populations differed by 0.408."""
    prop = _load("s_propensity", "experiments/lad_phenotype/propensity.py")
    cases = pd.Series([0.0, 1.0], index=["c0", "c1"])
    controls = pd.Series([0.0, 0.0, 0.0, 1.0], index=["x0", "x1", "x2", "x3"])
    res = prop.caliper_match(cases, controls, caliper_sd=0.2, k=3)
    assert res.pairs.groupby("case_pid").size().to_dict() == {"c0": 3, "c1": 1}
    assert res.smd_post_pair_weighted == pytest.approx(0.0)
    assert res.smd_post == pytest.approx(0.4082, abs=1e-3)


# ─────────────────────── lesion stability ───────────────────────


def test_c8_jaccard_counts_lesions_created_by_regrouping():
    stab = _load("s_lesion_stab", "experiments/lesion_morphology/lesion_stability_d033.py")
    base_c8 = {("7", "LAD", "0")}
    draw = pd.DataFrame({"pid": ["7", "7"], "vessel": ["LAD", "LAD"], "lesion_idx": [0, -1],
                         "cluster_draw": [4, 4]})
    assert stab.c8_jaccard(base_c8, draw, 4) == pytest.approx(0.5)
    assert stab.c8_jaccard(base_c8, draw.iloc[:1], 4) == pytest.approx(1.0)


def test_baseline_parity_is_enforced_on_every_clustered_input():
    stab = _load("s_lesion_stab2", "experiments/lesion_morphology/lesion_stability_d033.py")
    base = pd.DataFrame({"pid": ["7"], "vessel": ["LAD"], "lesion_idx": [0], "n_rois": [3],
                         "total_area_mm2": [5.0], "volume_mm3": [10.0],
                         "mean_hu_weighted": [300.0], "max_hu": [500.0]})
    assert stab.require_baseline_parity(base, base.copy())["n_lesions"] == 1
    with pytest.raises(SystemExit, match="max_hu"):
        stab.require_baseline_parity(base, base.assign(max_hu=520.0))
    with pytest.raises(SystemExit, match="keys differ"):
        stab.require_baseline_parity(base, base.assign(lesion_idx=1))
