"""scripts/08_analyse.py + 08d safeguards.

A small synthetic project (features.csv and three cohort folders with
stage 5 / stage 6 files and completion records) is run through the real
stage 7. Covered: a complete run publishes 07_analyse/ with a valid
COMPLETE_stage7.json and the D028 sensitivity outputs; a failed run keeps
the previous complete result; a features.csv other than the one stage 5
read is refused by stage 7 and by every consumer; a stage 5 record without
the features hash is refused; the report prints the verdicts the files hold
and none of the old hard-coded numbers.
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

from predict.analyse.stage7 import STAGE7_MARKER, require_stage7_complete, stage7_inputs
from predict.completion import file_sha, files_manifest, write_marker
from predict.config import DEFAULT_CONFIG_PATH, load_config
from predict.discover.stage6 import STAGE5_MARKER, STAGE6_MARKER, stage6_fingerprint

ROOT = Path(__file__).resolve().parents[2]
VESSELS = ("lad", "rca", "lcx", "lm")
SPATIAL = ("lesion_count_lad", "lesion_count_rca", "lesion_count_lcx", "lesion_count_lm",
           "lesion_count_total", "n_calcified_arteries", "gini_lesion_volume",
           "dist_from_top_max", "dist_from_top_mean", "center_of_mass_z",
           "inter_lesion_dist_mean_lad", "inter_lesion_dist_max_lad", "first_to_last_dist_lad")
COHORTS = {"full": ("", range(0, 80)), "Qr36d/2": ("stratified_Qr36d_2", range(0, 40)),
           "I30f/3": ("stratified_I30f_3", range(40, 80))}


def _load(name: str, script: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _features(n: int = 80) -> pd.DataFrame:
    """Odd pids diffuse (more arteries, more burden), even pids focal."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        diffuse = i % 2 == 1
        n_art = int(rng.integers(3, 5)) if diffuse else int(rng.integers(1, 3))
        per_vessel = {v: (float(rng.gamma(2.0, 150.0)) if k < n_art else 0.0)
                      for k, v in enumerate(VESSELS)}
        row = {"pid": str(i), "kernel": "Qr36d/2" if i < 40 else "I30f/3",
               "low_burden_flag": i % 4 == 0,
               "agatston_total": sum(per_vessel.values()),
               **{f"agatston_{v}": a for v, a in per_vessel.items()},
               **{f"n_rois_d{t}_{v}": int(rng.integers(0, 5)) for t in (1, 2, 3, 4) for v in VESSELS},
               "max_hu_global": float(rng.uniform(300, 900))}
        for f in SPATIAL:
            row[f] = float(rng.normal(5.0, 1.0))
        row["n_calcified_arteries"] = n_art
        row["lesion_count_lad"] = int(rng.integers(0, 3)) if diffuse else int(rng.integers(3, 7))
        row["dist_from_top_max"] = float(rng.uniform(60, 120) if diffuse else rng.uniform(5, 40))
        row["first_to_last_dist_lad"] = float(rng.uniform(40, 90) if diffuse else rng.uniform(0, 20))
        row["gini_lesion_volume"] = float(rng.uniform(0.0, 0.3) if diffuse else rng.uniform(0.5, 0.9))
        rows.append(row)
    return pd.DataFrame(rows)


def _project(tmp_path: Path, record_features_hash: bool = True):
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["paths"] = {"data_raw": str(tmp_path / "data" / "raw"),
                    "data_xml": str(tmp_path / "data" / "xml"),
                    "outputs": str(tmp_path / "outputs")}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(cfg_path)

    feats = _features()
    fdir = tmp_path / "outputs" / "03_features"
    fdir.mkdir(parents=True)
    feats.to_csv(fdir / "features.csv", index=False)
    reps = ["lesion_count_total", "max_hu_global"]
    root = tmp_path / "outputs" / "06_reduce"
    for cohort, (sub, idx) in COHORTS.items():
        cdir = root / sub if sub else root
        cdir.mkdir(parents=True, exist_ok=True)
        part = feats.iloc[list(idx)]
        part[["pid", "kernel", "low_burden_flag", "agatston_total"]].to_csv(
            cdir / "cohort_metadata.csv", index=False)
        pd.DataFrame({"feature": reps}).to_csv(cdir / "representative_features.csv", index=False)
        pd.DataFrame({"feature": reps, "block": ["spatial", "hu"]}).to_csv(
            cdir / "multi_block_assignments.csv", index=False)
        if cohort == "full":
            pd.DataFrame({"feature": ["original_glcm_Idm"], "kernel_r2_pre": [0.03],
                          "kernel_r2_post": [0.004], "passes_threshold": [True]}).to_csv(
                cdir / "combat_audit.csv", index=False)
        stage5 = sorted(f.name for f in cdir.iterdir() if f.is_file())
        rec5 = {"stage": 5, "files": files_manifest(cdir, stage5)}
        if record_features_hash:
            rec5["features_csv_sha"] = file_sha(fdir / "features.csv")
        write_marker(cdir, rec5, name=STAGE5_MARKER)

        labels = (part["pid"].astype(int) % 2).to_numpy()
        pd.DataFrame({"pid": part["pid"], "spatial_only_gmm_k2": labels}).to_csv(
            cdir / "cluster_labels_spatial_k2.csv", index=False)
        pd.DataFrame({"pid": part["pid"], "forced_k3_kmeans": labels}).to_csv(
            cdir / "cluster_labels_forced.csv", index=False)
        val = [{"test": "hennig_clusterboot", "algorithm": "gmm", "k": 2, "feature_space": "spatial_only",
                "cluster_id": c, "jaccard_median": 0.9, "jaccard_mean": 0.88, "stable": True} for c in (0, 1)]
        if cohort == "full":
            val.append({"test": "kernel_chi_square", "algorithm": "kmeans", "k": 3, "pval": 0.2,
                        "kernel_status": "pass", "p_cluster_given_burden": np.nan})
        pd.DataFrame(val).to_csv(cdir / "validity_checks.csv", index=False)
        (cdir / "hopkins.json").write_text(json.dumps({"H": 0.7, "n_total": len(part), "n_features": 3}))
        pd.DataFrame({"feature_space": ["full", "spatial_only"], "algorithm": ["kmeans", "gmm"],
                      "selected_k": [11, 3]}).to_csv(cdir / "gap_statistic_summary.csv", index=False)
        stage6 = ["cluster_labels_spatial_k2.csv", "cluster_labels_forced.csv", "validity_checks.csv",
                  "hopkins.json", "gap_statistic_summary.csv"]
        write_marker(cdir, {"stage": 6, "stage5_marker_sha": file_sha(cdir / STAGE5_MARKER),
                            "stage6_fingerprint": stage6_fingerprint(cfg),
                            "outputs": files_manifest(cdir, stage6)}, name=STAGE6_MARKER)
    return cfg_path, cfg, tmp_path / "outputs" / "07_analyse"


def _run(mod, monkeypatch, cfg_path):
    monkeypatch.setattr(sys, "argv", ["08_analyse.py", "--config", str(cfg_path)])
    return mod.main()


def test_complete_run_publishes_a_valid_record(tmp_path, monkeypatch):
    cfg_path, cfg, adir = _project(tmp_path)
    assert _run(_load("s08", "08_analyse.py"), monkeypatch, cfg_path) == 0
    marker = require_stage7_complete(adir, cfg)
    assert marker["inputs"] == stage7_inputs(cfg)
    assert not (tmp_path / "outputs" / "07_analyse.partial").exists()
    for f in ("burden_orthogonality_robust_sensitivity.csv", "directional_hypotheses_robust_sensitivity.csv",
              "signature_features_robust_sensitivity.csv", "robust_sensitivity_verdict.json",
              "phenotype_paper_table_robust_sensitivity.csv"):
        assert f in marker["outputs"], f
    verdict = json.loads((adir / "robust_sensitivity_verdict.json").read_text())
    assert verdict["n_patients"] == 60            # low_burden_flag False: 3 of every 4 pids
    rt = pd.read_csv(adir / "phenotype_paper_table_robust_sensitivity.csv").fillna("")
    assert (rt.loc[rt["partition"] == "spatial_k2", "top_signature_features"] != "").any()


def test_failed_run_keeps_the_previous_complete_result(tmp_path, monkeypatch):
    cfg_path, cfg, adir = _project(tmp_path)
    assert _run(_load("s08", "08_analyse.py"), monkeypatch, cfg_path) == 0
    before = (adir / STAGE7_MARKER).read_text()
    mod2 = _load("s08b", "08_analyse.py")

    def boom(*_a, **_k):
        raise RuntimeError("deliberate failure after the profiles")

    monkeypatch.setattr(mod2, "top_n_signatures", boom)
    with pytest.raises(RuntimeError, match="deliberate"):
        _run(mod2, monkeypatch, cfg_path)
    assert (adir / STAGE7_MARKER).read_text() == before
    require_stage7_complete(adir, cfg)              # previous result still complete and valid


def test_changed_features_csv_is_refused(tmp_path, monkeypatch):
    cfg_path, cfg, adir = _project(tmp_path)
    mod = _load("s08", "08_analyse.py")
    assert _run(mod, monkeypatch, cfg_path) == 0
    fcsv = tmp_path / "outputs" / "03_features" / "features.csv"
    df = pd.read_csv(fcsv, dtype={"pid": str})
    df.loc[0, "max_hu_global"] += 100.0
    df.to_csv(fcsv, index=False)
    with pytest.raises(SystemExit, match="is not the file stage 5 read"):
        require_stage7_complete(adir, cfg)        # consumers (report, stage 8, figures)
    with pytest.raises(SystemExit, match="is not the file stage 5 read"):
        _run(_load("s08c", "08_analyse.py"), monkeypatch, cfg_path)


def test_stage5_record_without_features_hash_is_refused(tmp_path, monkeypatch):
    cfg_path, cfg, _ = _project(tmp_path, record_features_hash=False)
    with pytest.raises(SystemExit, match="recorded nothing"):
        _run(_load("s08", "08_analyse.py"), monkeypatch, cfg_path)


def test_changed_output_or_missing_record_is_refused(tmp_path, monkeypatch):
    cfg_path, cfg, adir = _project(tmp_path)
    adir.mkdir(parents=True)
    with pytest.raises(SystemExit, match=STAGE7_MARKER):
        require_stage7_complete(adir, cfg)
    adir.rmdir()
    assert _run(_load("s08", "08_analyse.py"), monkeypatch, cfg_path) == 0
    prof = adir / "cluster_profiles.csv"
    prof.write_text(prof.read_text() + "\n")
    with pytest.raises(SystemExit, match="changed"):
        require_stage7_complete(adir, cfg)


def test_report_prints_the_verdicts_the_files_hold(tmp_path, monkeypatch, capsys):
    cfg_path, cfg, adir = _project(tmp_path)
    assert _run(_load("s08", "08_analyse.py"), monkeypatch, cfg_path) == 0
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["08d_stage7_results.py", "--config", str(cfg_path)])
    assert _load("s08d", "08d_stage7_results.py").main() == 0
    out = capsys.readouterr().out
    verdict = json.loads((adir / "directional_verdict.json").read_text())
    assert f"OVERALL VERDICT: {verdict['overall_verdict'].upper()}" in out
    if verdict["overall_verdict"] != "refuted":
        assert "REFUTED" not in out
    if verdict["primary"]["passes"]:
        assert "Passing D025 does not establish burden" in out or "D024 and D025 as listed" in out
    for stale in ("H = 0.717", "R^2 < 0.0025", "p = 0.048", "N=420", "doubly confirmed",
                  "monotonic to k=12", "PRESERVED from stage 6"):
        assert stale not in out, stale
    assert "descriptive outputs of the specified procedure" in out
    assert "Hopkins H 0.700" in out              # read from the cohort's hopkins.json
