"""scripts/06_reduce.py safeguards.

1. A failed run leaves the published cohort folder untouched (no mix of old
   and new files); sub-cohort folders survive a primary publish.
4. A gated feature missing from features.csv stops stage 5.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from predict.publish import publish, publish_keeping_subdirs

ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("scripts_06_reduce", ROOT / "scripts" / "06_reduce.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _setup(tmp_path: Path, monkeypatch, gated: list[str], features: list[str]):
    mod = _load()
    out = tmp_path / "outputs"
    (out / "03_features").mkdir(parents=True)
    gate = out / "05b_icc_realistic"
    gate.mkdir(parents=True)
    df = pd.DataFrame({"pid": ["1", "2"], "kernel": ["Qr36d/2", "Qr36d/2"],
                       "radiomics_status": ["ok", "ok"], **{f: [1.0, 2.0] for f in features}})
    df.to_csv(out / "03_features" / "features.csv", index=False)
    pd.DataFrame({"feature": gated}).to_csv(gate / "gated_features.csv", index=False)
    pd.DataFrame({"feature": gated, "icc": 0.9, "icc_source": "empirical"}).to_csv(
        gate / "icc_report.csv", index=False)
    pub = out / "06_reduce"
    pub.mkdir()
    (pub / "pca_scores.csv").write_text("OLD\n")
    (pub / "stratified_Qr36d_2").mkdir()
    cfg = SimpleNamespace(
        paths=SimpleNamespace(outputs=out),
        stability=SimpleNamespace(primary_gate="realistic"),
        raw={"reduce": {}, "hopkins": {"sample_frac": 0.1}},
        reduce=SimpleNamespace(variance_threshold=0.01),
    )
    monkeypatch.setattr(mod, "load_config", lambda *_a, **_k: cfg)
    monkeypatch.setattr(sys, "argv", ["06_reduce.py"])
    return mod, pub


def test_missing_gated_feature_stops_stage5(tmp_path, monkeypatch):
    mod, pub = _setup(tmp_path, monkeypatch,
                      gated=["original_glcm_Contrast", "agatston_total"],
                      features=["agatston_total"])
    with pytest.raises(SystemExit, match="gated feature"):
        mod.main()
    assert (pub / "pca_scores.csv").read_text() == "OLD\n"


def test_failed_run_leaves_published_files_untouched(tmp_path, monkeypatch):
    mod, pub = _setup(tmp_path, monkeypatch, gated=["agatston_total"], features=["agatston_total"])
    (tmp_path / "outputs" / "05b_icc_realistic" / "derived_icc.csv").write_text(
        "feature,icc,icc_source,passes_gate\n"
        "has_dense_calcium,1.0,invariant_by_construction,True\n"
        "high_density_fraction,0.99,empirical,True\n"
        "vessel_burden_gini,0.99,empirical,True\n")

    def boom(*_a, **_k):
        raise RuntimeError("deliberate failure after inputs were read")

    monkeypatch.setattr(mod, "run_matrix_prep", boom)
    with pytest.raises(RuntimeError, match="deliberate"):
        mod.main()
    assert (pub / "pca_scores.csv").read_text() == "OLD\n"
    assert not (pub / "run_header.json").exists()          # new file not published
    assert (pub / "stratified_Qr36d_2").is_dir()


def test_publish_keeping_subdirs_swaps_whole_folder_and_keeps_cohorts(tmp_path):
    target = tmp_path / "06_reduce"
    (target / "stratified_I30f_3").mkdir(parents=True)
    (target / "stratified_I30f_3" / "pca_scores.csv").write_text("stratum")
    (target / "stratified_Qr36d_2.partial").mkdir()                 # interrupted run
    (target / "multi_block_summary.json").write_text("old multi-block")   # stale stage 5 file
    (target / "gap_statistic.json").write_text("old stage 6")             # stale stage 6 file
    (target / "pca_scores.csv").write_text("old")
    partial = tmp_path / "06_reduce.partial"
    partial.mkdir()
    (partial / "pca_scores.csv").write_text("new")
    publish_keeping_subdirs(partial, target)
    assert (target / "pca_scores.csv").read_text() == "new"
    assert not (target / "multi_block_summary.json").exists()
    assert not (target / "gap_statistic.json").exists()
    assert (target / "stratified_I30f_3" / "pca_scores.csv").read_text() == "stratum"
    assert not (target / "stratified_Qr36d_2.partial").exists()
    assert not partial.exists()
    assert not (tmp_path / "06_reduce.old").exists()


def test_publish_recovers_from_a_crash_between_the_two_renames(tmp_path):
    """Target missing, .old present = the last complete version; the next
    publish restores it instead of deleting it."""
    target = tmp_path / "05b_icc_realistic"
    old = tmp_path / "05b_icc_realistic.old"
    old.mkdir()
    (old / "gated_features.csv").write_text("last complete")
    partial = tmp_path / "05b_icc_realistic.partial"
    partial.mkdir()
    (partial / "gated_features.csv").write_text("new")
    publish(partial, target)
    assert (target / "gated_features.csv").read_text() == "new"
    assert not old.exists()
