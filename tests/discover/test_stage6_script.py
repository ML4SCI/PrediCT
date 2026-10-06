"""scripts/07_discover.py publication safeguards.

A tiny synthetic cohort folder (stage 5 files + COMPLETE_stage5.json) is
run through the real stage 6 with production settings shrunk for speed.
Covered: a complete run publishes with a valid COMPLETE_stage6.json and keeps
the stage 5 files and other cohort folders; a failed run keeps the previous
complete result; a smoke run cannot publish; stages 7 and 8 (one helper)
refuse a missing or stale record.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from predict.completion import files_manifest, write_marker
from predict.config import DEFAULT_CONFIG_PATH, load_config
from predict.discover.stage6 import STAGE5_MARKER, STAGE6_MARKER, require_stage6_complete

ROOT = Path(__file__).resolve().parents[2]
SMALL = {"k_range": [1, 2, 3], "gap_bootstraps": 3, "consensus_subsamples": 3,
         "hennig_bootstraps": 3, "random_state": 0, "forced_k": 2}
SPATIAL = ["lesion_count_total", "n_calcified_arteries", "dist_from_top_max"]


def _load():
    spec = importlib.util.spec_from_file_location("scripts_07_discover", ROOT / "scripts" / "07_discover.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cohort(tmp_path: Path):
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["paths"] = {"data_raw": str(tmp_path / "data" / "raw"),
                    "data_xml": str(tmp_path / "data" / "xml"),
                    "outputs": str(tmp_path / "outputs")}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(cfg_path)

    rng = np.random.default_rng(0)
    n = 40
    pids = [str(i) for i in range(n)]
    cdir = tmp_path / "outputs" / "06_reduce"
    cdir.mkdir(parents=True)
    np.save(cdir / "pca_scores.npy", rng.normal(size=(n, 3)))
    pd.Series(pids, name="pid").to_csv(cdir / "pca_scores_pid_order.csv", index=False)
    prep = pd.DataFrame({"pid": pids, **{c: rng.normal(size=n) for c in SPATIAL}})
    prep.to_csv(cdir / "prepared_matrix.csv", index=False)
    pd.DataFrame({"pid": pids, "agatston_total": rng.gamma(2.0, 100.0, size=n),
                  "kernel": "Qr36d/2", "low_burden_flag": False, "category": "1-99"}).to_csv(
        cdir / "cohort_metadata.csv", index=False)
    stage5 = sorted(f.name for f in cdir.iterdir())
    write_marker(cdir, {"stage": 5, "files": files_manifest(cdir, stage5)}, name=STAGE5_MARKER)
    (cdir / "stratified_I30f_3").mkdir()
    (cdir / "stratified_I30f_3" / "keep.txt").write_text("other cohort")
    return cfg_path, cfg, cdir


def _run(mod, monkeypatch, cfg_path, cdir, settings):
    monkeypatch.setattr(mod, "PRODUCTION", dict(SMALL))
    argv = ["07_discover.py", "--config", str(cfg_path), "--cohort-dir", str(cdir), "--n-jobs", "1",
            "--k-range", *map(str, settings["k_range"]),
            "--gap-bootstraps", str(settings["gap_bootstraps"]),
            "--consensus-subsamples", str(settings["consensus_subsamples"]),
            "--hennig-bootstraps", str(settings["hennig_bootstraps"]),
            "--random-state", str(settings["random_state"]),
            "--forced-k", str(settings["forced_k"])]
    monkeypatch.setattr(sys, "argv", argv)
    return mod.main()


def test_complete_run_publishes_a_valid_record(tmp_path, monkeypatch):
    cfg_path, cfg, cdir = _cohort(tmp_path)
    mod = _load()
    assert _run(mod, monkeypatch, cfg_path, cdir, SMALL) == 0
    marker = require_stage6_complete(cdir, cfg)
    assert "cluster_labels_forced.csv" in marker["outputs"]
    assert marker["kernel_status"] == {a: "not_applicable_single_kernel" for a in mod.ALGORITHMS}
    assert (cdir / "pca_scores.npy").exists() and (cdir / STAGE5_MARKER).exists()
    assert (cdir / "stratified_I30f_3" / "keep.txt").read_text() == "other cohort"
    assert not (tmp_path / "outputs" / "06_reduce.partial").exists()


def test_failed_run_keeps_the_previous_complete_result(tmp_path, monkeypatch):
    cfg_path, cfg, cdir = _cohort(tmp_path)
    mod = _load()
    assert _run(mod, monkeypatch, cfg_path, cdir, SMALL) == 0
    before = (cdir / STAGE6_MARKER).read_text()

    def boom(*_a, **_k):
        raise RuntimeError("deliberate failure mid-run")

    mod2 = _load()
    monkeypatch.setattr(mod2, "hennig_clusterboot", boom)
    with pytest.raises(RuntimeError, match="deliberate"):
        _run(mod2, monkeypatch, cfg_path, cdir, SMALL)
    assert (cdir / STAGE6_MARKER).read_text() == before
    require_stage6_complete(cdir, cfg)          # previous result still complete and valid


def test_smoke_run_cannot_publish(tmp_path, monkeypatch):
    cfg_path, cfg, cdir = _cohort(tmp_path)
    mod = _load()
    smoke = {**SMALL, "gap_bootstraps": 2}        # differs from (patched) production
    assert _run(mod, monkeypatch, cfg_path, cdir, smoke) == 0
    assert not (cdir / STAGE6_MARKER).exists()
    assert not (cdir / "cluster_labels_forced.csv").exists()
    sdir = tmp_path / "outputs" / "07_discover_smoke" / "06_reduce"
    assert (sdir / "PARTIAL_SMOKE_DO_NOT_USE").exists()
    assert (sdir / "cluster_labels_forced.csv").exists()


def test_downstream_refuses_missing_or_stale_record(tmp_path, monkeypatch):
    cfg_path, cfg, cdir = _cohort(tmp_path)
    with pytest.raises(SystemExit, match=STAGE6_MARKER):
        require_stage6_complete(cdir, cfg)              # stage 6 never ran
    mod = _load()
    assert _run(mod, monkeypatch, cfg_path, cdir, SMALL) == 0
    labels = cdir / "cluster_labels_forced.csv"
    labels.write_text(labels.read_text() + "\n")      # an output changed afterwards
    with pytest.raises(SystemExit, match="changed"):
        require_stage6_complete(cdir, cfg)


def test_stage6_requires_a_stage5_record(tmp_path, monkeypatch):
    cfg_path, cfg, cdir = _cohort(tmp_path)
    (cdir / STAGE5_MARKER).unlink()
    mod = _load()
    with pytest.raises(SystemExit, match="COMPLETE_stage5"):
        _run(mod, monkeypatch, cfg_path, cdir, SMALL)
