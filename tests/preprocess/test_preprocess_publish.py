"""Tests for scripts/02_preprocess.py: exit status and publish semantics.

A patient failure must fail the stage, and a rerun
must never leave arrays from one run next to metadata from another.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def _load():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "scripts_02_preprocess", root / "scripts" / "02_preprocess.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ok(pid: str, out_dir: str) -> dict:
    Path(out_dir, f"{pid}_ct.npy").write_bytes(b"new")
    Path(out_dir, f"{pid}_mask.npy").write_bytes(b"new")
    return {"status": "ok", "pid": pid, "roundtrip_pass_rate": 1.0,
            "metal_artifact_flag": False, "n_excluded_by_roundtrip": 0, "trips": []}


def _setup(tmp_path: Path, monkeypatch, failing: set[str], argv: list[str] | None = None):
    mod = _load()
    outputs = tmp_path / "outputs"
    (outputs / "01_manifest").mkdir(parents=True)
    pd.DataFrame({"pid": ["1", "2", "3"]}).to_csv(outputs / "01_manifest" / "manifest.csv", index=False)
    # A previous complete run, including an array for a pid that will fail.
    old = outputs / "02_preprocessed"
    old.mkdir()
    for pid in ("1", "2", "3"):
        (old / f"{pid}_ct.npy").write_bytes(b"old")
    (old / "spacing.json").write_text(json.dumps({"target_spacing": [0.5, 0.5, 3.0]}))

    cfg = SimpleNamespace(
        paths=SimpleNamespace(outputs=outputs, data_raw=tmp_path / "raw", data_xml=tmp_path / "xml"),
        resample=SimpleNamespace(target_spacing=(0.375, 0.375, 3.0), slice_mm=3.0),
        hu=SimpleNamespace(clip_min=-200, clip_max=3000, metal_artifact_threshold=2000),
    )
    monkeypatch.setattr(mod, "load_config", lambda *_a, **_k: cfg)

    def fake(pid, _raw, _xml, out_dir, *_a, **_k):
        if pid in failing:
            return {"status": "error", "pid": pid, "error": "IrregularGeometryError: test"}
        return _ok(pid, out_dir)

    monkeypatch.setattr(mod, "_process_one", fake)
    monkeypatch.setattr(sys, "argv", ["02_preprocess.py", "--n-workers", "1", *(argv or [])])
    return mod, outputs


def test_failure_exits_nonzero_and_leaves_published_untouched(tmp_path, monkeypatch):
    mod, outputs = _setup(tmp_path, monkeypatch, failing={"2"})
    assert mod.main() == 1
    pub = outputs / "02_preprocessed"
    assert (pub / "2_ct.npy").read_bytes() == b"old"
    assert (pub / "1_ct.npy").read_bytes() == b"old"       # not mixed with new
    assert json.loads((pub / "spacing.json").read_text())["target_spacing"] == [0.5, 0.5, 3.0]
    partial = outputs / "02_preprocessed.partial"
    assert (partial / "errors.csv").exists()
    assert not (partial / "2_ct.npy").exists()


def test_success_replaces_published_folder_completely(tmp_path, monkeypatch):
    mod, outputs = _setup(tmp_path, monkeypatch, failing=set())
    assert mod.main() == 0
    pub = outputs / "02_preprocessed"
    assert all((pub / f"{p}_ct.npy").read_bytes() == b"new" for p in ("1", "2", "3"))
    assert not (outputs / "02_preprocessed.partial").exists()
    assert not (outputs / "02_preprocessed.old").exists()


def test_limit_smoke_run_never_publishes(tmp_path, monkeypatch):
    mod, outputs = _setup(tmp_path, monkeypatch, failing=set(), argv=["--limit", "1"])
    assert mod.main() == 0
    assert (outputs / "02_preprocessed" / "1_ct.npy").read_bytes() == b"old"
    assert (outputs / "02_preprocessed.partial" / "1_ct.npy").read_bytes() == b"new"
