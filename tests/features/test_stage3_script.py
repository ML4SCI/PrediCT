"""Tests for scripts/03_features.py.

1. A patient failure exits 1 and leaves the published folder untouched; a
   complete run replaces it; ``--limit`` never publishes.
3. Only a mask PyRadiomics' own ROI checks would reject is a radiomics
   "skip"; any exception from the extractor fails the patient.
4. The configured lesion-grouping parameters reach the grouping function.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from predict.features.feature_schema import zero_features


def _load():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "scripts_03_features", root / "scripts" / "03_features.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Extractor:
    settings = {"minimumROISize": 14, "minimumROIDimensions": 1}

    def __init__(self, exc: Exception | None = None):
        self.exc = exc

    def execute(self, *_a, **_k):
        if self.exc is not None:
            raise self.exc
        return {"original_firstorder_Mean": 400.0}


def _patch_inputs(mod, monkeypatch, tmp_path: Path, mask_voxels: int, captured: dict | None = None):
    pre = tmp_path / "pre"
    pre.mkdir(exist_ok=True)
    ct = np.full((4, 16, 16), -1000, dtype=np.int16)
    mask = np.zeros((4, 16, 16), dtype=np.uint8)
    mask.reshape(-1)[:mask_voxels] = 1
    ct[mask == 1] = 400
    np.save(pre / "1_ct.npy", ct)
    np.save(pre / "1_mask.npy", mask)
    meta = SimpleNamespace(kernel="Qr36d/2", scanner_model="X", pid="1")
    monkeypatch.setattr(mod, "load_patient_metadata", lambda *_a, **_k: meta)
    monkeypatch.setattr(mod, "parse_calcium_xml", lambda *_a, **_k: None)

    def fake_canonical(_parse, _meta, _excluded, **grouping):
        if captured is not None:
            captured.update(grouping)
        return zero_features(), "minimal", {"LAD": [], "RCA": [], "LCx": [], "LM": []}

    monkeypatch.setattr(mod, "compute_canonical_features", fake_canonical)
    return str(pre)


GROUPING = {"max_inplane_mm": 3.0, "max_slice_gap": 2}


def _run_one(mod, pre: str, extractor) -> dict:
    return mod._process_one("1", "data", "xml", pre, set(), (0.375, 0.375, 3.0), "ok",
                            GROUPING, {"e": extractor})


def test_extractor_exception_fails_the_patient(tmp_path, monkeypatch):
    mod = _load()
    pre = _patch_inputs(mod, monkeypatch, tmp_path, mask_voxels=200)
    out = _run_one(mod, pre, _Extractor(ValueError("Shape mismatch: boom")))
    assert out["status"] == "error"
    assert "boom" in out["error"]


def test_small_roi_is_a_skip_not_an_error(tmp_path, monkeypatch):
    mod = _load()
    pre = _patch_inputs(mod, monkeypatch, tmp_path, mask_voxels=5)
    out = _run_one(mod, pre, _Extractor(ValueError("must not be called")))
    assert out["status"] == "ok"
    assert out["row"]["radiomics_status"] == "skipped"
    assert "minimumROISize" in out["row"]["radiomics_reason"]


def test_configured_grouping_reaches_canonical_features(tmp_path, monkeypatch):
    mod = _load()
    seen: dict = {}
    pre = _patch_inputs(mod, monkeypatch, tmp_path, mask_voxels=200, captured=seen)
    out = _run_one(mod, pre, _Extractor())
    assert out["status"] == "ok"
    assert seen == GROUPING


def _setup_main(tmp_path: Path, monkeypatch, failing: set[str], argv: list[str] | None = None):
    mod = _load()
    outputs = tmp_path / "outputs"
    (outputs / "01_manifest").mkdir(parents=True)
    pd.DataFrame({"pid": ["1", "2"]}).to_csv(outputs / "01_manifest" / "manifest.csv", index=False)
    pre = outputs / "02_preprocessed"
    pre.mkdir()
    pd.DataFrame({"pid": ["1", "2"], "roundtrip_quality": ["ok", "ok"]}).to_csv(
        pre / "preprocess_report.csv", index=False)
    pd.DataFrame(columns=["pid", "passes", "matched_via", "image_index", "roi_idx_in_slice"]).to_csv(
        pre / "xml_roundtrip.csv", index=False)
    (pre / "spacing.json").write_text(json.dumps({"target_spacing": [0.375, 0.375, 3.0]}))
    old = outputs / "03_features"
    old.mkdir()
    (old / "features.csv").write_text("pid\nOLD\n")

    cfg = SimpleNamespace(
        paths=SimpleNamespace(outputs=outputs, data_raw=tmp_path / "raw", data_xml=tmp_path / "xml"),
        lesion_grouping=SimpleNamespace(max_inplane_dist_mm=5.0, max_slice_gap=1),
    )
    monkeypatch.setattr(mod, "load_config", lambda *_a, **_k: cfg)

    def fake(pid, *_a, **_k):
        if pid in failing:
            return {"status": "error", "pid": pid, "error": "ValueError: test"}
        return {"status": "ok", "row": {"pid": pid}, "lesions": [], "log": {"pid": pid}}

    monkeypatch.setattr(mod, "_process_one", fake)
    monkeypatch.setattr(sys, "argv", ["03_features.py", "--n-workers", "1", *(argv or [])])
    return mod, outputs


def test_failure_exits_nonzero_and_keeps_published(tmp_path, monkeypatch):
    mod, outputs = _setup_main(tmp_path, monkeypatch, failing={"2"})
    assert mod.main() == 1
    assert (outputs / "03_features" / "features.csv").read_text() == "pid\nOLD\n"
    assert (outputs / "03_features.partial" / "errors.csv").exists()


def test_success_publishes_complete_run_with_grouping_in_header(tmp_path, monkeypatch):
    mod, outputs = _setup_main(tmp_path, monkeypatch, failing=set())
    assert mod.main() == 0
    pub = outputs / "03_features"
    assert list(pd.read_csv(pub / "features.csv", dtype=str)["pid"]) == ["1", "2"]
    header = json.loads((pub / "run_header.json").read_text())
    assert header["lesion_grouping"] == {"max_inplane_mm": 5.0, "max_slice_gap": 1}
    assert not (outputs / "03_features.partial").exists()


def test_limit_never_publishes(tmp_path, monkeypatch):
    mod, outputs = _setup_main(tmp_path, monkeypatch, failing=set(), argv=["--limit", "1"])
    assert mod.main() == 0
    assert (outputs / "03_features" / "features.csv").read_text() == "pid\nOLD\n"
    assert (outputs / "03_features.partial" / "features.csv").exists()


def test_missing_stage2_patient_fails_before_extraction(tmp_path, monkeypatch):
    mod, outputs = _setup_main(tmp_path, monkeypatch, failing=set())
    pd.DataFrame({"pid": ["1"], "roundtrip_quality": ["ok"]}).to_csv(
        outputs / "02_preprocessed" / "preprocess_report.csv", index=False)
    assert mod.main() == 1
    assert (outputs / "03_features" / "features.csv").read_text() == "pid\nOLD\n"


def test_grouping_comes_from_config_and_has_no_silent_default():
    import inspect

    from predict.config import load_config
    from predict.features.canonical import compute_canonical_features, grouping_from_config

    assert grouping_from_config(load_config()) == {"max_inplane_mm": 5.0, "max_slice_gap": 1}
    params = inspect.signature(compute_canonical_features).parameters
    for name in ("max_inplane_mm", "max_slice_gap"):
        assert params[name].default is inspect.Parameter.empty, name
