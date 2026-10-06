"""Tests for scripts/09_validate.py orchestrator (lightweight, no real data)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def _load_orchestrator():
    """Load scripts/09_validate.py as a module."""
    root = Path(__file__).resolve().parents[2]
    p = root / "scripts" / "09_validate.py"
    spec = importlib.util.spec_from_file_location("scripts_09_validate", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────── helpers + flags ───────────────────────


def test_orchestrator_loads():
    mod = _load_orchestrator()
    assert hasattr(mod, "main")
    assert hasattr(mod, "run_d029_holdout")
    assert hasattr(mod, "run_d030_loo")
    assert hasattr(mod, "run_d031_consolidation")


def test_discover_holdout_pids_from_exclusions_csv(tmp_path: Path,
                                                     monkeypatch):
    mod = _load_orchestrator()
    # Build a minimal fake Config object with an outputs dir.
    excl_dir = tmp_path / "01_manifest"
    excl_dir.mkdir()
    pd.DataFrame({
        "pid": ["19", "28", "76", "77", "12"],
        "reason": ["ge_scanner"] * 4 + ["no_dicom"],
    }).to_csv(excl_dir / "exclusions.csv", index=False)

    class FakePaths:
        outputs = tmp_path

    class FakeCfg:
        paths = FakePaths()

    pids = mod._discover_holdout_pids(FakeCfg())
    assert pids == ["19", "28", "76", "77"]


def test_discover_holdout_pids_requires_the_manifest(tmp_path: Path):
    """No silent fallback list when stage 1's manifest is
    missing."""
    mod = _load_orchestrator()

    class FakePaths:
        outputs = tmp_path  # no exclusions.csv

    class FakeCfg:
        paths = FakePaths()

    with pytest.raises(SystemExit, match="exclusions.csv"):
        mod._discover_holdout_pids(FakeCfg())


def _holdout_cfg(tmp_path: Path):
    from types import SimpleNamespace
    excl_dir = tmp_path / "01_manifest"
    excl_dir.mkdir()
    pd.DataFrame({"pid": ["19", "28", "76", "77"], "reason": ["ge_scanner"] * 4}).to_csv(
        excl_dir / "exclusions.csv", index=False)
    return SimpleNamespace(
        paths=SimpleNamespace(outputs=tmp_path, data_raw=tmp_path / "raw", data_xml=tmp_path / "xml"),
        resample=SimpleNamespace(target_spacing=(0.5, 0.5, 0.5)),
        hu=SimpleNamespace(clip_min=-1024, clip_max=3071, metal_artifact_threshold=3000),
    )


def test_d029_holdout_processing_failure_stops_the_stage(tmp_path: Path, monkeypatch):
    """A holdout patient that fails processing is not skipped: the failure
    is written to holdout_failures.csv and the stage stops."""
    import logging
    from types import SimpleNamespace
    mod = _load_orchestrator()

    def fake_process_one(pid, *_a, **_k):
        if pid == "28":
            return {"status": "error", "error": "deliberate"}
        return {"status": "ok", "pid": pid, "roundtrip_quality": "full",
                "trips": [{"passes": True, "matched_via": "z", "image_index": 0,
                           "roi_idx_in_slice": 0}]}

    monkeypatch.setattr(mod, "_load_script_module",
                        lambda *_a, **_k: SimpleNamespace(_process_one=fake_process_one))
    out = tmp_path / "08_validate.partial"
    out.mkdir()
    with pytest.raises(SystemExit, match="failed for 1 of 4"):
        mod.run_d029_holdout(_holdout_cfg(tmp_path), out, logging.getLogger("t"), settings=None)
    failures = pd.read_csv(out / "holdout_failures.csv", dtype={"pid": str})
    assert failures["pid"].tolist() == ["28"]


def _main_cfg(tmp_path: Path) -> Path:
    import yaml
    from predict.config import DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["paths"] = {"data_raw": str(tmp_path / "data" / "raw"),
                    "data_xml": str(tmp_path / "data" / "xml"),
                    "outputs": str(tmp_path / "outputs")}
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p


def _stub_deliverables(mod, monkeypatch, fail_loo: bool = False):
    monkeypatch.setattr(mod, "stage8_inputs", lambda cfg: {"stage7_record_sha": "x"})
    monkeypatch.setattr(mod, "production_settings", lambda cfg: None)

    def d029(cfg, out_dir, log, settings):
        (out_dir / "external_holdout_report.csv").write_text("pid\n19\n")
        return {"verdict": "ok"}

    def d030(cfg, out_dir, log, **_k):
        if fail_loo:
            raise RuntimeError("deliberate D030 failure")
        (out_dir / "leave_k_out_ari.csv").write_text("fold\n0\n")
        return {"overall_pass": True}

    def d031(cfg, out_dir, log):
        (out_dir / "cross_cohort_ari_consolidated.csv").write_text("partition\n")
        return {"verdict": "ok"}

    monkeypatch.setattr(mod, "run_d029_holdout", d029)
    monkeypatch.setattr(mod, "run_d030_loo", d030)
    monkeypatch.setattr(mod, "run_d031_consolidation", d031)


def test_main_publishes_only_complete_runs_and_isolates_reduced_runs(tmp_path: Path, monkeypatch):
    import sys
    cfg_path = _main_cfg(tmp_path)
    target = tmp_path / "outputs" / "08_validate"

    mod = _load_orchestrator()
    _stub_deliverables(mod, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["09_validate.py", "--config", str(cfg_path)])
    assert mod.main() == 0
    marker = json.loads((target / "COMPLETE_stage8.json").read_text())
    assert {"external_holdout_report.csv", "leave_k_out_ari.csv",
            "cross_cohort_ari_consolidated.csv", "run_header_validate.json"} <= set(marker["outputs"])
    before = (target / "COMPLETE_stage8.json").read_text()

    # A failing run publishes nothing and keeps the previous complete result.
    mod2 = _load_orchestrator()
    _stub_deliverables(mod2, monkeypatch, fail_loo=True)
    with pytest.raises(RuntimeError, match="deliberate D030"):
        mod2.main()
    assert (target / "COMPLETE_stage8.json").read_text() == before

    # A reduced run goes to its own folder, flagged, never as stage 8.
    mod3 = _load_orchestrator()
    _stub_deliverables(mod3, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["09_validate.py", "--config", str(cfg_path), "--skip-loo"])
    assert mod3.main() == 0
    smoke = tmp_path / "outputs" / "08_validate_smoke"
    assert (smoke / "PARTIAL_SMOKE_DO_NOT_USE").exists()
    assert not (smoke / "COMPLETE_stage8.json").exists()
    assert (target / "COMPLETE_stage8.json").read_text() == before


def test_xml_pass_by_pid_aggregation():
    mod = _load_orchestrator()
    trips = pd.DataFrame({
        "pid": ["19", "19", "19", "28", "28", "76"],
        "passes": [True, True, False, True, True, True],
        "matched_via": ["z", "z", "z", "z", "z", "dirty"],
        "image_index": [0, 1, 2, 0, 1, 0],
        "roi_idx_in_slice": [0, 0, 0, 0, 0, 0],
    })
    out = mod._xml_pass_by_pid_from_trips(trips)
    assert out["19"] is False           # one non-dirty fail
    assert out["28"] is True
    # pid 76 has only one ROI which is matched_via=dirty -> all-dropped -> True
    # (empty after filter; .all() on empty returns True by NumPy convention).
    assert out["76"] is True


def test_d031_consolidation_smoke(tmp_path: Path, monkeypatch):
    mod = _load_orchestrator()
    # The stage 7 record check has its own tests (tests/analyse/test_stage7_script.py).
    import predict.analyse.stage7 as stage7
    monkeypatch.setattr(stage7, "require_stage7_complete", lambda *_a, **_k: {})
    # Synthesise a stage-7 cross_cohort_ari.csv.
    stage7 = tmp_path / "07_analyse"
    stage7.mkdir(parents=True)
    pd.DataFrame({
        "partition": ["spatial_k2", "burden_k3"],
        "stratum": ["Qr36d_2", "Qr36d_2"],
        "n_shared_pids": [220, 220],
        "ari": [0.91, 0.55],
        "passes": [True, False],
    }).to_csv(stage7 / "cross_cohort_ari.csv", index=False)

    class FakePaths:
        outputs = tmp_path

    class FakeCfg:
        paths = FakePaths()

    out = tmp_path / "08_validate"
    out.mkdir()
    import logging
    summary = mod.run_d031_consolidation(FakeCfg(), out, logging.getLogger("t"))
    assert summary["verdict"] == "ok"
    df = pd.read_csv(out / "cross_cohort_ari_consolidated.csv")
    assert list(df.columns) == ["partition", "stratum", "n_shared_pids",
                                 "ari", "pass_threshold", "pass_verdict"]
    assert summary["n_passing"] == 1


def test_d031_consolidation_refuses_without_a_stage7_record(tmp_path: Path):
    """Stage 8 reads stage 7 outputs only from a
    complete, current stage 7 run; it never skips silently."""
    mod = _load_orchestrator()

    class FakePaths:
        outputs = tmp_path

    class FakeCfg:
        paths = FakePaths()

    import logging
    out = tmp_path / "08_validate"
    out.mkdir()
    with pytest.raises(SystemExit, match="COMPLETE_stage7"):
        mod.run_d031_consolidation(FakeCfg(), out, logging.getLogger("t"))
