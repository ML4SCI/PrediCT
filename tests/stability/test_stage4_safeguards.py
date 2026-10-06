"""Stage 4 run safeguards.

1. Resume reuses a cached result only with a matching fingerprint, zero
   errors and every planned draw exactly once (4b) / exactly the eligible
   pids with no error text (4a).
2. Only an empty or too-small perturbed mask is a "broken" draw (D033.1);
   any other extraction exception propagates; the classic gate refuses
   error rows.
4. Every arm must hold its prescribed draw indices exactly once.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"scripts_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ───────────── 4b resume ─────────────

def _cache(tmp_path: Path, rows, meta: dict | None):
    csv = tmp_path / "1.csv"
    pd.DataFrame(rows, columns=["arm", "index", "status"]).to_csv(csv, index=False)
    js = tmp_path / "1.json"
    if meta is not None:
        js.write_text(json.dumps(meta))
    return csv, js


GOOD_META = {"fingerprint": "fp", "n_error": 0, "parity": {"pid": "1", "ok": True}}
GOOD_ROWS = [("baseline", -1, "ok"), ("primary", 0, "ok"), ("primary", 1, "broken")]
WANT = {("primary", 0), ("primary", 1)}


def test_4b_reuses_only_a_verified_cache(tmp_path):
    m = _load("04b_realistic_perturbations")
    csv, js = _cache(tmp_path, GOOD_ROWS, GOOD_META)
    assert m._reusable(csv, js, "fp", WANT) == {"pid": "1", "ok": True}


@pytest.mark.parametrize("rows, meta, fp", [
    (GOOD_ROWS, None, "fp"),                                           # no sidecar
    (GOOD_ROWS, GOOD_META, "other"),                                   # inputs changed
    (GOOD_ROWS, {**GOOD_META, "n_error": 1}, "fp"),                    # recorded errors
    (GOOD_ROWS, {**GOOD_META, "parity": {"ok": False}}, "fp"),         # failed parity
    ([("primary", 0, "error"), ("primary", 1, "ok")], GOOD_META, "fp"),  # failed rows only
    ([("primary", 0, "ok"), ("primary", 0, "ok"), ("primary", 1, "ok")], GOOD_META, "fp"),  # duplicate
    ([("primary", 0, "ok")], GOOD_META, "fp"),                         # missing draw
])
def test_4b_rejects_stale_or_failed_cache(tmp_path, rows, meta, fp):
    m = _load("04b_realistic_perturbations")
    csv, js = _cache(tmp_path, rows, meta)
    assert m._reusable(csv, js, fp, WANT) is None


# ───────────── 4b broken vs error (D033.1) ─────────────

class _Extractor:
    settings = {"minimumROISize": 14, "minimumROIDimensions": 1}

    def execute(self, *_a, **_k):
        raise ValueError("Shape mismatch: unexpected")


def _volumes(n_mask: int):
    ct = np.full((3, 16, 16), -1000, dtype=np.int16)
    mask = np.zeros((3, 16, 16), dtype=np.uint8)
    mask.reshape(-1)[:n_mask] = 1
    ct[mask == 1] = 400
    return ct, mask


def test_empty_or_small_mask_is_broken():
    m = _load("04b_realistic_perturbations")
    for n in (0, 5):
        status, why, feats = m._extract_draw(*_volumes(n), (0.375, 0.375, 3.0), _Extractor(), "1")
        assert status == "broken" and feats == {}


def test_other_extraction_errors_propagate():
    m = _load("04b_realistic_perturbations")
    with pytest.raises(ValueError, match="unexpected"):
        m._extract_draw(*_volumes(200), (0.375, 0.375, 3.0), _Extractor(), "1")


# ───────────── 4a resume and 5 gate ─────────────

def _pert_csv(tmp_path: Path, pids, errors, fp="fp", n_error=0):
    csv = tmp_path / "noise_5.csv"
    pd.DataFrame({"pid": pids, "original_x": 1.0, "error": errors}).to_csv(csv, index=False)
    csv.with_suffix(".json").write_text(json.dumps({"fingerprint": fp, "n_error": n_error}))
    return csv


def test_4a_resume_requires_fingerprint_and_clean_rows(tmp_path):
    m = _load("04_perturbations")
    ok = _pert_csv(tmp_path, ["1", "2"], ["", ""])
    assert m._csv_is_complete(ok, ["1", "2"], "fp") is True
    assert m._csv_is_complete(ok, ["1", "2"], "other") is False
    bad = _pert_csv(tmp_path, ["1", "2"], ["", "ValueError: boom"])
    assert m._csv_is_complete(bad, ["1", "2"], "fp") is False
    dup = _pert_csv(tmp_path, ["1", "1", "2"], ["", "", ""])
    assert m._csv_is_complete(dup, ["1", "2"], "fp") is False


def test_classic_gate_refuses_error_rows(tmp_path):
    g = _load("05_icc_gate")
    pd.DataFrame({"pid": ["1", "2"], "original_x": [1.0, np.nan],
                  "error": ["", "ValueError: boom"]}).to_csv(tmp_path / "noise_5.csv", index=False)
    with pytest.raises(ValueError, match="extraction errors"):
        g._load_perturbation_csvs(tmp_path, ["noise_5"], {"1", "2"})


# ───────────── 5b draw completeness ─────────────

def _draws(per_pid: dict[str, list[int]], status="ok"):
    rows = [{"pid": p, "index": i, "status": status} for p, idx in per_pid.items() for i in idx]
    return pd.DataFrame(rows)


def test_arm_with_2_of_10_draws_is_rejected():
    g = _load("05b_icc_gate_realistic")
    probs = g._check_draws(_draws({"1": list(range(10)), "2": [0, 1]}), ["1", "2"], 10, "S1")
    assert probs and "2 of 10" in probs[0]


def test_duplicate_and_error_draws_are_rejected():
    g = _load("05b_icc_gate_realistic")
    assert g._check_draws(_draws({"1": [0, 0, 1]}), ["1"], 2, "S1")
    assert g._check_draws(_draws({"1": [0, 1]}, status="error"), ["1"], 2, "S1")


def test_complete_arm_passes():
    g = _load("05b_icc_gate_realistic")
    assert g._check_draws(_draws({"1": list(range(10)), "2": list(range(10))}), ["1", "2"], 10, "S1") == []


# ───────────── 5b marker: calibration and code, not only features.csv ─────────────

def test_stale_calibration_marker_is_refused(tmp_path):
    """A recalibration after the draws must
    make the realistic gate refuse, even with features.csv unchanged."""
    from predict.completion import file_sha, require_marker, write_marker
    from predict.config import load_config
    m = _load("04b_realistic_perturbations")
    cfg = load_config()
    calib = tmp_path / "calibration.json"
    calib.write_text('{"version": 1}')
    write_marker(tmp_path, {"features_csv_sha": "f", "calibration_sha": file_sha(calib),
                            "run_fingerprint": m.run_fingerprint(cfg, calib)})
    calib.write_text('{"version": 2}')            # recalibrated after the draws
    with pytest.raises(SystemExit, match="calibration_sha|run_fingerprint"):
        require_marker(tmp_path, {"features_csv_sha": "f", "calibration_sha": file_sha(calib),
                                  "run_fingerprint": m.run_fingerprint(cfg, calib)})


def test_marker_binds_patient_inputs(tmp_path):
    """A CT changed after the draws must
    change the patients fingerprint the 4b marker carries, so 5b refuses
    even with features.csv unchanged."""
    import yaml

    from predict.completion import require_marker, write_marker
    from predict.config import DEFAULT_CONFIG_PATH, load_config

    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["paths"] = {"data_raw": str(tmp_path / "data" / "raw"),
                    "data_xml": str(tmp_path / "data" / "xml"),
                    "outputs": str(tmp_path / "outputs")}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(cfg_path)

    out = tmp_path / "outputs"
    (out / "03_features").mkdir(parents=True)
    (out / "02_preprocessed").mkdir(parents=True)
    (tmp_path / "data" / "xml").mkdir(parents=True)
    pd.DataFrame({"pid": ["1", "2"], "radiomics_status": ["ok", "ok"],
                  "original_firstorder_Mean": [1.0, 2.0]}).to_csv(out / "03_features" / "features.csv", index=False)
    pd.DataFrame({"pid": ["1"], "passes": [True], "matched_via": ["center"],
                  "image_index": [0], "roi_idx_in_slice": [0]}).to_csv(
        out / "02_preprocessed" / "xml_roundtrip.csv", index=False)
    for p in ("1", "2"):
        np.save(out / "02_preprocessed" / f"{p}_ct.npy", np.zeros((2, 4, 4), np.int16))
        np.save(out / "02_preprocessed" / f"{p}_mask.npy", np.ones((2, 4, 4), np.uint8))
        (tmp_path / "data" / "xml" / f"{p}.xml").write_text("<plist/>")
    calib = tmp_path / "calibration.json"
    calib.write_text("{}")

    m = _load("04b_realistic_perturbations")
    before = m.combine_patient_fingerprints(m.patient_fingerprints(cfg, calib, ["1", "2"], n_jobs=2))
    write_marker(tmp_path, {"patients_fingerprint": before})
    np.save(out / "02_preprocessed" / "2_ct.npy", np.ones((2, 4, 4), np.int16))   # CT changed
    after = m.combine_patient_fingerprints(m.patient_fingerprints(cfg, calib, ["1", "2"], n_jobs=2))
    assert after != before
    with pytest.raises(SystemExit, match="patients_fingerprint"):
        require_marker(tmp_path, {"patients_fingerprint": after})


# ───────────── robust feature sets per arm and the final set ─────────────

def _report(passes: dict[str, bool]) -> pd.DataFrame:
    return pd.DataFrame([{"feature": f, "family": "canonical" if not f.startswith("original_") else "glcm",
                          "icc": 0.9 if ok else 0.5, "ci_low": 0.85 if ok else 0.4,
                          "ci_high": 0.95 if ok else 0.6, "icc_source": "empirical",
                          "passes_gate": ok} for f, ok in passes.items()])


def test_robust_sets_per_arm_and_final_is_the_primary_gate(tmp_path):
    g = _load("05b_icc_gate_realistic")
    report = _report({"a": True, "b": True, "original_glcm_x": False})
    arms = pd.concat([_report({"a": True, "b": False, "original_glcm_x": True}).assign(arm="S1"),
                      _report({"a": False, "b": False, "original_glcm_x": False}).assign(arm="attr_M1")])
    m = g.robust_feature_sets(report, arms)
    assert list(m.columns) == ["family", "primary", "S1", "attr_M1", "final"]
    assert m["final"].equals(m["primary"])           # final = the locked D033 set
    g.write_robust_sets(tmp_path, report, m)
    read = lambda name: pd.read_csv(tmp_path / "robust_sets" / f"{name}.csv")["feature"].tolist()
    assert read("primary") == ["a", "b"]
    assert read("S1") == ["a", "original_glcm_x"]
    assert read("attr_M1") == []
    final = pd.read_csv(tmp_path / "robust_features_final.csv")
    assert final["feature"].tolist() == ["a", "b"]
    assert {"icc", "ci_low", "ci_high"} <= set(final.columns)
    assert (pd.read_csv(tmp_path / "robust_feature_matrix.csv").shape) == (3, 6)


def test_robust_sets_refuse_an_arm_missing_features():
    g = _load("05b_icc_gate_realistic")
    report = _report({"a": True, "b": True})
    arms = _report({"a": True}).assign(arm="S2")
    with pytest.raises(SystemExit, match="S2"):
        g.robust_feature_sets(report, arms)
