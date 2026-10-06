"""Stage 10 (figures) safeguards.

Covered: a complete run publishes outputs/figures with COMPLETE_stage10.json;
a failing figure publishes nothing and keeps the previous set; --only runs go
to figures_smoke; the figure text follows the files and the decisions
(Figure 1 and 2: no continuum claim; Figure 3: Hennig and D024 read from the
files; Figure 4: classes named by their threshold rule, no "maturation";
Figure 5: superior-inferior wording, stage 9's own comparison, no fallback).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402

from predict.config import DEFAULT_CONFIG_PATH  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _text(fig) -> str:
    parts = [fig._suptitle.get_text() if fig._suptitle else ""]
    for ax in fig.axes:
        parts += [ax.get_title(), ax.get_xlabel(), ax.get_ylabel()]
        parts += [t.get_text() for t in ax.texts]
        leg = ax.get_legend()
        if leg is not None:
            parts += [t.get_text() for t in leg.get_texts()]
    plt.close(fig)
    return "\n".join(parts)


# ─────────────────────── publication ───────────────────────


def _load_builder():
    spec = importlib.util.spec_from_file_location("s10_figures", ROOT / "scripts" / "10_figures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cfg(tmp_path: Path) -> Path:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["paths"] = {"data_raw": str(tmp_path / "data" / "raw"),
                    "data_xml": str(tmp_path / "data" / "xml"),
                    "outputs": str(tmp_path / "outputs")}
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p


def _stub(mod, monkeypatch, fail: str | None = None):
    import predict.analyse.stage7 as s7
    import predict.discover.stage6 as s6
    import predict.stage9 as s9
    for m, name in ((s6, "require_stage6_complete"), (s7, "require_stage7_complete"),
                    (s9, "require_stage9_complete")):
        monkeypatch.setattr(m, name, lambda *_a, **_k: {})

    def make(fid):
        def build(**_k):
            if fid == fail:
                raise RuntimeError(f"deliberate failure in {fid}")
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            return fig
        return build

    for fid, attr in (("F1", "build_f1"), ("F2", "build_f2"), ("F3", "build_f3"), ("F4", "build_f4"),
                      ("F5", "build_f5"), ("F6", "build_f6"), ("SF1", "build_s1"), ("SF2", "build_s2"),
                      ("SF3", "build_s3"), ("SF4", "build_s4"), ("SF5", "build_s5")):
        monkeypatch.setattr(mod, attr, make(fid))


def test_figures_publish_only_complete_sets(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    target = tmp_path / "outputs" / "figures"
    mod = _load_builder()
    _stub(mod, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["10_figures.py", "--config", str(cfg)])
    assert mod.main() == 0
    marker = json.loads((target / "COMPLETE_stage10.json").read_text())
    assert "figure_1_patient_pca.png" in marker["outputs"]
    assert "figure_4_burden_composition.pdf" in marker["outputs"]
    before = (target / "COMPLETE_stage10.json").read_text()

    mod2 = _load_builder()
    _stub(mod2, monkeypatch, fail="F3")
    assert mod2.main() == 1
    assert (target / "COMPLETE_stage10.json").read_text() == before   # previous set kept

    mod3 = _load_builder()
    _stub(mod3, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["10_figures.py", "--config", str(cfg), "--only", "F1"])
    assert mod3.main() == 0
    smoke = tmp_path / "outputs" / "figures_smoke"
    assert (smoke / "PARTIAL_SMOKE_DO_NOT_USE").exists()
    assert not (smoke / "COMPLETE_stage10.json").exists()
    assert (target / "COMPLETE_stage10.json").read_text() == before


# ─────────────────────── figure text ───────────────────────


def _reduce_dir(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    d = tmp_path / "06_reduce"
    d.mkdir()
    n = 30
    pids = [str(i) for i in range(n)]
    np.save(d / "pca_scores.npy", rng.normal(size=(n, 3)))
    pd.Series(pids, name="pid").to_csv(d / "pca_scores_pid_order.csv", index=False)
    pd.DataFrame({"pid": pids, "agatston_total": rng.gamma(2, 100, n)}).to_csv(
        d / "cohort_metadata.csv", index=False)
    recs = [{"feature_space": s, "algorithm": a, "k_range": [1, 2, 3, 4],
             "gap_values": [0.1, 0.2, 0.3, 0.35], "sk_values": [0.05] * 4, "selected_k": 4}
            for s in ("full", "burden_residualised", "spatial_only") for a in ("kmeans", "ward", "gmm")]
    (d / "gap_statistic.json").write_text(json.dumps(recs))
    return d


def test_figures_1_and_2_make_no_continuum_claim(tmp_path):
    from predict.figures.figure_1_patient_pca import build as f1
    from predict.figures.figure_2_gap_grid import build as f2
    d = _reduce_dir(tmp_path)
    for text in (_text(f1(reduce_dir=d)), _text(f2(reduce_dir=d))):
        low = text.lower()
        assert "continuum" not in low and "continuous" not in low and "elbow" not in low
    assert "D021.1" in _text(f2(reduce_dir=d))


def _lesion_dir(tmp_path: Path, deep_cluster: int = 3, with_c8: bool = True) -> Path:
    d = tmp_path / "lesion_morphology"
    (d / "finalise").mkdir(parents=True)
    prof = pd.DataFrame({"cluster": [0, 1, 2, 3], "n_lesions": [50, 40, 30, 10],
                         "volume_mm3_median": [5.0, 30.0, 80.0, 290.0 if with_c8 else 50.0],
                         "max_hu_median": [180.0, 350.0, 700.0, 840.0],
                         "n_rois_median": [1, 2, 3, 10]})
    prof.to_csv(d / "cluster_profiles.csv", index=False)
    vc = pd.DataFrame({"cluster": [0, 1, 2, 3], "n_lesions": [50, 40, 30, 10],
                       "chi2_p": [0.5, 0.5, 0.1, 1e-9], "cramers_v": [0.05, 0.05, 0.1, 0.6],
                       "vessel_biased": [False, False, False, True],
                       "lad_obs_over_exp": [1.0, 1.0, 1.2, 0.2], "rca_obs_over_exp": [1.0, 1.0, 0.8, 2.5],
                       "lcx_obs_over_exp": [1.0, 1.0, 1.0, 0.6], "lm_obs_over_exp": [1.0, 1.0, 1.0, 0.0]})
    vc.to_csv(d / "cluster_vessel_chi_square.csv", index=False)
    (d / "finalise" / "c8_deep_dive.json").write_text(json.dumps({
        "c8_like_cluster_id": deep_cluster, "n_c8_patients": 12, "c8_patient_burden_median": 1600.0,
        "non_c8_patient_burden_median": 80.0, "mannwhitney_p_one_sided_c8_higher": 1e-20,
        "cliffs_delta_burden_c8_vs_non": 0.92, "c8_vs_other_RCA_z_mw_p_two_sided": 1e-6}))
    pd.DataFrame({"pid": ["1", "1", "2", "2"], "in_c8": [True, False, True, False],
                  "relative_z_within_RCA": [0.8, 0.1, 0.7, 0.2]}).to_csv(
        d / "finalise" / "c8_rca_relative_z.csv", index=False)
    pd.DataFrame({"tertile": ["low", "mid", "high"], **{f"frac_C{c}": [0.25] * 3 for c in range(4)}}).to_csv(
        d / "mixture_by_burden_tertile.csv", index=False)
    pd.DataFrame({"cluster": [0, 1, 2, 3], "JT_p_two_sided": [1e-5, 0.2, 1e-4, 1e-6]}).to_csv(
        d / "finalise" / "jonckheere_terpstra_trends.csv", index=False)
    return d


def test_figure_5_uses_neutral_position_and_stage9_comparison(tmp_path):
    from predict.figures.figure_5_c8_anatomy import build as f5
    text = _text(f5(lesion_morph_dir=_lesion_dir(tmp_path)))
    low = text.lower()
    assert "proximal" not in low.replace("no proximal", "") and "distal" not in low
    assert "superior" in low
    assert "0.75 (c8-like) vs 0.15" in low            # medians of stage 9's per-carrier table


def test_figure_5_refuses_a_missing_or_mismatched_c8(tmp_path):
    from predict.figures.figure_5_c8_anatomy import build as f5
    with pytest.raises(ValueError, match="C8-like signature"):
        f5(lesion_morph_dir=_lesion_dir(tmp_path / "a", with_c8=False))
    with pytest.raises(ValueError, match="stage 9 C8 cluster"):
        f5(lesion_morph_dir=_lesion_dir(tmp_path / "b", deep_cluster=2))


def test_figure_4_names_classes_by_rule_without_progression(tmp_path):
    from predict.figures.figure_4_burden_composition import build as f4
    text = _text(f4(lesion_morph_dir=_lesion_dir(tmp_path)))
    low = text.lower()
    assert "gmm" not in low and "maturation" not in low and "trajector" not in low
    assert "max hu >= 500" in low and "cross-sectional" in low
