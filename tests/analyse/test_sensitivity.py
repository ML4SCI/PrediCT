"""Tests for predict.analyse.sensitivity (D028 low-burden sensitivity)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predict.analyse.derived_features import augment_raw_with_derived
from predict.analyse.sensitivity import low_burden_sensitivity, robust_pids


def _inputs(n: int = 80, seed: int = 0):
    rng = np.random.default_rng(seed)
    pids = [str(i) for i in range(n)]
    diffuse = np.arange(n) % 2 == 1
    raw = pd.DataFrame({
        "lesion_count_lad": np.where(diffuse, rng.integers(0, 3, n), rng.integers(3, 7, n)),
        "n_calcified_arteries": np.where(diffuse, rng.integers(3, 5, n), rng.integers(1, 3, n)),
        "dist_from_top_max": np.where(diffuse, rng.uniform(60, 120, n), rng.uniform(5, 40, n)),
        "gini_lesion_volume": np.where(diffuse, rng.uniform(0, .3, n), rng.uniform(.5, .9, n)),
        "first_to_last_dist_lad": np.where(diffuse, rng.uniform(40, 90, n), rng.uniform(0, 20, n)),
        **{f"agatston_{v}": rng.gamma(2.0, 100.0, n) for v in ("lad", "rca", "lcx", "lm")},
    }, index=pd.Index(pids, name="pid"))
    raw = augment_raw_with_derived(raw)
    labels = pd.Series(np.where(diffuse, "diffuse", "focal"), index=raw.index)
    meta = pd.DataFrame({"pid": pids,
                         "agatston_total": np.where(diffuse, rng.gamma(4, 200, n), rng.gamma(2, 50, n)),
                         "low_burden_flag": np.arange(n) % 4 == 0})
    return raw, labels, meta


def test_restricts_to_low_burden_false_and_keeps_labels():
    raw, labels, meta = _inputs()
    res = low_burden_sensitivity(raw, labels, meta, list(raw.columns),
                                 full_d024_interpretation="confounded", full_d025_primary_passes=True)
    v = res["verdict"]
    assert v["n_patients"] == 60
    assert v["n_focal"] + v["n_diffuse"] == 60
    assert set(res["hypotheses"]["cohort"]) == {"robust"}
    assert len(res["hypotheses"]) == 6
    assert res["orthogonality"].iloc[0]["n_focal"] == v["n_focal"]
    assert v["d024_same_as_full"] == (v["d024_interpretation"] == "confounded")
    assert v["d025_primary_same_as_full"] == bool(v["d025_primary"]["passes"])


def test_signatures_carry_the_robust_label():
    raw, labels, meta = _inputs()
    res = low_burden_sensitivity(raw, labels, meta, list(raw.columns),
                                 full_d024_interpretation="confounded", full_d025_primary_passes=True)
    assert len(res["signatures"]) > 0
    assert set(res["signatures"]["cohort"]) == {"robust"}


def test_unrecognised_low_burden_flag_raises():
    meta = pd.DataFrame({"pid": ["1", "2"], "low_burden_flag": ["False", "maybe"]})
    with pytest.raises(ValueError, match="low_burden_flag"):
        robust_pids(meta)
