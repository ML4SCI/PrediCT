"""Tests for predict.analyse.derived_features.

The crucial regression test is `test_matches_prepare_matrix_byte_for_byte`:
the stage-7 re-derivation MUST produce the exact same numeric values as
``predict.reduce.prepare_matrix.compute_*``. If these diverge, the stage-7
analysis silently differs from stage 5, which is a publication-grade
failure mode.

Other tests cover edge cases (no vessels with calcium, single vessel,
multiple vessels, missing input columns).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predict.analyse.derived_features import (
    augment_raw_with_derived,
    derive_high_density_fraction,
    derive_vessel_burden_gini,
)
from predict.reduce.prepare_matrix import (
    apply_d018_binarisation,
    compute_high_density_fraction,
    compute_vessel_burden_gini,
)


# ─────────────────────── helpers ───────────────────────


def _synthetic_cohort(seed: int = 0, n: int = 30) -> pd.DataFrame:
    """Synthetic cohort with the 4 vessel agatston columns and 16
    density-tier columns, plus some unrelated columns to test column
    selectivity."""
    rng = np.random.default_rng(seed)
    pids = [f"p{i:02d}" for i in range(n)]
    df = pd.DataFrame({"pid": pids}).set_index("pid")
    # Per-vessel agatston (heavy-tailed, with zeros)
    for v in ("lad", "rca", "lcx", "lm"):
        # Mix of zero patients and substantial-burden patients
        burst = rng.exponential(50, n)
        burst[rng.random(n) < 0.3] = 0.0
        df[f"agatston_{v}"] = burst
    # Density tier counts
    for t in (1, 2, 3, 4):
        for v in ("lad", "rca", "lcx", "lm"):
            df[f"n_rois_d{t}_{v}"] = rng.integers(0, 5, n).astype(float)
    # Unrelated column to verify column selectivity
    df["unused_column"] = rng.normal(0, 1, n)
    return df


# ─────────────────────── byte-identity regression ───────────────────────


def test_high_density_fraction_matches_prepare_matrix_byte_for_byte():
    df = _synthetic_cohort(seed=0)
    stage7 = derive_high_density_fraction(df)
    stage5 = compute_high_density_fraction(df)
    assert stage7 is not None
    assert stage5 is not None
    pd.testing.assert_series_equal(stage7, stage5, check_names=False)


def test_vessel_burden_gini_matches_prepare_matrix_byte_for_byte():
    df = _synthetic_cohort(seed=0)
    stage7 = derive_vessel_burden_gini(df)
    stage5 = compute_vessel_burden_gini(df)
    assert stage7 is not None
    assert stage5 is not None
    pd.testing.assert_series_equal(stage7, stage5, check_names=False)


def test_matches_prepare_matrix_across_seeds():
    """Multiple random seeds in case one seed happens to mask a bug."""
    for seed in (1, 7, 42, 123, 999):
        df = _synthetic_cohort(seed=seed, n=50)
        s7 = derive_vessel_burden_gini(df)
        s5 = compute_vessel_burden_gini(df)
        pd.testing.assert_series_equal(s7, s5, check_names=False)
        s7 = derive_high_density_fraction(df)
        s5 = compute_high_density_fraction(df)
        pd.testing.assert_series_equal(s7, s5, check_names=False)


# ─────────────────────── high_density_fraction edge cases ───────────────────────


def test_high_density_fraction_zero_when_no_lesions():
    df = pd.DataFrame({
        f"n_rois_d{t}_{v}": [0]
        for t in (1, 2, 3, 4) for v in ("lad", "rca", "lcx", "lm")
    })
    out = derive_high_density_fraction(df)
    assert out.iloc[0] == 0.0


def test_high_density_fraction_one_when_all_in_d34():
    df = pd.DataFrame({
        f"n_rois_d{t}_{v}": [1 if t in (3, 4) else 0]
        for t in (1, 2, 3, 4) for v in ("lad", "rca", "lcx", "lm")
    })
    out = derive_high_density_fraction(df)
    assert out.iloc[0] == 1.0


def test_high_density_fraction_half_when_even_split():
    df = pd.DataFrame({
        f"n_rois_d{t}_{v}": [1] for t in (1, 2, 3, 4)
        for v in ("lad", "rca", "lcx", "lm")
    })
    out = derive_high_density_fraction(df)
    assert out.iloc[0] == 0.5


def test_high_density_fraction_returns_none_on_missing_columns():
    df = pd.DataFrame({"agatston_lad": [100]})
    assert derive_high_density_fraction(df) is None


# ─────────────────────── vessel_burden_gini edge cases ───────────────────────


def test_vessel_burden_gini_zero_for_no_burden():
    df = pd.DataFrame({
        "agatston_lad": [0.0], "agatston_rca": [0.0],
        "agatston_lcx": [0.0], "agatston_lm": [0.0],
    })
    out = derive_vessel_burden_gini(df)
    assert out.iloc[0] == 0.0


def test_vessel_burden_gini_single_vessel_is_maximal():
    """D040: one calcified vessel of four -> Gini 0.75 (most focal)."""
    df = pd.DataFrame({
        "agatston_lad": [500.0], "agatston_rca": [0.0],
        "agatston_lcx": [0.0], "agatston_lm": [0.0],
    })
    out = derive_vessel_burden_gini(df)
    assert out.iloc[0] == pytest.approx(0.75)


def test_vessel_burden_gini_two_equal_vessels_of_four():
    """Two equal vessels, two calcium-free: half the vessels carry all the
    burden -> Gini 0.5 (D040 counts the calcium-free vessels)."""
    df = pd.DataFrame({
        "agatston_lad": [100.0], "agatston_rca": [100.0],
        "agatston_lcx": [0.0], "agatston_lm": [0.0],
    })
    out = derive_vessel_burden_gini(df)
    assert out.iloc[0] == pytest.approx(0.5)


def test_vessel_burden_gini_positive_for_two_unequal_vessels():
    """Two vessels with very different burden -> Gini > 0."""
    df = pd.DataFrame({
        "agatston_lad": [1000.0], "agatston_rca": [10.0],
        "agatston_lcx": [0.0], "agatston_lm": [0.0],
    })
    out = derive_vessel_burden_gini(df)
    assert out.iloc[0] > 0.4


def test_vessel_burden_gini_increases_with_concentration():
    """Two vessels: as imbalance grows, Gini grows."""
    df = pd.DataFrame({
        "agatston_lad": [100.0, 500.0, 990.0],
        "agatston_rca": [100.0,  50.0,  10.0],
        "agatston_lcx": [0.0, 0.0, 0.0],
        "agatston_lm":  [0.0, 0.0, 0.0],
    })
    out = derive_vessel_burden_gini(df)
    assert out.iloc[0] == pytest.approx(0.5)   # two equal of four
    assert out.iloc[0] < out.iloc[1] < out.iloc[2] < 0.75


def test_vessel_burden_gini_returns_none_on_missing_columns():
    df = pd.DataFrame({"agatston_lad": [100], "agatston_rca": [50]})
    # Missing lcx + lm
    assert derive_vessel_burden_gini(df) is None


# ─────────────────────── augment_raw_with_derived ───────────────────────


def test_augment_adds_both_derived_columns():
    df = _synthetic_cohort()
    out = augment_raw_with_derived(df)
    assert "high_density_fraction" in out.columns
    assert "vessel_burden_gini" in out.columns


def test_augment_preserves_existing_column():
    df = _synthetic_cohort()
    df["high_density_fraction"] = 0.999
    out = augment_raw_with_derived(df)
    # Pre-existing column is not overwritten
    assert (out["high_density_fraction"] == 0.999).all()


def test_augment_skips_when_inputs_missing():
    """If the input columns aren't present, the augment is a no-op for
    the affected derived feature."""
    df = pd.DataFrame({
        # has agatston columns but no density tier columns
        "agatston_lad": [100.0], "agatston_rca": [50.0],
        "agatston_lcx": [10.0], "agatston_lm": [0.0],
    })
    out = augment_raw_with_derived(df)
    assert "vessel_burden_gini" in out.columns
    assert "high_density_fraction" not in out.columns


def test_augment_does_not_mutate_input():
    df = _synthetic_cohort()
    cols_before = set(df.columns)
    _ = augment_raw_with_derived(df)
    assert set(df.columns) == cols_before


# ─────────────────────── has_dense_calcium (D018 at raw scale) ───────────────────────


def test_augment_adds_has_dense_calcium_identical_to_stage5():
    """Stage 7 must see the same has_dense_calcium stage 5 clusters on."""
    df = _synthetic_cohort()
    df["dense_calcium_count"] = [0, 1, 3, 0, 2] * 6
    out = augment_raw_with_derived(df)
    stage5, _, applied = apply_d018_binarisation(
        df.reset_index(), ["dense_calcium_count"])
    assert applied
    assert out["has_dense_calcium"].tolist() == stage5["has_dense_calcium"].tolist()
    assert "dense_calcium_count" in out.columns          # raw count kept


def test_augment_skips_has_dense_calcium_without_its_source():
    out = augment_raw_with_derived(_synthetic_cohort())
    assert "has_dense_calcium" not in out.columns
