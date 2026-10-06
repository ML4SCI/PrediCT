"""Tests for predict.analyse.monotonicity (D026).

Coverage:
* classify_feature: 4-class rule on all corner cases
* compute_monotonicity: Spearman + Kendall match scipy reference
* burden block + |rho| > 0.5 -> burden_tracking
* spatial block + |rho| < 0.3 -> spatial_tracking
* structure blocks + |rho| < 0.3 -> structure_tracking
* mixed-band catches 0.3 <= |rho| < 0.5
* density_tier block at |rho| < 0.3 -> mixed (per D026 strict reading)
* classification_summary produces expected counts
* NaN / constant / missing input is not_evaluable, never a class (D026.1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from predict.analyse.monotonicity import (
    BURDEN_TRACKING_RHO,
    SPATIAL_BLOCK,
    STRUCTURE_BLOCKS,
    WEAK_RHO_THRESHOLD,
    classification_summary,
    classify_feature,
    compute_monotonicity,
)


# ─────────────────────── classify_feature rules ───────────────────────


def test_classify_burden_tracking_at_rho_0_5():
    assert classify_feature(0.5, "burden") == "burden_tracking"
    assert classify_feature(-0.5, "spatial") == "burden_tracking"
    assert classify_feature(0.99, "any") == "burden_tracking"


def test_classify_burden_tracking_overrides_block_at_high_rho():
    """A spatial feature with high rho is still burden_tracking."""
    assert classify_feature(0.7, "spatial") == "burden_tracking"


def test_classify_spatial_tracking_low_rho_in_spatial_block():
    assert classify_feature(0.1, "spatial") == "spatial_tracking"
    assert classify_feature(-0.2, "spatial") == "spatial_tracking"
    assert classify_feature(0.0, "spatial") == "spatial_tracking"


def test_classify_structure_tracking_low_rho_in_structure_blocks():
    for block in ("hu_statistics", "texture", "shape"):
        assert classify_feature(0.15, block) == "structure_tracking"
        assert classify_feature(-0.25, block) == "structure_tracking"


def test_classify_mixed_in_intermediate_rho_band():
    # 0.3 <= |rho| < 0.5 is mixed regardless of block
    for block in ("spatial", "hu_statistics", "texture", "shape", "burden",
                  "density_tier"):
        assert classify_feature(0.4, block) == "mixed"
        assert classify_feature(-0.35, block) == "mixed"
        assert classify_feature(0.49, block) == "mixed"


def test_classify_mixed_for_density_tier_low_rho():
    """D026: density_tier features at |rho| < 0.3 should be mixed
    (only spatial / structure blocks get specific labels at low rho)."""
    assert classify_feature(0.1, "density_tier") == "mixed"
    assert classify_feature(-0.1, "burden") == "mixed"


def test_classify_mixed_for_unknown_block_low_rho():
    assert classify_feature(0.1, "unknown") == "mixed"


def test_classify_not_evaluable_on_nan():
    assert classify_feature(float("nan"), "spatial") == "not_evaluable"


def test_classify_boundary_exactly_at_0_3():
    # |rho| == 0.3 is the lower boundary of mixed; NOT structure / spatial
    assert classify_feature(0.3, "spatial") == "mixed"
    assert classify_feature(-0.3, "hu_statistics") == "mixed"


def test_classify_boundary_exactly_at_0_5():
    # |rho| == 0.5 IS burden_tracking
    assert classify_feature(0.5, "spatial") == "burden_tracking"
    assert classify_feature(-0.5, "shape") == "burden_tracking"


# ─────────────────────── compute_monotonicity ───────────────────────


def _synthetic_monotonicity(seed: int = 0, n: int = 100):
    """Build a cohort with features at known correlation strengths."""
    rng = np.random.default_rng(seed)
    pids = [f"p{i:03d}" for i in range(n)]
    agatston = pd.Series(np.linspace(0, 1000, n), index=pids)
    df = pd.DataFrame({
        # Perfect rank correlation -> Spearman = 1
        "burden_strong": np.arange(n, dtype=float),
        # Negative perfect rank correlation
        "burden_strong_neg": -np.arange(n, dtype=float),
        # Moderate correlation: noise added
        "mixed_signal": np.arange(n, dtype=float) + rng.normal(0, 50, n),
        # Independent
        "spatial_indep": rng.normal(0, 1, n),
        # Independent + structure block
        "structure_indep": rng.normal(0, 1, n),
    }, index=pids)
    return df, agatston


def test_compute_monotonicity_returns_row_per_feature():
    df, agatston = _synthetic_monotonicity()
    block_lookup = {
        "burden_strong": "burden",
        "burden_strong_neg": "burden",
        "mixed_signal": "burden",
        "spatial_indep": "spatial",
        "structure_indep": "hu_statistics",
    }
    result = compute_monotonicity(
        df, list(df.columns), agatston, "t", block_lookup,
    )
    assert len(result) == 5
    assert set(result["feature"]) == set(df.columns)


def test_compute_monotonicity_spearman_matches_scipy():
    df, agatston = _synthetic_monotonicity()
    result = compute_monotonicity(
        df, ["mixed_signal"], agatston, "t",
        block_lookup={"mixed_signal": "spatial"},
    )
    row = result.iloc[0]
    ref_rho, ref_p = stats.spearmanr(
        df["mixed_signal"].to_numpy(), agatston.to_numpy(),
    )
    assert abs(row["spearman_rho"] - ref_rho) < 1e-12
    assert abs(row["spearman_p"] - ref_p) < 1e-12


def test_compute_monotonicity_kendall_matches_scipy():
    df, agatston = _synthetic_monotonicity()
    result = compute_monotonicity(
        df, ["mixed_signal"], agatston, "t",
        block_lookup={"mixed_signal": "spatial"},
    )
    row = result.iloc[0]
    ref_tau, ref_p = stats.kendalltau(
        df["mixed_signal"].to_numpy(), agatston.to_numpy(), variant="b",
    )
    assert abs(row["kendall_tau"] - ref_tau) < 1e-12


def test_compute_monotonicity_burden_strong_classified_as_burden_tracking():
    df, agatston = _synthetic_monotonicity()
    result = compute_monotonicity(
        df, ["burden_strong"], agatston, "t",
        block_lookup={"burden_strong": "burden"},
    )
    assert result.iloc[0]["classification"] == "burden_tracking"
    assert result.iloc[0]["spearman_rho"] >= BURDEN_TRACKING_RHO


def test_compute_monotonicity_burden_strong_neg_classified_as_burden_tracking():
    df, agatston = _synthetic_monotonicity()
    result = compute_monotonicity(
        df, ["burden_strong_neg"], agatston, "t",
        block_lookup={"burden_strong_neg": "burden"},
    )
    assert result.iloc[0]["classification"] == "burden_tracking"
    assert result.iloc[0]["spearman_rho"] <= -BURDEN_TRACKING_RHO


def test_compute_monotonicity_spatial_indep_classified_as_spatial_tracking():
    df, agatston = _synthetic_monotonicity()
    result = compute_monotonicity(
        df, ["spatial_indep"], agatston, "t",
        block_lookup={"spatial_indep": SPATIAL_BLOCK},
    )
    assert result.iloc[0]["classification"] == "spatial_tracking"
    assert abs(result.iloc[0]["spearman_rho"]) < WEAK_RHO_THRESHOLD


def test_compute_monotonicity_structure_indep_classified_as_structure_tracking():
    df, agatston = _synthetic_monotonicity()
    block = list(STRUCTURE_BLOCKS)[0]
    result = compute_monotonicity(
        df, ["structure_indep"], agatston, "t",
        block_lookup={"structure_indep": block},
    )
    assert result.iloc[0]["classification"] == "structure_tracking"


def test_compute_monotonicity_handles_constant_feature():
    pids = [f"p{i:03d}" for i in range(50)]
    df = pd.DataFrame({"constant": [3.14] * 50}, index=pids)
    agatston = pd.Series(np.arange(50, dtype=float), index=pids)
    result = compute_monotonicity(
        df, ["constant"], agatston, "t",
        block_lookup={"constant": "spatial"},
    )
    row = result.iloc[0]
    assert np.isnan(row["spearman_rho"])
    assert row["classification"] == "not_evaluable"


def test_compute_monotonicity_handles_missing_feature():
    df, agatston = _synthetic_monotonicity()
    result = compute_monotonicity(
        df, ["nonexistent"], agatston, "t",
        block_lookup={"nonexistent": "spatial"},
    )
    assert len(result) == 1
    assert np.isnan(result.iloc[0]["spearman_rho"])
    assert result.iloc[0]["n_used"] == 0
    assert result.iloc[0]["classification"] == "not_evaluable"


def test_compute_monotonicity_drops_nans_pairwise():
    pids = [f"p{i:03d}" for i in range(50)]
    feature_vals = np.linspace(0, 100, 50)
    feature_vals[5] = np.nan
    feature_vals[15] = np.nan
    df = pd.DataFrame({"f": feature_vals}, index=pids)
    agatston_vals = np.linspace(0, 100, 50)
    agatston_vals[7] = np.nan
    agatston = pd.Series(agatston_vals, index=pids)
    result = compute_monotonicity(
        df, ["f"], agatston, "t", block_lookup={"f": "spatial"},
    )
    # 50 - 3 NaN positions = 47 used
    assert result.iloc[0]["n_used"] == 47


def test_compute_monotonicity_raises_on_no_shared_pids():
    df = pd.DataFrame({"f": [1.0]}, index=["a"])
    agatston = pd.Series([10.0], index=["b"])
    with pytest.raises(ValueError, match="no shared pids"):
        compute_monotonicity(
            df, ["f"], agatston, "t", block_lookup={"f": "spatial"},
        )


# ─────────────────────── classification_summary ───────────────────────


def test_classification_summary_counts_per_cohort():
    df = pd.DataFrame({
        "cohort": ["A"] * 4 + ["B"] * 3,
        "feature": [f"f{i}" for i in range(7)],
        "classification": [
            "burden_tracking", "burden_tracking", "spatial_tracking", "mixed",
            "burden_tracking", "structure_tracking", "spatial_tracking",
        ],
    })
    summary = classification_summary(df)
    a_row = summary[summary["cohort"] == "A"].iloc[0]
    assert a_row["burden_tracking"] == 2
    assert a_row["spatial_tracking"] == 1
    assert a_row["mixed"] == 1
    assert a_row["structure_tracking"] == 0
    assert a_row["total"] == 4

    b_row = summary[summary["cohort"] == "B"].iloc[0]
    assert b_row["burden_tracking"] == 1
    assert b_row["structure_tracking"] == 1
    assert b_row["spatial_tracking"] == 1
    assert b_row["total"] == 3


def test_classification_summary_ensures_all_columns_present():
    df = pd.DataFrame({
        "cohort": ["A"],
        "feature": ["f0"],
        "classification": ["burden_tracking"],
    })
    summary = classification_summary(df)
    # Every classification column + total
    assert {"burden_tracking", "structure_tracking", "spatial_tracking",
            "mixed", "not_evaluable", "total"} <= set(summary.columns)
