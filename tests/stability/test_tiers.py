"""Tests for predict.stability.tiers (D034)."""
from __future__ import annotations

import pandas as pd
import pytest

from predict.stability.tiers import TIERS, assign_tiers

ARMS = ("S1", "S2", "S3", "S4", "S5", "S6")


def _frames():
    feats = ["fail", "primary_only", "ci_ok", "z_ok", "arms_ok", "excellent"]
    primary = pd.DataFrame({
        "passes_gate": [False, True, True, True, True, True],
        "icc":         [0.50, 0.76, 0.85, 0.85, 0.85, 0.95],
        "ci_low":      [0.45, 0.72, 0.80, 0.80, 0.80, 0.93],
    }, index=feats)
    arm = pd.DataFrame(True, index=feats, columns=list(ARMS))
    arm.loc["ci_ok", "S1"] = False                     # stops at z-shift
    arm.loc["z_ok", "S5"] = False                      # stops at other arms
    arm["S3"] = False                                  # S3 must be ignored
    return primary, arm


def test_ladder_assignment():
    primary, arm = _frames()
    t = assign_tiers(primary, arm, 0.75)
    assert t.loc["fail", "tier"] == "T0_fail"
    assert t.loc["primary_only", "tier"] == "T1_primary"
    assert t.loc["ci_ok", "tier"] == "T2_ci"
    assert t.loc["z_ok", "tier"] == "T3_zshift"
    assert t.loc["arms_ok", "tier"] == "T4_all_arms"
    assert t.loc["excellent", "tier"] == "T5_excellent"


def test_tiers_are_nested():
    primary, arm = _frames()
    t = assign_tiers(primary, arm, 0.75)
    for lo, hi in zip(TIERS, TIERS[1:]):
        assert not (t[hi] & ~t[lo]).any()


def test_stops_at_reason():
    primary, arm = _frames()
    t = assign_tiers(primary, arm, 0.75)
    assert t.loc["primary_only", "stops_at"] == "lower CI < 0.75"
    assert t.loc["ci_ok", "stops_at"] == "fails S1 z-shift"
    assert t.loc["excellent", "stops_at"] == ""


def test_missing_arm_raises():
    primary, arm = _frames()
    with pytest.raises(KeyError):
        assign_tiers(primary, arm.drop(columns=["S4"]), 0.75)
