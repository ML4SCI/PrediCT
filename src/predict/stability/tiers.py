"""D034 robustness tiers on top of the D033 gate.

A nested ladder; each rung adds one criterion to the one below it:

    T1 primary    passes D033 (ICC(1,1) >= 0.75, <= 5% excluded)
    T2 ci         T1 and the lower 95% CI bound >= 0.75
    T3 zshift     T2 and passes secondary arm S1 (slice-position shift)
    T4 all_arms   T3 and passes S2, S4, S5, S6 (every arm where ICC is measurable)
    T5 excellent  T4 and ICC(1,1) >= 0.90 (Koo and Li 2016 "excellent")

S3 (130 HU threshold mask) is never used: it fails every radiomics feature
on the D033 exclusion rule, not on ICC, so it carries no stability
information (D034).

A feature's tier is the highest rung it reaches; T0 means it fails D033.
Exploratory (D034): the gate carried downstream is chosen later on
criteria that do not use the downstream findings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TIERS: tuple[str, ...] = ("T1_primary", "T2_ci", "T3_zshift", "T4_all_arms", "T5_excellent")
ZSHIFT_ARM = "S1"
OTHER_ARMS: tuple[str, ...] = ("S2", "S4", "S5", "S6")
EXCELLENT_ICC = 0.90


def assign_tiers(primary: pd.DataFrame, arm_pass: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Per-feature criteria flags and highest tier.

    ``primary``: indexed by feature, columns ``passes_gate``, ``icc``, ``ci_low``
    (stage-5b ``icc_report_full.csv``). ``arm_pass``: indexed by feature, one
    boolean column per arm (stage-5b ``arms.csv`` pivoted). Missing arms raise.
    """
    need = [ZSHIFT_ARM, *OTHER_ARMS]
    missing = [a for a in need if a not in arm_pass.columns]
    if missing:
        raise KeyError(f"arms missing from arms.csv: {missing}")
    ap = arm_pass.reindex(primary.index)
    if ap[need].isna().any().any():
        bad = ap.index[ap[need].isna().any(axis=1)].tolist()
        raise KeyError(f"features missing from arms.csv: {bad[:5]}")

    out = pd.DataFrame(index=primary.index)
    out["icc"] = primary["icc"]
    out["ci_low"] = primary["ci_low"]
    out["c_primary"] = primary["passes_gate"].astype(bool)
    out["c_ci"] = primary["ci_low"].fillna(-np.inf) >= threshold
    out["c_zshift"] = ap[ZSHIFT_ARM].astype(bool)
    out["c_other_arms"] = ap[list(OTHER_ARMS)].astype(bool).all(axis=1)
    out["c_excellent"] = primary["icc"].fillna(-np.inf) >= EXCELLENT_ICC

    ladder = [out["c_primary"]]
    ladder.append(ladder[-1] & out["c_ci"])
    ladder.append(ladder[-1] & out["c_zshift"])
    ladder.append(ladder[-1] & out["c_other_arms"])
    ladder.append(ladder[-1] & out["c_excellent"])
    for name, member in zip(TIERS, ladder):
        out[name] = member

    tier = pd.Series("T0_fail", index=out.index)
    for name in TIERS:
        tier[out[name]] = name
    out["tier"] = tier

    # First rung the feature fails, for the reason column.
    first_fail = pd.Series("", index=out.index)
    labels = ("fails D033", "lower CI < 0.75", "fails S1 z-shift",
              "fails an S2/S4/S5/S6 arm", "ICC < 0.90")
    for crit, label in zip(("c_primary", "c_ci", "c_zshift", "c_other_arms", "c_excellent"), labels):
        hit = (first_fail == "") & ~out[crit]
        first_fail[hit] = label
    out["stops_at"] = first_fail
    return out


__all__ = ["TIERS", "ZSHIFT_ARM", "OTHER_ARMS", "EXCELLENT_ICC", "assign_tiers"]
