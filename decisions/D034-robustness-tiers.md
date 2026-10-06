# D034 Robustness tiers on top of the D033 gate (exploratory)

**Date**: 2026-09-25
**Stage**: stability
**Status**: Active, exploratory. Defined after the D033 pass counts were seen, as part of the study's declared method exploration. Which gate feeds the final analysis is chosen later on criteria that do not use the downstream findings (realism, parity, seed stability, tier behaviour), and confirmed on data not used to choose it.
**Module**: `src/predict/stability/tiers.py`, `scripts/05c_robustness_tiers.py`

## Decision

Grade every one of the 175 features on a nested ladder built from D033 outputs (no new extraction). Each rung adds one criterion:

| Tier | Adds | Why |
|------|------|-----|
| T1 primary | passes D033: ICC(1,1) >= 0.75, <= 5% of patients excluded | the locked gate |
| T2 ci | lower 95% CI bound >= 0.75 | the point estimate alone lets features sit on the threshold (two D033 failures had CIs spanning 0.75) |
| T3 zshift | passes secondary arm S1 (slice-position shift) | slice position is the biggest real rescan factor the primary gate leaves out |
| T4 all_arms | passes S2, S4, S5, S6 | robust to every other variation we can measure |
| T5 excellent | ICC(1,1) >= 0.90 | Koo and Li 2016 "excellent" |

A feature's tier is the highest rung it reaches (T0 = fails D033). S3 (130 HU threshold mask) is never used: it fails every radiomics feature on the exclusion rule, not on ICC, so it carries no stability information.

Each tier is written as a stage-6-compatible feature list, so any tier can be carried through stages 6 to 8.

## Check included

ICC(3,1) absolute agreement is recomputed on the same 20-draw matrices and compared with ICC(1,1), with any pass-status change listed. Random draws have no systematic rater effect, so the two forms should nearly coincide; this answers whether the change of ICC form between D014 and D033 matters.

## Relation to earlier decisions

D033 is unchanged and remains the locked primary gate. D013 and D014 unchanged.

## Verified by

- `tests/stability/test_tiers.py`: ladder assignment, nesting, first-failed-rung reason, missing-arm error.
