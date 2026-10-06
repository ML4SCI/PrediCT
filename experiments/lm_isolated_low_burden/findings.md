# Isolated left-main calcification at low total burden: findings

**Status**: exploratory, pre-registered (`plan.md`; thresholds recorded in
`run_header.json`).
**Date**: 2026-10-06.
**Run**: stage 9 of the reported pipeline run, published as
`outputs/exploratory/lm_isolated_low_burden/` under
`outputs/exploratory/COMPLETE_stage9.json` (D043). All numbers below are
read from that folder.
**Inputs**: the stage 5 patient PCA scores and cohort metadata (436
patients), `features.csv`, the k=12 lesion partition and this run's
LAD-signature clusters (2 and 11).

## Summary

In the low-burden Agatston tertile (146 patients), a pre-registered rule on
the patient PCA (P1: PC1 > 0 or PC2 > 2.5) selects 11 displaced patients.
9 of the 11 have left-main calcification, against 16 of the other 135
low-tertile patients.

| Criterion | Pre-registered rule | Result | Verdict |
|---|---|---|---|
| P2 LM enrichment | displaced LM rate >= 0.5 and one-sided Fisher p < 0.001 | 9 / 11 (82%; Wilson 95% 52 to 95%) vs 16 / 135 (12%); OR 33.5, p 1.4e-6 | PASS |
| P3 stratum replication | displaced LM rate >= 0.5 in each kernel stratum | I30f/3 7 / 7 (Wilson 65 to 100%); Qr36d/2 2 / 4 (15 to 85%) | PASS |
| P4 density profile | descriptive | 9 LM lesions at Agatston density weight 3 or 4 (max HU >= 300), none at 1 or 2; median max HU 318 | dense |
| P5 distinct from the LAD signature | < 20% of displaced LM lesions in the LAD-signature clusters | 0 of 11 LM lesions | PASS |

All pre-registered criteria pass.

## The displaced patients

| pid | kernel | Agatston total | Agatston LM | LM lesions | calcified arteries |
|---|---|---|---|---|---|
| 198 | I30f/3 | 8.9 | 8.9 | 1 | 1 |
| 105 | I30f/3 | 9.7 | 9.7 | 1 | 1 |
| 200 | I30f/3 | 12.4 | 12.4 | 1 | 1 |
| 170 | I30f/3 | 17.5 | 17.5 | 1 | 1 |
| 290 | Qr36d/2 | 17.9 | 16.6 | 3 | 2 |
| 21 | I30f/3 | 19.1 | 19.1 | 1 | 1 |
| 43 | I30f/3 | 24.7 | 24.7 | 1 | 1 |
| 204 | Qr36d/2 | 29.3 | 0 | 0 | 1 (LAD) |
| 311 | Qr36d/2 | 30.7 | 30.7 | 1 | 1 |
| 109 | Qr36d/2 | 46.6 | 0 | 0 | 1 (LAD) |
| 426 | I30f/3 | 48.3 | 48.3 | 1 | 1 |

(`displaced_patients.csv`). Ten of the eleven have calcium in a single
artery; in eight of them it is a single LM lesion and nothing else.

## Reading

- The rule picks out low-burden patients whose calcium sits almost
  entirely in the left main. In that sense the result partly restates the
  selection: a patient whose only calcium is an LM lesion has an unusual
  feature profile and projects away from the low-burden core.
- What the analysis adds is that this group is real and specific: it is
  strongly enriched for LM calcification compared with other low-burden
  patients, its LM lesions are dense rather than soft microspots, and none
  of them falls in the LAD-signature lesion clusters.
- Replication is strong in I30f/3 (7 of 7) and weak in Qr36d/2 (2 of 4,
  exactly at the 0.5 threshold, with a wide interval). The subgroup is
  small (11 patients).
- Conventional Agatston categories place these patients in the lowest
  burden group. Whether isolated LM calcification at low total burden
  carries different risk cannot be tested here: COCA has no outcomes. This
  is a hypothesis for a cohort with follow-up, not a clinical finding.

## Limitations

- 11 patients; one stratum has 4.
- The PCA, the displacement thresholds and the tertiles come from this
  cohort; no external replication.
- Two displaced patients (109, 204) have no LM calcium; their displacement
  comes from a single LAD lesion.
- Lesion-cluster membership depends on the k=12 lesion partition.
