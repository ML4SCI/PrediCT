# Pre-registration: isolated LM calcification at low total Agatston burden

**Locked**: 2026-06-15
**Status**: pre-systematic-query. The thresholds and hypotheses below
were locked BEFORE running any systematic analysis. Three patients
(pids 290, 426, 252) were inspected manually as outliers in the PCA
scatter plot prior to lock; that visual inspection informed the choice
of the displacement criterion but no full-cohort query was run before
the criterion was written down here.

If a threshold is later found to disqualify a clinically obvious case
or to admit a clinically irrelevant case, the failure mode is
documented in `findings.md` and the threshold is NOT relaxed to
produce a positive verdict.

---

## P1. Displaced-patient identification (Step 1, run.py)

A patient `p` is classified as **displaced** within the low-burden
tertile iff all three of the following hold:

| Criterion | Threshold | Source |
|---|---|---|
| Burden tier | `qcut(agatston_total, q=3, labels=['low','mid','high']) == 'low'` | `outputs/06_reduce/cohort_metadata.csv` |
| Displaced along PC1 (rightward) | `pc1 > 0`, **OR** | `outputs/06_reduce/pca_scores.npy` |
| Displaced along PC2 (upward) | `pc2 > 2.5` | (same) |

The OR connective in the second criterion means a patient qualifies
if displaced on either principal axis. Both thresholds (0 on PC1 and
2.5 on PC2) sit well outside the dense core of the low-burden cluster,
which centres near PC1 ≈ -3, PC2 ≈ 0.

### Rationale for thresholds

* **PC1 > 0**: PC1 is empirically the burden axis (35% of variance,
  positive correlation with agatston_total). PC1 = 0 sits at the
  cohort mean. A low-burden patient projecting to PC1 > 0 has a
  feature profile consistent with at least mean burden despite a
  low total Agatston. This catches the "diffuse multi-vessel disease
  at low burden" pattern (e.g. pid 252 was the manually-inspected
  example, though pid 252's actual PC1 = -0.31 placed them just
  inside the cluster - the threshold did not catch them).
* **PC2 > 2.5**: PC2 is empirically the anatomical-location axis
  (11% of variance, loadings dominated by LM features). PC2 > 2.5
  is more than 2 standard deviations above the cohort mean. A
  low-burden patient at PC2 > 2.5 has unusual concentration along
  the LM axis. This catches the "LM-isolated" pattern (e.g. pids
  290 and 426 inspected manually).

The OR connective is biologically motivated: the two axes capture
qualitatively different unusual patterns. A patient unusual on
either axis is biologically displaced even if not on both.

## P2. Pre-registered hypothesis on the displaced subgroup

The biological hypothesis (locked before systematic query):

> **Displaced low-burden patients have a higher rate of left-main
> calcification than non-displaced low-burden patients.**

Test: one-sided Fisher exact test on the 2x2 contingency table
of (displaced / non-displaced) x (LM-positive / LM-negative), with
alternative = "greater" (displaced LM rate > non-displaced LM rate).

**PASS criterion**: Fisher exact p < 0.001 AND displaced LM rate
≥ 50%.

The 50% effect-size floor prevents declaring victory on a
statistically significant difference of small clinical magnitude.

### Why this is the right hypothesis

PC2's top loadings (`lesion_count_lm` +0.43, `max_hu_lm` +0.43,
`n_rois_d4_lm` +0.36, `agatston_lm` +0.33) make the LM enrichment
prediction the natural consequence of how PC2 was constructed. A
displaced subgroup with no LM enrichment would mean PC2's loading
structure does not translate into patient-level LM identification -
itself an informative null.

## P3. Cross-stratum replication (Step 2, run.py)

The displaced-vs-non-displaced LM rate comparison must hold
independently in both kernel strata (Qr36d/2 and I30f/3).

**Replication PASS criterion**: in each stratum independently:

* displaced LM rate ≥ 50%
* non-displaced LM rate < 50% (so the contrast is real, not a
  baseline-rate artifact)

If the pattern holds in only one stratum, the finding is
"kernel-dependent" and not eligible for paper inclusion.

## P4. Density profile of the displaced subgroup's LM lesions

Pre-registered classification thresholds for LM lesion density
distribution by Agatston tier:

| Tier | max-HU range |
|---|---|
| W1 (soft microspot) | 130 - 199 |
| W2 (moderate-soft) | 200 - 299 |
| W3 (moderate-dense) | 300 - 399 |
| W4 (dense) | ≥ 400 |

The density profile is descriptive (no PASS/FAIL criterion). It
distinguishes between two biologically-distinct framings:

* If the median max-HU is in the W1/W2 range: framing is "early-stage
  LM disease detected by morphology"
* If the median max-HU is in the W3/W4 range: framing is "advanced
  isolated LM disease distinct from soft microspot biology"

## P5. Lesion-cluster overlap analysis

For every LM lesion belonging to a displaced patient, look up its
cluster id in `outputs/exploratory/lesion_morphology/lesion_cluster_labels.csv`.

**Pre-registered question**: do any displaced patients' LM lesions
fall in clusters 10 or 11 (the LAD/LM-region compact-dense clusters
identified in `experiments/lad_phenotype/`)?

* If 0 LM lesions fall in clusters 10/11: the LM-isolated finding is
  fully distinct from the high-burden LAD/LM phenotype, and the
  framing of "three independent findings across the burden
  spectrum" stands.
* If a non-trivial fraction (≥20%) of displaced LM lesions fall in
  clusters 10/11: the LM-isolated finding is a low-burden expression
  of the LAD/LM phenotype, and the framing collapses into one
  biological entity at two burden levels.

## Termination conditions (no negotiation)

| Condition | Outcome |
|---|---|
| Displaced subgroup is empty (no patient meets P1) | Experiment terminates; `findings.md` records "no displaced low-burden subgroup at pre-registered thresholds." |
| Fisher exact p ≥ 0.001 (P2 fails) | Pattern null; `findings.md` records the negative result. |
| Displaced LM rate < 50% (P2 effect-size floor) | Pattern small-effect; `findings.md` records as "weakly significant but not clinically substantive." |
| One stratum fails P3 | Finding labelled "kernel-dependent"; not eligible for paper inclusion. |
| ≥ 20% of displaced LM lesions fall in clusters 10/11 (P5 negative) | Reframe as low-burden expression of LAD/LM phenotype rather than independent entity. |

## Hash of this document at lock time

```bash
sha256sum experiments/lm_isolated_low_burden/plan.md
```

The hash will be recorded in `run_header.json` at first successful
run so any post-hoc edit is detectable.
