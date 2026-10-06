# Pre-registration: LAD-specific morphological phenotype experiment

**Locked**: 2026-06-10
**Status**: pre-data-inspection. Every threshold and hypothesis direction
below is fixed BEFORE the modules in this folder are executed against
real cohort outputs. Iterating on these thresholds after seeing which
clusters match is hidden p-hacking and undoes the pre-registration.

If a threshold is later found to be unworkable (e.g. zero clusters
match), the failure mode is documented in `findings.md` and the
experiment terminates with a NEGATIVE verdict. The threshold is NOT
relaxed to produce a positive verdict.

---

## P1. LAD-dominant lesion-cluster signature (Step 1, run.py)

A cluster `c` from the lesion-morphology k=12 partition qualifies as
LAD-dominant iff **all four** criteria hold simultaneously:

| Criterion | Threshold | Source column |
|---|---|---|
| LAD over-representation | `lad_obs_over_exp > 1.30` | `cluster_vessel_chi_square.csv` |
| RCA NOT over-represented (distinguishes from C8) | `rca_obs_over_exp < 1.00` | `cluster_vessel_chi_square.csv` |
| Substantively dense (not microspots) | `max_hu_median > 400` | `cluster_profiles.csv` |
| Substantively sized (not d1 noise) | `volume_mm3_median > 50` | `cluster_profiles.csv` |

Rationale:

* **LAD obs/exp > 1.30** matches the per-cluster heuristic the lesion
  experiment already uses for vessel bias. C8 RCA obs/exp = 1.99
  (strong); C6 LAD obs/exp ~ 1.37 (moderate). 1.30 catches both C6 and
  any cluster of similar magnitude.
* **RCA obs/exp < 1.00** prevents picking up clusters that are
  multi-vessel-biased (would catch C8 if not for the LAD constraint).
* **max_HU > 400** is the IBSI Agatston W4 floor + 1 step. Soft
  microspots (~160-185 HU per the lesion findings) and moderate
  nodules (~200-300 HU) fail. Only mature mineralised lesions pass.
* **volume > 50 mm³** sits above the cohort median (~5 mm³) and below
  C8's median (250 mm³). Catches "substantive but not massive" plaque.

If zero clusters match all four criteria, the experiment reports
"no LAD-dominant cluster qualifies under the pre-registered signature"
and terminates. The signature is NOT relaxed.

## P2. Within-LAD axial localisation (Step 2, analyse.py)

For each patient `p` with two or more LAD lesions, every LAD lesion `l`
gets a relative-z position within the patient's LAD z-range:

```
rel_z[p, l] = (max_z[p_LAD] - centroid_z[l]) /
              (max_z[p_LAD] - min_z[p_LAD])
```

`rel_z = 0` is most proximal (top of the LAD, closest to the LM origin
- clinically dangerous segment), `rel_z = 1` is most distal (apex).

**Coordinate-convention note (added 2026-06-15 after first run)**: In
DICOM patient coordinates, +z is SUPERIOR (toward the head). The heart
base sits superiorly, the apex inferiorly. Within the LAD: proximal =
HIGH z, distal = LOW z. The formula above inverts the raw min-max
scaling so that the convention "rel_z = 0 is proximal" matches the
underlying anatomy. An initial implementation used
`(centroid_z - min_z) / range` which flipped the axis (rel_z = 0
ended up at apex = distal). That was a coordinate-direction bug, not
a hypothesis change; the fix re-aligns code with plan intent. The
hypothesis direction is unchanged.

### Pre-registered hypothesis

LAD-dominant cluster lesions concentrate **proximally** (rel-z < 0.5).

Test: one-sided Mann-Whitney comparing rel-z of LAD-cluster lesions
against rel-z of all other LAD lesions, alternative = "less."

**PASS criterion**: MW p < 0.01 AND median rel-z difference >= 0.10
(LAD-cluster median at least 10% closer to proximal than other-LAD
median).

Rationale: proximal LAD is the widowmaker segment (anatomically
nearest to the LM bifurcation, supplies the largest myocardial
territory). If the LAD-dominant cluster represents an
anatomically-meaningful phenotype, proximal localisation is the
biologically expected direction. The 10% effect-size floor prevents
declaring victory on a statistically significant but practically
trivial difference.

## P3. Carrier patient signature (Step 3, analyse.py)

A patient is a **carrier** of cluster `c` iff they have one or more
lesions assigned to `c`. Otherwise non-carrier.

For each pre-registered comparison, report Cliff's delta + Mann-Whitney
p-value with FDR-BH correction across the bundle:

| Feature | Pre-registered direction (carriers vs non-carriers) | PASS contribution |
|---|---|---|
| `agatston_total` | carriers > non-carriers | expected; not load-bearing |
| `agatston_lad` | carriers > non-carriers, MUCH more than agatston_total ratio | direct LAD-bias evidence |
| `n_calcified_arteries` | carriers slightly higher | weak; expected at higher burden |
| `agatston_rca / agatston_total` ratio | carriers LOWER than non-carriers | LAD-specific concentration evidence |
| `agatston_lm` | carriers similar or lower | LM-sparing pattern |

**PASS criterion at this step (advisory only)**: at least 3 of 5
directional predictions confirmed at FDR p < 0.05.

This step is exploratory characterisation. The independence-from-burden
question is settled at Step 4 (matched comparison), not here.

## P4. Burden-propensity-matched comparison (Step 4, matched.py)

The KEY test of whether the LAD bias is independent of total burden,
not a side-effect of carriers having more calcium overall.

### Matching specification

* **Matching variable**: `log(agatston_total + 1)` (single covariate)
* **Caliper**: 0.2 standard deviations of the matching variable
* **Match ratio**: 1 case to up to 3 controls (1:k with k_max = 3)
* **Replacement**: WITHOUT replacement (each control used at most once)
* **Selection order**: by ascending log-Agatston rank within cases
* **Diagnostic**: post-match standardised mean difference (SMD) must
  be < 0.1 to consider the match successful

If the match diagnostic fails (SMD >= 0.1 or fewer than 50% of cases
matched), the experiment reports "match infeasible at pre-registered
caliper" and terminates. The caliper is NOT relaxed.

### Pre-registered hypotheses on the matched set

After matching cases (carriers) to controls (non-carriers) on burden:

| Test | Pre-registered hypothesis (case - control) | PASS contribution |
|---|---|---|
| `agatston_lad / agatston_total` ratio | cases > controls, Cliff's delta >= 0.20 | LAD bias survives matching |
| LAD-lesion proximal proportion (rel-z < 0.5) | cases > controls, chi-square p < 0.05 | proximal localisation survives matching |
| max_hu_lad | cases > controls, Cliff's delta >= 0.15 | density character survives matching |

**Overall PASS**: at least 2 of 3 confirmed at the listed effect-size
thresholds. If 0 or 1 confirm, the LAD bias is "burden-confounded"
(parallel to how C8 was retrospectively reframed in Stage 7).

## P5. Cross-stratum replication (Step 5, finalise.py)

Every quantitative verdict above is re-tested independently in the
Qr36d/2 stratum (N ~ 220) and the I30f/3 stratum (N ~ 200).

**Replication PASS criterion**: each verdict must hold (PASS) in BOTH
strata independently, OR the effect-size estimate in each stratum
must be within +/-0.10 of the full-cohort estimate.

If a verdict holds in the full cohort but flips in one stratum, the
finding is "kernel-dependent" and not publication-eligible.

---

## Termination conditions (no negotiation)

| Condition | Outcome |
|---|---|
| P1 finds 0 LAD-dominant clusters | Experiment terminates. `findings.md` records "no LAD-dominant cluster under pre-registered signature." Negative result, still publishable as null. |
| P4 match diagnostic SMD >= 0.1 OR match yield < 50% | Experiment terminates. `findings.md` records "burden-propensity matching infeasible at pre-registered caliper." |
| P5 verdict flips in one stratum | Finding labelled "kernel-dependent." Not eligible as a lead claim. |

## Hash of this document at lock time

Track this file's SHA in `run_header.json` so any post-hoc edit is
detectable.

```bash
sha256sum experiments/lad_phenotype/plan.md
# Expected: recorded at first successful run
```
