# LAD-phenotype experiment: findings

**Status**: exploratory, pre-registered (`plan.md`; thresholds recorded in
each `run_header.json`).
**Date**: 2026-10-06.
**Run**: stage 9 of the reported pipeline run, published as
`outputs/exploratory/lad_phenotype/` under
`outputs/exploratory/COMPLETE_stage9.json` (D043). All numbers below are
read from that folder; paths are relative to it.
**Inputs**: the k=12 lesion-morphology partition
(`../lesion_morphology/findings.md`); 439 patients with lesions, 117 of
them carriers of the LAD-signature clusters, 322 non-carriers.

## Summary

| Step | Pre-registered rule | Result | Verdict |
|---|---|---|---|
| P1 signature | cluster with LAD obs/exp > 1.3, RCA obs/exp < 1.0, median max HU > 400, median volume > 50 mm3 | clusters 2 and 11 | found |
| P2 axial position | median rel_z difference >= 0.10 and one-sided MW p < 0.01 | 0.145 vs 0.277 (difference 0.132), p 5.8e-4 | PASS |
| P3 carrier signature | >= 3 of 5 directional predictions at FDR p < 0.05 | 3 of 5 | PASS (advisory) |
| P4 burden matching | post-match SMD < 0.1 and yield >= 0.5 | SMD 0.080, yield 0.46 (54 of 117) | match infeasible; step terminates |
| P5 stratum replication | each component judged against the full-cohort verdict (D043) | signature and axial replicate; matched not evaluable | overall None |

The two LAD-signature clusters (cluster 2: 170 lesions, median 102 mm3,
max HU 750, LAD obs/exp 1.50; cluster 11: 82 lesions, 165 mm3, max HU 1056,
LAD obs/exp 1.38) are large, dense lesions over-represented in the LAD.
Within each patient's LAD lesions they sit nearer the top of the vessel.
Their carriers have much higher burden. Whether the LAD bias survives
matching on total burden could not be tested: fewer than half of the
carriers have a non-carrier within the caliper, so the pre-registered
matching step terminates (`matched_diagnostics.json`).

## P2: position within the LAD

`rel_z = (max z - z) / (max z - min z)` over each patient's LAD lesions
(patients with two or more LAD lesions): 0 is the highest lesion (nearest
the LM origin), 1 the lowest. Height approximates position along the
descending LAD; it is not a centreline measurement.

| Cohort | signature lesions | other LAD lesions | rel_z median (signature / other) | difference | one-sided MW p | rule |
|---|---|---|---|---|---|---|
| Full | 144 | 1038 | 0.145 / 0.277 | 0.132 | 5.8e-4 | PASS |
| Qr36d/2 | 79 | 609 | 0.145 / 0.273 | 0.128 | 2.3e-3 | PASS |
| I30f/3 | 65 | 424 | 0.145 / 0.304 | 0.158 | 0.045 | fails p < 0.01; effect within +/-0.10 of the full cohort, so it replicates (D043) |

## P3: carrier signature

| Feature | predicted | carriers | non-carriers | Cliff's delta | FDR q | confirmed |
|---|---|---|---|---|---|---|
| `agatston_total` | carriers higher | 716 | 55 | 0.77 | 7.8e-35 | yes |
| `agatston_lad` | carriers higher | 344 | 31 | 0.74 | 2.4e-32 | yes |
| `n_calcified_arteries` | carriers higher | 3 | 2 | 0.54 | 2.7e-19 | yes |
| RCA share of Agatston | carriers lower | 0.20 | 0.00 | +0.31 | 1.0 | no (reversed) |
| `agatston_lm` | carriers similar or lower | 1.5 | 0.0 | +0.31 | 1.0 | no (reversed) |

The three confirmed predictions are the burden predictions. The two
predictions specific to an LAD-concentrated pattern (lower RCA share, LM
sparing) are reversed: carriers have more calcium in every vessel, as
expected of higher-burden patients. Both kernel strata give the same 3 of
5 (Qr36d/2: 64 carriers; I30f/3: 53).

## P4: burden matching

Nearest non-carriers on log Agatston within kernel, caliper 0.2 SD (0.385),
up to 3 controls per case. Balance is met (SMD 1.73 before, 0.080 after,
measured on matched cases against matched controls), but only 54 of 117
carriers match (yield 0.46 < 0.50). The strata fail the same way (Qr36d/2
yield 0.47, I30f/3 0.47). By the plan the step terminates and no matched
comparison is reported.

## Reading

- A dense, large, LAD-over-represented lesion type exists in this
  partition, replicates within both kernel strata, and its lesions sit
  higher within the LAD than other LAD lesions.
- Its carriers are high-burden patients, and the LAD-specific carrier
  predictions are reversed. Whether the LAD bias is more than a
  consequence of high total burden is **not established**: the
  pre-registered burden-matched test could not be run.
- Like the C8 core, it is best described as a lesion-level feature of
  advanced calcification, not a burden-independent patient phenotype.

## Limitations

- Relative height within the LAD is a proxy for position along the vessel,
  not a centreline coordinate.
- Signature membership depends on the k=12 partition; the lesion
  experiment's k sweep tests robustness for C8, not for these clusters.
- No outcomes are available in COCA, so no clinical risk claim is made.
