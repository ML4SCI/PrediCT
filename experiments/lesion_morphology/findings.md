# Lesion morphology experiment: findings (D037 features, k=12)

**Status**: exploratory. Outside the production pipeline.
**Date**: 2026-10-06.
**Run**: stage 9 of the reported pipeline run, published as `outputs/exploratory/lesion_morphology/` under `outputs/exploratory/COMPLETE_stage9.json` (decision D043). All numbers below are read from that folder; paths are relative to it.
**Inputs**: `03_features/lesions.csv` (3171 clustered lesions in 439 patients; 3158 of them in the 436-patient analysis cohort), `06_reduce/{cohort_metadata, cluster_labels_spatial_k2}.csv`, `03_features/features.csv`. Clustering uses every lesion in lesions.csv; patient-level comparisons use the 436-patient cohort, so carrier counts differ by one (52 over all lesions, 51 in the cohort).
**Features (D037)**: log1p volume, mean HU (area-weighted XML ROI means), max HU, log1p n_rois, z-scored.
**Machinery**: `predict.discover.{clusterability, cluster_discovery, validity}` as in stages 6 and 7. Cluster identity (broad class, C8-like) is found per run from cluster medians by the locked signature rules in `predict.figures._lesion_classes`; no cluster id is hardcoded.
**Primary k**: 12, the gap vote of `run.py` (kmeans 12, ward 12, gmm 2). "C8" means the C8-like cluster found by signature; in this run it is cluster 3. Numeric cluster ids are arbitrary; "cluster 8" below is the soft cluster and is unrelated to the C8 name.

## Headline findings

| # | Finding | Strength | Evidence |
|---|---|---|---|
| L1 | Lesion Hopkins 0.953 (3171 lesions, 4 features). The gap rises to the k_max boundary for kmeans and ward (selected k 12); gmm selects 2. Per D021.1 the gap result is descriptive: k=12 is the edge of the searched range, not an identified number of lesion types. The C8 claims do not depend on k (L3). | descriptive | `hopkins.json`, `gap_statistic.json` |
| L2 | A three-class split {soft, moderate, dense} by the signature rule agrees only weakly with an unsupervised GMM k=3 fit (ARI 0.207). At k=12 there is 1 soft, 4 moderate and 7 dense clusters; the counts change with k (1/2/4 at k=7). Not claimed as a taxonomy. All 12 k=12 clusters are Hennig-stable (median Jaccard 0.807 to 0.915). | weak | `gmm_k3_summary.json`, `k_sensitivity/summary.csv`, `hennig_stability.json` |
| **L3** | **The C8-like core is robust.** 62 large (median 290 mm3), dense (max HU 842), multi-slice (10 ROIs) lesions, RCA-dominant (obs/exp 2.56, Cramer's V 0.62, chi-square p 2.2e-15) with zero LM lesions; found by the strict signature rule at every k from 7 to 12 and within both kernel strata at every k. Carriers have median Agatston 1600 vs 80 (Cliff's delta 0.92). Within the carriers' RCA lesions it sits higher on the superior-inferior axis (relative z 0.733 vs 0.145; 0 = lowest lesion, +z superior); z alone does not establish proximal / distal position. | **strong, replicated, robust to k** | `finalise/c8_deep_dive.json`, `finalise/per_stratum_c8_replication.json`, `k_sensitivity/` |
| L4 | **C8 marks heavy disease beyond itself.** Carriers' non-C8 lesions: 15 vs 4 per patient, 990 vs 79 mm3, 4 vs 2 vessels (Cliff's delta 0.77 / 0.83 / 0.63). Pre-registered rule: SUPPORTED, at every k from 7 to 12. | strong | `c8_other_lesions/` |
| L5 | **Information beyond total burden: inconclusive.** At matched Agatston (38 of 51 carriers matched, SMD 2.32 to 0.019), carriers' other lesions are slightly less dense (219 vs 230 HU, p 0.045, delta -0.30) with a lower RCA share (0.27 vs 0.37, p 0.033, delta -0.21). Neither primary outcome meets the rule (p < 0.025 and abs(delta) >= 0.33), and 38 pairs are below the pre-registered minimum of 40. No run has shown a primary effect. | inconclusive | `c8_burden_matched/` |
| L6 | **Lesion features and C8 membership are stable under the D033 annotation perturbation.** 20 replayed M1 draws, 436 patients: ICC(1,1) volume 0.996, mean HU 0.999, max HU 0.9998, n_rois 0.995; 97.5% of lesions keep their cluster; C8 membership Jaccard median 0.937, counting lesions lost and created by regrouping. Pre-registered rules: both STABLE. | strong | `lesion_stability_d033/` |
| L7 | Composition differs across burden tertiles (cross-sectional): 9 of 12 cluster fractions rise at Jonckheere-Terpstra p < 0.001 (all 7 dense clusters and moderate clusters 4 and 5). The soft cluster 8 declines, not significantly (p 0.40). Fractions are compositional, so a decline is partly forced by increases elsewhere. Different patients per tertile; no time course is implied. | strong (descriptive) | `finalise/jonckheere_terpstra_trends.csv` |
| L8 | Patient-level lesion mixtures form no stable discrete groups: GMM k=4 on CLR mixtures gives 0 of 4 Hennig-stable clusters (median Jaccard 0.09 to 0.67); the mixture gap selects the k_max boundary (8) for all three algorithms, and 0 of 8 clusters are stable there (at most 0.74). The k=4 mixture groups follow burden (median Agatston 805, 97, 24, 26). | no stable groups at the Hennig threshold | `finalise/mixture_k4_hennig.json`, `mixture_hennig.json`, `finalise/mixture_k4_profile.csv` |

## Cluster reference (k=12)

Broad class from the signature rule (`finalise/broad_class_mapping.csv`). Sorted by median volume. JT = Jonckheere-Terpstra trend of the patient fraction across burden tertiles.

| ID | broad | n (%) | vol med (mm3) | mean HU med | max HU med | n_rois med | JT direction (p) | Note |
|---|---|---|---|---|---|---|---|---|
| 8 | soft_microspots | 494 (15.6%) | 3.5 | 150 | 169 | 1 | dec (0.40) n.s. | the only soft cluster |
| 1 | moderate_nodules | 533 (16.8%) | 8.4 | 179 | 233 | 1 | inc (0.38) n.s. | |
| 9 | moderate_nodules | 297 (9.4%) | 14.2 | 172 | 236 | 2 | inc (0.018) | |
| 4 | moderate_nodules | 399 (12.6%) | 18.4 | 220 | 341 | 1 | inc (1.3e-6) | |
| 7 | dense_plaques | 202 (6.4%) | 33.7 | 278 | 508 | 1 | inc (2.7e-14) | |
| 5 | moderate_nodules | 426 (13.4%) | 34.7 | 218 | 385 | 2 | inc (9.5e-13) | |
| 0 | dense_plaques | 272 (8.6%) | 56.0 | 276 | 578 | 2 | inc (< 1e-15) | strongest trend (Z 9.2) |
| 10 | dense_plaques | 141 (4.4%) | 87.7 | 229 | 502 | 5 | inc (2.9e-15) | LCx-biased (o/e 1.52, V 0.20) |
| 2 | dense_plaques | 170 (5.4%) | 102.2 | 341 | 750 | 1 | inc (3.4e-13) | LAD-biased (o/e 1.50, LM 1.71, V 0.29); LAD-phenotype cluster |
| 6 | dense_plaques | 93 (2.9%) | 144.4 | 320 | 842 | 4 | inc (3.9e-11) | |
| 11 | dense_plaques | 82 (2.6%) | 164.9 | 418 | 1056 | 2 | inc (1.5e-5) | densest; LAD o/e 1.38 (not vessel-biased, p 0.016); LAD-phenotype cluster |
| **3 (C8-like)** | **dense_plaques** | **62 (2.0%)** | **290.3** | **300** | **842** | **10** | inc (2.2e-7) | **RCA-dominant (o/e 2.56, V 0.62), zero LM** |

Vessel-biased clusters (chi-square p < 0.001 and Cramer's V >= 0.20): 2 (LAD and LM), 3 (RCA, zero LM), 10 (LCx); `finalise/vessel_biased_clusters_full_cohort.csv`.

## C8 deep dive (k=12)

| Quantity | C8 carriers | non-carriers | Test |
|---|---|---|---|
| Agatston median | 1600 (52 carriers over all lesions; 51 in the 436 cohort) | 80 | one-sided MW p 3.4e-27, Cliff's delta 0.92 |
| Relative z within RCA (per carrier; 0 = lowest, +z superior) | 0.733 (C8 RCA lesions) | 0.145 (other RCA lesions of the same carriers; 47 carriers in `finalise/c8_rca_relative_z.csv`) | two-sided MW p 5.1e-9, Cliff's delta 0.53 |

Relative z is the superior-inferior coordinate within each carrier's RCA lesions ((z - min z) / (max z - min z); +z superior in DICOM). It does not establish proximal / distal position along the curving RCA (D044).

## Per-stratum replication (refit at k=12 within each kernel)

| Cohort | cluster | n | vol med | max HU med | n_rois med | RCA o/e | LM o/e | Cramer's V |
|---|---|---|---|---|---|---|---|---|
| Full | 3 | 62 | 290.3 | 842 | 10 | 2.56 | 0.00 | 0.62 |
| Qr36d/2 (231 patients, 1883 lesions) | 8 | 33 | 315.9 | 876 | 11 | 2.48 | 0.00 | 0.59 |
| I30f/3 (205 patients, 1275 lesions) | 3 | 25 | 241.5 | 865 | 10 | 2.56 | 0.00 | 0.61 |

## Robustness to k (pre-registered, `open/lesion-k-sensitivity`)

Rule fixed before running: C8 is robust to k iff a cluster matches the signature at every k in 7..12 in the full cohort AND in both kernel strata. Result: **ROBUST**.

| k | C8 lesions | vol | max HU | n_rois | RCA o/e | Hennig J | Jaccard with k=12 core | carriers | Agatston carrier / non |
|---|---|---|---|---|---|---|---|---|---|
| 7 | 91 | 244 | 849 | 8 | 2.06 | 0.91 | 0.68 | 70 | 1251 / 71 |
| 8 | 100 | 238 | 830 | 8 | 2.03 | 0.83 | 0.62 | 77 | 1226 / 68 |
| 9 | 79 | 261 | 871 | 8 | 2.17 | 0.86 | 0.70 | 63 | 1290 / 72 |
| 10 | 88 | 255 | 852 | 8 | 2.13 | 0.83 | 0.69 | 69 | 1347 / 71 |
| 11 | 56 | 294 | 866 | 10 | 2.60 | 0.80 | 0.90 | 47 | 1697 / 81 |
| **12** | 62 | 290 | 842 | 10 | 2.56 | 0.86 | 1.00 | 52 | 1600 / 80 |

C8 is a stable core whose boundary depends on k: at higher k it sheds borderline lesions and the remaining core is larger, denser and more RCA-biased. Report it as a core, not as a fixed-size cluster. LM o/e is 0 at every k and in every stratum (strata over k 7 to 12: vol 241 to 316, max HU 813 to 916, RCA o/e 1.87 to 2.67).

## Disease outside the C8 lesions (pre-registered, `open/c8-other-lesions`)

Per patient, carriers' non-C8 lesions vs all lesions of non-carriers, k=12 (51 carriers in the cohort; every carrier has lesions outside C8):

| Metric | carriers (non-C8 only) | non-carriers | one-sided MW p | Cliff's delta | incl. C8 lesions (delta) |
|---|---|---|---|---|---|
| lesion count | 15 | 4 | 9.3e-20 | 0.77 | 0.81 |
| total volume (mm3) | 990 | 79 | 2.1e-22 | 0.83 | 0.92 |
| mass proxy (vol x mean HU) | 250508 | 18777 | 2.0e-22 | 0.83 | 0.93 |
| vessels with a lesion | 4 | 2 | 1.6e-14 | 0.63 | 0.66 |

Rule (count and volume both p < 0.05 and delta >= 0.33): **SUPPORTED**, at every k from 7 to 12 (count delta 0.74 to 0.78, volume delta 0.80 to 0.84). Removing the C8 lesions lowers the volume delta only from 0.92 to 0.83.

## C8 at matched burden (pre-registered, `open/c8-burden-matched`)

Nearest non-carrier on log1p(Agatston), exact kernel, caliper 0.2 SD (0.381), with replacement: 38 of 51 carriers matched, 26 distinct controls; SMD 2.32 before, 0.019 after. Outcomes on non-C8 lesions only.

| Outcome | role | carrier | control | p (Wilcoxon) | Cliff's delta |
|---|---|---|---|---|---|
| median density of other lesions (HU) | primary | 218.8 | 229.7 | 0.045 | -0.30 |
| RCA share of other lesion volume | primary | 0.27 | 0.37 | 0.033 | -0.21 |
| other lesion count | secondary | 14 | 15 | 0.32 | -0.15 |
| other lesion volume (mm3) | secondary | 752 | 1046 | 7.8e-9 | -0.32 |
| vessels with other lesions | secondary | 4 | 4 | 0.83 | 0.07 |
| any LM lesion | secondary | 1 | 1 | 0.49 | 0.08 |
| max HU of other lesions | secondary | 828 | 989 | 0.032 | -0.33 |

Rule: **INCONCLUSIVE** (38 < 40 matched; neither primary outcome meets p < 0.025 and abs(delta) >= 0.33). Lower other-lesion volume in carriers at equal Agatston is expected: the C8 plaque is part of their burden. Secondary outcomes are descriptive and carry no verdict.

## Lesion stability under D033 (pre-registered, `lesion_stability_d033/`)

The 20 D033 primary draws' M1 polygon jitter replayed with production code (the only D033 factor that reaches a lesion), lesions regrouped and matched per vessel (Hungarian on centroids, 3 mm gate). Parity with lesions.csv enforced (3158 lesions of the 436 patients; keys, n_rois, area, volume, mean and max HU equal), stored labels rebuilt exactly by the production k-means. Matched fraction per draw median 0.993 [0.991, 0.995]; 3036 lesions matched in every draw; 19 [13, 28] lesions created by regrouping per draw.

| Feature | ICC(1,1) [95% CI] |
|---|---|
| log1p volume | 0.9962 [0.9960, 0.9964] |
| mean HU | 0.9994 [0.9994, 0.9995] |
| max HU | 0.9998 [0.9998, 0.9998] |
| log1p n_rois | 0.9946 [0.9943, 0.9949] |

97.5% [97.0, 98.1] of lesions keep their k-means cluster per draw (median); C8 membership Jaccard median 0.937 [0.906, 1.000] over 61 baseline C8 lesions (the cohort's), with created lesions counted. Rules (every ICC >= 0.75; C8 median Jaccard >= 0.75): **STABLE / STABLE**. HU values are XML ROI values; they move when M1 changes ROI areas (mean) or regroups ROIs into different lesions (both); max |draw - base| is 328 HU for max HU and 63.5 HU for mean HU. Inter-scan HU variability is outside D033.

## Narrative for the paper supplement

> At the lesion level, one lesion type is specific and stable: a core of large (median about 240 to 315 mm3 across k and kernel strata), dense (peak about 810 to 920 HU), multi-slice (about 8 to 11 ROIs) plaques that are RCA-dominant (obs/exp 1.9 to 2.7), absent from the left main (zero LM lesions at every k and in both kernel strata) and located higher on the superior-inferior axis within the carrier's RCA lesions (relative z 0.73 vs 0.15). A fixed signature rule recovers this core at every k from 7 to 12 and within each kernel stratum, and its membership survives annotation-boundary perturbation (Jaccard 0.94). Patients carrying it have heavy, multi-vessel disease elsewhere (non-C8 calcified volume 990 vs 79 mm3). Whether it carries information beyond total Agatston could not be established: at matched burden too few carriers could be paired, and no primary outcome met the pre-registered rule. We therefore describe the C8 core as a robust lesion-level signature of advanced coronary calcification. Lesion-type composition differs monotonically across burden tertiles (cross-sectional; 9 of 12 clusters, Jonckheere-Terpstra p < 0.001), and patient-level lesion mixtures form no stable discrete groups (0 of 4 Hennig-stable at k=4, 0 of 8 at the gap-selected k).

## Related lesion experiments (same run)

- **LAD phenotype** (`experiments/lad_phenotype/`): the pre-registered signature matches clusters 2 and 11 in the full cohort and in both kernel strata. Axial position within the LAD replicates (full-cohort median difference 0.132; Qr36d/2 0.128; I30f/3 0.158, within the +/-0.10 tolerance). The burden-propensity match is infeasible at the pre-registered caliper (54 of 117 cases matched, yield 0.46 < 0.50; SMD 0.080), so that step terminates as the plan requires and the overall replication verdict is None (`lad_phenotype/cross_stratum_replication.json`).
- **Isolated LM at low burden** (`experiments/lm_isolated_low_burden/`): 11 displaced patients in the 146-patient low tertile, 9 LM-positive (82%, Wilson 52 to 95%), Fisher p 1.4e-6; P2 to P5 pass, P5 with this run's LAD clusters (`lm_isolated_low_burden/summary.json`).

## Net update to the overall study findings

| Study finding | Update from lesion morphology |
|---|---|
| Finding 1: no stable patient-level subtypes; burden dominates | Consistent: mixture groups are unstable and follow burden; composition differs with burden. |
| Finding 3: spatial k=2 = burden dichotomy | Consistent: lesion composition tracks burden. |
| Lesion-mixture phenotype claim | Not supported (0 of 4 and 0 of 8 Hennig-stable). |
| C8 RCA-dominant dense plaque core | Robust to k, replicated in both strata, stable under annotation perturbation; a signature of advanced disease; information beyond Agatston not established. |

## Open follow-ups

- A detectability check (as D036.1) on this cohort, and a lesion-level analogue, before reading L1 beyond description.
- C8 leave-one-patient-out refits.
- Anatomical centreline mapping instead of the relative-z coordinate.
- Outcome association needs a cohort with outcomes; COCA has none.
