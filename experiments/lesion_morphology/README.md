# Exploratory: lesion-level morphology clustering

**Status**: exploratory, pre-registered analyses run as part of stage 9
(`run_pipeline.sh`). Produces no file consumed by stages 1 to 8 and does not
modify any decision. Results: [findings.md](findings.md).

## Question

At the patient level, total burden dominates every feature space and no
stable burden-independent subtypes are found (stages 6 and 7). Lesion-type
structure (dense nodular vs soft speckled vs sheet-like calcification) could
be hidden by averaging over heterogeneous lesions within a patient, so this
experiment clusters the lesions themselves.

1. **Lesion-level structure**: clusterability and k-means clusters of the
   lesions on four morphology features (D037: log volume, mean HU, max HU,
   log ROI count).
2. **A specific lesion type**: the C8-like core (large, dense, multi-slice,
   RCA-dominant) found by a locked signature rule, tested for robustness to
   k, replication within each kernel stratum, disease outside it, its
   relation to total burden, and stability under the D033 annotation
   perturbation.
3. **Patient-level mixtures**: whether patients' distributions over lesion
   clusters form stable groups.

## Inputs

| File | Source | Used for |
|---|---|---|
| `outputs/03_features/lesions.csv` | stage 3 | the lesions to cluster |
| `outputs/03_features/features.csv` | stage 3 | per-patient Agatston and kernel |
| `outputs/06_reduce/cluster_labels_spatial_k2.csv` | stage 6 | patient focal / diffuse labels for cross-tabs |
| `outputs/06_reduce/cohort_metadata.csv` | stage 5 | kernel, low_burden_flag |

## Scripts (run in this order by stage 9)

| Script | Purpose | Main outputs |
|---|---|---|
| `run.py` | features, Hopkins, gap statistic, k-means at the gap-selected k, Hennig, vessel chi-square | `lesion_features.csv`, `lesion_cluster_labels.csv`, `cluster_profiles.csv`, `cluster_vessel_chi_square.csv`, `hopkins.json`, `gap_statistic.json`, `hennig_stability.json` |
| `analyse.py` | patient lesion-mixture signatures, mixture clusterability, burden trends | `patient_lesion_signatures.csv`, `mixture_*.json/csv` |
| `finalise.py` | broad classes, C8 deep dive, per-stratum replication, Jonckheere-Terpstra trends, mixture k=4 | `finalise/` |
| `k_sensitivity.py` | C8 at every k from 7 to 12, full cohort and strata | `k_sensitivity/` |
| `c8_other_lesions.py` | disease outside the C8 lesions of carriers | `c8_other_lesions/` |
| `c8_burden_matched.py` | C8 carriers vs burden-matched non-carriers | `c8_burden_matched/` |
| `lesion_stability_d033.py` | lesion features and C8 membership under D033 M1 jitter | `lesion_stability_d033/` |

All outputs are written under `outputs/exploratory/lesion_morphology/` and
published with the stage 9 completion record (D043).

## Methodological consistency with the production pipeline

The experiment reuses the production helpers from `predict.discover`
(`assess_clusterability`, `gap_statistic`, `fit_cluster` / `fit_kmeans`,
`hennig_clusterboot`), so lesion-level and patient-level results use the
same definitions (including the D021.1 corrections). Cluster identities are
found per run by the locked signature rules in
`predict.figures._lesion_classes`; no cluster id is hard-coded.
