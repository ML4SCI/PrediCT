# Exploratory: LAD-specific lesion morphology

**Status**: exploratory, pre-registered (`plan.md`), run as part of stage 9
(`run_pipeline.sh`). Produces no file consumed by stages 1 to 8 and does not
modify any decision. Results: [findings.md](findings.md).

## Question

The lesion-morphology experiment finds an RCA-dominant dense lesion core
(C8). This experiment asks the parallel question for the LAD: does a
lesion cluster with an LAD-dominant, dense, large profile exist, where do
its lesions sit within the LAD, what do its carriers look like, and does
the LAD bias survive matching on total burden?

## Steps (thresholds and directions fixed in `plan.md`)

1. **Signature discovery**: which lesion clusters match the locked
   LAD-dominant profile (LAD obs/exp > 1.3, RCA obs/exp < 1.0, median max
   HU > 400, median volume > 50 mm3).
2. **Axial position within the LAD**: `rel_z = (max z - z) / (max z - min z)`
   over each patient's LAD lesions (0 = highest lesion, nearest the LM
   origin; 1 = lowest, towards the apex). Height approximates position along
   the descending LAD; it is not a centreline measurement.
3. **Carrier profile**: carriers vs non-carriers on burden and vessel
   distribution.
4. **Burden-matched comparison**: propensity matching on log Agatston within
   kernel (`propensity.py`); a match that misses the pre-registered balance
   or yield rule terminates the step and is recorded as the result (D043).
5. **Kernel-stratum replication** (Qr36d/2, I30f/3): each component judged
   against the full-cohort verdict (D043).

## Inputs (read-only)

| File | Source |
|---|---|
| `outputs/03_features/lesions.csv`, `features.csv` | stage 3 |
| `outputs/exploratory/lesion_morphology/{lesion_cluster_labels, cluster_profiles, cluster_vessel_chi_square}.csv` | lesion-morphology experiment |
| `outputs/06_reduce/cohort_metadata.csv` | stage 5 |

## Scripts and outputs

| Script | Outputs under `outputs/exploratory/lad_phenotype/` |
|---|---|
| `run.py` (full cohort and `--stratum`) | `lad_cluster_signature.json`, `axial_within_lad.csv`, `axial_summary.json`, `carrier_profile.csv`, `carrier_summary.json` |
| `matched.py` | `matched_pairs.csv`, `matched_diagnostics.json`, `matched_comparison*.{csv,json}` (only when the match is feasible) |
| `finalise.py` | `cross_stratum_replication.json`, `report.txt` |

Stratum runs write to `stratified_<kernel>/`. Tests:
`experiments/lad_phenotype/tests/test_propensity.py` (run with
`pytest experiments --import-mode=importlib`).

## Methodological consistency

Effect sizes and tests reuse `predict.analyse` (`cliffs_delta`,
`mannwhitney_u_pval`, `apply_fdr_bh`). Balance is measured on the compared
population: matched cases once against matched controls once (D043).
