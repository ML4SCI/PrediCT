# D028 Stage 7 low-burden sensitivity output

**Date**: 2026-06-06
**Stage**: analyse (stage 7)
**Status**: Active
**Module**: `src/predict/analyse/paper_table.py`, `scripts/08_analyse.py`

## Decision

The 422-patient stage-4-gated cohort includes 142 patients with `low_burden_flag = True` (mask < 100 voxels per D010). These patients are concentrated in the Agatston 1-99 category and have small lesion masses that contribute mostly noise to the burden axis. They are NOT excluded from the main stage 7 analysis (which uses N=420 = 422 minus 2 singleton-kernel patients, per D019 ComBat filter), but a **robust-cohort sensitivity rerun** restricts the analysis to `low_burden_flag = False` (N approximately 280, depending on the kernel filter cascade).

The robust-cohort outputs are written as a SEPARATE file alongside the main paper table:

| File | Cohort | Row count |
|---|---|---|
| `phenotype_paper_table.csv` | full + Qr36d/2 + I30f/3 (each at N >= 200) | 15 (3 cohorts x 5 phenotype clusters: 2 spatial + 3 burden) |
| `phenotype_paper_table_robust_sensitivity.csv` | low_burden_flag == False subset of the full cohort, N approx 280 | 5 (1 cohort x 2 spatial + 3 burden) |

Plus a robust-cohort version of `burden_orthogonality.csv` named `burden_orthogonality_robust_sensitivity.csv`, and a robust-cohort version of `directional_hypotheses.csv` named `directional_hypotheses_robust_sensitivity.csv`.

The robust cohort outputs are run by the orchestrator as a second pass after the main 3-cohort sweep, gated by `--include-robust-sensitivity` (default True).

## Rationale

### Why separate file instead of 4th cohort

- The main paper table communicates "the phenotypes replicate across the three cohort definitions we built in stage 6". Adding a 4th cohort would dilute that message; readers would conflate the cross-kernel replication argument (D027) with the low-burden sensitivity argument (D028) and might wonder why we have 4 cohorts when only 3 were prospectively designed.
- Putting it in a separate file lets the paper Methods section reference it explicitly: "as a sensitivity analysis, we re-ran the characterisation on the 280 patients with low_burden_flag = False (mask >= 100 voxels) and found the phenotype signatures preserved at the verdict level (see Supplementary Table X)".

### Why low_burden_flag = False is the right cut

- D010 set `low_burden_flag = True` at mask_voxels < 100. This is the threshold below which PyRadiomics texture results become noisy (mask too small for stable GLCM / GLRLM matrices). It is a per-patient quality flag, not a per-feature one.
- 142 of 422 stage-4-gated patients carry this flag; nearly all are in Agatston 1-99. Removing them gives a cohort dominated by clinically-significant calcium burden.
- The 280 remaining patients still span Agatston tiers 1-99 to >=400 (the high-burden patients also have low_burden_flag = False, by definition).

### Why don't we just run the main analysis on the 280-patient cohort

- The main analysis is the prospectively-designed pipeline: stage 5 ran on N=420 with the kernel filter; stage 6 produced labels for N=420. Running stage 7 on a subset of those 420 (the 280 with low_burden_flag = False) re-uses the existing cluster labels and is the most rigorous sensitivity test.
- An entirely separate stage 5 / 6 run on N=280 would re-derive different multi-block representatives (D022 has cohort-dependent tiebreaks) and different cluster labels, which would change the analysis and make the comparison unclean.

### Why this output is not part of the main paper table row count (15 stays 15)

The 15-row main table is structured as (3 cohorts x 5 clusters). Including the robust sensitivity would create asymmetry (a 4th cohort with the same 5 clusters but a different N and different kernel mix). The separate file with the same 5-cluster schema is cleaner.

## Alternatives considered

- **Embed as 4 extra rows in the main paper table**: rejected; keeps the main table clean.
- **Run an entirely separate stage 5 / 6 pipeline on the 280-patient cohort and produce a full set of stage 7 outputs**: rejected for the relabelling complication described above.
- **Skip the robust sensitivity entirely**: rejected; it is the hedge against the low-burden patients' noise contaminating the main findings.

## Verified by

- `tests/analyse/test_paper_table.py` covering: 15-row main table count; 5-row robust sensitivity count; required columns present in both; `--include-robust-sensitivity False` skips the second pass.
