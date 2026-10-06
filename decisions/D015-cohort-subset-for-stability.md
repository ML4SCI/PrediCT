# D015 Cohort subset for stability analysis

**Date**: 2026-06-03
**Stage**: stability
**Status**: Active
**Module**: `scripts/04_perturbations.py`

## Decision

Compute ICC across the **422 patients with `radiomics_status == "ok"`** in `outputs/03_features/features.csv` (the full 444-patient cohort minus the 22 patients whose whole-mask is below PyRadiomics' `minimumROISize = 14` per D009/D010).

Reliability matrix shape: **422 patients by 14 perturbations**.

The 22 skipped patients are not perturbed and not represented in the ICC computation. Their PyRadiomics columns in `features.csv` remain NaN as they were after stage 3. In the gated feature set written by stage 5, these rows survive with the same NaN pattern; downstream code (reduce / analyse) decides how to handle them via the `radiomics_status` and `low_burden_flag` metadata columns.

## Rationale

- **Same population as downstream consumes**. Estimating the gate on a stratified subset of ~200 patients for compute economy and then applying it to the full cohort would be a population mismatch: ICC estimated on a small slice may not generalise to features computed across the full cohort. Using the full eligible cohort eliminates the gap.
- **Skip the 22**. Their PyRadiomics row is already NaN at stage 3. Re-running PyRadiomics on perturbed versions of their CT volumes will fail with the same `ValueError` from `minimumROISize`. We would generate NaN-only columns for them, which contribute nothing to ICC and waste roughly 22 * 14 = 308 extractions. Skip them at stage 4 and document the exclusion.
- **Compute budget**. 422 patients * 14 perturbations = 5,908 PyRadiomics extractions. At ~10 to 30 seconds per extraction on the remote machine with 16 workers in parallel, full-cohort stage 4 runs in roughly 1.5 to 3 hours wall time, well within budget, so a smaller subset is not needed for compute.
- **Sample size for ICC**. Bonett 2002 gives the variance of ICC estimates as a function of n_subjects, n_raters, and the true ICC. At true ICC = 0.75, n_raters = 14, the 95% CI half-width drops below 0.05 at n_subjects ~ 50. With 422 we are far past that, giving tight CIs for the gate decision.

## Alternatives considered

- **Stratified 200 by Agatston category**. Smaller computer budget but introduces a population mismatch between ICC and downstream. Rejected.
- **Include the 22 skipped patients with NaN columns**. They cannot produce numeric PyRadiomics under perturbation either; including them adds rows that ICC computation must skip, no benefit. Rejected.
- **Random subsample (e.g., 200 random patients)**. Same compute saving as stratified, worse coverage of the Agatston distribution. Rejected.

## Verified by

- `scripts/04_perturbations.py` reads `outputs/03_features/features.csv`, filters `radiomics_status == "ok"`, and asserts the row count is 422 before starting the perturbation loop.
- `scripts/05_icc_gate.py` reads the perturbation outputs and asserts the per-feature matrix is 422 by 14 before computing ICC.
