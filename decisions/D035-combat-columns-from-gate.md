# D035 ComBat harmonises every gated non-shape PyRadiomics feature

**Date**: 2026-10-01
**Stage**: reduce
**Status**: Active (fixes an implementation defect in D019's ComBat step)
**Module**: `src/predict/reduce/prepare_matrix.py` (`harmonisable_columns`, `combat_harmonise`, `run_matrix_prep`)

## Decision

ComBat (D019 step 5) harmonises, with kernel as the batch covariate, **every PyRadiomics feature in the gated set except shape features**, after the variance filter. The column list is derived from the gated feature set at run time by `harmonisable_columns()`; there is no hardcoded list. Shape features (mask-only) and canonical XML-derived features pass through unharmonised, as before. The D019 acceptance check is unchanged: any harmonised column with post-ComBat kernel R^2 above 0.02 stops the run.

## Rationale

D019's intent was to harmonise the texture features that survive the stability gate. It was implemented as a fixed tuple of 6 names, the survivors of an earlier gate. When the gate changed, the tuple went stale:

- production D014 at 0.375 mm: 7 non-shape PyRadiomics features gated, 6 harmonised, 1 not;
- D033: 84 gated, 6 harmonised, 78 not; T5: 68 gated, 62 not.

Unharmonised texture features exceeded D019's own 0.02 bound (glcm Imc2 0.035, MCC 0.034, Imc1 0.031, ngtdm Busyness 0.032), raised kernel predictability from the representatives (AUC .618 production, .725 D033, .740 T5) and made kmeans forced k=3 fail the D021 kernel chi-square under D033 and T5 (`scripts/diagnose_kernel_leak.py`). A list that duplicates a gate's outcome breaks one-rule-one-implementation; deriving it removes the failure mode.

Firstorder features are included: they are intensity statistics and carry kernel effects just as texture does (the old list already included firstorder Range).

## Not changed

- Canonical XML features stay unharmonised (D019). Their residual kernel signal (LCx lesion count R^2 0.035) may be population confounding rather than measurement, and is reported, not corrected.
- D029.1: the external holdout's spatial projection uses canonical features only, so it remains ComBat-free.

## Consequence

Production results change slightly (one more feature harmonised); every stage 6 to 9 output must be regenerated after this fix, including the D014 baseline.

## Verified by

- `tests/reduce/test_prepare_matrix.py::test_harmonisable_columns_is_every_non_shape_radiomics_feature`
- `tests/reduce/test_prepare_matrix.py::test_run_matrix_prep_harmonises_texture_outside_old_list`
- `scripts/diagnose_kernel_leak.py` rerun after the fix (reads the harmonised set from each run's `matrix_prep_log.json`).
