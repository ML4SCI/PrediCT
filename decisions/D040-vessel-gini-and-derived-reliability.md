# D040 Vessel Gini over all four vessels; stage-5 transformed features carry measured D033 reliability

**Date**: 2026-10-04
**Stage**: reduce (stage 5), with the stage 5b report
**Status**: Active
**Module**: `src/predict/reduce/prepare_matrix.py` (`compute_vessel_burden_gini`, `stage5_transformed_columns`, `run_matrix_prep`), `scripts/05b_icc_gate_realistic.py`, `src/predict/reduce/redundancy.py` (`build_icc_lookup`), `scripts/06_reduce.py`

## Decision

1. **`vessel_burden_gini` is the Gini coefficient of the four per-vessel Agatston values, calcium-free vessels included**, as D019 states ("gini across 4 per-vessel Agatston values"). Range 0 (burden spread evenly over LAD, RCA, LCx, LM) to 0.75 (all burden in one vessel). The stage 7 helper uses the same function.
2. **The three features stage 5 creates (`has_dense_calcium` by D018 binarisation; `high_density_fraction` and `vessel_burden_gini` by D019) get measured reliability.** Stage 5b computes each one per primary draw with stage 5's own `stage5_transformed_columns` and gates it with the D033 rule (ICC(1,1) >= 0.75, at most 5% of patients excluded), writing `derived_icc.csv`. Stage 5 admits a transformed feature only if it passes that gate (and, for the two derived features, D019's R^2 < 0.95 rule), and uses the measured ICC when choosing representatives. With a D014 classic gate there is no `derived_icc.csv`; there the canonical inputs are invariant by construction and so are the transformed features.

## Rationale

**Gini.** The code dropped calcium-free vessels before computing the Gini, a convention that lived only in code docstrings and was never decided. It gives a patient with all calcium in one artery 0, the value of perfectly even disease: [100, 0, 0, 0] and [25, 25, 25, 25] both scored 0. In the analysis cohort 140 of 436 patients have single-vessel calcium. This contradicts D019's definition and the construct D025 hypothesis 5 tests ("focal disease concentrates burden in one vessel"). Finding F3 lists `vessel_burden_gini` among D025's refuted predictions while its focal group has a median of one calcified artery, exactly the patients the old convention set to 0; that refutation may be an artefact of the definition. This decision follows D019's text, so it implements a closed decision rather than amending it. It is fixed on construct grounds before any result under the corrected definition exists; D025 hypothesis 5 is evaluated under this definition and reported whichever way it falls.

**Reliability.** D019 gave derived features `invariant_by_construction` (ICC 1.0) because they are functions of XML-derived inputs. That was true under D014, whose CT-only perturbations cannot move canonical features. Under D033 (primary since 2026-09-25) polygon jitter moves per-vessel Agatston and tier counts, so the derived features vary: `vessel_burden_gini` was measured varying across draws for 297 of 437 patients, ICC about 0.998. The invariant label was unsupported, and stage 5 used it when ranking representatives. Separately, the D018 rename to `has_dense_calcium` lost the feature's ICC and its canonical classification. D033 says the feature set carried downstream is the set that passes its gate; measuring the transformed features on the same draws applies that rule to them.

## Alternatives considered

- **Keep the dropped-zero Gini and rename it "Gini among calcified vessels".** Rejected: it then measures inequality conditional on involvement, a different construct from D019's and D025's, and it is degenerate (always 0) for a third of the cohort.
- **Normalise the Gini to [0, 1] by n / (n - 1).** Not needed: the feature is z-scored in stage 5 and D025 is directional, so a constant factor changes nothing.
- **Inherit reliability from the inputs (e.g. the minimum ICC of the per-vessel Agatston values).** Rejected: a function of reliable inputs need not be equally reliable; the draws allow a direct measurement.

## Consequence

Stage 5 and everything after it use the corrected Gini; stage 5b is rerun first (minutes, no stage 4 draws) to produce `derived_icc.csv`. Results produced before D040 used the dropped-zero Gini.

## Verified by

- `tests/reduce/test_prepare_matrix.py`: `test_vessel_burden_gini_single_vessel_is_maximal`, `test_vessel_burden_gini_separates_focal_from_even`, `test_stage5_transformed_columns_cover_the_registry`, `test_unreliable_derived_feature_is_rejected_before_r2`.
- `tests/analyse/test_derived_features.py`: stage 7 equals stage 5 (one implementation), single vessel 0.75, two equal of four 0.5.
- `tests/reduce/test_redundancy.py`: measured derived reliability is used; no silent default.
