# D016 Canonical feature bypass with `icc_source` tagging

**Date**: 2026-06-03
**Stage**: stability
**Status**: Active (revised 2026-06-03 after tracing how HU values flow through the pipeline)
**Module**: `src/predict/stability/icc.py`, `scripts/05_icc_gate.py`

## Decision

Every feature column in `outputs/03_features/features.csv` is tagged with one of two ICC sources in the stage-5 report:

| `icc_source`                | Meaning                                                                                                  |
|-----------------------------|----------------------------------------------------------------------------------------------------------|
| `invariant_by_construction` | Computed from XML annotation fields (`roi.max_hu`, `roi.mean_hu`, `roi.area_cm2`, polygon vertices). Does not read the CT pixel array. ICC asserted = 1.0. |
| `empirical`                 | Reads the perturbed CT array via the PyRadiomics extractor. ICC measured from the 422-by-15 reliability matrix (baseline + 14 perturbations). |

Both sources are subject to the same `passes_gate` rule (ICC >= 0.75 per D013). The bypass does not exempt any feature from the gate; it short-circuits the computation when the answer is provably 1.0.

### Bypass list: 68 features tagged `invariant_by_construction`

All canonical features. This is exactly the set returned by `predict.features.feature_schema.feature_names()`, used as the single source of truth.

Per vessel (14 stems x 4 vessels = 56):

- `lesion_count_{lad,rca,lcx,lm}`
- `max_hu_{lad,rca,lcx,lm}`
- `mean_hu_{lad,rca,lcx,lm}`
- `volume_{lad,rca,lcx,lm}_mm3`
- `mass_{lad,rca,lcx,lm}`
- `agatston_{lad,rca,lcx,lm}`
- `inter_lesion_dist_mean_{lad,rca,lcx,lm}`
- `inter_lesion_dist_max_{lad,rca,lcx,lm}`
- `first_to_last_dist_{lad,rca,lcx,lm}`
- `diffusivity_{lad,rca,lcx,lm}`
- `n_rois_d{1,2,3,4}_{lad,rca,lcx,lm}`

Globals (12):

- `lesion_count_total`
- `n_calcified_arteries`
- `gini_lesion_volume`
- `dist_from_top_max`
- `dist_from_top_mean`
- `dense_calcium_count`
- `agatston_total`
- `volume_total_mm3`
- `mass_total`
- `mean_hu_weighted_global`
- `max_hu_global`
- `center_of_mass_z`

### Empirical list: 107 features

All `original_*` PyRadiomics features across the 7 IBSI families: shape (14), firstorder (18), glcm (24), glszm (16), glrlm (16), ngtdm (5), gldm (14).

Total accounted for: 68 + 107 = 175 features (plus 9 metadata columns not subject to stability).

## Rationale

A feature is invariant under our perturbation set (D014, mask fixed, CT only) if and only if its computation does not read the perturbed array.

**The XML is the radiologist's reading.** The COCA dataset's calcium XML is produced by OsiriX, a clinical-grade tool, by trained annotators. The `Max` and `Mean` HU fields stored in the XML for each ROI are the radiologist's measurement of record. In this pipeline:

1. `xml_parser.py` reads the XML once at parse time. `roi.max_hu` and `roi.mean_hu` are populated from the XML's `Max` and `Mean` fields (lines 122-123) and never mutated afterward.
2. `agatston.py`, `density_tiers.py`, `per_vessel_aggregates.py` all read `roi.max_hu` / `roi.mean_hu` / `roi.area_cm2` from `parse_result`. None of them index into the CT array.
3. `lesion_ccl.py` and `spatial.py` operate on polygon vertices and `Lesion` objects derived from the XML. No CT access.
4. `slice_matcher.py` (imported by `lesion_ccl`) is pure-XML logic; the only mention of `load_patient_dicom` is in a docstring comment.

When the CT is perturbed in stage 4, the XML does not change. There is no radiologist in the loop to re-annotate. The 68 canonical feature values are therefore bit-identical across all 14 perturbations.

Forcing the canonical features through the empirical gate would require **re-deriving** HU statistics by sampling the perturbed CT through the polygon (a different code path than stage 3 uses). That would measure the stability of a feature **that does not exist in the production pipeline**. Stage 5 / 6 / 7 only ever see the XML-derived values; an ICC computed from CT-resampled values describes a hypothetical alternative pipeline, not ours.

The PyRadiomics 107 features are the only features in the pipeline that index into the CT pixel array (via `extractor.execute(ct_sitk, mask_sitk)`). They are the only features that can change under our perturbations, so they are the only ones for which empirical ICC has any meaning.

### Why not bypass without explicit tagging

Quietly inserting ICC = 1.0 rows without a source column would hide the assertion. A reader could not tell whether 1.0 was measured or asserted. The `icc_source` column makes the audit trail explicit: any reader who doubts the bypass can grep the listed modules for `ct_array` and `sitk.GetArrayFromImage` and verify zero hits in executable code.

### What the bypass does NOT claim

The bypass asserts that the XML-derived canonical features are invariant to **CT-side image perturbations** under the D014 design. It does not claim:

- Invariance to **mask redefinition** (a different radiologist re-segmenting on a different scan would produce different XML values; that is a separate reproducibility axis our perturbation set does not test).
- Invariance to **OsiriX vs our rasterisation** (the binary mask vs partial-volume offset is a known systematic effect, audited via D002's XML round-trip).
- That the XML values themselves are "correct" (we accept them as the ground-truth-equivalent in an unsupervised pipeline because there is no clinical outcome label).

## Alternatives considered

- **31 pure-geometry bypass + 37 HU-touching canonical through empirical gate**. Treats Agatston / mass / density tiers as CT-derived. They are not; they read XML. Including them in the empirical gate would require re-deriving HU from the perturbed CT, a code path that does not exist in production. Rejected.
- **Bypass nothing, run every feature through the empirical pipeline**. Geometric and HU-touching canonical features would produce 14 identical values per patient (input does not depend on CT), so ICC variance is zero and ICC(3,1) is degenerate (0/0). Numerically unstable, no information gain. Rejected.
- **Bypass without source tagging**. Loses the audit trail. Rejected.
- **Full pipeline redesign: re-derive HU from CT cohort-wide (Option D)**. Would abandon the XML's Max / Mean as the source of truth and re-sample HU through the polygon at every stage. Invalidates current stage-3 output, conflicts with D002, and changes the clinical interpretation of the features. Rejected.

## Verified by

- `tests/stability/test_icc.py`: registry tests: length 68, matches `feature_names()`, excludes PyRadiomics and metadata.
- `tests/stability/test_bypass_truly_invariant.py`: static check that grep finds zero CT-array references in executable code across the seven canonical-feature modules; functional check that each canonical-feature function produces byte-identical output across repeated calls.
- Empirical verification (2026-06-03): ran `compute_agatston`, `compute_per_vessel_aggregates`, `compute_density_tiers`, `group_rois_into_lesions`, `compute_spatial_features` twice on patient 306. All 68 canonical keys produced byte-identical values across the two calls. `keys: 68, diffs across two calls: 0`. Confirms the canonical pipeline is deterministic and CT-independent.
- `scripts/05_icc_gate.py` will write the `icc_source` column for every feature and assert the bypass list count equals 68 and the empirical list count equals 107 (sum = 175 = total non-metadata feature columns).
