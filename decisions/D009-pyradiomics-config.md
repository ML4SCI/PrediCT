# D009 PyRadiomics extractor configuration

**Date**: 2026-06-02
**Stage**: features
**Status**: Active
**Module**: `src/predict/features/radiomics.py`, `params.yaml`

## Decision

Locked PyRadiomics extractor configuration in `params.yaml` at the repository root:

```yaml
imageType:
  Original: {}

featureClass:
  shape:        # 14 features
  firstorder:   # 18 features
  glcm:         # 24 features
  glszm:        # 16 features
  glrlm:        # 16 features
  ngtdm:        #  5 features
  gldm:         # 14 features

setting:
  binWidth: 25
  interpolator: sitkBSpline
  label: 1
  geometryTolerance: 0.0001
  correctMask: true
  normalize: false
  force2D: false
  minimumROIDimensions: 1
  minimumROISize: 14
```

Total: **107 features** per mask (whole-mask and optionally per-artery).

Input is raw HU from `outputs/02_preprocessed/{pid}_ct.npy` (int16, clipped to `[-200, 3000]`). Mask is the corresponding `_mask.npy` (uint8, binary). Spacing comes from `outputs/02_preprocessed/spacing.json` (`0.5 × 0.5 × 3.0 mm`).

A guardrail `validate_ct_for_radiomics(ct_array)` runs before extraction and raises on suspicious input ranges (max ≤ 1.5 → normalised; min ≥ −100 → not raw HU; max < 130 → no calcium possible).

## Rationale

- **All 7 IBSI-compliant families enabled**: proposal commits to ≥80 raw features. Limiting families would prejudge what survives the ICC gate downstream. Let stage 4 decide what's robust.
- **`binWidth = 25`**: standard for CT calcium texture (Lin 2022, Kolossváry 2025). Discretisation independent of mask intensity range.
- **`normalize: false`**: critical. Normalising the CT inside PyRadiomics destroys HU semantics. Raw HU is the only valid input.
- **`force2D: false`**: texture features are computed in 3D throughout. The known 3 mm slice anisotropy is a property of the cohort, not something PyRadiomics should mask.
- **`minimumROISize: 14`**: IBSI floor for stable texture statistics. PyRadiomics will refuse to compute on smaller ROIs and return NaN; downstream handles it via `low_burden_flag` (D010).
- **`correctMask: true`**: auto-fix minor mask-geometry issues (rare; safety net).

The `imageType: Original` block means no Laplacian-of-Gaussian / wavelet derivations, those would 5×–10× the feature count and explode the multiple-testing burden without strong evidence of added signal on calcium-only NCCT.

## Alternatives considered

- **LoG / wavelet derivations**: adds hundreds of features. Lin 2022 needed them for CCTA culprit lesions (subtle texture); calcium on NCCT is dominated by intensity and shape, not subtle texture filters. Rejected; can be added as an amendment if reduce-stage analysis shows the original features insufficient.
- **`force2D: true` per-slice texture**: would sidestep the z anisotropy. But COCA lesions are often 1–2 slices thick; per-slice texture on a 1-slice mask is degenerate. Rejected.
- **`minimumROISize=1`**: produces values on tiny ROIs that are noise. Rejected.

## Verified by

- `tests/features/test_radiomics.py`: feature count and prefix check on a synthetic CT + mask; `validate_ct_for_radiomics` guardrail tests.
- Empirical feature count per patient = 107 (after dropping `diagnostics_*` keys).
