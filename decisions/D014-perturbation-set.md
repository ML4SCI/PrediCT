# D014 Perturbation set (14 perturbations)

**Date**: 2026-06-03
**Stage**: stability
**Status**: Active
**Module**: `src/predict/stability/perturbations.py`, `scripts/04_perturbations.py`

## Decision

Apply **14 deterministic perturbations** to each patient's CT volume. The mask is held fixed; only the CT array is transformed. Each (patient, perturbation) pair re-extracts the empirical-track features and contributes one column to the 422-by-14 reliability matrix consumed by D013.

| # | Family       | Parameter                  | Generator                                |
|---|--------------|----------------------------|------------------------------------------|
| 1 | rotation     | +5 deg about z (axial)     | `rotate(ct, +5, axis="z")`               |
| 2 | rotation     | -5 deg about z             | `rotate(ct, -5, axis="z")`               |
| 3 | rotation     | +10 deg about z            | `rotate(ct, +10, axis="z")`              |
| 4 | rotation     | -10 deg about z            | `rotate(ct, -10, axis="z")`              |
| 5 | translation  | +2 mm in x                 | `translate(ct, +2, axis="x")`            |
| 6 | translation  | -2 mm in x                 | `translate(ct, -2, axis="x")`            |
| 7 | translation  | +5 mm in x                 | `translate(ct, +5, axis="x")`            |
| 8 | translation  | -5 mm in x                 | `translate(ct, -5, axis="x")`            |
| 9 | translation  | +2 mm in y                 | `translate(ct, +2, axis="y")`            |
| 10| translation  | -2 mm in y                 | `translate(ct, -2, axis="y")`            |
| 11| translation  | +5 mm in y                 | `translate(ct, +5, axis="y")`            |
| 12| translation  | -5 mm in y                 | `translate(ct, -5, axis="y")`            |
| 13| noise        | additive Gaussian, sigma=5 HU  | `add_noise(ct, sigma=5, seed=pid)`   |
| 14| noise        | additive Gaussian, sigma=10 HU | `add_noise(ct, sigma=10, seed=pid)`  |

### Implementation rules

- **Rotation**: SimpleITK `Euler3DTransform` about the volume centre, linear interpolation, constant background fill = -1024 (air HU). Z axis only; no x/y rotations because the cohort is axial-gated and clinically plausible mis-registration is in-plane.
- **Translation**: SimpleITK `TranslationTransform`, linear interpolation, constant fill = -1024. Translation is in physical millimetres, converted to voxel offsets per patient's resampled spacing (0.5 x 0.5 x 3.0 mm per D005). z is not translated; slice spacing of 3 mm makes sub-slice z shifts ambiguous and gated cardiac acquisition limits z mis-registration in real workflows.
- **Noise**: independent Gaussian added pixel-wise with deterministic seed `seed = int(pid) * 1_000 + sigma_int` for reproducibility across reruns. Clipped to the same `[-200, 3000]` HU range as the source preprocessed array.
- **Mask is not transformed**. This is the test-retest reproducibility convention (Mackin 2015, Lin 2022 for cardiac calcium), not the IBSI-2 convention. The two designs answer different questions: co-perturbing the mask tests whether the extractor is numerically equivariant under rigid transforms (a software correctness check); holding the mask fixed tests whether features survive small mis-registrations of the CT relative to a fixed reference segmentation (the clinically relevant question for test-retest stability in downstream phenotype discovery). Our pipeline is downstream-driven, so we adopt the fixed-mask design. Co-perturbing the mask is a weaker test for this use case (most rigid-transform perturbations become near-identity and ICC is dominated by noise).
- **Order is fixed**: perturbations are indexed 0 through 13 in the table above so that the 422-by-14 reliability matrix is reproducible across reruns.

## Rationale

- **Why these magnitudes**: ±5 to ±10 degrees of rotation, ±2 to ±5 mm of translation, and 5 to 10 HU of additive noise span the clinically plausible registration-error envelope for gated cardiac CT. Larger values would test the radiomics pipeline outside its intended operating range. Smaller values produce uniformly high ICC and would not discriminate.
- **Why 14 perturbations**: enough degrees of freedom for stable ICC estimation per Bonett 2002 (n_subjects >> n_raters, n_raters >= 6 is sufficient).
- **Z-axis rotation only**: cardiac NCCT is gated axially; left-right and head-foot tilts that produce out-of-axial rotations are extremely uncommon in clinical reality and would primarily test the resampling code, not biological feature stability.
- **No z translation**: slice spacing is 3 mm. A sub-slice z shift is either (a) negligible after re-resampling or (b) loses an entire slice's worth of calcium for the +/-2 mm probe. Neither outcome teaches us about feature stability; it teaches us about interpolator choice. We exclude it.
- **Deterministic noise seed per (pid, sigma)**: the same patient produces the same noisy array on a rerun. This is necessary for D013's ICC to be a fixed property of the feature set, not a function of RNG state.

## Alternatives considered

- **More rotations including ±1, ±2 degrees**: too small to move features off their precision floor; adds compute, no information.
- **More noise levels (sigma=2, 20)**: sigma=2 is below the residual noise of the source acquisition (CTDIvol heterogeneity in our cohort already implies ~5 HU baseline noise per the dataset report). sigma=20 is above clinically plausible regimes and would distort HU thresholding.
- **Random rather than deterministic perturbation parameters**: ICC computed on random draws estimates a population mean ICC, which sounds more general but is not what downstream code consumes. We need ICC of a fixed feature pipeline, so the design matrix is fixed.
- **Geometric mask perturbations** (erosion / dilation by 1 voxel): tests segmentation robustness, not registration. Out of scope for this stage; can be added as a stage 7 sensitivity probe.

## Verified by

- `tests/stability/test_perturbations.py`: each transform produces a CT of the same shape; noise has the expected std on a constant image; rotation by 360 degrees is near-identity (interpolation round-trip); translation by 0 is identity.
- `scripts/04_perturbations.py`: writes 14 per-patient extraction logs, one per perturbation, with a manifest mapping `(pid, pert_id)` to the saved feature row.
