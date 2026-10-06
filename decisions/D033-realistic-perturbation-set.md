# D033 Realistic composite perturbation set (primary stability gate)

**Date**: 2026-09-24 (drafted and locked)
**Stage**: stability
**Status**: Active. LOCKED 2026-09-24, before any D033 ICC was computed. After lock, the only permitted edits are dated fixes of implementation defects found by the tests or the stage-4b parity check, and only while no D033 ICC exists.
**Module**: `src/predict/stability/realistic.py`, `src/predict/stability/icc.py` (`icc_1_1`), `src/predict/features/canonical.py`, `scripts/calibrate_realistic.py`, `scripts/04b_realistic_perturbations.py`, `scripts/05b_icc_gate_realistic.py`

## Decision

The feature set carried downstream (stages 5 to 8) is the set that passes **this** gate. Each eligible patient (D015) receives **20 composite draws**. Every draw applies every primary factor below at a random magnitude, because real errors occur together. A feature passes if **ICC(1,1) >= 0.75** across the 20 draws.

The organising rule for what enters the gate: **primary factors are variation that already exists inside COCA's own acquisition, reconstruction and annotation protocol**. Protocol changes COCA does not contain, and variation that cannot be simulated faithfully from 3 mm data, are reported in a secondary arm and never gate.

D014 is retired from downstream use. It is not amended (closed decisions are never edited); its gate is still computed and reported as a supplementary comparison.

## Primary factors (all applied in every draw)

Voxel = 0.375 mm in-plane (D005). U(a, b) is uniform.

| ID | Real process | Simulation | Magnitude | Anchor |
|----|--------------|------------|-----------|--------|
| P1 | Patient positioning | Rotation about z through the volume centre, CT **and** mask together | theta ~ U(-5, 5) deg | Positioning varies by a few degrees between scans; co-perturbation per IBSI-2 |
| P2 | A rescan lands anatomy on a new voxel grid | In-plane translation, CT and mask together | dx, dy ~ U(-0.5, 0.5) voxel each | Grid position is arbitrary: any sub-voxel offset is equally likely |
| P3 | Mask vs image misalignment (model masks, mask transfer, synthetic mask/image pairs) | In-plane translation of the **CT only**, mask held fixed (kept from D014) | direction ~ U(0, 360) deg, length ~ U(0, 1) voxel | Boundary ambiguity is about 1 voxel (binary mask vs partial volume). Larger shifts move most lesions off their own outline (`scripts/probe_perturbation_displacement.py`: median lesion radius 1.22 mm) |
| P4 | Field-of-view differences change in-plane pixel size | CT and mask resampled 0.375 mm -> s* -> 0.375 mm | s* drawn from the empirical distribution of COCA native in-plane spacings (eligible cohort); no-op when s* <= the patient's own native spacing | COCA's own FOV variation (0.26 to 0.71 mm); an image cannot be made sharper than it was acquired |
| P5 | Reconstruction kernel (COCA mixes Qr36d/2, n=235, and I30f/3, n=207) | In-plane Gaussian blur of the CT | sigma_b ~ U(0, sigma_delta); sigma_delta from calibration C2. Factor dropped if sigma_delta < 0.1 voxel | The measured difference between COCA's two kernels |
| P6 | Dose variation within protocol | Additive correlated noise | dose fraction f ~ U(0.5, 1.0); added sd = sigma0 x sqrt(1/f - 1), i.e. 0 to sigma0 (half dose); white noise filtered in-plane by a Gaussian of sd sigma_n,k (kernel-specific, C1b), rescaled to the target sd, independent per slice | Noise sd scales with 1/sqrt(dose); CT noise is textured by the kernel, so white noise is not realistic |
| P7 | Scanner HU calibration drift | Constant HU offset on the CT | c ~ U(-5, 5) HU | Inside the ACR accreditation tolerances: water plug -7 to +7 HU (Module 1) and uniformity-module centre 0 +/- 5 HU, edge-to-centre <= 5 HU (Module 3). See External anchors |
| M1 | Annotator boundary | Smooth displacement of every XML polygon along its local normal, at native resolution, **before** mask building; the mask is then rebuilt with production `build_3d_mask` and resampled with production `resample_to_target` | Displacement field = sum of 3 random low-order Fourier terms along the contour, scaled to RMS A ~ U(0, 0.5) native pixel, hard cap 1 native pixel; if a polygon's area falls below 50% of the original, the field is scaled down until it does not | Rasterisation ambiguity of about 0.5 pixel per side (binary mask vs partial volume). No lesion is ever deleted by boundary noise |

### Order of operations (one draw)

1. **M1** jitters the polygons; the D002 exclusion set from the unperturbed round-trip is reused unchanged. Mask rebuilt at native resolution, resampled to 0.375 mm (production code). The CT is the cached production-resampled CT.
2. **P1, P2, P3** composed into one resampling per image (no double interpolation): CT transform = P3 o P2 o P1, mask transform = P2 o P1. CT linear interpolation with fill `stability.background_fill_hu` (as D014); mask linearly interpolated and thresholded at 0.5.
3. **P4** CT B-spline and mask nearest-neighbour both ways (the D005 interpolators). Components that vanish are counted (see "Broken draws").
4. **P5** blur, then **P6** noise, then **P7** offset, then clip to `[hu.clip_min, hu.clip_max]` (as stage 2).
5. Feature extraction with the production PyRadiomics configuration (D009) and the production canonical-feature code on the jittered annotation.

### Randomness

`rng = numpy.random.default_rng([int(pid), draw_index, 33])`, factors drawn in the fixed order M1, P1, P2, P3, P4, P5, P6, P7, so any draw is reproducible in isolation. Draw indices 0 to 19 are primary.

### Implementation precision (fixed at lock)

- **M1 field**: 7 coefficients N(0, 1): a0 (a systematic dilation or erosion, as a second annotator drawing tighter or looser would) plus cos and sin terms for k = 1, 2, 3 over normalised arc length; scaled to RMS A, capped at 1 native pixel per vertex, outward normals from the polygon's orientation. The XML `Area` is multiplied by the polygon's own area ratio, so a zero field reproduces the XML exactly. Dirty, D002-excluded and < 3-point ROIs are untouched.
- **Seeds**: primary `default_rng([int(pid), draw_index, 33])`; every other arm appends an arm code, `[int(pid), draw_index, 33, arm_code]`. All magnitudes are always drawn in the fixed order, so switching a factor off never shifts the stream.
- **P4**: s* is one value drawn from the C3 list; if s* is not larger than the patient's own native spacing the draw leaves resolution unchanged.
- **Final CT**: rounded to the nearest integer and stored as int16, the stage-2 dtype.
- **S3 / S4** act on the voxel mask only; canonical features in those arms come from M1 as in every draw.
- **S6**: s* ~ U(native spacing, 0.90 mm), replacing P4; no-op if the native spacing is already >= 0.90 mm.
- **Canonical values on broken draws**: canonical features are computed before extraction, so a draw broken only for PyRadiomics still contributes its canonical values.
- **Parity hard stop**: for every patient, stage 4b rebuilds the unperturbed mask and features with the D033 code path; the mask must equal the stage-2 mask exactly and the features must equal `features.csv`. Stage 5b refuses to gate if any eligible patient failed.
- **Compute**: 160 extractions per patient (20 primary + 6 x 10 secondary + 8 x 10 attribution).

## Canonical (XML-derived) features: extension of D016

D016's bypass rests on the canonical features never reading the CT. M1 changes the annotation itself, so the canonical features are now recomputed per draw from the jittered polygons: polygon area (shoelace x native spacing), lesion grouping (D007), centroids and every quantity derived from them. The XML `Max` and `Mean` values stay frozen: regenerating them would mean sampling the CT through the polygon, which D016 rejected as a different pipeline.

Rule: a canonical feature whose within-patient variance across the 20 draws is zero for every patient is tagged `invariant_by_construction` (ICC asserted 1.0, as D016). Every other canonical feature gets an empirical ICC(1,1) and faces the same gate as the PyRadiomics features. Limitation, stated in the paper: annotator variation in the frozen XML HU fields is not tested.

## ICC and gate

- **ICC(1,1)**, one-way random effects, single measurement, computed over the 20 draws. The unperturbed baseline is excluded: it is not a draw from the error distribution.
- Why not D013's ICC(3,1): the draws are random samples, not the same fixed raters across patients, so the two-way mixed model does not apply. D013 stays closed and continues to describe D014.
- Pass: point estimate >= 0.75 (D013's threshold, kept). The 95% CI (F-distribution) is reported for every feature.
- New function `icc_1_1` in `src/predict/stability/icc.py`, tested against a hand-computed example and against the textbook one-way ANOVA identity.

## Broken draws and undefined features

- A draw is broken for a patient if extraction fails (for example the mask drops below `minimumROISize` 14, D009) or a feature is NaN.
- For each feature, patients with any broken draw are excluded from that feature's ICC and counted. **A feature fails if more than 5% of the eligible cohort is excluded for it.** This depends only on definedness, never on ICC values.
- Vanished lesion components (P4) are counted per patient and reported.

## Edge-case strata (reported, never gating)

Low burden (D010), metal flag, kernel, native-spacing tertile. Per stratum: ICC(1,1) and within-subject CV. ICC falls mechanically in narrow strata (restricted between-subject range), so worst-stratum gating would penalise the wrong thing. Features fragile in the low-burden stratum are listed for the segmentation and synthetic-mask teams.

## Secondary arm (reported, never gating)

Each runs 10 draws of the primary factors plus one extra factor:

| ID | Factor | Why secondary |
|----|--------|---------------|
| S1 | z-phase shift U(-1.5, 1.5) mm, CT and mask, interpolated | Real and important, but 3 mm data cannot recreate true partial volume; no thinner series exists in COCA (`scripts/diagnose_multi_series.py`) |
| S2 | Directional blur, random in-plane direction, length U(0, 1.5) mm | Scans are ECG-gated and real motion is local; a global blur is a weak model |
| S3 | M2: mask = voxels >= 130 HU inside a 1-voxel dilation of the expert mask, keeping only components that overlap it | Protocol change (automatic Agatston tools), absent from COCA |
| S4 | M3: Agatston 1 mm^2 per-slice minimum-area lesion dropout | Protocol change |
| S5 | Quarter dose, f ~ U(0.25, 0.5) | Low-dose protocol, outside COCA's |
| S6 | P4 up to 0.90 mm | Outside COCA's spacing range (coarser in-plane resolution) |

## Attribution arm (explanation only, never gating)

Each primary factor alone, 10 draws. Reports which factor removes which feature.

## Calibration (computed before any D033 ICC; reads images and headers only)

`scripts/calibrate_realistic.py` writes `outputs/04b_calibration/calibration.json`; its values are appended to this document when it becomes Active. It must not read any feature or ICC output.

- **C1 sigma0** per patient: Immerkaer (1996) estimator on the preprocessed CT, soft-tissue band [-100, 200] HU, only pixels whose full 3x3 neighbourhood is in band, median over slices.
- **C1b sigma_n,k** per kernel: lag-1 autocorrelation rho1 along x of the high-pass residual (CT minus 5x5 median) at pixel pairs whose 5x5 neighbourhoods are in band; kernel median rho1; sigma_n,k is the filter sd whose filtered white noise has that rho1, solved exactly for the discrete scipy Gaussian kernel used in P6 (continuous limit rho1 = exp(-1 / (4 sigma^2))). rho1 <= 0 gives sigma_n = 0 (white).
- **C2 sigma_delta** = sqrt(|sigma_n,Qr36d/2^2 - sigma_n,I30f/3^2|) voxels.
- **C3** empirical distribution of native in-plane spacing over the eligible cohort.

## Realism check (reported, no retuning)

Metric, matched to the published one: mean percentage absolute difference between two measurements, |a - b| / ((a + b) / 2), over pairs of draws, patients with both values zero excluded. Computed for (a) recomputed Agatston total and (b) calcium volume.

- **Primary composite vs rescan.** Published rescan variability includes causes the primary gate deliberately leaves out (z-phase, motion; Detrano 2005 found reproducibility lower with misregistration or motion artifacts). So the primary composite must come out **at or below** the published rescan figures (Agatston about 20%, volume about 18%). Above them means the primary magnitudes are too aggressive.
- **Secondary S1 (z-phase) plus primary** is expected to move toward the rescan figures; reported for context.
- **M1 alone vs readers.** The M1 attribution arm's Agatston variability is compared with published intra-observer variability (about 2%).

Outcomes are reported, including as a limitation if they fall outside. Magnitudes are **not** retuned after this check.

## External anchors (sourced 2026-09-24)

- **ACR CT accreditation phantom.** Water plug acceptable range -7 to +7 HU (listed in Table 5 of the ACR-phantom extension-plate evaluation, J Appl Clin Med Phys, PMC5690207). Module 3 uniformity: centre CT number 0 +/- 5 HU and edge-to-centre <= 5 HU (ACR CT Phantom Testing Instructions, 2004; read from an unofficial copy, so the current ACR manual should be checked if P7 is challenged).
- **Rescan reproducibility, MESA.** Detrano et al., Radiology 2005;236:477-84. 3,355 participants with calcium, duplicate scans, electron-beam and multi-detector CT. Mean relative differences 20.1 (Agatston), 18.3 (volume), 18.3 (interpolated volume); presence-of-calcium concordance 96%, kappa 0.92; reproducibility lower with misregistration or motion artifacts.
- **Rescan reproducibility, 16-MDCT.** Sabour et al., Eur J Epidemiol 2007 (PMC1914270). 76 women (55 with calcium), 1.5 and 3.0 mm slices. At 1.5 mm: relative difference 14.5% (Agatston), 14.6% (volume), 12.4% (mass); ICC 0.98 to 0.99. Absolute differences grow with calcium burden.
- **Readers and scanners.** Ghadri et al., Eur Heart J 2011;32:1865. 30 patients on 64-MDCT (GE, 2.5 mm) and dual-source CT (Siemens Definition, 3 mm). Intra-observer CV for Agatston 2.0% and 2.1%; inter-scanner CV 15.1% (Agatston), 21.6% (mass), 44.9% (volume, falling to 11.5% when both were read with the same software).

## Relation to earlier decisions

- **D014**: retired from downstream use as the gate, not amended. Its gate is still computed and reported as a supplementary comparison (survivor overlap).
- **D013**: unchanged; describes D014. D033 defines its own ICC form above.
- **D015**: same eligible cohort.
- **D016**: extended as described above; its CT-side argument still holds.
- **Disclosure**: D033 was designed after the D014 perturbation-subset probe (`scripts/probe_perturbation_subsets.py`) had been seen. Its rationale rests on lesion displacement geometry and on COCA's protocol, not on pass counts. The D033 ICC outcome was unknown when it was locked.

## Rationale

- D014's magnitudes were set as a "clinically plausible registration-error envelope". The displacement probe showed every D014 spatial perturbation moves most lesions beyond their own radius (77% to 99%), so its fixed-mask arm measured signal loss rather than registration robustness. Its fixed-mask idea was sound and is kept (P3) at a magnitude where lesions stay inside their outline.
- D014 rejected random draws because "ICC on random draws estimates a population mean ICC ... not what downstream code consumes". Downstream here is new scans, new reconstructions and new masks, whose errors are random. A population ICC over the realistic error distribution is exactly the quantity downstream meets; an ICC over 14 fixed settings is a property of those settings.
- Composite draws follow Zwanenburg et al. 2019 (combined rotation, translation, volume adaptation and contour randomisation), adapted to the fact that COCA masks are drawn on the same image (D002 max-exact), so image-mask misregistration is small and boundary uncertainty is the dominant annotation error.
- Under D014 the 14 shape features passed automatically because they read only the fixed mask. M1, P2 and P4 give them their first real test.

## Alternatives considered

- **Keep D014 as primary**: rejected; see Rationale.
- **D014 minus the +/-5 mm translations**: rejected; chosen after seeing pass counts, and +/-2 mm and +/-5 deg are also signal-loss tests geometrically.
- **Single-factor arms as the gate**: rejected; no real scan varies one factor at a time.
- **Voxel-level boundary jitter instead of polygon-level M1**: rejected; polygon-level lets the mask and the canonical features vary coherently, which is what a second annotator does.
- **Worst-stratum gating**: rejected; ICC range restriction.
- **Include z-phase in the gate**: rejected; not simulable faithfully from 3 mm slices.

## Before lock (all required)

- [x] D006 confirmed 2026-09-24: on all 17 alternates with >= 2 slices the XML round-trip matches only the selected series (`scripts/diagnose_multi_series.py`).
- [x] ACR tolerance sourced (External anchors): +/-5 HU is inside both the water-plug and uniformity criteria. P7 unchanged.
- [x] Published rescan and reader Agatston variability sourced (External anchors); realism-check metric and direction fixed above.
- [x] Draw count confirmed: 160 extractions per patient.
- [x] Locked 2026-09-24, including the canonical-feature extension above.

## Verified by

- `tests/stability/test_realistic.py`: zero draw is an exact identity; P1/P2 move CT and mask together; P3 moves only the CT, by <= 1 voxel; M1 respects the 1-pixel cap and 50% area floor and leaves excluded / dirty ROIs and XML HU untouched; noise sd and lag-1 correlation match their targets; the rho1 inversion is exact; M2 / M3 behave as specified; seeds match the text above; arm draw counts.
- `tests/stability/test_realistic.py`: `icc_1_1` against a hand-computed example, perfect agreement, pure noise and listwise deletion.
- Stage-4b parity on every eligible patient (runtime hard stop).
- Calibration outputs checked for plausibility before the ICC run (sigma0 by kernel, rho1 by kernel).
