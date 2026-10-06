# Stage 4, stability (ICC perturbation gate)

## What this stage does

Filters the 175-feature output of stage 3 down to a stability-gated subset.
Features that are unstable under plausible acquisition, reconstruction and
annotation variation are dropped before any clustering or phenotype analysis
in stage 5 and beyond. Two gates are computed:

1. **D033 realistic gate (primary; feeds stage 5).** Each eligible patient
   receives 20 composite draws; every draw applies positioning, grid offset,
   mask/image misalignment, field-of-view resampling, kernel blur, dose noise,
   HU calibration drift and annotator-boundary jitter (M1) at random
   magnitudes anchored to COCA's own protocol. A feature passes at
   ICC(1,1) >= 0.75 over the draws. Canonical features are recomputed from the
   jittered annotation; those that never vary are tagged
   `invariant_by_construction`.
2. **D014 fixed-mask gate (supplementary).** 14 deterministic CT
   perturbations with the mask held fixed, ICC(3,1) >= 0.75. The 68 canonical
   features never read the CT and are asserted ICC 1.0
   (`icc_source = invariant_by_construction`, D016); the 107 PyRadiomics
   features are measured (`icc_source = empirical`).

`configs/default.yaml` `stability.primary_gate` selects the gate that stage 5
reads (`predict.config.gate_dir`).

## Module list

- `src/predict/stability/perturbations.py`  the 14 perturbation specs and
  the SimpleITK transforms.
- `src/predict/stability/icc.py`  ICC(3,1) absolute-agreement formula,
  reliability-matrix assembly, gate, bypass registry.
- `scripts/04_perturbations.py`  parallel re-extraction of PyRadiomics on
  perturbed CTs.
- `scripts/05_icc_gate.py`  ICC computation, gating, report writers.
- `scripts/04b_realistic_perturbations.py`, `scripts/05b_icc_gate_realistic.py`
  the D033 realistic gate (primary).
- `src/predict/completion.py`  input fingerprints and `COMPLETE.json`
  markers; `src/predict/publish.py`  all-or-nothing gate publication.

## Why these design choices

### D013, ICC(3,1) absolute agreement at 0.75

- ICC(3,1), not (2,1): the 14 perturbations are a fixed deterministic design,
  not a random sample of raters.
- Absolute agreement, not consistency: a systematic shift under rotation
  (e.g. all values bias up by 5 units) is a reliability concern, not a noise
  channel to be ignored. Penalising it is the conservative choice.
- 0.75: Koo and Li 2016 "good reliability" floor, standard in radiomics
  reproducibility studies (IBSI-2, Lin 2022, Kolossvary 2025).

### D014, 14 perturbations, mask held fixed

- 4 rotations (+/- 5 and +/- 10 degrees about z), 8 translations (+/- 2 and
  +/- 5 mm in x and y separately), 2 noise levels (sigma = 5 and 10 HU).
- z is not translated; 3 mm slice spacing makes sub-slice z shifts ambiguous
  and gated cardiac acquisition limits z mis-registration in real workflows.
- Magnitudes span the clinically plausible registration-error envelope.
  Smaller would not discriminate; larger would test out-of-distribution.
- Mask is **not** transformed. This tests mis-registration robustness ("if
  the patient was positioned slightly differently, would the feature
  change?"). The co-perturbation design (IBSI-2 convention) would make most
  perturbations near-identity and the gate would only filter noise-sensitive
  features. D033 combines both: positioning moves CT and mask together (P1,
  P2), and a sub-voxel CT-only shift (P3) models mask/image misalignment.

### D015, eligible cohort (437 patients)

- The 439 stage 1 patients minus the 2 with `radiomics_status == "skipped"`
  from stage 3 (whole-mask voxel count at or below PyRadiomics'
  minimumROISize = 14).
- These already have NaN PyRadiomics columns in baseline; ICC is undefined
  for them and they would contribute no information.
- Includes the 97 `low_burden_flag == True` patients (mask below 100
  voxels). Their PyRadiomics is numerically defined but biologically
  suspect; we keep them in the ICC computation because the gate is meant
  to identify features that are unstable across the whole cohort, not the
  cleanest subset.

### D016, bypass scope, 68 canonical features

The bypass list is **all 68 canonical features**, not just the 31
pure-geometry features. The reasoning, verified by code inspection and
empirical confirmation:

- All canonical features read from `parse_result` (XML) or `Lesion` objects
  derived from XML polygons. None index into the CT pixel array.
- `roi.max_hu` and `roi.mean_hu` are populated from the XML's Max / Mean
  fields at parse time (`xml_parser.py` lines 122 and 123) and never mutated.
  These are the radiologist's measurement, not a CT-derived statistic.
- Perturbing the CT does not change the XML, therefore cannot change the
  canonical feature values. Empirically confirmed on patient 306, 2026-06-03:
  68 keys, 0 diffs across two calls.

Splitting the canonical features into pure-geometry (bypass) and
HU-touching (empirical gate) would be wrong: the HU-touching canonical
features read XML, not CT. Forcing them through the empirical gate would
require re-deriving HU statistics by sampling the perturbed CT through the
polygon, a code path that does not exist in production. The ICC measured on
such a re-derived quantity would describe a hypothetical alternative
pipeline, not this one. Under D033 the annotation itself is perturbed (M1),
so canonical features are recomputed per draw and gated empirically unless
they never vary.

## Module contracts

### `perturbations.py`

```python
@dataclass(frozen=True)
class PerturbationSpec:
    name: str                # stable string key, e.g. "rotate_+5", "noise_10"
    kind: Literal["rotate", "translate", "noise"]
    rotation_deg: float = 0.0
    tx_mm: float = 0.0
    ty_mm: float = 0.0
    sigma_hu: float = 0.0

def enumerate_perturbations(cfg: Config) -> tuple[PerturbationSpec, ...]:
    """Return the locked 14 specs in deterministic order. Raises if the
    config does not produce exactly 14."""

def rotate(ct_img: sitk.Image, degrees: float, *, background_hu: float = -1024.0) -> sitk.Image:
    """Z-axis rotation about volume centre, linear interpolation."""

def translate(ct_img: sitk.Image, tx_mm: float, ty_mm: float, *, background_hu: float = -1024.0) -> sitk.Image:
    """Physical x/y translation in mm; z is never translated."""

def add_gaussian_noise(ct_img: sitk.Image, sigma_hu: float, *, seed: int,
                       clip_min: float = -200.0, clip_max: float = 3000.0) -> sitk.Image:
    """Per-voxel Gaussian noise, deterministic seed, clipped to HU range."""

def noise_seed(pid: str | int, sigma_hu: float, multiplier: int) -> int:
    """Per-(pid, sigma) deterministic seed: int(pid) * multiplier + int(sigma)."""

def apply_perturbation(ct_img: sitk.Image, spec: PerturbationSpec, *,
                       pid: str | int, cfg: Config) -> sitk.Image:
    """Dispatch one spec; returns the perturbed CT. Mask is NOT modified."""
```

### `icc.py`

```python
def icc_3_1_absolute(matrix: np.ndarray) -> float:
    """ICC(3,1) absolute agreement on (n_subjects, k_raters). Listwise NaN
    deletion. Returns NaN if degenerate."""

def build_reliability_matrix(feature_name: str,
                             baseline_df: pd.DataFrame,
                             perturbation_dfs: dict[str, pd.DataFrame],
                             *, pid_col: str = "pid") -> tuple[np.ndarray, list[str], list[str]]:
    """Stack baseline + each perturbation's column for one feature.
    Patients are intersected; baseline first, then dict iteration order."""

@dataclass(frozen=True)
class IccRecord:
    feature: str
    icc: float
    icc_source: Literal["empirical", "invariant_by_construction"]
    n_subjects: int
    n_raters: int
    passes_gate: bool

def gate_features(records: list[IccRecord], *, threshold: float) -> tuple[list[IccRecord], list[str]]:
    """Apply threshold; returns updated records (passes_gate recomputed) and
    the list of feature names that pass. NaN ICC always fails."""

def invariant_by_construction_features() -> tuple[str, ...]:
    """All 68 canonical features. Returns feature_schema.feature_names() so
    the registry is locked to the schema and cannot drift."""
```

### Outputs

```
outputs/04_perturbations/
  rotate_+5.csv             one row per eligible patient, 107 original_* cols + error
  rotate_-5.csv             ...
  ... (14 files total)
  perturbation_log.csv      per-(pid, pert) runtime / status row
  run_header.json           git hash + lib versions + perturbation names

outputs/05_icc/
  icc_report.csv            one row per feature: feature, icc, icc_source,
                            n_subjects, n_raters, passes_gate
  gated_features.csv        single column 'feature', features passing the gate
  icc_summary.json          aggregate counts and threshold info
```

## How to run

Smoke test first (one perturbation, five patients):

```bash
cd <repo-root>
conda activate predict_env
python scripts/04_perturbations.py --only-perturbation noise_5 --limit 5 --no-resume
```

Full run (normally through `run_pipeline.sh`, stage 4):

```bash
python scripts/04_perturbations.py            # D014 draws
python scripts/05_icc_gate.py                 # D014 gate
python scripts/calibrate_realistic.py         # D033 calibration
python scripts/04b_realistic_perturbations.py --arms all
python scripts/05b_icc_gate_realistic.py      # D033 gate (primary)
python scripts/05c_robustness_tiers.py        # D034 tiers
```

Resume is automatic but verified (see "Run safeguards" below); `--no-resume`
forces re-extraction.

## Run safeguards

Both gates (D014 classic, D033 realistic) follow the same rules.

- **Verified resume.** A cached result is reused only if its sidecar JSON
  carries the current fingerprint and recorded zero errors. The fingerprint
  covers all `src/predict` code, `params.yaml`, `configs/default.yaml` and the
  stage script, plus, for D014, the perturbation spec, the eligible pid list
  and the published stage-2 folder; for D033, the calibration and each
  patient's stage-2 CT and mask, XML, `features.csv` row and D002
  exclusions. The CSV must also be complete: every eligible pid once with no
  error text (D014), every planned (arm, index) once (D033). The sidecar is
  written after the CSV, so an interrupted unit is recomputed. Checking only
  that the pids or (arm, index) pairs are present would reuse files of failed
  rows or results from older inputs.
- **Errors fail the stage.** D033.1: a draw is `broken` (allowed, counted by
  the 5% exclusion rule) only when the perturbed mask is empty or rejected
  by PyRadiomics' own ROI checks (14 voxels or fewer, a single voxel, or too
  few dimensions), decided before extraction. Any other
  exception is an `error`. One error (or a failed parity check) makes 4b
  exit 1 without `COMPLETE.json`. In D014 any patient error, including an
  empty mask, makes 4a exit 1 without `COMPLETE.json`, and 5 refuses
  perturbation CSVs with error text (instead of silently dropping the
  patient from that feature's ICC).
- **Completion markers.** 4a and 4b clear `COMPLETE.json` at start and write
  it only after a full, error-free run (never for `--limit`, `--pids`,
  `--max-draws` or `--only-perturbation`). 5 and 5b refuse to gate unless
  the marker exists and matches the current `features.csv` and the current
  run fingerprint, recomputed by the stage script's own `run_fingerprint`
  (5b also checks the calibration hash). The 4b marker also carries a
  patients fingerprint (every patient's stage-2 CT and mask, XML,
  `features.csv` row and exclusions); 5b recomputes it with 4b's own
  `patient_fingerprints` (hashing the patient files, about a minute), so a
  changed input refuses the gate even when `features.csv` is unchanged. 4a's
  run fingerprint already covers the stage-2 folder. 5b evaluates only the
  arms the marker lists.
- **Fingerprint scope.** The code and config part of every stage 4
  fingerprint is `predict.completion.upstream_fingerprint`: the stage 1-4
  packages (`io`, `preprocess`, `features`, `stability`), `config.py` and
  the package `__init__.py` (the only top-level modules stage 1-4
  computation imports; the bookkeeping modules `completion`, `provenance` and
  `publish` change no computed value and are excluded), `params.yaml`, the
  stage script,
  and the config sections `seed`,
  `cohort`, `resample`, `hu`, `lesion_grouping`, `stability`. Stage 5+ code
  and config sections cannot change a draw, so editing them does not
  invalidate the (many-hour) stage 4 results; editing anything stage 4
  depends on does.
- **Prescribed draw counts.** 5b checks that every gated patient has draw
  indices 0..N-1 exactly once in the primary arm (20) and in every secondary
  or attribution arm (10), with N from `arm_draw_plan`, never inferred from
  the largest index present.
- **Smoke isolation.** `05b --allow-partial` never writes to
  `05b_icc_realistic/` (default `05b_icc_realistic_smoke/`) and marks its
  output `PARTIAL_SMOKE_DO_NOT_USE`; `predict.config.gate_dir`, which every
  downstream stage uses to find the gate, refuses a marked folder.
- **Atomic publication.** 5 and 5b write to `<gate>.partial/` and replace the
  gate folder only when the run finishes.

## Tests

`tests/stability/`:

- `test_perturbations.py`: perturbation enumeration, individual transforms
  (rotation / translation / noise), determinism, geometry preservation,
  dispatch, full-set smoke.
- `test_icc.py`: ICC formulas on toy matrices, NaN handling, degeneracy,
  matrix assembly, gate application, bypass registry.
- `test_bypass_truly_invariant.py`: static check (zero CT-array refs in
  canonical-feature modules) and functional check (each canonical function
  produces byte-identical output across repeated calls).
- realistic-gate, tier and stage 4 safeguard tests (draw plan, seeds,
  parity, markers, smoke isolation).

## Results on the analysis cohort

437 eligible patients (PyRadiomics `ok`, D015), 175 features (68 canonical,
107 PyRadiomics). Numbers are from the published stage 4 and 5 outputs
(`outputs/05b_icc_realistic/`, `outputs/05_icc/`).

### D033 realistic gate (primary)

| | value |
|---|---|
| Draws | 69,920 (20 primary per patient plus the secondary and attribution arms) |
| Broken draws | 542, all perturbed masks below PyRadiomics' minimum ROI size (32 in the primary arm) |
| Parity | 437 / 437 patients rebuild the stage 2 mask and `features.csv` exactly; 0 errors |
| **Features passing ICC(1,1) >= 0.75** | **166 of 175** (68 canonical, 98 PyRadiomics) |
| Realism (primary arm, mean pairwise relative difference between draws) | Agatston 4.60%, volume 4.43%, mask voxels 3.82% |

The 9 failing features are PyRadiomics low-intensity and zone-size
descriptors (`firstorder_10Percentile` .499, `firstorder_Minimum` .745,
`gldm_LowGrayLevelEmphasis` .746, `gldm_LargeDependenceLowGrayLevelEmphasis`
.540, `glrlm_LongRunLowGrayLevelEmphasis` .709,
`glszm_LargeAreaLowGrayLevelEmphasis` .500,
`glszm_SizeZoneNonUniformityNormalized` .720, `glszm_SmallAreaEmphasis`
.655, `glszm_SmallAreaLowGrayLevelEmphasis` .744). Three sit within .006 of
the threshold; the locked point rule applies and none is rescued.

The primary composite stays below published rescan variability (Agatston
20.1%, volume 18.3%; Detrano 2005), as D033 requires. Secondary arms (passing
features): S1 z-phase 153, S2 167, S3 68 (excluded from the D034 tiers),
S4 168, S5 168, S6 168. Robustness tiers (D034) T1 to T5: 166 / 164 / 153 /
153 / 150. ICC(3,1) and ICC(1,1) give the same pass / fail for every
feature.

### D014 fixed-mask gate (supplementary)

89 of 175 features pass (68 canonical, 21 PyRadiomics). Every D014 survivor
also passes D033; 77 features pass only D033. The fixed-mask translations of
up to 5 mm move most lesions off their own outline (median lesion radius
1.22 mm), so voxel-identity texture features collapse under D014; D033's
magnitudes are anchored to COCA's own acquisition and annotation variation
instead.

Threshold sensitivity of the D014 report (`scripts/05c_threshold_sensitivity.py`)
and the per-arm and per-tier tables are written next to each gate.

## Edge cases

- **Empty mask**: stage 2 guarantees a non-empty mask, so `_process_one`
  records an empty mask as an error, which fails stage 4a.
- **Perturbation that rotates calcium out of the FOV**: corner voxels get
  background fill (-1024). PyRadiomics still extracts on whatever calcium
  is inside the fixed mask; if the mask now overlaps mostly background-filled
  voxels, the texture features will be unstable, which is exactly what the
  gate is designed to detect.
- **noise_seed collision**: seed = int(pid) * 1000 + int(sigma). For
  sigma 5 vs 10 and any two patient IDs that differ, the seeds are unique
  by construction. Non-numeric pids fall back to a hash. Tested.
- **Listwise NaN deletion**: a feature with NaN in any rater drops that
  patient from the ICC computation for that feature only. A PyRadiomics
  failure never reaches this path: error rows stop stage 5.
- **ICC degeneracy**: if a feature is constant across all subjects and
  raters (zero total variance), ICC is NaN, treated as a gate failure.
  Should not happen for any non-trivial feature on this cohort.

## Known limitations

- D014 tests robustness to CT-side mis-registration only. D033 adds
  annotator-boundary jitter, but the frozen XML HU fields (Max, Mean) are
  not perturbed, and true rescan effects such as z-phase and motion cannot
  be simulated faithfully from 3 mm data (reported in a secondary arm).
- Neither perturbation set is exhaustive of clinically plausible noise
  sources. They are a representative subset designed to discriminate
  reliable from unreliable features, not to certify a feature as stable
  under every conceivable acquisition variation.
- PyRadiomics shape features should have ICC = 1.0 exactly under our
  fixed-mask design. If they do not, the perturbation implementation has
  a bug (the mask is moving when it should not). The 0/0 invariants in
  the gate code are designed to catch this.

## Decisions referencing this stage

- D013, ICC formulation and threshold.
- D014, 14-perturbation set (mask held fixed).
- D015, eligible cohort (PyRadiomics `ok`).
- D016, bypass scope (68 canonical bypass, 107 PyRadiomics empirical).
- D033 / D033.1, the realistic gate and its broken-draw definition.
- D034, robustness tiers on the D033 gate.
