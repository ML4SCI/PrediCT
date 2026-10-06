# Stage 3: `features`

XML-derived scalar features + 3D lesion identification + PyRadiomics on the whole-mask. Produces one row per patient with 184 columns: 68 canonical scalar features, 9 metadata + status columns, 107 PyRadiomics features.

## Modules

| Module | Purpose |
|---|---|
| `feature_schema.py` | Canonical 68-key feature name registry. Single source of truth. |
| `agatston.py` | Per-vessel + total Agatston with the **D011 single helper** (`agatston_roi_score`). |
| `lesion_ccl.py` | BFS-based 3D lesion identification (D007) with Z-matched slice indices. |
| `per_vessel_aggregates.py` | Per-vessel volume / mass / mean HU / max HU (XML-derived). |
| `spatial.py` | Lesion counts, diffusivity (D016), Gini, inter-lesion distances, COM. |
| `density_tiers.py` | Per-vessel ROI counts in 4 Agatston HU tiers + dense-calcium count. |
| `per_artery_mask.py` | Filter-then-rasterise per vessel (D008): reuses `mask_builder`. |
| `radiomics.py` | PyRadiomics wrapper (D009 locked `params.yaml`). |

## What this stage does

Per patient (orchestrated by `scripts/03_features.py`, multi-process):

1. Load native DICOM headers (`load_patient_metadata`) for `slice_positions` and native pixel spacing: header-only, ~10× faster than full image load.
2. Parse XML (`parse_calcium_xml`).
3. Derive `excluded_roi_ids` for this patient from `outputs/02_preprocessed/xml_roundtrip.csv` (non-dirty trips that failed the D002 Max-exact gate).
4. **XML-only paths** (all take `excluded_roi_ids` per D012):
   - Agatston per vessel + total (D011 helper).
   - Per-vessel volume / mass / mean HU / max HU + globals.
   - Density-tier ROI counts + `dense_calcium_count`.
5. **Lesion CCL**: per-vessel BFS with Z-matched slice indices (D001), using the configured `lesion_grouping` values (`max_inplane_dist_mm`, `max_slice_gap`; recorded in `run_header.json`). Produces 3D `Lesion` dataclasses.
6. **Spatial features** from the `Lesion` list: count, diffusivity, Gini, inter-lesion distances, distance-from-top, center of mass.
7. **PyRadiomics** on the resampled whole-mask using locked config (all 7 IBSI families). Graceful-degrade per **D010**, decided before extraction by `roi_too_small` from the locked `minimumROISize` (14) and `minimumROIDimensions`: leave PyRadiomics columns NaN, record `radiomics_status="skipped"` and the reason. Any exception the extractor raises (input guardrail, shape mismatch, PyRadiomics error) fails the patient.
8. Aggregate into one row; write the row + lesion audit + log.

All files go to `outputs/03_features.partial/`; only a run in which every manifest patient succeeds replaces `outputs/03_features/` (`predict.publish`, the stage 2 policy). Any patient failure exits 1 and leaves the published folder untouched; a manifest patient missing from stage 2 fails the stage before extraction; `--limit` never publishes.

Per-stage outputs in `outputs/03_features/`:

```
features.csv                 # one row per patient × 184 columns (the main artefact)
lesions.csv                  # one row per 3D lesion (audit trail)
feature_extraction_log.csv   # per-patient runtime + counts + radiomics_status
lesion_grouping_probe.csv    # sensitivity probe output (offline)
```

## Why these choices

### Hybrid per-patient feature stack (whole-mask + per-lesion + per-vessel)

- **First-order intensity + texture**: whole-mask PyRadiomics. Texture statistics need many voxels; pooling all calcified voxels per patient gives 100s–1000s of voxels, stable. Per-lesion texture on 15-voxel lesions would be noise.
- **Shape**: whole-mask via PyRadiomics for the aggregatable shape features; per-lesion volume/area via `Lesion` dataclass for spatial aggregation (Gini, lesion count). PyRadiomics shape features on a multi-lesion mask are by convention.
- **Spatial distribution**: pure custom code from XML + lesion CCL. Lesion count, inter-lesion distances, diffusivity (D016), Gini, COM. These are voxel-count-independent and stable at any burden.
- **Per-vessel**: scalar features only (XML-derived). Per-artery masks (D008, filter-then-rasterise) are built at native resolution by `scripts/03b_cohort_sanity.py`, whose union check confirms they sum to the whole mask. Per-vessel PyRadiomics was removed on 2026-10-04 (stage 3 review): it rasterised native-pixel polygons onto an image of the resampled shape, so its masks did not match the CT (Dice 0.28 with the saved whole mask on a real scan). It was off by default and nothing used it.

### Z-matched slice indices everywhere

The `Lesion` dataclass stores CT array slice indices from `slice_matcher.resolve_roi_slice` (D001), not the raw `ImageIndex`. Linking by slice index (`max_slice_gap`) is only a statement about millimetres because D039 guarantees every included series has evenly spaced, contiguous slices; without D039 two slices 15 mm apart could be adjacent indices. This is what makes the spatial Z values correct; an inverted `(num_slices-1) - ImageIndex` mapping would put lesions on wrong slices for the Z computations in `spatial_features._compute_dist_from_top` and the Lesion centroid z.

### Sub-pixel polygon rasterisation everywhere

The per-artery masks (stage 3b sanity check) reuse `mask_builder` which uses `cv2.fillPoly` with `shift=4`. Consistent with stage 2's whole-mask. Boundary precision is the same.

### `excluded_roi_ids` is an explicit input to every feature module (D012)

The orchestration script reads `outputs/02_preprocessed/xml_roundtrip.csv` once at start, builds a per-patient exclusion set, and passes it to:

  - `compute_agatston`
  - `group_rois_into_lesions`
  - `compute_per_vessel_aggregates`
  - `compute_density_tiers`
  - `build_per_artery_masks` (stage 3b sanity check)

Result: every feature value in a row is computed from the same input set. Whole-mask PyRadiomics consumes the stage-2 mask which was already cleaned. Consistency guaranteed.

### `low_burden_flag` instead of pre-gating (D010)

`low_burden_flag = (mask_voxels < 100)` is metadata; no feature is pre-gated. PyRadiomics' own `minimumROISize=14` filters the bottom of the distribution; for those patients `roi_too_small` says so before extraction, the PyRadiomics columns are NaN, and the row's XML-derived features are kept. (Until 2026-10-04 the orchestration caught every `ValueError` as a skip, which also hid real extraction failures.) Downstream filtering on `low_burden_flag` or `radiomics_status` is one column away.

### Single Agatston helper (D011)

`agatston_roi_score(area_cm2, max_hu, slice_thickness_mm)` lives in `agatston.py` and is the only Agatston formula in the codebase. `per_vessel_aggregates.py` does not duplicate it. A regression test asserts both Agatston paths agree at multiple slice thicknesses.

## Module contracts

```python
# feature_schema.py
VESSEL_SUFFIXES: tuple[str, ...]      # ("lad", "rca", "lcx", "lm")
DENSITY_TIERS: tuple[str, ...]         # ("d1", "d2", "d3", "d4")
PER_VESSEL_STEMS: tuple[str, ...]      # 10 stems
GLOBAL_FEATURES: tuple[str, ...]       # 12 keys
def feature_names() -> tuple[str, ...]: ...      # full ordered name list
def zero_features() -> dict[str, float]: ...     # fresh schema-complete zero dict
def n_features() -> int: ...                     # 68

# agatston.py
def density_factor(max_hu: float) -> int: ...
def agatston_roi_score(area_cm2: float, max_hu: float,
                       slice_thickness_mm: float) -> float: ...   # D011 helper
def classify_risk(total: float) -> str: ...                       # "0" / "1-99" / "100-399" / "400+"

@dataclass(frozen=True)
class AgatstonResult:
    total: float
    per_vessel: dict[str, float]
    category: str
    def to_feature_dict(self) -> dict[str, float]: ...

def compute_agatston(parse_result: ParseResult, *,
                     slice_thickness_mm: float,
                     excluded_roi_ids: set | None = None) -> AgatstonResult: ...

# lesion_ccl.py
@dataclass(frozen=True)
class Lesion:
    vessel: str
    roi_keys: tuple[tuple[int, int], ...]
    slice_indices: tuple[int, ...]
    centroid_mm: tuple[float, float, float]
    total_area_mm2: float
    mean_hu_weighted: float
    max_hu: float
    volume_mm3: float
    @property
    def n_rois(self) -> int: ...

def group_rois_into_lesions(parse_result: ParseResult, *,
    slice_positions: Sequence[float],
    pixel_spacing_xy: tuple[float, float],
    slice_thickness_mm: float,
    excluded_roi_ids: set | None = None,
    max_inplane_mm: float = 5.0,
    max_slice_gap: int = 1,
    tolerance_mm: float = 1.5,
) -> dict[str, list[Lesion]]: ...

# per_vessel_aggregates.py
def compute_per_vessel_aggregates(parse_result: ParseResult, *,
                                  slice_thickness_mm: float,
                                  excluded_roi_ids: set | None = None,
                                  ) -> dict[str, float]: ...   # 20 keys

# spatial.py
def gini_coefficient(values: Sequence[float]) -> float: ...
def diffusivity(n_lesions: int, d_first_last_mm: float) -> float: ...   # D016
def compute_spatial_features(lesions_per_vessel: dict[str, list[Lesion]], *,
                             slice_positions: Sequence[float],
                             slice_thickness_mm: float = 3.0,
                             ) -> dict[str, float]: ...   # 26 keys

# density_tiers.py
def density_tier(max_hu: float) -> str | None: ...
def compute_density_tiers(parse_result: ParseResult, *,
                          excluded_roi_ids: set | None = None,
                          ) -> dict[str, float]: ...   # 17 keys

# per_artery_mask.py
def build_per_artery_masks(parse_result: ParseResult,
                           loaded: LoadedPatient, *,
                           excluded_roi_ids: set | None = None,
                           tolerance_mm: float = 1.5,
                           ) -> dict[str, np.ndarray]: ...

# radiomics.py
def create_extractor(params_yaml: Path = DEFAULT_PARAMS_YAML) -> RadiomicsFeatureExtractor: ...
def validate_ct_for_radiomics(ct_array: np.ndarray, pid: str = "") -> None: ...
def extract_pyradiomics(ct_array: np.ndarray, mask_array: np.ndarray,
                        spacing: tuple[float, float, float],
                        extractor, *,
                        label: int = 1, pid: str = "") -> dict[str, float]: ...
```

### Orchestration flow (`scripts/03_features.py` per patient)

```
load_patient_metadata(pid)       ─►  meta (slice_positions, spacing, thickness)
parse_calcium_xml(pid)           ─►  parse_result
excluded_by_pid[pid]             ─►  excluded  (from outputs/02_preprocessed/xml_roundtrip.csv)
np.load(*_ct.npy, *_mask.npy)    ─►  ct, mask

row = zero_features()
row += compute_agatston(...)
row += compute_per_vessel_aggregates(...)
row += compute_density_tiers(...)
lesions = group_rois_into_lesions(..., **grouping_from_config(cfg))
row += compute_spatial_features(lesions, ...)

reason = roi_too_small(mask, extractor)      # D010, from the locked params
if reason is None:
    row += extract_pyradiomics(ct, mask, target_spacing, extractor)   # raises -> patient error
    radiomics_status = "ok"
else:
    radiomics_status = "skipped"             # radiomics_reason = reason

row + lesions + log → CSV writers
```

## Decisions

- **D007** Lesion grouping rule (BFS, ≤5 mm in-plane, gap=1, skip dirty + excluded).
- **D008** Per-artery masks via filter-then-rasterise.
- **D009** PyRadiomics extractor configuration (locked `params.yaml`).
- **D010** `low_burden_flag = (mask_voxels < 100)` + graceful-degrade for PyRadiomics on tiny masks.
- **D011** Single Agatston thickness-correction helper.
- **D012** `excluded_roi_ids` as explicit input to every feature path.

## Tests (`tests/features/`)

98 unit tests, all passing:

- `test_feature_schema.py` (12): canonical name list, uniqueness, ordering, zero dict.
- `test_agatston.py` (14): density factor boundaries, single-helper formula, thickness scaling (D011), classify_risk, per-vessel breakdown, dirty + excluded skipping, empty parse.
- `test_lesion_ccl.py` (17): BFS edge predicate, same-slice / cross-vessel / gap behaviour, dirty + excluded + too-few-points + unmatched ROIs all skipped, area-weighted centroid, deterministic output keys.
- `test_per_vessel_aggregates.py` (13): volume/mass formula, area-weighted mean HU, max-of-max, empty-vessel zeros, dirty/excluded skip, thickness scaling.
- `test_spatial.py` (16): Gini edge cases, diffusivity (D016) full edge-case table, distance metrics, z-sorted first-to-last, dist_from_top clamping, area-weighted COM.
- `test_density_tiers.py` (10): tier boundary HUs, below-threshold rejection, strict `> 1000` for dense, dirty/excluded skip.
- `test_per_artery_mask.py` (6): vessel_filter isolation, voxel-count sum, backward-compat with whole-mask, exclusion consistency.
- `test_radiomics.py`: guardrail rejects normalised / low-max / wrong-dtype inputs, geometry wrap, all 7 IBSI families present, feature count in expected range, empty mask returns empty dict, `roi_too_small` (voxel count, dimensions, reads the locked params).
- `test_stage3_script.py`: an extractor exception fails the patient while a small ROI is a skip; configured grouping reaches canonical features and the run header; failure exits 1 and keeps the published folder; `--limit` never publishes; a patient missing from stage 2 fails the stage.

## Publication-readiness audits

Run after stage 3 via `scripts/03b_cohort_sanity.py`. Outputs
`outputs/03_features/cohort_sanity.json`.

| Audit | Result on the 439-patient cohort |
|---|---|
| `row_count_matches_manifest` | OK (439 = 439) |
| `canonical_schema_present` | OK (68 schema keys, none missing) |
| `nan_audit` | OK (NaN only on PyRadiomics columns of `radiomics_status="skipped"` rows) |
| `per_vessel_totals_consistent` | OK (differences at numerical roundoff only) |
| `per_vessel_union_equals_whole` | OK (per-vessel masks OR'd together = whole-mask voxel for voxel, on every patient) |

Inter-vessel annotation overlap (a few voxels, for example in PID 116) is the LM-vs-proximal-LAD/LCx annotation overlap phenomenon documented in the dataset analysis report (§ 5.3); the per-vessel masks correctly handle it (their union equals the whole-mask), the small `per_vessel_sum > whole_mask` discrepancy is the overlap arithmetic, not a pipeline bug.

A `run_header.json` is written alongside the features CSV containing the run timestamp, git commit hash, `params.yaml` SHA, Python version, and PyRadiomics / numpy / pandas / SimpleITK / pydicom / OpenCV versions. Anyone reproducing the result has the full dependency baseline.

## Results on the 439-patient cohort (0.375 mm grid)

- 439 rows x 184 columns (68 canonical + 9 metadata + 107 PyRadiomics), 0 errors, no duplicate rows; the same 439 pids as the manifest, the stage 2 report and the CT/mask pairs.
- PyRadiomics `ok` for 437 patients; 2 `skipped` (masks of 10 and 12 voxels; PyRadiomics rejects masks at or below `minimumROISize` 14), their XML-derived features kept.
- 97 patients `low_burden_flag=True`. 3,171 lesions.
- Agatston category: 214 at 1-99, 104 at 100-399, 121 at 400+.
- Canonical features have no missing values; radiomics values in successful rows are all finite; per-vessel Agatston, volume and mass sum to the patient totals.
- The six patients with a failing ROI (126, 184, 238, 298, 347, 433) carry their stage 2 quality labels.
- Runtime about 80 s on 12 workers.

### Lesion-grouping sensitivity probe (D007 verification)

10 patients across Agatston quartiles; grouping run at `{3, 5, 8} mm × {gap 1, 2}` (6 settings).

| Region | Behaviour |
|---|---|
| 5 mm / gap=1 vs 5 mm / gap=2 | max ±11.1%: gap parameter has secondary effect |
| 5 mm vs 3 mm | up to +46% lesions (3 mm over-splits) |
| 5 mm vs 8 mm | up to −67% lesions (8 mm over-merges) |
| Low-burden patients (Agatston < 100) | no sensitivity to any threshold (lesion count = 1) |

The default (5 mm, gap=1) sits in the safe middle: conservatively over-split per D007's stated rationale, neighbours-sensitive only for high-burden patients. Accepted as the default for COCA.

## Edge cases handled

- Patients with masks at or below 14 voxels (2 in this cohort): PyRadiomics skipped, XML-derived features intact.
- Empty vessel (no ROIs after exclusion): all per-vessel scalars = 0 (D017).
- Single lesion in a vessel: `diffusivity = 1.0`, distances = 0 (D016, D017).
- Dirty vessel names: never enter any feature path (`vessel=None` upstream).
- Round-trip-failed ROIs (7 ROIs across 6 patients): never enter any feature path (D012).
- ROIs whose `Center` is unmatched in Z (no centre + ImageIndex fallback also fails): not in any lesion, not in any mask.

## Known limitations

- **Texture features on tiny masks**: even where PyRadiomics didn't refuse (mask 15–99 voxels), texture statistics are noisy. `low_burden_flag` flags these. Stage 7 reports a sensitivity analysis without the `low_burden_flag=True` patients (D028, D028.1).
- **No per-vessel PyRadiomics.** Patient-level whole-mask features are the primary unit; a per-vessel version would have to build native per-artery masks and resample them exactly as stage 2 does.
- **3 mm slice anisotropy**: inherited from preprocess (D005). Affects 3D texture features; consistent across the cohort so phenotyping is not biased, only the absolute interpretation.
- **Binary-vs-partial-volume mask bias**: inherited from preprocess (D002). PyRadiomics consumes the binary mask we built; the systematic boundary-pixel inclusion is a constant across all patients.
- **Lesion grouping is threshold-sensitive at high burden**: the sensitivity probe shows ±50% swings for high-burden patients between 3 mm and 8 mm. The 5 mm/gap=1 default is defensible per D007 but is the single largest parameter to revisit on a new cohort.
