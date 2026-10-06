# Stage 1: `io`

DICOM loading, XML parsing, patient discovery, spacing metadata. The entry point of the pipeline; everything downstream consumes its outputs.

## Modules

| Module | Purpose |
|---|---|
| `dicom_loader.py` | Load patient CT series into SimpleITK with Z-ascending slice order |
| `xml_parser.py` | Parse Apple plist calcium annotation XMLs |
| `patient_discovery.py` | Build the cohort manifest with exclusions applied |
| `spacing.py` | Persist target voxel grid metadata as JSON |

## What this stage does

1. **Discovery**: intersect `paths.data_raw` (DICOM patient folders) with `paths.data_xml` (XML annotation files), drop patients in the configured exclusion list, select each patient's series (D006), drop GE scanners, drop series that fail the slice geometry check (D039), and label each remaining patient with scanner/kernel/thickness/series UID/slice count/z step from the selected series. Produces `outputs/01_manifest/manifest.csv` and `exclusions.csv`.
2. **Per-patient load**: given a `pid`, return a `LoadedPatient` containing the CT image and the per-slice physical Z positions (the load-bearing primitive for D001 slice matching). It makes the same `select_series` call as discovery, so the same files and the same checks.
3. **Per-patient XML parse**: return a `ParseResult` with all non-empty ROIs, vessel names normalised, dirty-name set surfaced, and the `Center` field carried through verbatim.

## Why these choices

### Z-ascending slice sort, not InstanceNumber or filename
Empirically: `InstanceNumber` is not always monotonic in Z (patient 78 shows this). Filename order is scanner- and protocol-dependent. `ImagePositionPatient[2]` (IPP[2]) is the only authoritative physical Z per DICOM. Sorting by IPP[2] ascending makes `array[0]` correspond to the most-negative-Z slice, and that ordering is what we hand to `SimpleITK.ImageSeriesReader`. Downstream, the round-trip check (D002) verified this is the order the XML's `Center.z` expects.

### Multi-series rule: largest slice count (D006)
"Lowest SeriesNumber" fails on two real patients in the cohort:
- **388**: SeriesNumber 2 is a 1-slice scout; SeriesNumber 3 is the 44-slice diagnostic scan. Lowest SeriesNumber picks the scout.
- **159**: one DICOM has a missing `SeriesInstanceUID` and is isolated in a 1-file group; lowest SeriesNumber selects that group and SimpleITK cannot read the file.

The annotated scan is reliably the longest series in the COCA folder structure. D006 sorts by slice count (desc) first, then SeriesNumber, then UID. Patient 78 (two same-length series) falls through to the SeriesNumber tiebreak.

### Kernel normalisation
Siemens stores `ConvolutionKernel` as a multi-valued tag (`Qr36d\2`). pydicom hands it back as a `MultiValue` whose `str()` is the list repr `"['Qr36d', '2']"`, ugly and unstable for joins/groupbys later. `_normalise_kernel` joins it with `/` (→ `"Qr36d/2"`) so the manifest column and downstream kernel-harmonisation grouping are deterministic.

### Exclusions at discovery (D004)
Every paper-relevant exclusion is applied once, at `discover_patients`. Downstream stages read the manifest and never re-derive eligibility. The audit trail (`exclusions.csv`) records every excluded PID with reason (`no_dicom`, `no_xml`, `config_excluded`, `ge_scanner`, `irregular_geometry`) and, for geometry exclusions, the failing check in `detail`.

### Slice geometry check (D039)
SimpleITK assigns a uniform spacing of (z_last - z_first) / (n - 1) to whatever slice list it receives and does not warn. A series with missing slices is then stretched as a whole, resampling treats the false spacing as true, and lesion grouping (D007, by array slice index) links slices that are far apart. The D002 round-trip cannot see it. `check_geometry` therefore requires, on the selected series: a position on every file, axial orientation, every z step within 0.05 mm of the median, one slice thickness, and a median step within 5% of the thickness. Failing patients are excluded at discovery (33, 135, 159, 388; per-patient findings in D039) and both loaders raise `IrregularGeometryError`. Split series are not merged.

D039.1 adds one in-plane grid per series: the same `PixelSpacing` (within 1e-4 mm), the same matrix, and the same (x, y) origin (within 0.05 mm) on every file. Pid 156 passed the z-only check at discovery but its series mixes 0.38086 and 0.59570 mm pixels with an origin shift; SimpleITK measures slice spacing in 3D and assigned 3.79 mm, which the loader's spacing check caught in stage 2. The check now runs at discovery, so discovery and the loaders agree.

### One series selection for discovery and both loaders
Labelling a patient from the alphabetically first `.dcm` in the folder could describe a series the loader does not select. Discovery, `load_patient_dicom` and `load_patient_metadata` therefore all call `read_selected_headers` / `select_series`. `load_patient_metadata` reports the checked z step as its z spacing, matching the SimpleITK volume, instead of `SliceThickness`. Discovery reads every header per patient, which costs seconds for the cohort.

### Configured paths
Discovery takes `paths.data_raw` and `paths.data_xml` explicitly, and every XML read uses `paths.data_xml`. The loaders take `data_raw.parent` and append `raw/<pid>`, so `load_config` rejects a `data_raw` folder not named `raw` rather than letting discovery and the loaders read different folders.

### XML parser exposes `Center`, not just `ImageIndex`
The parser keeps `Center[2]` for Z-coordinate matching (D001), not only `ImageIndex`, plus `Point_px` (2D) and `Point_mm` (3D). The XML's pre-computed HU stats (`Mean`, `Max`, `Min`, `Total`) are carried verbatim so the round-trip checker can compare without re-reading the original CT through OsiriX's renderer.

### Dirty vessel names
Five entries in the cohort have garbage names (`"555614876"`, `"Unnamed"`, `"1"`, …). The parser sets `roi.vessel = None` for any name not in `VESSEL_NAME_MAP` and surfaces them as a set in `ParseResult.dirty_vessel_names`. Mask building skips these by default; round-trip records them with `matched_via="dirty"` and excludes them from pass-rate counts.

## Decisions
- D001: Z-coordinate matching (this stage provides the `slice_positions` it needs).
- D004: Cohort exclusions applied at discovery.
- D006: Multi-series reduction rule.
- D039: slice geometry check on the selected series; one shared selection.
- D039.1: the check also requires one in-plane grid.

## Module contracts

```python
# dicom_loader.py
@dataclass(frozen=True)
class LoadedPatient:
    pid: str
    ct_sitk: sitk.Image
    slice_positions: tuple[float, ...]   # IPP[2] per array slice, ascending
    pixel_spacing: tuple[float, float, float]
    slice_thickness: float
    n_slices: int
    scanner_model: str
    kernel: str                          # normalised, e.g. "Qr36d/2"
    manufacturer: str
    series_uid: str

class IrregularGeometryError(ValueError): ...

@dataclass(frozen=True)
class SelectedSeries:
    headers: tuple[dict, ...]            # D006-selected, z ascending
    dz: float                            # checked z step (mm)

def check_geometry(headers: list[dict], pid: str) -> float: ...       # D039
def read_selected_headers(patient_dir: Path, pid: str) -> list[dict]: ...
def select_series(patient_dir: Path, pid: str) -> SelectedSeries: ...
def load_patient_dicom(pid: str, data_root: Path) -> LoadedPatient: ...
def load_patient_metadata(pid: str, data_root: Path) -> PatientMetadata: ...
def _select_single_series(file_meta: list[dict], pid: str) -> list[dict]: ...
def _normalise_kernel(value) -> str: ...

# xml_parser.py
@dataclass(frozen=True)
class ROI:
    vessel_raw: str
    vessel: str | None                   # LAD/RCA/LCx/LM or None if dirty
    area_cm2: float
    mean_hu: float; max_hu: float; min_hu: float; total_hu: float
    n_points: int
    points_px: tuple[tuple[float, float], ...]
    points_mm: tuple[tuple[float, float, float], ...]
    center_xyz: tuple[float, float, float] | None

@dataclass(frozen=True)
class SliceAnnotation:
    image_index: int
    rois: tuple[ROI, ...]

@dataclass(frozen=True)
class ParseResult:
    pid: str
    slices: tuple[SliceAnnotation, ...]
    dirty_vessel_names: tuple[str, ...]
    n_active_rois: int
    n_dropped_zero_point: int

def parse_calcium_xml(pid: str, xml_dir: Path) -> ParseResult: ...

# patient_discovery.py
@dataclass(frozen=True)
class PatientRecord:
    pid: str; raw_path: Path; xml_path: Path
    manufacturer: str; scanner_model: str; kernel: str
    slice_thickness: float
    series_uid: str; n_slices: int; z_step_mm: float

@dataclass(frozen=True)
class DiscoveryResult:
    included: tuple[PatientRecord, ...]
    excluded_no_dicom: tuple[str, ...]
    excluded_no_xml: tuple[str, ...]
    excluded_by_config: tuple[str, ...]
    excluded_ge: tuple[str, ...]
    excluded_geometry: tuple[tuple[str, str], ...]   # (pid, reason), D039

def discover_patients(
    raw_dir: Path, xml_dir: Path, *,
    exclude_pids: tuple[str, ...] = (),
    exclude_ge_scanners: bool = True,
) -> DiscoveryResult: ...
def manifest_rows(result: DiscoveryResult) -> list[dict]: ...

# spacing.py
def parse_spacing(value: str) -> tuple[float, float, float]: ...
def save_spacing_metadata(path: Path, target_spacing: tuple[float, float, float]) -> None: ...
def load_spacing_metadata(path: Path) -> tuple[float, float, float]: ...
```

## Tests (`tests/io/`)
- `test_config.py`: config loader, default exclusions, vessel map round-trip, `data_raw` must be named `raw`.
- `test_dicom_loader.py`: `_select_single_series` rules (scout/scan, equal counts, UID tiebreak, modality filter), kernel normalisation, multi-series warning logging, `check_geometry` (D039: missing slice, duplicate z, overlap, 2.95/3.0 accepted, missing position, tilt, single slice, mixed thickness), integration on patients 1 and 78.
- `test_xml_parser.py`: vessel normalisation, dirty-name detection, zero-point ROI filtering, `Center` parsing (valid / missing / malformed), HU stat preservation.
- `test_patient_discovery.py`: intersection logic, exclusion application, GE filter, manifest metadata, D039 exclusions, configured XML folder, both loaders refuse irregular series, both loaders agree on z spacing.
- `test_spacing.py`: parse/save/load round-trip, validation errors.

## Results on the COCA cohort

```
439 included | irregular_geometry 5 (33, 135, 156, 159, 388) | ge_scanner 4 | no_dicom 2 | config_excluded 1 | no_xml 0
Kernel breakdown:
  Qr36d/2    232 (SOMATOM Force)
  I30f/3     206 (SOMATOM Definition Flash)
  I36f/3       1
```

The 2 no-DICOM patients (12, 197) match the dataset report. The config exclusion (268) handles the replaced-series patient. The 4 GE patients are held out from the primary analysis for external validation (D029). D039 and D039.1 exclude 5 patients for slice geometry.

## Edge cases handled
- Patient 78 (multi-series, same Z): D006 SeriesNumber tiebreak → 34 slices, correct series.
- Patient 388: D006 picks the 44-slice series. The 1-slice "scout" is in fact pid 135's missing slice (same series UID and z). Excluded by D039 (its own slice at z -174.5 is missing).
- Patient 159: the scan is split over two series, with 11 slices missing from the selected one, and one file has no position. Excluded by D039.
- Patient 135: one slice was exported as a separate series. Excluded by D039.
- Patient 33: overlapping reconstruction (1.5 mm steps, 3.0 mm slices). Excluded by D039.
- Patients with non-CT modality entries: `_select_single_series` filters them out, warns if no CT remains.
- Dirty vessel names (5 entries across 4 patients): `vessel=None`, surfaced in `dirty_vessel_names`.

## Known limitations
- A series with duplicate z, gaps or overlap is excluded (D039), never repaired. Pid 159's UID-less file has no position at all, and its annotated ROIs sit on slices missing from the selected series.
- pydicom emits noisy "Invalid value for VR UI" warnings for many COCA files (non-standard UID format). These are cosmetic and do not affect loading.
