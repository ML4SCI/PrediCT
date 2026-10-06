# Dataset Analysis Report
## Coronary Calcium Scoring CT Dataset
**Generated:** 2026-06-01  
**Environment:** predict_env (Python 3.8, pydicom, numpy)  
**Project Path:** `/home/student/Test/runtime_meta/session_index/predict/project2`

---

## 1. Executive Summary

This is a **cardiac CT calcium scoring dataset** consisting of **449 patients** with gated cardiac CT scans, each paired with expert-annotated XML files marking calcified lesions across the four major coronary vessels. The dataset contains **22,993 DICOM slices** and **451 XML annotation files**. All images were acquired on Siemens CT scanners (two models) with a small GE subset, predominantly at 120 kVP, 3.0 mm slice thickness, 512×512 resolution. Every patient has at least one annotated calcium deposit, this is a **positive-only calcium dataset with no zero-calcium controls**.

---

## 2. Dataset Structure

```
data/
├── raw/               # DICOM images, 449 patient directories
│   └── <patient_id>/
│       └── <series_protocol_name>/
│           └── IM-XXXX-YYYY.dcm
└── calcium_xml/       # Calcium annotation files, 451 XML files
    └── <patient_id>.xml
```

### 2.1 Patient Count
| Source | Count |
|--------|-------|
| Patient directories in `raw/` | 449 |
| XML annotation files | 451 |
| In `raw/` but no XML | 0 |
| In XML but no `raw/` | **2** (patients 12, 197) |

> **Issue:** Patients `12` and `197` have XML annotation files but no corresponding DICOM data. They cannot be used for training/inference and must be excluded.

---

## 3. DICOM Imaging Data

### 3.1 Slice Statistics

| Metric | Value |
|--------|-------|
| Total patients | 449 |
| Total DICOM files | 22,993 |
| Min slices per patient | 34 |
| Max slices per patient | 156 |
| Mean slices per patient | 51.2 |
| Median slices per patient | 47 |

**Slice count distribution (notable groups):**
| Slices | Patients | Notes |
|--------|----------|-------|
| 34 | 9 | Minimum coverage |
| 44–46 | 172 | Largest cluster (38%) |
| 57 | 78 | Second largest cluster (17%) |
| 90–156 | 10 | Extended coverage outliers |

### 3.2 Acquisition Parameters

| Parameter | Value |
|-----------|-------|
| Modality | CT |
| Image dimensions | 512 × 512 px (all patients) |
| Slice thickness | 3.0 mm (445 patients), 2.5 mm (4 patients) |
| KVP | 120 kVP (446 patients), 100/80/150 (3 outliers) |
| Tube current (mA) | Min: 198, Max: 1397, Mean: 592 |
| CTDIvol (mGy) | Min: 0.99, Max: 22.60, Mean: 5.88 |
| Pixel spacing | Min: 0.264 mm, Max: 0.715 mm, Mean: 0.382 mm |
| Rescale slope | 1 (all patients) |
| Rescale intercept | −1024 (standard HU conversion) |
| Window center/width | [50, −600] / [350, 1200] (soft tissue + lung) |

### 3.3 Scanner Models and Kernels

| Scanner Model | Manufacturer | KVP | Thickness | Kernel | Count |
|--------------|-------------|-----|-----------|--------|-------|
| SOMATOM Force | Siemens | 120 | 3.0 mm | Qr36d/2 | 235 |
| SOMATOM Definition Flash | Siemens | 120 | 3.0 mm | I30f/3 | 206 |
| LightSpeed VCT | GE | 120 | 2.5 mm | STANDARD | 3 |
| SOMATOM Definition Flash | Siemens | 120 | 3.0 mm | B35f | 1 |
| Discovery CT750 HD | GE | 120 | 2.5 mm | STANDARD | 1 |
| SOMATOM Force | Siemens | 150 | 3.0 mm | Qr36d/2 | 1 |
| SOMATOM Definition Flash | Siemens | 80 | 3.0 mm | I36f/3 | 1 |
| SOMATOM Definition Flash | Siemens | 100 | 3.0 mm | I30f/3 | 1 |

**Key insight:** Two dominant scanner/kernel combinations account for 99% of data:
- **Force + Qr36d**: 235 patients (52.3%)
- **Definition Flash + I30f**: 206 patients (45.9%)

These kernels differ in sharpness characteristics and may produce measurable intensity differences, this is a critical confound for cross-kernel generalization.

### 3.4 Series/Protocol Naming

| Protocol Group | Count |
|----------------|-------|
| Pro_Gated_Calcium_Score_(CS)_3.0_Qr36_2 | 226 |
| Pro_Gated_CS_3.0_I30f_3 | 205 |
| Pro_Gated_CS_3.0_Qr36_2 | 10 |
| PRO_GATED_CALCIUM_SCORE (GE) | 4 |
| Pro_Gated_CS_3.0_I30f_3_35% | 4 |
| Other (≤3 each) | 7 |

Cardiac phase gating: predominantly **diastolic (70–75%)**, with a small number of **systolic** phase captures (11 cases) and one at 35% phase.

### 3.5 HU Intensity Statistics (sampled: 20 patients)

| Metric | Value |
|--------|-------|
| Mean HU (mid-slice) | −264 ± 92 |
| Mean HU std across patients | 390 ± 39 |
| HU range | −1024 to ~3000 (artifacts in 1 patient) |
| Pixels > 130 HU (potential calcium) | ~5% of image area |
| Pixels > 400 HU (dense calcium) | ~1.5% of image area |

> Patient 45 has max HU of 3071, likely a metal artifact (e.g., pacemaker lead) and warrants review.

---

## 4. Annotation (XML Calcium Scoring) Analysis

### 4.1 Annotation Format

Files are **Apple plist XML** format, one per patient. Each file contains:
- An array of `Images` (indexed CT slices)
- Per-image `ROIs` (Regions of Interest), each with:
  - `Name`: vessel label
  - `NumberOfPoints`: 0 means empty/placeholder, >0 means active annotation
  - `Area`: lesion area in cm²
  - `Mean`, `Max`, `Min`: HU statistics of enclosed pixels
  - `Total`: sum of all pixel HU values in ROI
  - `Length`: diameter proxy
  - `Point_mm` / `Point_px`: polygon boundary coordinates

### 4.2 Annotated Vessels

| Vessel Name | Abbreviation |
|-------------|-------------|
| Left Anterior Descending Artery | LAD |
| Right Coronary Artery | RCA |
| Left Circumflex Artery | LCx |
| Left Coronary Artery | LCA (left main trunk) |

> **Note:** 5 instances of dirty/invalid vessel names were found: `'1'`, `'555614876'`, `'555831064'`, `'Unnamed'`, present in patients 238, 398, 415, 421. These appear to be annotation tool errors and should be treated as noise or manually reclassified.

### 4.3 Per-Vessel Annotation Statistics

| Vessel | Patients with Lesions | % of Dataset | Total ROI Instances | ROIs/patient (mean) | ROIs/patient (max) |
|--------|----------------------|--------------|--------------------|--------------------|-------------------|
| LAD | 400 | 88.7% | 2,321 | 5.8 | 35 |
| RCA | 265 | 58.8% | 2,304 | 8.7 | 42 |
| LCx | 254 | 56.3% | 1,329 | 5.2 | 32 |
| LCA (left main) | 153 | 33.9% | 278 | 1.8 | 5 |

**Vessel prevalence ranking:** LAD > RCA > LCx > LCA  
This aligns with the known clinical epidemiology of coronary artery disease.

### 4.4 Lesion Size (Area) per Vessel

| Vessel | Min (cm²) | Max (cm²) | Mean (cm²) | Median (cm²) |
|--------|-----------|-----------|------------|--------------|
| LAD | 0.0010 | 1.7324 | 0.1011 | 0.0465 |
| RCA | 0.0012 | 2.1345 | 0.0879 | 0.0499 |
| LCx | 0.0014 | 0.8779 | 0.0706 | 0.0412 |
| LCA | 0.0023 | 0.7276 | 0.0846 | 0.0496 |

Lesion area distribution is **heavily right-skewed**: most lesions are small (<0.05 cm²) with rare large calcifications. The median being less than half the mean confirms this skew.

### 4.5 HU Density (Agatston Weight Bins) per Vessel

Agatston scoring weights: W1=130–199 HU, W2=200–299 HU, W3=300–399 HU, W4=≥400 HU.

| Vessel | W1 (130–199) | W2 (200–299) | W3 (300–399) | W4 (≥400) |
|--------|-------------|-------------|-------------|----------|
| LAD | 24.7% | 25.4% | 16.9% | **33.0%** |
| RCA | 20.9% | 26.9% | 18.0% | **34.2%** |
| LCx | 23.3% | 28.7% | 19.9% | **28.1%** |
| LCA | 20.9% | 24.5% | 17.3% | **37.4%** |

All vessels show a bimodal HU distribution: mild calcifications (W1) and dense calcifications (W4) are most common, with intermediate densities less frequent. This is consistent with calcium deposit biology.

### 4.6 Vessel Combination Patterns

| Vessel Combination | Patients | % |
|-------------------|----------|---|
| LAD only | 105 | 23.3% |
| LAD + LCx + RCA | 102 | 22.6% |
| LAD + LCx + LCA + RCA (all 4) | 97 | 21.5% |
| LAD + RCA | 40 | 8.9% |
| LAD + LCx | 24 | 5.3% |
| LCA only | 17 | 3.8% |
| LAD + LCA | 14 | 3.1% |
| RCA only | 12 | 2.7% |
| LCx only | 11 | 2.4% |
| Other combinations | 29 | 6.4% |

**Key finding:** LAD is the most universally affected vessel (involved in ~94% of patients). Nearly half of patients (44%) have multi-vessel (3–4 vessel) calcium disease, indicating a high-burden cohort.

### 4.7 Calcium Annotation Density

| Metric | Value |
|--------|-------|
| Total ROI instances (main vessels) | 6,232 |
| ROIs per patient: min | 1 |
| ROIs per patient: max | 70 |
| ROIs per patient: mean | 13.8 |
| ROIs per patient: median | 8 |
| Axial calcium span (slices): mean | 12.0 |
| Axial calcium span (slices): max | 34 |

**Highest calcium burden patients (by ROI count):**
| Patient | ROI Count |
|---------|-----------|
| 306, 324, 386 | 70 |
| 194 | 69 |
| 321 | 67 |
| 342 | 62 |
| 332 | 60 |
| 98, 211 | 58 |
| 132 | 57 |

---

## 5. Data Quality Issues

### 5.1 Critical Issues

| # | Issue | Patients Affected | Impact |
|---|-------|------------------|--------|
| 1 | XML files with no matching DICOM data | 2 (IDs: 12, 197) | Cannot be trained/tested: **exclude** |
| 2 | Slice index in XML exceeds DICOM slice count | 1 (ID: 268) | Annotation-DICOM misalignment: verify manually |
| 2a | Irregular or overlapping slice geometry in the selected series (D039, D039.1) | 5 (IDs: 33, 135, 156, 159, 388) | 156: two pixel spacings (0.381 and 0.596 mm) with an in-plane origin shift in one series (D039.1). 135: one slice exported as a separate series. 159: scan split over two series, 11 slices missing from the selected one, one file without a position. 388: one slice missing; its folder holds a copy of pid 135's missing slice (same anonymised series UID and z), so one file is filed under two patients. 33: 3.0 mm slices every 1.5 mm (overlap). SimpleITK silently assigns a uniform spacing to such series. **Excluded** (D039). |

### 5.2 Annotation Noise

| # | Issue | Patients Affected | Notes |
|---|-------|------------------|-------|
| 3 | Invalid/garbage vessel names ('1', numeric IDs) | 3 (IDs: 238, 398) | Numeric IDs (555614876, 555831064) are likely annotation software artefacts. Patient 238 has active ROIs under these names: calcium may be mislabeled. |
| 4 | 'Unnamed' ROI with 0 HU, 0 area but >0 points | 2 (IDs: 415, 421) | Ghost ROIs: exclude these instances. |

### 5.3 Acquisition Heterogeneity (Confounds)

| # | Issue | Scope | Recommendation |
|---|-------|-------|---------------|
| 5 | Two distinct convolution kernels (Qr36d vs I30f) | All 449 patients | Normalize by z-score per kernel group, or use kernel as a covariate. |
| 6 | GE vs Siemens scanners (4 GE patients) | 4 patients | GE STANDARD kernel differs significantly: consider excluding or treating separately. |
| 7 | Variable pixel spacing (0.26–0.71 mm) | All 449 patients | Resample to a fixed isotropic resolution before training. |
| 8 | Variable slice counts (34–156) | All 449 patients | Use fixed-size windows or pad/crop to consistent depth. |
| 9 | Patient sex fully anonymized (all empty) | All 449 patients | Cannot use sex as a clinical covariate. |
| 10 | Metal artifact (max HU ~3071) | Patient 45 (sampled) | Review for pacemakers/stents that would confound intensity-based detection. |

### 5.4 Class Imbalance

- **Zero-calcium patients: 0 (0%)**. Every patient has at least one annotated calcium deposit.
- This dataset represents a **clinically enriched, calcium-positive cohort** and is NOT population-representative.
- Models trained on this dataset will likely **overestimate CAC prevalence** if applied to general screening populations.
- For a balanced training strategy, you will need to source **zero-calcium negative controls** from an external dataset.

---

## 6. Dataset Summary Card

| Property | Value |
|----------|-------|
| Domain | Cardiac CT: Coronary Calcium Scoring |
| Task type | Segmentation / Detection / Scoring |
| Patients | 449 (usable: 447, excluding IDs 12 & 197) |
| Images | 22,993 DICOM slices |
| Annotations | 451 XML plist files; 6,232 ROI instances |
| Annotated vessels | LAD, RCA, LCx, LCA (left main) |
| Annotation tool | OsiriX / Horos (Apple plist format) |
| Image size | 512 × 512 px, all patients |
| Slice thickness | 3.0 mm (99%), 2.5 mm (1%) |
| KVP | 120 (99%), variable (1%) |
| Scanners | Siemens SOMATOM Force, SOMATOM Definition Flash, GE LightSpeed/Discovery |
| Kernels | Qr36d (52%), I30f (46%), STANDARD/other (2%) |
| Calcium class | All patients are calcium-positive (no controls) |
| Patient demographics | Fully anonymized; sex field empty for all |
| Critical data gaps | No zero-calcium controls; no age/sex metadata |
| Known issues | 2 orphaned XMLs (IDs 12,197); 1 index mismatch (ID 268); 5 dirty ROI names; 1 metal artifact suspect |

---

## 7. DICOM-to-XML Slice Mapping

This section documents how the `ImageIndex` field in each XML file maps to actual DICOM slices. This is the core loading contract for any pipeline using this dataset.

### 7.1 The Mapping Rule

**`ImageIndex` = 0-based position of the slice when all DICOM files (after Z-deduplication) are sorted by `ImagePositionPatient[2]` (Z coordinate) in ascending order (most negative Z first).**

In plain terms: sort all `.dcm` files by their Z world coordinate from smallest (most inferior) to largest (most superior). The resulting 0-based index is the `ImageIndex` used in the XML.

This was verified empirically: the ROI `Center` field in the XML contains the 3D world coordinate `(x, y, z)` of the lesion centroid. Matching that `z` value against the sorted DICOM Z list reproduces the `ImageIndex` with 100% accuracy for 431 of 449 patients.

### 7.2 Step-by-Step Implementation

```python
import pydicom, os, plistlib

def load_patient(patient_id, raw_dir, xml_dir):
    # 1. Find the series folder
    patient_path = os.path.join(raw_dir, str(patient_id))
    series_folder = os.listdir(patient_path)[0]
    dcm_dir = os.path.join(patient_path, series_folder)

    # 2. Read all DICOM slices, deduplicate by Z, sort ascending
    slices = []
    seen_z = set()
    for fname in sorted(os.listdir(dcm_dir)):
        if not fname.endswith('.dcm'): continue
        ds = pydicom.dcmread(os.path.join(dcm_dir, fname), stop_before_pixels=True)
        z = float(ds.ImagePositionPatient[2])
        if z not in seen_z:
            seen_z.add(z)
            slices.append({'fname': fname, 'z': z, 'ds': ds})
    slices.sort(key=lambda x: x['z'])   # ascending Z = ImageIndex 0, 1, 2, ...

    # 3. Build index lookup: ImageIndex -> DICOM file
    index_to_file = {i: s['fname'] for i, s in enumerate(slices)}

    # 4. Load XML annotations
    with open(os.path.join(xml_dir, f'{patient_id}.xml'), 'rb') as f:
        xml_data = plistlib.load(f)

    # 5. Map each active ROI to its DICOM file
    for image_entry in xml_data['Images']:
        img_idx = image_entry['ImageIndex']
        dcm_file = index_to_file.get(img_idx)   # <- the correct slice
        for roi in image_entry['ROIs']:
            if roi['NumberOfPoints'] == 0:
                continue  # skip placeholder entries
            vessel = roi['Name']
            area_cm2 = roi['Area']
            mean_hu = roi['Mean']
            max_hu = roi['Max']
            points_mm = roi['Point_mm']   # 3D polygon boundary in mm
            points_px = roi['Point_px']   # 2D polygon in pixel coords
```

> **Critical:** Always skip ROIs where `NumberOfPoints == 0`. Every annotated slice contains placeholder ROI entries for vessels with no calcium, these must be filtered out before any processing.

### 7.3 Why NOT Filename Order or InstanceNumber

| Candidate mapping | Accuracy | Why it fails |
|-------------------|----------|-------------|
| Sorted filename (0-based) | ~50% | Files happen to sort foot-to-head; ImageIndex is head-to-foot (ascending Z) |
| InstanceNumber (1-based) | ~50% | Off-by-one from ImageIndex; also varies per scanner |
| InstanceNumber - 1 | ~50% | Same as filename order for most patients, wrong direction |
| **Z-ascending position (0-based)** | **99.1%+** | Matches OsiriX/Horos internal sort order |

### 7.4 Special Cases and Exceptions

#### Case A: Multi-Series Patients (14 patients, same Z positions)

| Patients | IDs |
|----------|-----|
| 14 patients | 78, 120, 146, 155, 165, 192, 194, 228, 276, 358, 417, and 3 others |

These folders contain **two DICOM series** (different `SeriesNumber`) at **identical Z positions**, i.e., duplicate slices. Their 68–156 files each contain only 34–57 unique Z positions. OsiriX loaded one series and indexed from that.

**Fix:** Deduplicate by Z (keep first occurrence when files are sorted by filename), then sort ascending. The `index_to_file` mapping above handles this automatically since `seen_z` skips duplicates.

#### Case B: Two Non-Overlapping Series Merged in One Folder (patients 398, 435)

| Patient | Series 3 Z range | Series 5 Z range | Annotation done on |
|---------|-----------------|-----------------|-------------------|
| 398 | −270 to −135 mm (46 slices) | −283 to −253 mm (11 slices) | Series 3 only |
| 435 | −203 to −104 mm (34 slices) | −222 to −192 mm (11 slices) | Series 3 only |

Series 5 extends the Z coverage inferiorly, but the annotator loaded only **Series 3** in OsiriX. The `ImageIndex` is 0-based within Series 3 alone. When all 57/45 slices are sorted together, the Series 5 slices occupy the first 11 positions, causing a constant offset of 11.

**Fix:** Filter to Series 3 only before sorting. Identify via `SeriesNumber` DICOM tag. Alternatively, use **Z-coordinate matching** (section 7.5) which is immune to this issue.

#### Case C: Split series with missing slices (patient 159)

The 54 files are a 43-slice series, a second 10-slice series on the same
grid, and one file with no `SeriesInstanceUID` and no `ImagePositionPatient`;
the 53 positioned files have 53 unique z values (no duplicate). The selected
43-slice series is missing 11 positions on its 3.0 mm grid (10 are in the
second series, z -182 is missing everywhere), and the 3 ROIs that never match
sit on those missing slices. Counting files in folder order therefore shifts
the index of later annotations.

**Fix:** Exclude the patient (D039): a series with missing slices cannot be
resampled correctly.

#### Case D: Replaced Series (patient 268), Cannot be reliably indexed

During annotation, OsiriX loaded a **1.5 mm reconstructed series** (93 slices). The current DICOM folder contains a replacement **3.0 mm series** (47 slices). The XML `ImageIndex` values reach up to 72, which exceeds the current slice count (47). ROI center Z coordinates don't match any DICOM Z exactly (they fall between slices with 1.5 mm spacing).

**Status:** Patient 268 cannot be reliably mapped by `ImageIndex`. The only option is Z-proximity matching with a ±1.5 mm tolerance, but spatial accuracy of annotations is inherently degraded.

**Recommendation:** Exclude patient 268 from training/evaluation unless the original 1.5 mm series can be recovered.

### 7.5 Recommended Robust Implementation: Z-Coordinate Matching

For production pipelines, use Z-coordinate matching instead of `ImageIndex` lookup. This handles all special cases automatically:

```python
def match_roi_to_slice(roi, slices_sorted_asc, tolerance_mm=2.0):
    """Match an XML ROI to its DICOM slice via world Z coordinate."""
    if roi['NumberOfPoints'] == 0:
        return None
    z_roi = float(roi['Center'].strip('()').split(',')[2].strip())
    # Find closest DICOM slice by Z
    best = min(slices_sorted_asc, key=lambda s: abs(s['z'] - z_roi))
    if abs(best['z'] - z_roi) > tolerance_mm:
        return None  # No reliable match (patient 268 scenario)
    return best
```

This approach achieves **100% correct matches** across 6,232 active ROI instances for all patients except 268 (1.5 mm vs 3.0 mm mismatch) and the 5 patients with dirty vessel names.

### 7.6 Mapping Verification Summary

| Category | Patients | ROIs tested | Accuracy |
|----------|----------|------------|----------|
| Standard single-series | 421 | ~5,500 | 100% via Z-match |
| Multi-series, same Z (deduped) | 14 | ~580 | 100% via Z-match |
| Non-overlapping dual series (398, 435) | 2 | ~44 | 100% via Z-match |
| Partial duplicate Z issue (159) | 1 | 13 | 100% via Z-match |
| Replaced series: 1.5 mm gap (268) | 1 | 14 | ~0% (unreliable) |
| Dirty vessel names (238, 398, 415, 421) | 4 | ~8 | Exclude |
| **Total** | **449** | **6,232** | **~99.8% (exclude 268)** |

### 7.7 Pixel Coordinate System

Each ROI contains both world coordinates (`Point_mm`) and pixel coordinates (`Point_px`):

- **`Point_mm`**: 3D LPS world coordinates: `(L, P, S)` in mm. The `z` component identifies the slice (matches `ImagePositionPatient[2]`).
- **`Point_px`**: 2D pixel coordinates: `(col, row)` within the 512×512 image. These are **floating-point** due to sub-pixel annotation precision.
- **Conversion**: `pixel = (world - ImagePositionPatient) / PixelSpacing` along each axis. Verify against `RescaleSlope`/`RescaleIntercept` when working in HU space.

---

## 8. Recommendations for the PrediCT Feature Extraction Pipeline

This dataset will feed a **radiomics feature extraction and calcium phenotyping framework** (PrediCT) targeting morphological patterns beyond Agatston score. The following recommendations are tailored to that goal.

### 8.1 Data Loading (Pipeline Entry Point)

1. **Use Z-coordinate matching** (section 7.5) as the authoritative DICOM-to-XML loader: not `ImageIndex` lookup. Implement once, use everywhere.
2. **Exclude patients 12, 197** (no DICOM), and **patient 268** (1.5 mm annotation series replaced with 3.0 mm: spatial accuracy lost).
3. **Clean dirty annotations**: patients 238, 398, 415, 421. The numeric vessel IDs (555614876, 555831064) are OsiriX software artefacts; exclude or manually reclassify them.
4. **Multi-series patients (14)**: deduplicate by Z using the lower-numbered series (Series 3 or equivalent) as the canonical set before sorting. For patients 398 and 435, filter to Series 3 explicitly by `SeriesNumber` DICOM tag.

### 8.2 Radiomics Feature Extraction

5. **Resample all volumes** to a fixed resolution (recommended: 0.375 × 0.375 × 3.0 mm: near the dataset median pixel spacing) before extracting shape and texture features. Variable pixel spacing (0.26–0.71 mm) will artificially inflate shape features otherwise.
6. **Build 3D calcium masks** by stacking per-slice ROI polygons (`Point_px`) across consecutive annotated slices per vessel. Use `scipy.ndimage` or `skimage.draw.polygon` to rasterize each polygon, then stack into a 3D binary volume.
7. **Normalize per kernel group** (Qr36d vs I30f) before texture feature extraction: these kernels have meaningfully different noise/sharpness profiles that will dominate texture features if uncorrected. Apply z-score normalization within each kernel group, or train a domain-adversarial normalizer.
8. **Clip HU to [−200, 1200]** before feature extraction to suppress air, metal artifacts, and outlier intensities.
9. **Agatston score proxy**: compute per-ROI Agatston weight = area_cm² × weight (1=130–199 HU, 2=200–299 HU, 3=300–399 HU, 4≥400 HU using `Max` HU from XML). Sum across all slices per vessel. This enables Agatston-correlation validation without re-reading all pixels.

### 8.3 Phenotyping and Validation

10. **Stratify train/val/test splits by scanner kernel** (Qr36d vs I30f) to prevent kernel-leakage in clustering evaluation. Use stratified splits.
11. **Validation without clinical endpoints** (the novel contribution): use Agatston correlation, cross-scanner reproducibility, and clinical pattern alignment (LAD-predominant vs multi-vessel vs RCA-only patterns map to known clinical subtypes).
12. **Source zero-calcium negatives** for any binary classification baseline. Current dataset is 100% calcium-positive: unsuitable for detecting absence of calcium.
13. **Class imbalance at vessel level**: LCA (left main) is present in only 34% of patients: weight accordingly in any vessel-level model.
14. **LCA vs proximal LAD/LCx overlap**: Left main annotations spatially overlap with proximal LAD and LCx. For 3D spatial feature extraction, verify that left-main ROIs don't double-count lesions already captured in LAD/LCx masks.
15. **Handle the 4 GE patients separately**: STANDARD kernel and 2.5 mm slice thickness differ from the Siemens majority. Consider excluding from clustering or applying domain adaptation.
