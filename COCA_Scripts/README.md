# COCA Dataset Processing Pipeline

Scripts for processing the COCA Dataset into training-ready CT volumes, binary masks, multi-class artery masks, and Agatston scores.

## Updates
#### Last Updated on: 10/06/2026

1. Agatston scores are now calculated directly from Area and HU information stored in XML files.
2. Added multi-class artery masks for future multi-class segmentation tasks.
3. Fixed multi-series DICOM handling. Patients 135 and 763 contained multiple series; the processor now detects all series and selects the one with the largest number of Z-slices.
4. Added utility scripts and dataset documentation for exploration and debugging.
5. Minor pipeline improvements and usage instructions.


## File Structure

```text
COCA_Setup_Files/
│
├── COCA_pipeline.py
├── COCA_processor.py
├── COCA_resampler.py
├── README.md
│
└── Utilities/
    ├── COCA_Sample_Dicom_Metadata.md
    ├── COCA_Voxel_Spacing.md
    ├── UnLabelled_Arteries.md
    ├── get_image_sizes.py
    └── visualizer_3D.py
```

## Files

* **COCA_pipeline.py** : Runs both Processor and Resampler.
* **COCA_processor.py** : Combines DICOM slices into 3D volumes, generates binary and multi-class artery masks, and calculates Agatston scores from XML annotations.
* **COCA_resampler.py** : Resamples all scans to a uniform voxel spacing.
* **README.md** : Setup instructions.
* **Documentation.md** : Development, debugging, logging, and usage notes.

### Utilities

* **COCA_Sample_Dicom_Metadata.md** : Sample DICOM metadata.
* **COCA_Voxel_Spacing.md** : Voxel spacing information for all patients.
* **UnLabelled_Arteries.md** : Files containing unlabeled arteries.
* **visualizer_3D.py** : GPU-accelerated 3D visualizer.
* **get_image_sizes.py** : Extract volume dimensions.

## Usage

### 1. Download COCA Dataset

[https://stanfordaimi.azurewebsites.net/datasets/e8ca74dc-8dd4-4340-815a-60b41f6cb2aa](https://stanfordaimi.azurewebsites.net/datasets/e8ca74dc-8dd4-4340-815a-60b41f6cb2aa)

### 2. Setup

Place:

```text
COCA_pipeline.py
COCA_processor.py
COCA_resampler.py
```

in the same directory that contains the COCA dataset folder (see `setup_image.png`).

### 3. Configure Output Directory

Inside `COCA_pipeline.py`, specify the location where the processed dataset will be stored.

### 4. Run

```bash
python COCA_pipeline.py
```

Outputs will be saved in the `data_canonical` directory.

## Known Issues

1. Patient 263 currently has issue with binary mask generation and requires further investigation.
2. Resampling method has to be taken care of Agatston scores before and after resampling should ideally remain identical, but interpolation effects may alter measurements.
