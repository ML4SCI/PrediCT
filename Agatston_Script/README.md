# Agatston Coronary Artery Calcium (CAC) Calculator

A lightweight Python utility to calculate standard Agatston CAC scores from 3D NIfTI CT scans and binary calcium segmentation masks.

## Features

- **Standard Agatston Scaling**: Applies 130+ HU density weighting (130-199 -> 1, 200-299 -> 2, 300-399 -> 3, >= 400 -> 4).
- **Slice Thickness Adjustment**: Automatically normalizes slice thickness against the standard 3.0 mm reference (sz / 3.0).
- **Noise Filtering**: Ignores connected components under 1.0 mm² (partial volume/noise).
- **Spatial Alignment**: Automatically re-orients CT and mask volumes to RAS orientation before slice-by-slice processing.

## File Structure

├── agatston_script.py      # Core script and CLI calculator (includes usage instructions)   
└── mask_gen_problem.png    # Reference diagram highlighting common calcium mask generation issues

## Requirements

- Python 3.8+
- numpy
- SimpleITK
- scipy

## Usage

### 1. Command Line Interface

Run directly on your pair of NIfTI files:

```text
python agatston_script.py --ct path/to/scan_img.nii.gz --mask path/to/scan_calcium_mask.nii.gz