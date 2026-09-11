# RERO: Coronary Artery Calcium (CAC) Segmentation & Scoring

An end-to-end deep learning pipeline for coronary artery calcium (CAC) lesion segmentation and Agatston scoring from cardiac CT scans. The pipeline covers dataset preparation, heart ROI masking, dual-window HU intensity handling, model training (custom RERO architecture and SwinUNETR), evaluation with clinically relevant metrics, and comparison against nnU-Net baselines.

---

## Pipeline Overview

1. **Preprocessing (`pre_process.py`)** — Generates and caches Heart ROI masks for faster inference. These masks are used both for ROI-based masking in the pipeline and, optionally, as an additional input channel. The dataset is also split according to a configurable strategy — **Distribution-Optimally-Balanced Stratified Cross-Validation (DOB-SCV)** or **Simple Stratified Cross-Validation** — producing a MONAI-compatible split dictionary saved to `MetaData/splits.json`.
2. **Dataset construction (`dataset.py`)** — Reads `MetaData/splits.json` and builds MONAI datasets and dataloaders. Includes:
   - Coordinate convolution (CoordConv) channels
   - Heart ROI mask channel
   - Dual HU windowing (separate windows for calcium and soft tissue)
   - A strong augmentation suite: `RandCropByPosNegLabeld` (pos=3, neg=1), `RandAffined`, `Rand3DElasticd`, `RandGaussianSmoothd`, `RandGaussianNoised`, `RandAdjustContrastd`, `RandShiftIntensityd`, and others
3. **Training (`train.py`)** — Trains the RERO model (or SwinUNETR variant) using the configured splits and augmentations.
4. **Evaluation (`eval.py`, `resampled_eval.py`)** — Loads a trained checkpoint by experiment name and computes segmentation and scoring metrics via `metric.py`. `resampled_eval.py` additionally evaluates robustness to resampling (resampling invariance).
5. **Inference (`wrapper.py`)** — Takes a NIfTI file as input and returns model predictions end-to-end.

All configuration — dataloader, dataset, preprocessing, model, training, evaluation, and dual-HU-windowing parameters — is centralized in **`config.py`**.

---
 
## Repository Structure
 
```
.
├── config.py                         # Central configuration: dataloader, dataset, preprocess,
│                                     #   model, train, eval, and dual-HU-windowing settings
├── pre_process.py                    # Generates/caches ROI masks; produces MetaData/splits.json
├── dataset.py                        # MONAI dataset/dataloader construction + augmentations
├── train.py                          # Model training entry point
├── eval.py                           # Evaluation entry point (loads checkpoint by experiment name)
├── resampled_eval.py                 # Evaluation on resampled data (resampling-invariance check)
├── metric.py                         # All evaluation metrics (see below)
├── model.py                          # RERO model combining all rero_modules
├── model_with_swin.py                # SwinUNETR variant of the model
├── wrapper.py                        # Inference wrapper: NIfTI in -> predictions out
├── visualizations.py                 # Code to generate qualitative results
├── agatston_script.py                # Computes Agatston scores for CT scans
│                                     #   (source: Agatston_Script/agatston_script.py)
├── cited_papers.md                   # References and cited articles
├── requirements.txt                  # Dependencies for local execution
├── requirements_for_remote.txt       # Dependencies for remote execution (requires PyTorch pre-installed)
│
├── rero_modules/                     # Model submodules used by model.py
├── LW_UNET_TVERSKY/                  # Lightweight U-Net (Tversky loss) used to generate Heart ROI masks
│                                     #   consumed by pre_process.py
├── runs/<experiment_name>/           # Logs, metrics, checkpoints, and outputs per experiment
│
├── EDA_Extra/
│   ├── Codes/                        # Scripts used to generate the EDA plots below
│   ├── all_cac_tsne_plot.png         # t-SNE plot across all patients (with and without mask)
│   ├── cac_positieve_log_tsne.png    # t-SNE of per-patient features (RCA, LCA, LAD, LCX,
│   │                                 #   lesion count), colored by Agatston risk category:
│   │                                 #   Low (0–10), Medium (10–100), High (100–400),
│   │                                 #   Very High (400–1000), Extreme (1000+)
│   ├── freq_log_plot.plot            # Frequency plot with log transform
│   ├── id_146_proof_of_small_HU.png  # Example of faulty segmentation motivating COCA
│   │                                 #   script refactoring
│   ├── id_192_proof_of_small_HU.png  # Example of faulty segmentation motivating COCA
│   │                                 #   script refactoring
│   ├── biggest_valid_lesions.png     # Largest valid lesion (by ground truth) in the dataset
│   └── ...                           # Additional EDA outputs and artifacts
│
└── nnUnet/
    ├── setup_files/
    │   └── setup.py                  # Generates an nnU-Net-ready folder structure from splits.json
    ├── nnUnet_dob_scv.ipynb          # Notebook used for training nnU-Net DOB SCV version
    └── nnUnet_simple_scv.ipynb       # Notebook used for training nnU-Net Simple SCV version


```
 
---

## Evaluation Metrics (`metric.py`)

| Metric | Description |
|---|---|
| **Lesion F1** | Plaque-wise F1 score |
| **Dice** | Voxel-wise Dice coefficient |
| **Lesion Dice** | Averge Dice Score across Mathced lesions per Scan (aka Macro Dice in Code) |
| **Lesion PQ** | Panoptic Quality — `Lesion F1 × Lesion Dice` |
| **MAE** | Mean Absolute Error over Agatston scores |
| **F1 / Recall / Precision** | Per Agatston risk category, with confusion matrix |
| **Weighted Kappa** | Agreement over risk-category scoring |

---

## Getting Started

### 1. Install dependencies

**Local:**
```bash
pip install -r requirements.txt
```

**Remote:** ensure PyTorch is installed first (matching your CUDA version), then:
```bash
pip install -r requirements_for_remote.txt
```

### 2. Configure the pipeline

Edit `config.py` to set preprocessing, dataset, dataloader, model, training, evaluation, and dual-HU-windowing parameters — including the choice between DOB-SCV and Simple SCV splitting strategies.

### 3. Preprocess and generate splits

```bash
python pre_process.py
```

This generates/caches Heart ROI masks and writes the split dictionary to `MetaData/splits.json`.

### 4. Train

```bash
python train.py <your_experiment_name>
```

### 5. Evaluate

```bash
python eval.py  <your_experiment_name>
```

The checkpoint is loaded from the file please edit resampled_eval.py`.

For resampling-invariance checks:
```bash
python resampled_eval.py 
```

### 6. Run inference on a new scan

Use `wrapper.py` to run predictions on a single NIfTI file end-to-end.

---

## Exploratory Data Analysis

Supplementary EDA plots and the scripts used to generate them are available in `EDA_Extra/`, including t-SNE projections of patient-level calcium features by Agatston risk category, log-transformed frequency distributions, and examples of segmentation errors that motivated refactoring of the COCA preprocessing scripts.

---

## References

See [`cited_papers.md`](./cited_papers.md) for the full list of cited papers and articles.