# Data & Model Utilities

A collection of operational scripts designed for dataset preprocessing, spatial resampling diagnostic testing, model sanity checking, and remote GPU path dynamic updating.

## File Structure
  
├── big_lesion_test.py                # Diagnostic test for rectangular blob artifacts caused by nearest-neighbor resampling   
├── generate_dataset_for_reporting.py  # Dataset generation utility to resample CT scans at various voxel spacings   
├── model_sanity_check.py             # Performance check utility to evaluate model output across different configurations   
├── overfit_check.py                  # Diagnostic script to overfit the model on 10 batches (Cache/Live mode) 
├── Visualize_Bottleneck.py                  # Use it Visualize the bottlenck layer stored for every few epochs    
└── update_splits.py                  # Utility script to remap dataset paths for training on remote GPU servers

## Module Overview

### 1. Resampling Diagnostic (`big_lesion_test.py`)
Identifies and resolves spatial interpolation artifacts.
- **Problem**: Input volumes showed abnormal rectangular calcification blobs when resampled via nearest-neighbor interpolation.
- **Function**: Tests spatial fidelity before and after preprocessing to confirm that alternative resampling methods eliminate geometric distortion.

### 2. Multi-Resolution Dataset Builder (`generate_dataset_for_reporting.py`)
Generates resampled test datasets from canonical CT volumes.
- **Source Directory**: `data_canonical/`
- **Function**: Resamples 3D CT scans across varying target voxel spacings (e.g., isotropic vs. anisotropic resolutions) to build standardized benchmarking datasets.

### 3. Configuration Checker (`model_sanity_check.py`)
Verifies model stability across hyperparameter setups.
- **Function**: Executes quick forward/backward evaluation passes across diverse experimental configurations to ensure architectural integrity prior to large-scale training.

### 4. Overfit Verification (`overfit_check.py`)
Validates model capacity and gradient flow.
- **Function**: Overfits the network on a minimal 10-batch subset to confirm convergence capability.
- **Execution Modes**:
  - `CACHE`: Loads pre-processed volumes directly into RAM.
  - `LIVE`: Performs real-time augmentations and loading per iteration.
  
### 5. FNO Bottleneck Visualizer (`Visualize_Bottleneck.py`)
Inspects high-dimensional latent representations extracted from Fourier Neural Operator (FNO) layers.
- **Function**: Loads intermediate tensor representations checkpointed during training to visualize spatial-frequency feature maps, monitor mode retention, and diagnose spectral energy decay or gradient bottlenecks across epochs.

### 6. Remote Path Remapper (`update_splits.py`)
Manages dataset split paths across heterogeneous environments.
- **Function**: Dynamically rewrites relative/absolute paths, and split metadata (`train.json`, `val.json`, etc.) to allow seamless migration between local machines and remote GPU compute nodes.
