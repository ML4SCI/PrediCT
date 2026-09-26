# PrediCT: Physiologically-Driven Synthetic Calcium Generation

PrediCT is a comprehensive pipeline for generating hyper-realistic, physiologically accurate synthetic coronary artery calcium (CAC) directly into non-contrast CT (NCCT) or contrast-enhanced CT angiography (CCTA) scans. 

Unlike traditional data augmentation techniques that rely on hardcoded phenotypes or simple image processing, PrediCT uses a mathematically rigorous, multi-stage pipeline grounded in fluid dynamics and clinical statistics.

## Project Structure

The codebase is organized into four distinct biological phases:

* **`src/phase1_segmentation/`**: Vessel extraction and binary masking.
* **`src/phase2_hemodynamics/`**: Endothelial Shear Stress (ESS) calculation via Physics-Informed Neural Networks (PINNs).
* **`src/phase3_plaque_growth/`**: Stochastic calcium seeding and anisotropic Breadth-First Search (BFS) growth based on Negative Binomial and Log-Normal distributions.
* **`src/phase4_texturing/`**: Dual-stage Gaussian Alpha Blending and Hounsfield Unit (HU) texturing to simulate CT quantum noise and blooming artifacts.

*See the `README.md` inside each of these folders for detailed mathematical and technical documentation of that specific phase.*

## Entry Points

* **`scripts/physio_twin.py`**: The primary end-to-end execution script. Runs the entire Phase 1 -> Phase 4 pipeline on a single patient scan.
* **`scripts/run_batch_pinn.py`**: Batch processor for running the expensive Phase 2 PINN models across multiple patients.
* **`analysis/`**: Various auditing, validation, and diagnostic scripts.
* **`docs/`**: Generated reports and reproducibility logs.

## Legacy Code
Older iterations of the pipeline have been moved to `_legacy_archive/` to keep the main source tree clean.

## Usage
1. Install dependencies via `pip install -r requirements.txt`
2. Configure paths in `src/phase2_hemodynamics/config.py`
3. Run `python scripts/physio_twin.py <patient_id> <target_agatston>`
