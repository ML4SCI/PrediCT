import os
import sys
import csv
import shutil
import tempfile
import gc
from pathlib import Path

import SimpleITK as sitk
import torch

# 1. Fix ModuleNotFoundError: Add project root and subfolders to sys.path
PROJECT_ROOT = Path(r"E:\MyProjects\Gsoc_2026_Official")
for p in [
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "Segmentation_Rajat"),
    str(PROJECT_ROOT / "nnunet_dob_scv"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Import prediction wrappers and Agatston script
from Segmentation_Rajat.wrapper import predict as reronet_predict
from nnunet_dob_scv.predict import predict as nnunet_predict
from Segmentation_Rajat.agatston_script import calculate_agatston

# Configuration
RERONET_CKPT = str(PROJECT_ROOT / "Segmentation_Rajat" / "runs" / "run_no_coord" / "ckpts" / "best_dice.pth")

def cleanup_vram():
    """Aggressively clear VRAM and system memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def process_patients(root_folder: str, model_choice: str, output_base_dir: str):
    root_path = Path(root_folder).resolve()
    out_base_path = Path(output_base_dir).resolve()
    out_base_path.mkdir(parents=True, exist_ok=True)

    # Patient folders are the first level of directories in root_folder
    patient_folders = [d for d in root_path.iterdir() if d.is_dir()]

    if not patient_folders:
        print(f"No patient folders found in {root_folder}")
        return

    print(f"Found {len(patient_folders)} patients. Using model: {model_choice}")

    results_data = []

    for patient_dir in patient_folders:
        patient_id = patient_dir.name
        print(f"\nProcessing Patient: {patient_id}")

        try:
            # Create patient-specific output directory
            patient_out_dir = out_base_path / patient_id
            patient_out_dir.mkdir(parents=True, exist_ok=True)

            # 1. Run Prediction
            if model_choice.lower() == "reronet":
                # wrapper.predict handles 'dicom' mode by converting folder to nii and predicting
                mask_path = reronet_predict(
                    mode="dicom",
                    input_path=str(patient_dir),
                    output_dir=str(patient_out_dir),
                    riunet_ckpt_path=RERONET_CKPT
                )
            elif model_choice.lower() == "nnunet":
                # nnunet_dob_scv.predict expects 'dicom' mode for folder input
                nnunet_predict("dicom", str(patient_dir), str(patient_out_dir))
                # nnU-Net saves as prediction_<case_id>.nii.gz
                mask_path = next(patient_out_dir.glob("prediction_*.nii.gz"), None)
                if not mask_path:
                    raise RuntimeError(f"nnU-Net failed to produce mask for {patient_id}")
                mask_path = str(mask_path)
            else:
                raise ValueError("Invalid model choice. Use 'reronet' or 'nnunet'.")

            # 2. Calculate Agatston Score
            # We need the original image to calculate the score.
            # Since the predictors often use temp files for DICOM->NII,
            # we'll do a quick conversion of the DICOM folder to NII here for the Agatston script.
            with tempfile.TemporaryDirectory() as tmp_dir:
                temp_nii_path = Path(tmp_dir) / "temp_scan.nii.gz"

                # Re-use the SimpleITK logic from wrappers to get a NIfTI scan
                series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(patient_dir))
                if not series_ids:
                    raise RuntimeError(f"No DICOM series found for {patient_id}")

                best_series_id = max(series_ids, key=lambda sid: len(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(patient_dir), sid)))
                dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(patient_dir), best_series_id)

                reader = sitk.ImageSeriesReader()
                reader.SetFileNames(dicom_names)
                image = reader.Execute()
                sitk.WriteImage(image, str(temp_nii_path))

                # Calculate Agatston
                agatston_res = calculate_agatston(str(temp_nii_path), mask_path)
                score = agatston_res["agatston_total"]
                print(f"  Agatston Score: {score}")

                results_data.append({
                    "PatientID": patient_id,
                    "Model": model_choice,
                    "AgatstonScore": score,
                    "MaskPath": mask_path
                })

        except Exception as e:
            print(f"  [Error] Failed to process {patient_id}: {e}")
            results_data.append({
                "PatientID": patient_id,
                "Model": model_choice,
                "AgatstonScore": "Error",
                "MaskPath": "N/A"
            })
        finally:
            cleanup_vram()

    # 3. Save to CSV
    csv_path = out_base_path / "predicted_agatston_scores.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["PatientID", "Model", "AgatstonScore", "MaskPath"])
        writer.writeheader()
        writer.writerows(results_data)

    print(f"\nProcessing complete. CSV saved to: {csv_path}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("\nUsage: python dicom_all.py <root_folder> <model_choice> <output_dir>")
        print("\nExample:")
        print("  python dicom_all.py ./patients reronet ./results")
        print("  python dicom_all.py ./patients nnunet ./results")
        sys.exit(1)

    root_f = sys.argv[1]
    model_c = sys.argv[2]
    out_d = sys.argv[3]

    process_patients(root_f, model_c, out_d)

#Sample Usage: python Agentic_interface/dicom_all.py E:\MyProjects\Gsoc_2026_Official\Agentic_interface\test reronet E:\MyProjects\Gsoc_2026_Official\output
