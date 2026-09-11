import os
import shutil
import sys
import tempfile
from pathlib import Path

import SimpleITK as sitk
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

USAGE_HELP = """
================================================================================
                              nnU-Net Inference Tool
================================================================================
Usage:
  python predict.py <mode> <input_path> <output_dir>

Arguments:
  <mode>        : 'dicom' (if input is a folder of DICOM files) 
                  'nii'   (if input is a .nii.gz or .nii volume file)
  <input_path>  : Path to the DICOM folder OR .nii.gz file
  <output_dir>  : Directory where the segmentation mask will be saved

Examples:
  python predict.py dicom ./patient_001_dicom_folder ./output/
  python predict.py nii   ./patient_001_dicom_folder/image.nii.gz ./output/
================================================================================
"""

THIS_DIR = Path(__file__).resolve().parent
os.environ.setdefault("nnUNet_results", str(THIS_DIR / "nnUNet_results"))


def read_dicom_folder(dicom_dir: Path, output_nifti_path: Path):
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_dir))
    best_series_id = max(
        series_ids,
        key=lambda sid: len(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir), sid)),
    )
    dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir), best_series_id)

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, str(output_nifti_path))


def predict(mode: str, input_path: str | os.PathLike, output_dir: str | os.PathLike):
    target_path = Path(input_path).resolve()
    target_out_dir = Path(output_dir).resolve()
    target_out_dir.mkdir(parents=True, exist_ok=True)

    mode = mode.strip().lower()
    temp_dir = None

    if mode == "dicom":
        case_id = target_path.name
        temp_dir = tempfile.mkdtemp()
        input_file = Path(temp_dir) / f"{case_id}.nii.gz"
        print(f"Reading DICOM series from: {target_path}")
        read_dicom_folder(target_path, input_file)
    elif mode in ("nii", "nii.gz", "nifti"):
        case_id = (
            target_path.parent.name
            if target_path.stem in ("image", "image.nii") or target_path.name == "image.nii.gz"
            else target_path.name.replace(".nii.gz", "").replace(".nii", "").replace("_0000", "")
        )
        input_file = target_path
    else:
        print(f"Error: Unknown mode '{mode}'. Use 'dicom' or 'nii'.")
        sys.exit(1)

    # Prefix the truncated output path so nnU-Net generates 'prediction_<case_id>.nii.gz' directly
    output_truncated_path = str(target_out_dir / f"prediction_{case_id}")

    # Detect execution device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing inference on: {device}")

    # Initialize Predictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=device,
    )

    model_folder = (
        THIS_DIR / "nnUNet_results" / "Dataset001_CAC" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    )
    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=(0,),
        checkpoint_name="checkpoint_final.pth",
    )

    print(f"Running inference for: {case_id}")
    predictor.predict_from_files(
        list_of_lists_or_source_folder=[[str(input_file)]],
        output_folder_or_list_of_truncated_output_files=[output_truncated_path],
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
    )

    # Clean up metadata JSONs
    for json_file in target_out_dir.glob("*.json"):
        json_file.unlink()

    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    print(f"\n[SUCCESS] Saved prediction to: {target_out_dir / f'prediction_{case_id}.nii.gz'}")


if __name__ == "__main__":
    # Check CUDA / Device status for user feedback
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\nActive Execution Device: {device_name}\n")

    if(sys.argv[1] in ("-h", "--help")):
        print(USAGE_HELP)
        sys.exit(0)

    if len(sys.argv) < 4:
        sys.exit(1)

    predict(sys.argv[1], sys.argv[2], sys.argv[3])