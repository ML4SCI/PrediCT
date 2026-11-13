import os
import shutil
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import Segmentation_Rajat.config as cfg

from Segmentation_Rajat.eval import lesion_post_process, roi_post_process
from Segmentation_Rajat.LW_UNET_TVERSKY.lw_model import LightweightUNet3D as Heart_Seg_Model
from Segmentation_Rajat.pre_process import generate_roi_masks

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
  <output_dir>  : Directory where the CAC prediction will be saved

Output:
  <output_dir>/prediction_<case_id>.nii.gz : post-processed CAC binary mask

Examples:
  python predict.py dicom ./patient_001_dicom_folder ./output/
  python predict.py nii   ./patient_001_dicom_folder/image.nii.gz ./output/
================================================================================
"""

THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent
os.environ.setdefault("nnUNet_results", str(THIS_DIR / "nnUNet_results"))

MAX_VOXEL_THRESHOLD = 15000  # lesions with more voxels than this are removed (big-lesion filter)


# -----------------------------------------------------------------------------
# Error handling
# -----------------------------------------------------------------------------
class PredictError(RuntimeError):
    """Raised for any failure in this script; message carries file, function and context."""


def _fail(func: str, reason: str, cause: BaseException = None, **ctx):
    lines = [
        "",
        f"[PredictError] file     : {THIS_FILE}",
        f"[PredictError] function : {func}()",
        f"[PredictError] reason   : {reason}",
    ]
    if cause is not None:
        lines.append(f"[PredictError] cause    : {type(cause).__name__}: {cause}")
    for k, v in ctx.items():
        lines.append(f"[PredictError] {k:<8} : {v}")
    err = PredictError("\n".join(lines))
    if cause is not None:
        raise err from cause
    raise err


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def read_dicom_folder(dicom_dir: Path, output_nifti_path: Path):
    fn = "read_dicom_folder"
    if not dicom_dir.is_dir():
        _fail(fn, "DICOM path is not a directory", dicom_dir=dicom_dir)
    try:
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_dir))
    except Exception as e:
        _fail(fn, "SimpleITK failed to scan the DICOM folder", cause=e, dicom_dir=dicom_dir)
    if not series_ids:
        _fail(fn, "No DICOM series found in directory", dicom_dir=dicom_dir)

    best_series_id = max(
        series_ids,
        key=lambda sid: len(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir), sid)),
    )
    dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir), best_series_id)
    print(f"      DICOM series chosen: {best_series_id} ({len(dicom_names)} slices, "
          f"{len(series_ids)} series found)")
    try:
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        sitk.WriteImage(image, str(output_nifti_path))
    except Exception as e:
        _fail(fn, "Failed to read DICOM series / write NIfTI", cause=e,
              dicom_dir=dicom_dir, output=output_nifti_path)


def _count(mask: np.ndarray) -> int:
    return int((mask > 0).sum())


def _axcodes(nii) -> tuple:
    return tuple(nib.aff2axcodes(nii.affine))


def _load_heart_model(device: torch.device):
    fn = "_load_heart_model"
    try:
        heart_ckpt_path = cfg.dataloader_config["HEART_MODEL_PATH"]
    except Exception as e:
        _fail(fn, "cfg.dataloader_config['HEART_MODEL_PATH'] not available", cause=e,
              config_file=getattr(cfg, "__file__", "config.py"))
    if not Path(heart_ckpt_path).exists():
        _fail(fn, "Heart model checkpoint does not exist", checkpoint=heart_ckpt_path)
    try:
        model = Heart_Seg_Model().to(device)
        ckpt = torch.load(heart_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()
    except Exception as e:
        _fail(fn, "Failed to build/load the heart segmentation model", cause=e,
              checkpoint=heart_ckpt_path, device=device)
    print(f"      Heart model loaded from: {heart_ckpt_path}")
    return model


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------
def predict(mode: str, input_path: str | os.PathLike, output_dir: str | os.PathLike):
    fn = "predict"
    target_path = Path(input_path).resolve()
    target_out_dir = Path(output_dir).resolve()
    target_out_dir.mkdir(parents=True, exist_ok=True)

    mode = mode.strip().lower()
    if not target_path.exists():
        _fail(fn, "Input path does not exist", input_path=target_path, mode=mode)

    # Temp workspace: converted input, raw nnU-Net output, and the (unsaved) heart ROI mask.
    work_dir = Path(tempfile.mkdtemp())

    try:
        # ---------------------------------------------------------------------
        # 0. Input handling -> '<case_id>_img.nii.gz' in the temp dir
        #    (name is what generate_roi_masks expects; no resampling is done)
        # ---------------------------------------------------------------------
        if mode == "dicom":
            case_id = target_path.name
            input_file = work_dir / f"{case_id}_img.nii.gz"
            print(f"[0/5] Reading DICOM series from: {target_path}")
            read_dicom_folder(target_path, input_file)
        elif mode in ("nii", "nii.gz", "nifti"):
            case_id = (
                target_path.parent.name
                if target_path.stem in ("image", "image.nii") or target_path.name == "image.nii.gz"
                else target_path.name.replace(".nii.gz", "").replace(".nii", "").replace("_0000", "")
            )
            input_file = work_dir / f"{case_id}_img.nii.gz"
            print(f"[0/5] Reading NIfTI input: {target_path}")
            try:
                sitk.WriteImage(sitk.ReadImage(str(target_path)), str(input_file))
            except Exception as e:
                _fail(fn, "Failed to read NIfTI input with SimpleITK", cause=e,
                      input_path=target_path, temp_copy=input_file)
        else:
            _fail(fn, f"Unknown mode '{mode}'. Use 'dicom' or 'nii'.", mode=mode)

        try:
            in_img = sitk.ReadImage(str(input_file))
        except Exception as e:
            _fail(fn, "Failed to re-read prepared input image", cause=e, input_file=input_file)
        print(f"      case_id={case_id} | input spacing={in_img.GetSpacing()} size={in_img.GetSize()}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"      Executing inference on: {device}")

        # ---------------------------------------------------------------------
        # 1. nnU-Net CAC inference (nnU-Net resamples internally)
        # ---------------------------------------------------------------------
        print("[1/5] Running nnU-Net CAC inference")
        model_folder = (
            THIS_DIR / "nnUNet_results" / "Dataset001_CAC" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
        )
        if not model_folder.exists():
            _fail(fn, "nnU-Net model folder not found", model_folder=model_folder,
                  nnUNet_results=os.environ.get("nnUNet_results"))

        raw_truncated = work_dir / f"raw_prediction_{case_id}"
        raw_pred_path = Path(f"{raw_truncated}.nii.gz")
        try:
            predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=False,
                perform_everything_on_device=True,
                device=device,
            )
            predictor.initialize_from_trained_model_folder(
                str(model_folder),
                use_folds=(0,),
                checkpoint_name="checkpoint_final.pth",
            )
            predictor.predict_from_files(
                list_of_lists_or_source_folder=[[str(input_file)]],
                output_folder_or_list_of_truncated_output_files=[str(raw_truncated)],
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1,
            )
        except Exception as e:
            _fail(fn, "nnU-Net inference failed", cause=e, model_folder=model_folder,
                  input_file=input_file, device=device)

        if not raw_pred_path.exists():
            _fail(fn, "nnU-Net finished but produced no output file", expected=raw_pred_path)

        pred_nii = nib.load(str(raw_pred_path))
        pred_sitk = sitk.ReadImage(str(raw_pred_path))
        pred_mask = (np.asarray(pred_nii.dataobj) > 0).astype(np.uint8)
        final_spacing = [float(s) for s in pred_sitk.GetSpacing()]  # (x, y, z) SimpleITK order

        print(f"      Predicted mask: shape={pred_mask.shape} spacing={tuple(final_spacing)} "
              f"CAC voxels={_count(pred_mask)}")
        pred_axcodes = _axcodes(pred_nii)
        print(f"      Predicted mask orientation (nibabel axcodes): {pred_axcodes} "
              f"-> {'RAS' if pred_axcodes == ('R', 'A', 'S') else 'NOT RAS'}")
        print("      (nnU-Net exports in the input image's own orientation; no RAS reorientation applied.)")

        # ---------------------------------------------------------------------
        # 2. Heart ROI mask (temporary, not saved). FINAL_SPACING = predicted mask spacing
        # ---------------------------------------------------------------------
        print(f"[2/5] Generating heart ROI mask (temporary) with FINAL_SPACING={final_spacing}")
        heart_model = _load_heart_model(device)
        try:
            roi_mask_paths = generate_roi_masks(
                [str(input_file)], model=heart_model, FINAL_SPACING=final_spacing
            )
        except Exception as e:
            _fail(fn, "generate_roi_masks() raised an exception", cause=e,
                  image=input_file, FINAL_SPACING=final_spacing)
        finally:
            del heart_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        roi_mask_path = Path(roi_mask_paths[0]) if roi_mask_paths and roi_mask_paths[0] else None
        if roi_mask_path is None:
            _fail(fn, "No heart detected: generate_roi_masks() returned no ROI mask "
                      "(heart segmentation was empty)", image=input_file,
                  FINAL_SPACING=final_spacing)
        if not roi_mask_path.exists():
            _fail(fn, "generate_roi_masks() reported a path that does not exist",
                  roi_mask_path=roi_mask_path)

        roi_nii = nib.load(str(roi_mask_path))
        roi_arr = (np.asarray(roi_nii.dataobj) > 0).astype(np.uint8)
        roi_axcodes = _axcodes(roi_nii)
        print(f"      ROI mask: shape={roi_arr.shape} voxels={_count(roi_arr)} "
              f"orientation={roi_axcodes} -> {'RAS' if roi_axcodes == ('R', 'A', 'S') else 'NOT RAS'}")

        # Grid consistency checks (ROI must live on the prediction grid)
        if roi_arr.shape != pred_mask.shape:
            _fail(fn, "ROI mask shape does not match predicted mask shape",
                  roi_shape=roi_arr.shape, pred_shape=pred_mask.shape,
                  roi_mask=roi_mask_path, pred_mask=raw_pred_path,
                  FINAL_SPACING=final_spacing)
        if not np.allclose(roi_nii.affine, pred_nii.affine, atol=1e-3):
            _fail(fn, "ROI mask affine does not match predicted mask affine",
                  roi_affine=roi_nii.affine.round(4).tolist(),
                  pred_affine=pred_nii.affine.round(4).tolist(),
                  roi_mask=roi_mask_path, pred_mask=raw_pred_path)
        print("      ROI mask and prediction share the same shape/affine: OK")

        # ---------------------------------------------------------------------
        # 3. Post-processing: big-lesion removal
        # ---------------------------------------------------------------------
        post_big = cfg.eval_config.get("POST_PROCESS_BIG_LESIONS", True)
        print(f"[3/5] Big-lesion post-processing (POST_PROCESS_BIG_LESIONS={post_big}, "
              f"max_voxel_threshold={MAX_VOXEL_THRESHOLD}, connectivity=26)")
        if post_big:
            before = _count(pred_mask)
            try:
                pred_mask = lesion_post_process(
                    binary_mask=pred_mask,
                    max_voxel_threshold=MAX_VOXEL_THRESHOLD,
                    connectivity=26,
                )
            except Exception as e:
                _fail(fn, "lesion_post_process() raised an exception", cause=e,
                      max_voxel_threshold=MAX_VOXEL_THRESHOLD, mask_shape=pred_mask.shape)
            after = _count(pred_mask)
            print(f"      CAC voxels: {before} -> {after} (removed {before - after})")
        else:
            print("      Skipped.")

        # ---------------------------------------------------------------------
        # 4. Post-processing: ROI masking
        # ---------------------------------------------------------------------
        apply_roi = cfg.eval_config.get("APPLY_ROI_MASK", False)
        edge = cfg.eval_config.get("ROI_EXCLUDE_EDGE_VOXELS", 10)
        print(f"[4/5] ROI post-processing (APPLY_ROI_MASK={apply_roi}, exclude_edge_voxels={edge})")
        if apply_roi:
            before = _count(pred_mask)
            try:
                pred_mask = roi_post_process(
                    binary_mask=pred_mask,
                    roi_mask=roi_arr,
                    exclude_edge_voxels=edge,
                )
            except Exception as e:
                _fail(fn, "roi_post_process() raised an exception", cause=e,
                      exclude_edge_voxels=edge, mask_shape=pred_mask.shape,
                      roi_shape=roi_arr.shape)
            after = _count(pred_mask)
            print(f"      CAC voxels: {before} -> {after} (removed {before - after})")
        else:
            print("      Skipped.")

        # ---------------------------------------------------------------------
        # 5. Save final prediction
        # ---------------------------------------------------------------------
        output_path = target_out_dir / f"prediction_{case_id}.nii.gz"
        try:
            out_nii = nib.Nifti1Image(
                pred_mask.astype(np.uint8), affine=pred_nii.affine, header=pred_nii.header
            )
            out_nii.set_data_dtype(np.uint8)
            nib.save(out_nii, str(output_path))
        except Exception as e:
            _fail(fn, "Failed to save final prediction", cause=e, output_path=output_path)

        print(f"[5/5] Final CAC voxels: {_count(pred_mask)}")
        print(f"\n[SUCCESS] Saved prediction to: {output_path}")
        return str(output_path)

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(USAGE_HELP)
        sys.exit(0)

    if len(sys.argv) < 4:
        print(USAGE_HELP)
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\nActive Execution Device: {device_name}\n")

    predict(sys.argv[1], sys.argv[2], sys.argv[3])