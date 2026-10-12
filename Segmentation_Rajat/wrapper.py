from pathlib import Path
import nibabel as nib
import numpy as np
import torch
import SimpleITK as sitk
import os
import sys
import shutil
import tempfile
from monai.inferers import sliding_window_inference

import config as cfg
from dataset import get_transforms
from eval import lesion_post_process, roi_post_process
from LW_UNET_TVERSKY.lw_model import LightweightUNet3D as Heart_Seg_Model
from model import RIUnet
from pre_process import generate_roi_masks


def read_dicom_folder(dicom_dir: Path, output_nifti_path: Path):
    """
    Reads a DICOM series from a directory and saves it as a NIfTI file.
    Following the pattern in nnunet_dob_scv/predict.py.
    """
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in directory: {dicom_dir}")

    best_series_id = max(
        series_ids,
        key=lambda sid: len(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir), sid)),
    )
    dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir), best_series_id)

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, str(output_nifti_path))


def predict_single_nifti(
    input_nifti_path: str,
    output_dir: str,
    case_id: str = None,
    riunet_ckpt_path: str = None,
    heart_model_ckpt_path: str = cfg.dataloader_config["HEART_MODEL_PATH"],
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_voxel_threshold: int = 15000,
) -> str:
    """End-to-end wrapper to generate and save CAC binary predictions for a single NIfTI image."""
    device = torch.device(device_str)
    input_path = Path(input_nifti_path).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)


    # -------------------------------------------------------------------------
    # 1. Heart Segmentation & ROI Mask Generation (pre_process.py logic)
    # -------------------------------------------------------------------------
    print(f"[1/4] Generating Heart ROI Mask for: {input_path.name}")
    heart_model = Heart_Seg_Model().to(device)
    heart_ckpt = torch.load(heart_model_ckpt_path, map_location=device)
    heart_model.load_state_dict(heart_ckpt["model"])
    heart_model.eval()

    # Generates '<case_id>_roi_mask.nii.gz' adjacent to the image
    roi_mask_paths = generate_roi_masks([str(input_path)], model=heart_model)
    roi_mask_path = Path(roi_mask_paths[0]) if roi_mask_paths else None

    # -------------------------------------------------------------------------
    # 2. Data Loading & MONAI Preprocessing Pipeline (dataset.py logic)
    # -------------------------------------------------------------------------
    print("[2/4] Applying evaluation transform pipeline...")
    transforms = get_transforms(mode="val")
    data_dict = {"image": str(input_path), "label": str(input_path)}
    if roi_mask_path and roi_mask_path.exists():
        data_dict["roi_mask"] = str(roi_mask_path)

    processed_data = transforms(data_dict)
    x = processed_data["image"].unsqueeze(0).to(device, dtype=torch.float32)

    # -------------------------------------------------------------------------
    # 3. Model Loading & Sliding Window Inference (eval.py logic)
    # -------------------------------------------------------------------------
    print("[3/4] Running ReRoNet sliding window inference...")
    in_channels = x.shape[1]
    model = RIUnet(in_channels=in_channels).to(device)
    riunet_state = torch.load(riunet_ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(riunet_state["model"])
    model.eval()

    roi_size = tuple(cfg.dataloader_config["ROI_SIZE"])
    sw_batch_size = int(cfg.train_config.get("VAL_SW_BATCH_SIZE", 4))
    sw_overlap = float(cfg.train_config.get("VAL_SW_OVERLAP", 0.25))
    threshold = float(cfg.train_config.get("VAL_THRESHOLD", 0.5))

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float32):
            logits = sliding_window_inference(
                inputs=x,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                overlap=sw_overlap,
                predictor=model,
                mode="gaussian",
                device=device,
            )
        probs = torch.sigmoid(logits)
        pred_mask = (probs > threshold).squeeze().detach().cpu().numpy().astype(np.uint8)

    # -------------------------------------------------------------------------
    # 4. Post-Processing (eval.py logic)
    # -------------------------------------------------------------------------
    print("[4/4] Post-processing predictions...")
    if cfg.eval_config.get("POST_PROCESS_BIG_LESIONS", True):
        pred_mask = lesion_post_process(
            binary_mask=pred_mask,
            max_voxel_threshold=max_voxel_threshold,
            connectivity=26,
        )

    if cfg.eval_config.get("APPLY_ROI_MASK", False) and roi_mask_path and roi_mask_path.exists():
        roi_nii = nib.load(roi_mask_path)
        roi_arr = roi_nii.get_fdata().astype(np.uint8)
        pred_mask = roi_post_process(binary_mask=pred_mask, roi_mask=roi_arr, exclude_edge_voxels=cfg.eval_config.get("ROI_EXCLUDE_EDGE_VOXELS", 10))

    # -------------------------------------------------------------------------
    # 5. Spatial Reorientation & Output Saving
    # -------------------------------------------------------------------------
    ref_nii = nib.load(str(input_path))
    ras_ornt = nib.orientations.axcodes2ornt(("R", "A", "S"))
    orig_ornt = nib.io_orientation(ref_nii.affine)
    ras2orig = nib.orientations.ornt_transform(ras_ornt, orig_ornt)

    # Map target RAS prediction back into raw orientation coordinates
    corrected_pred_arr = nib.orientations.apply_orientation(pred_mask, ras2orig)

    pred_nii = nib.Nifti1Image(
        corrected_pred_arr.astype(np.uint8),
        affine=ref_nii.affine,
        header=ref_nii.header,
    )

    # Use provided case_id for output naming if available
    stem = case_id if case_id else input_path.stem.split('.')[0]
    output_file_path = out_dir / f"{stem}_prediction.nii.gz"
    nib.save(pred_nii, str(output_file_path))
    print(f"Prediction successfully saved to: {output_file_path}")

    return str(output_file_path)



def predict(
    mode: str,
    input_path: str,
    output_dir: str,
    riunet_ckpt_path: str,
    heart_model_ckpt_path: str = cfg.dataloader_config["HEART_MODEL_PATH"],
    **kwargs
) -> str:
    """
    Routes the request based on mode ('dicom' or 'nii'), handles
    temporary file creation/cleanup, and calls predict_single_nifti.
    """
    mode = mode.strip().lower()
    target_path = Path(input_path).resolve()
    target_out_dir = Path(output_dir).resolve()

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
        raise ValueError(f"Unknown mode '{mode}'. Use 'dicom' or 'nii'.")

    try:
        output_path = predict_single_nifti(
            input_nifti_path=str(input_file),
            output_dir=str(target_out_dir),
            case_id=case_id,
            riunet_ckpt_path=riunet_ckpt_path,
            heart_model_ckpt_path=heart_model_ckpt_path,
            **kwargs
        )
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    return output_path


import time

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("\nUsage: python wrapper.py <mode> <input_path> <output_dir> <riunet_ckpt_path> [label_path]")
        print("\nModes:")
        print("  dicom : input_path is a folder containing DICOM slices")
        print("  nii   : input_path is a .nii or .nii.gz file")
        print("\nExample:")
        print("  python wrapper.py dicom ./data/patient_01 ./output ./ckpts/best.pth")
        print("  python wrapper.py nii ./data/patient_01/img.nii.gz ./output ./ckpts/best.pth")
        sys.exit(1)

    mode_arg = sys.argv[1]
    input_path_arg = sys.argv[2]
    output_dir_arg = sys.argv[3]
    riunet_ckpt_arg = sys.argv[4]
    label_path_arg = sys.argv[5] if len(sys.argv) > 5 else None

    t_start_total = time.perf_counter()

    # 1. Run inference pipeline
    t0 = time.perf_counter()
    try:
        output_mask_path = predict(
            mode=mode_arg,
            input_path=input_path_arg,
            output_dir=output_dir_arg,
            riunet_ckpt_path=riunet_ckpt_arg
        )
    except Exception as e:
        print(f"Inference failed: {e}")
        sys.exit(1)
    t_inference = time.perf_counter() - t0

    # 2. Optional Evaluation
    if label_path_arg:
        t0 = time.perf_counter()
        prediction_path = output_mask_path
        label_path = label_path_arg

        gt_img = nib.load(label_path)
        pred_img = nib.load(prediction_path)

        gt_arr = (gt_img.get_fdata() > 0).astype(np.uint8)
        pred_arr = (pred_img.get_fdata() > 0).astype(np.uint8)

        # Compute Dice score
        intersection = np.logical_and(gt_arr, pred_arr).sum()
        total_voxels = gt_arr.sum() + pred_arr.sum()
        dice_score = 1.0 if total_voxels == 0 else (2.0 * intersection) / total_voxels
        t_evaluation = time.perf_counter() - t0
    else:
        dice_score = None
        t_evaluation = 0

    t_total = time.perf_counter() - t_start_total

    # 3. Print Results & Timings
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    if label_path_arg:
        print(f"Target Label Scan : {label_path_arg}")
    print(f"Prediction Scan   : {output_mask_path}")
    if dice_score is not None:
        print(f"Voxel-wise Dice   : {dice_score:.4f}")
    print("-" * 50)
    print("EXECUTION TIMINGS")
    print("-" * 50)
    print(f"Inference Pipeline : {t_inference:.3f} s")
    if label_path_arg:
        print(f"Dice Computation   : {t_evaluation:.3f} s")
    print(f"Total Elapsed Time : {t_total:.3f} s")
    print("=" * 50)