from pathlib import Path
import nibabel as nib
import numpy as np
import torch
import SimpleITK as sitk
from monai.inferers import sliding_window_inference

import config as cfg
from dataset import get_transforms
from eval import lesion_post_process, roi_post_process
from LW_UNET_TVERSKY.lw_model import LightweightUNet3D as Heart_Seg_Model
from model import RIUnet
from pre_process import generate_roi_masks


def predict_single_nifti(
    input_nifti_path: str,
    output_dir: str,
    riunet_ckpt_path: str,
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
    print("[3/4] Running RIUnet sliding window inference...")
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

    output_file_path = out_dir / f"{input_path.stem.split('.')[0]}_prediction.nii.gz"
    nib.save(pred_nii, str(output_file_path))
    print(f"Prediction successfully saved to: {output_file_path}")

    return str(output_file_path)



import time
import nibabel as nib
import numpy as np

if __name__ == "__main__":

    t_start_total = time.perf_counter()

    # 1. Run inference pipeline
    t0 = time.perf_counter()
    output_mask_path = predict_single_nifti(
        input_nifti_path="/pscratch/sd/r/rajatns/gsoc_2026_official/try1/data_resampled/32bd921a6965/32bd921a6965_img.nii.gz",
        output_dir="/pscratch/sd/r/rajatns/gsoc_2026_official/try1/output",
        riunet_ckpt_path="/pscratch/sd/r/rajatns/gsoc_2026_official/try1/Segmentation_Rajat/runs/run_id6/ckpts/best_dice.pth",
    )
    t_inference = time.perf_counter() - t0

    # 2. Load NIfTI volumes & calculate metrics
    t0 = time.perf_counter()
    prediction_path = output_mask_path
    label_path = "/pscratch/sd/r/rajatns/gsoc_2026_official/try1/data_resampled/32bd921a6965/32bd921a6965_binary_seg.nii.gz"

    gt_img = nib.load(label_path)
    pred_img = nib.load(prediction_path)

    gt_arr = (gt_img.get_fdata() > 0).astype(np.uint8)
    pred_arr = (pred_img.get_fdata() > 0).astype(np.uint8)

    # Compute Dice score
    intersection = np.logical_and(gt_arr, pred_arr).sum()
    total_voxels = gt_arr.sum() + pred_arr.sum()
    dice_score = 1.0 if total_voxels == 0 else (2.0 * intersection) / total_voxels
    t_evaluation = time.perf_counter() - t0

    t_total = time.perf_counter() - t_start_total

    # 3. Print Results & Timings
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"Target Label Scan : {label_path}")
    print(f"Prediction Scan   : {prediction_path}")
    print(f"Voxel-wise Dice   : {dice_score:.4f}")
    print("-" * 50)
    print("EXECUTION TIMINGS")
    print("-" * 50)
    print(f"Inference Pipeline : {t_inference:.3f} s")
    print(f"Dice Computation   : {t_evaluation:.3f} s")
    print(f"Total Elapsed Time : {t_total:.3f} s")
    print("=" * 50)