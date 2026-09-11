"""
eval.py — Evaluate a trained RIUnet checkpoint on val or test split.

CLI:
    python Segmentation_Rajat/eval.py EXPERIMENT_NAME
    python Segmentation_Rajat/eval.py EXPERIMENT_NAME --split val
    python Segmentation_Rajat/eval.py EXPERIMENT_NAME --split test
    python Segmentation_Rajat/eval.py EXPERIMENT_NAME --split val --num-samples 5
    python Segmentation_Rajat/eval.py EXPERIMENT_NAME --checkpoint best_dice.pth
    python Segmentation_Rajat/eval.py EXPERIMENT_NAME --overwrite
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import nibabel as nib
import numpy as np
import psutil
import torch
import torch.nn as nn
import cc3d
from monai.inferers import sliding_window_inference

import config as cfg
from dataset import build_dataloaders
from model import RIUnet



# Import evaluation metrics engine from metrics.py
from metrics import (
    voxel_wise_f1,
    plaque_wise_f1,
    compute_agatston_score,
    compute_basic_metrics_for_score,
    output_confusion_matrix,
)

# Match train.py environment setup
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"
SPLITS_JSON = Path(cfg.preprocessing_config["SPLITS_JSON"])


# ══════════════════════════════════════════════════════════════════
#  Paths / Logging / Utilities
# ══════════════════════════════════════════════════════════════════

def setup_eval_dirs(experiment_name: str, split_name: str) -> Dict[str, Path]:
    run_dir = RUNS_DIR / experiment_name
    eval_preds = run_dir / "eval_preds" / split_name
    paths = {
        "run": run_dir,
        "ckpts": run_dir / "ckpts",
        "logs": run_dir / "logs",
        "eval_preds": eval_preds,
        "visualizations": eval_preds / "Visualizations",
    }
    for k, p in paths.items():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def get_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger


def set_seed(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def auto_detect_in_channels(loader) -> int:
    return int(next(iter(loader))["image"].shape[1])


def _shrink_dataset(loader, max_samples: int):
    """Limit evaluation to the first N samples if --num-samples is provided."""
    base_ds = loader.dataset
    if hasattr(base_ds, "data"):
        base_ds.data = base_ds.data[:max_samples]
    from torch.utils.data import DataLoader
    return DataLoader(
        base_ds,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=0,
        sampler=None,
        collate_fn=loader.collate_fn,
    )


def _infer_nii_path(case: dict) -> Optional[Path]:
    """Pull the original image .nii.gz path from a dataset entry."""
    p = case.get("image")
    return Path(p) if p is not None else None

def lesion_post_process(
        binary_mask: np.ndarray,
        max_voxel_threshold: int = 15000,
        connectivity: int = 26,
    ) -> np.ndarray:
        """Filters out large unwanted 3D components exceeding a voxel threshold."""
        binary_input = (binary_mask > 0).astype(np.uint8)

        if np.sum(binary_input) == 0:
            return binary_input

        # 1. Run 3D Connected Components Analysis
        labels_out = cc3d.connected_components(binary_input, connectivity=connectivity)
        stats = cc3d.statistics(labels_out)
        voxel_counts = stats["voxel_counts"]

        # 2. Identify label IDs exceeding threshold (ignoring background index 0)
        large_component_ids = np.where(voxel_counts > max_voxel_threshold)[0]
        large_component_ids = large_component_ids[large_component_ids != 0]

        # 3. Create cleaned mask
        cleaned_binary = binary_input.copy()

        # 4. Zero out voxels belonging to large components
        if len(large_component_ids) > 0:
            print(
                f"[POSTPROC: LARGE LESIONS] Removing {len(large_component_ids)} large components "
                f"exceeding {max_voxel_threshold} voxels."
            )
            large_blobs_mask = np.isin(labels_out, large_component_ids)
            cleaned_binary[large_blobs_mask] = 0

        return cleaned_binary


import numpy as np
from scipy.ndimage import binary_erosion


def roi_post_process(
    binary_mask: np.ndarray, roi_mask: np.ndarray, exclude_edge_voxels: int = 0
) -> np.ndarray:
  """Applies a heart ROI mask to the binary prediction mask after peeling back

  the outer surface inward by `exclude_edge_voxels` on all 3D sides.
  """
  if binary_mask.shape != roi_mask.shape:
    raise ValueError(
        f"Shape mismatch: binary_mask {binary_mask.shape} vs roi_mask"
        f" {roi_mask.shape}."
    )

  processed_roi = roi_mask > 0

  # Peel the outer layers of the 3D heart ROI inward
  if exclude_edge_voxels > 0:
    processed_roi = binary_erosion(
        processed_roi, iterations=exclude_edge_voxels
    )

  # Mask the binary prediction (e.g., coronary artery segmentation)
  return (binary_mask > 0) * processed_roi

def morphological_cleanup(binary_mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Applies morphological operations to clean up the binary mask."""
    from scipy.ndimage import binary_opening, binary_closing

    # Apply binary opening followed by closing
    opened_mask = binary_opening(binary_mask, structure=np.ones((kernel_size,) * 3))
    cleaned_mask = binary_closing(opened_mask, structure=np.ones((kernel_size,) * 3))

    return cleaned_mask.astype(np.uint8)


def predict_one_case(
    model: nn.Module,
    batch: dict,
    device: torch.device,
    roi_size: tuple,
    roi_path: str,
    sw_batch_size: int = 4,
    overlap: float = 0.25,
    threshold: float = 0.5,
    max_voxel_threshold: int = 15000,
    connectivity: int = 26,
) -> tuple[np.ndarray, torch.Tensor, float, float]:
    """Runs sliding window inference on a single volume and returns pred_mask array and metrics."""
    x = batch["image"].to(device, non_blocking=True).to(torch.float32)
    y = batch["label"].to(device, non_blocking=True)

    with torch.amp.autocast("cuda", enabled=True, dtype=torch.float32):
        out = sliding_window_inference(
            inputs=x,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            predictor=model,
            mode="gaussian",
            device=device,
        )

    probs = torch.sigmoid(out)
    pred_mask = probs > threshold
    target_mask = y > 0

    # Extract NumPy array once on CPU for sequential post-processing operations
    pred_np = pred_mask.squeeze().detach().cpu().numpy().astype(np.uint8)

    # 1. Post-process large lesions
    if cfg.eval_config.get("POST_PROCESS_BIG_LESIONS", True):
        pred_np = lesion_post_process(
            binary_mask=pred_np,
            max_voxel_threshold=max_voxel_threshold,
            connectivity=connectivity,
        )

    # 2. Apply ROI mask post-processing
    if cfg.eval_config.get("APPLY_ROI_MASK", False):

        if roi_path.exists():
            roi_nii = nib.load(roi_path)
            roi_mask = roi_nii.get_fdata().astype(np.uint8)

            voxel_count_before = int(np.sum(pred_np))

            how_many_to_exclude = cfg.eval_config.get("ROI_EXCLUDE_EDGE_VOXELS", 10)

            pred_np = roi_post_process(binary_mask=pred_np, roi_mask=roi_mask, exclude_edge_voxels=how_many_to_exclude)

            voxel_count_after = int(np.sum(pred_np))

            print(
                f"[POSTPROC: ROI MASK] Applied ROI mask. (Exlude Voxels on all side: {how_many_to_exclude}) "
                f"Voxels before: {voxel_count_before}, after: {voxel_count_after}"
            )

        else:
            print(f"[WARNING] ROI mask not found at {roi_path}. Skipping ROI post-processing.")

    # Reconstruct modified tensor on device for Dice metric computation
    pred_mask = torch.from_numpy(pred_np).to(device, dtype=torch.bool).reshape(pred_mask.shape)

    # Dice metric computation
    intersection = (pred_mask & target_mask).sum()
    total = pred_mask.sum() + target_mask.sum()
    vol_dice = torch.where(total > 0, (2.0 * intersection) / total, torch.tensor(1.0, device=device))

    fg_m = probs[target_mask].mean().item() if target_mask.any() else float("nan")
    bg_m = probs[~target_mask].mean().item() if (~target_mask).any() else float("nan")

    # Final formatted prediction array (D, H, W) uint8
    pred_arr = pred_np.astype(np.uint8)

    del x, y, out, probs, pred_mask, target_mask
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return pred_arr, vol_dice, fg_m, bg_m


# ══════════════════════════════════════════════════════════════════
#  Step 1: Prediction Generation Engine
# ══════════════════════════════════════════════════════════════════
@torch.no_grad()
def generate_predictions(
    model: nn.Module,
    loader,
    device: torch.device,
    cfg_train: dict,
    logger: logging.Logger,
    eval_preds_dir: Path,
    num_samples: Optional[int] = None,
) -> float:
    """Generates and saves prediction masks alongside original images and ground truths."""
    model.eval()
    roi_size = tuple(cfg.dataloader_config["ROI_SIZE"])
    sw_batch_size = int(cfg_train.get("VAL_SW_BATCH_SIZE", 4))
    sw_overlap = float(cfg_train.get("VAL_SW_OVERLAP", 0.25))
    threshold = float(cfg_train.get("VAL_THRESHOLD", 0.5))

    if num_samples is not None:
        logger.info(f"Limiting prediction generation to top {num_samples} volume(s).")
        loader = _shrink_dataset(loader, num_samples)

    base_ds = loader.dataset
    total_volumes = len(loader) if hasattr(loader, "__len__") else 0
    t_start = time.time()

    logger.info(f"Generating predictions for {total_volumes} volume(s)...")

    # Target orientation from MONAI's Orientationd transform
    ras_ornt = nib.orientations.axcodes2ornt(('R', 'A', 'S'))

    for idx, batch in enumerate(loader):
        case = base_ds.data[idx] if hasattr(base_ds, "data") else {}
        case_id = case.get("id") or case.get("image", f"case_{idx}")
        case_id = str(Path(case_id).name)

        # Define patient output directory: run_dir/eval_preds/<split>/<case_id>/
        patient_dir = eval_preds_dir / case_id
        patient_dir.mkdir(parents=True, exist_ok=True)

        img_path = _infer_nii_path(case)
        lbl_path = Path(case.get("label")) if case.get("label") else None

        # Directly calculate ROI path from parent directory & case_id
        roi_path = img_path.parent / f"{case_id}_roi_mask.nii.gz"
        if not roi_path.exists():
            print(f"ERROR: ROI mask not found at {roi_path}")
            roi_path = None

        t0 = time.time()
        pred_arr, vol_dice, fg_m, bg_m = predict_one_case(
            model,
            batch,
            device,
            roi_size,
            roi_path=roi_path,
            sw_batch_size=sw_batch_size,
            overlap=sw_overlap,
            threshold=threshold,
        )
        dt = time.time() - t0

        

        # 1. Copy original image & label files
        if img_path and img_path.exists():
            shutil.copy(img_path, patient_dir / "image.nii.gz")
        if lbl_path and lbl_path.exists():
            shutil.copy(lbl_path, patient_dir / "label.nii.gz")

        ref_nii_path = img_path if (img_path and img_path.exists()) else lbl_path

        if ref_nii_path and ref_nii_path.exists():
            ref_nii = nib.load(ref_nii_path)

            # --- Re-orient RAS pred_arr back to raw image space ---
            orig_ornt = nib.io_orientation(ref_nii.affine)
            ras2orig = nib.orientations.ornt_transform(ras_ornt, orig_ornt)
            corrected_pred_arr = nib.orientations.apply_orientation(pred_arr, ras2orig)

            # Build prediction NIfTI with original affine and header metadata
            pred_nii = nib.Nifti1Image(
                corrected_pred_arr.astype(np.uint8), 
                affine=ref_nii.affine, 
                header=ref_nii.header
            )

            output_pred_path = patient_dir / "prediction.nii.gz"
            nib.save(pred_nii, output_pred_path)

            # Verification log
            saved_pred_nii = nib.load(output_pred_path)
            print(f"[{case_id}] Saved Pred Shape: {saved_pred_nii.shape} | Ref Image Shape: {ref_nii.shape}")

        logger.info(
            f"  [{idx + 1:02d}/{total_volumes:02d}] Generated: {case_id} | "
            f"Inf Dice: {vol_dice.item():.4f} | prob@fg: {fg_m:.4f} | prob@bg: {bg_m:.4f} | {dt:.2f}s"
        )
        gc.collect()

    dt_total = time.time() - t_start
    return dt_total


# ══════════════════════════════════════════════════════════════════
#  Step 2: Metric Evaluation Engine (using metrics.py)
# ══════════════════════════════════════════════════════════════════
import logging
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np
from monai.metrics import compute_hausdorff_distance

# Optional fallback if MONAI isn't installed: SimpleITK can be used instead

def compute_hd95_mm(label_path: Path, pred_path: Path) -> float:
    """Computes the 95th Percentile Hausdorff Distance in millimeters.

    Handles spatial spacing (pixdim) from the NIfTI header.
    """
    img_gt = nib.load(str(label_path))
    img_pred = nib.load(str(pred_path))

    gt_arr = img_gt.get_fdata() > 0
    pred_arr = img_pred.get_fdata() > 0

    # Handle edge cases where ground truth or prediction is empty
    if not np.any(gt_arr) or not np.any(pred_arr):
        return float("inf") if np.any(gt_arr) != np.any(pred_arr) else 0.0

    # Extract spacing and cast NumPy floats to native Python floats
    spacing = tuple(float(x) for x in img_gt.header.get_zooms()[:3])

    # Format tensors for MONAI: (Batch=1, Channel=1, X, Y, Z)
    gt_tensor = torch.from_numpy(gt_arr).unsqueeze(0).unsqueeze(0).float()
    pred_tensor = torch.from_numpy(pred_arr).unsqueeze(0).unsqueeze(0).float()

    # Compute HD95
    hd95 = compute_hausdorff_distance(
        y_pred=pred_tensor,
        y=gt_tensor,
        include_background=False,
        percentile=95,
        spacing=spacing,
    ).item()

    return float(hd95)


def evaluate_metrics(
    eval_preds_dir: Path,
    visualizations_dir: Path,
    logger: logging.Logger,
) -> Dict[str, float]:
    """Runs metrics.py functions over saved evaluation prediction folders."""
    logger.info("\n" + "=" * 60)
    logger.info(" RUNNING FULL METRIC EVALUATION (metrics.py)")
    logger.info("=" * 60)

    # Collect patient case directories
    patient_dirs = [
        p
        for p in eval_preds_dir.iterdir()
        if p.is_dir() and p.name != "Visualizations"
    ]
    if not patient_dirs:
        logger.error(f"No patient case directories found in {eval_preds_dir}")
        return {}

    eval_results = []
    gts_agatston = []
    preds_agatston = []

    for patient_dir in patient_dirs:
        case_id = patient_dir.name
        label_path = patient_dir / "label.nii.gz"
        pred_path = patient_dir / "prediction.nii.gz"
        ct_path = patient_dir / "image.nii.gz"

        if not (label_path.exists() and pred_path.exists()):
            logger.warning(
                f"  [skip] Missing prediction/label for: {case_id}"
            )
            continue

        # Execute voxel, plaque matching, HD95, and Agatston metrics
        voxel = voxel_wise_f1(label_path, pred_path)
        plaque = plaque_wise_f1(label_path, pred_path)
        hd95_mm = compute_hd95_mm(label_path, pred_path)
        gt_agat = compute_agatston_score(label_path, ct_path)
        pr_agat = compute_agatston_score(pred_path, ct_path)

        gts_agatston.append(gt_agat["agatston_total"])
        preds_agatston.append(pr_agat["agatston_total"])

        pq_score = plaque["macro_dice"] * voxel["dice"]

        rule = "-" * 60
        logger.info(f"\n{rule}")
        logger.info(f"  Scan: {case_id}")
        logger.info(f"{rule}")
        logger.info(
            f"  Voxel-wise  : F1={voxel['f1']:.4f}  P={voxel['precision']:.4f}  "
            f"R={voxel['recall']:.4f}    (TP={voxel['tp']:,}  FP={voxel['fp']:,}  FN={voxel['fn']:,}  TN={voxel['tn']:,})"
        )
        logger.info(
            f"  Plaque-wise : F1={plaque['f1']:.4f}  P={plaque['precision']:.4f}  "
            f"R={plaque['recall']:.4f} "
            f"(GT={plaque['n_gt_plaques']}, Pred={plaque['n_pred_plaques']}, "
            f"matched={plaque['n_matched']}, "
            f"unmatched_GT={plaque['n_unmatched_gt']}, "
            f"unmatched_Pred={plaque['n_unmatched_pred']})"
        )
        logger.info(
            f"  Dice        : {voxel['dice']:.4f}\n"
            f"  HD95 (mm)   : {hd95_mm:.4f}\n"
            f"  MacroDice   : {plaque['macro_dice']:.4f}\n"
            f"  PQ Score    : {pq_score:.4f}  "
        )
        logger.info(
            f"  Agatston    : GT={gt_agat['agatston_total']:7.2f}  "
            f"Pred={pr_agat['agatston_total']:7.2f}  "
            f"d={(pr_agat['agatston_total'] - gt_agat['agatston_total']):+.2f}"
        )

        eval_results.append({
            "scan_id": case_id,
            "voxel": voxel,
            "plaque": plaque,
            "hd95_mm": hd95_mm,
            "pq_score": pq_score,
            "gt_agatston": gt_agat,
            "pred_agatston": pr_agat,
        })

    # Aggregated Agatston metrics & Confusion Matrix output
    score_metrics = {}
    if gts_agatston and preds_agatston:
        score_metrics = compute_basic_metrics_for_score(
            gts_agatston, preds_agatston
        )
        reg = score_metrics["regression"]
        cat = score_metrics["category"]
        agr = score_metrics["agreement"]

        rule = "=" * 60
        logger.info(f"\n{rule}")
        logger.info(f"  Score-level Aggregate (n={len(eval_results)})")
        logger.info(f"{rule}")
        logger.info(
            f"  Regression  : MAE={reg['mae']:.2f}  RMSE={reg['rmse']:.2f}  "
            f"MAPE={reg['mape']:.2f}%  r={reg['pearson_r']:.4f}  "
            f"ρ={reg['spearman_r']:.4f}  R²={reg['r2']:.4f}"
        )
        logger.info(
            f"  Category    : F1(macro)={cat['f1_macro']:.4f}  "
            f"F1(weighted)={cat['f1_weighted']:.4f}  "
            f"Acc={cat['accuracy']:.4f}"
        )
        logger.info(
            f"  Agreement   : κ_quadratic={agr['kappa_quadratic']:.4f}  "
            f"κ_linear={agr['kappa_linear']:.4f}  "
            f"exact={agr['exact_agreement']:.4f}"
        )

        output_confusion_matrix(
            cm=cat["confusion_matrix"],
            categories=cat["categories"],
            out_dir=visualizations_dir,
            filename="risk_category_confusion_matrix",
            normalize=False,
        )

        logger.info(
            f"\n  Confusion matrix saved to {visualizations_dir}"
        )
        logger.info(
            "  PQ Score is product of Recognition Quality (Plaque wise F1) and Segmentation Quality for Matched Lesions (Macro Dice)"
        )

    # Filter out inf/nan values when computing average HD95 across valid scans
    valid_hd95s = [
        r["hd95_mm"]
        for r in eval_results
        if not np.isinf(r["hd95_mm"]) and not np.isnan(r["hd95_mm"])
    ]

    mean_voxel_dice = (
        float(np.mean([r["voxel"]["dice"] for r in eval_results]))
        if eval_results
        else 0.0
    )
    mean_macro_dice = (
        float(np.mean([r["plaque"]["macro_dice"] for r in eval_results]))
        if eval_results
        else 0.0
    )
    mean_plaque_f1 = (
        float(np.mean([r["plaque"]["f1"] for r in eval_results]))
        if eval_results
        else 0.0
    )
    mean_pq_score = (
        float(np.mean([r["pq_score"] for r in eval_results]))
        if eval_results
        else 0.0
    )
    mean_hd95_mm = float(np.mean(valid_hd95s)) if valid_hd95s else 0.0

    return {
        "mean_voxel_dice": mean_voxel_dice,
        "mean_plaque_f1": mean_plaque_f1,
        "mean_pq_score": mean_pq_score,
        "mean_hd95_mm": mean_hd95_mm,
        "score_metrics": score_metrics,
        "mean_macro_dice": mean_macro_dice,
        "total_cases": len(eval_results),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate RIUnet on val or test split."
    )
    ap.add_argument(
        "experiment_name", type=str, help="Name of experiment in runs/"
    )
    ap.add_argument(
        "--split",
        type=str,
        choices=["val", "test"],
        default="val",
        help="Dataset split to evaluate on",
    )
    ap.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Evaluate on only the first N samples",
    )
    ap.add_argument(
        "--checkpoint",
        type=str,
        default="best_dice.pth",
        help="Checkpoint filename inside run/ckpts/",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Automatically overwrite existing predictions without prompting",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_paths = setup_eval_dirs(args.experiment_name, split_name=args.split)
    logger = get_logger(run_paths["logs"] / f"eval_{args.split}.log")

    logger.info(
        f"=== eval.py | experiment: {args.experiment_name} | split: {args.split} | device: {device} === | Threshold: {cfg.train_config['VAL_THRESHOLD']}"
    )
    logger.info(
        f"PostProcessing Includes: POST_PROCESS_BIG_LESIONS={cfg.eval_config.get('POST_PROCESS_BIG_LESIONS', True)}, APPLY_ROI_MASK={cfg.eval_config.get('APPLY_ROI_MASK', False)}, MORPHOLOGICAL_CLEANUP={cfg.eval_config.get('MORPHOLOGICAL_CLEANUP', False)}, ANISTROPIC_GRAPH_CUTTING={cfg.eval_config.get('ANISTROPIC_GRAPH_CUTTING', False)}"
    )

    # Check for existing prediction folders to prompt for overwrite permission
    eval_preds_dir = run_paths["eval_preds"]
    existing_cases = [
        p
        for p in eval_preds_dir.iterdir()
        if p.is_dir() and p.name != "Visualizations"
    ]

    should_generate = True
    if existing_cases and not args.overwrite:
        print(
            f"\n[?] Found {len(existing_cases)} existing prediction(s) in {eval_preds_dir}."
        )
        resp = (
            input(
                "    Do you want to re-generate predictions and overwrite them? (y/N): "
            )
            .strip()
            .lower()
        )
        if resp not in ["y", "yes"]:
            should_generate = False
            logger.info(
                "Skipping prediction generation phase. Running metrics evaluation on existing files..."
            )

    proc = psutil.Process()
    cpu_t0, t0 = proc.cpu_times(), time.perf_counter()
    inference_time_s = 0.0
    ckpt_epoch = -1

    if should_generate:
        logger.info("Building dataloaders...")
        train_loader, val_loader, test_loader = build_dataloaders(
            str(SPLITS_JSON)
        )

        cfg.model_config["IN_CHANNELS"] = auto_detect_in_channels(train_loader)
        logger.info(
            f"Auto-detected IN_CHANNELS = {cfg.model_config['IN_CHANNELS']}"
        )

        loader = val_loader if args.split == "val" else test_loader
        if loader is None:
            logger.error(
                f"Requested split '{args.split}' returned None. Check dataset splits config."
            )
            sys.exit(1)

        model = RIUnet(in_channels=cfg.model_config["IN_CHANNELS"]).to(device)

        ckpt_path = run_paths["ckpts"] / args.checkpoint
        if not ckpt_path.exists():
            logger.error(f"Checkpoint not found at: {ckpt_path}")
            sys.exit(1)

        logger.info(f"Loading checkpoint state from: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        ckpt_epoch = state.get("epoch", -1)
        logger.info(
            f"Successfully loaded checkpoint trained up to epoch {ckpt_epoch}."
        )

        cfg_train = dict(cfg.train_config)

        # Generate and save predictions to disk
        inference_time_s = generate_predictions(
            model=model,
            loader=loader,
            device=device,
            cfg_train=cfg_train,
            logger=logger,
            eval_preds_dir=eval_preds_dir,
            num_samples=args.num_samples,
        )

    # Run metrics on saved predictions directory
    summary_eval = evaluate_metrics(
        eval_preds_dir=eval_preds_dir,
        visualizations_dir=run_paths["visualizations"],
        logger=logger,
    )

    t1, cpu_t1 = time.perf_counter(), proc.cpu_times()
    total_time = t1 - t0
    cpu_time = (cpu_t1.user - cpu_t0.user) + (cpu_t1.system - cpu_t0.system)
    peak_gpu_mem = (
        torch.cuda.max_memory_allocated() / 1e6
        if device.type == "cuda"
        else 0.0
    )

    logger.info("\n" + "=" * 60)
    logger.info(f" EVALUATION SUMMARY ({args.split.upper()} SPLIT)")
    logger.info("=" * 60)
    logger.info(
        f"  Total Volumes Evaluated : {summary_eval.get('total_cases', 0)}"
    )
    logger.info(
        f"  Mean Voxel Dice Score   : {summary_eval.get('mean_voxel_dice', 0.0):.4f}"
    )
    logger.info(
        f"  Mean Plaque-wise F1     : {summary_eval.get('mean_plaque_f1', 0.0):.4f}"
    )
    logger.info(
        f"  Mean Macro Dice Score        : {summary_eval.get('mean_macro_dice', 0.0):.4f}"
    )
    logger.info(
        f"  Mean PQ Score           : {summary_eval.get('mean_pq_score', 0.0):.4f}"
    )
    logger.info(
        f"  Mean HD95 (mm)          : {summary_eval.get('mean_hd95_mm', 0.0):.4f}"
    )
    logger.info(f"  Wall Time               : {total_time:.2f} s")
    logger.info(f"  CPU Time                : {cpu_time:.2f} s")
    logger.info(f"  Peak GPU Memory         : {peak_gpu_mem:.1f} MB")
    logger.info("=" * 60)

    # Custom JSON serializer for NumPy and float/inf compatibility
    def json_serializer(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, float)):
            if np.isinf(x) or np.isnan(x):
                return None  # Replaces inf/nan with null in JSON
            return float(x)
        return str(x)

    # Dump full dictionary to JSON
    out_file = eval_preds_dir / "summary_metrics.json"
    dump_data = {
        "experiment_name": args.experiment_name,
        "split": args.split,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": ckpt_epoch,
        "num_samples": args.num_samples,
        "summary_metrics": summary_eval,
        "system_stats": {
            "wall_time_s": total_time,
            "inference_time_s": inference_time_s,
            "cpu_time_s": cpu_time,
            "peak_gpu_mem_mb": peak_gpu_mem,
        },
    }
    with open(out_file, "w") as f:
        json.dump(dump_data, f, indent=2, default=json_serializer)
    logger.info(f"Saved evaluation summary to {out_file}")


if __name__ == "__main__":
    main()