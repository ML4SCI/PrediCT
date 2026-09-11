import logging
from pathlib import Path
import time
import traceback
import numpy as np
from pathlib import Path


# Adjust imports to match your project directory structure
from metrics import (
    compute_agatston_score,
    compute_basic_metrics_for_score,
    plaque_wise_f1,
    voxel_wise_f1,
)
from wrapper import predict_single_nifti


def setup_logger(log_file_path: Path):
    logger = logging.getLogger(str(log_file_path))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")

    formatter = logging.Formatter("%(message)s")
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger, f_handler


def run_batch_pipeline(image_paths: list, label_paths: list, riunet_ckpt_path: str):
    results = []

    all_dices = []
    all_lesion_dices = []
    all_lesion_f1s = []
    all_lesion_pqs = []
    gt_agatston_scores = []
    pred_agatston_scores = []

    for img_path_str, label_path_str in zip(image_paths, label_paths):
        img_path = Path(img_path_str).resolve()
        label_path = Path(label_path_str).resolve()

        patient_dir = img_path.parent
        patient_id = patient_dir.name

        # Target file: id_137_pred.nii.gz
        target_pred_path = patient_dir / f"{patient_id}_pred.nii.gz"

        log_file_path = patient_dir / "evaluation_results.txt"
        logger, f_handler = setup_logger(log_file_path)

        logger.info("=" * 60)
        logger.info(f"Processing Patient: {patient_id}")
        logger.info("=" * 60)

        try:
            # 1. Run single prediction
            t0 = time.perf_counter()
            generated_out_str = predict_single_nifti(
                input_nifti_path=str(img_path),
                output_dir=str(patient_dir),
                riunet_ckpt_path=riunet_ckpt_path,
            )

            # Atomic replace (Windows-safe file overwriting)
            generated_pred_path = Path(generated_out_str).resolve()
            if (
                generated_pred_path.exists()
                and generated_pred_path != target_pred_path
            ):
                generated_pred_path.replace(target_pred_path)

            t_inference = time.perf_counter() - t0

            # 2. Compute metrics
            t0 = time.perf_counter()
            voxel = voxel_wise_f1(label_path, target_pred_path)
            plaque = plaque_wise_f1(label_path, target_pred_path)
            gt_agat = compute_agatston_score(label_path, img_path)
            pr_agat = compute_agatston_score(target_pred_path, img_path)

            lesion_pq = plaque["f1"] * plaque["macro_dice"]
            t_eval = time.perf_counter() - t0

            # Accumulate metrics
            all_dices.append(voxel["dice"])
            all_lesion_dices.append(plaque["macro_dice"])
            all_lesion_f1s.append(plaque["f1"])
            all_lesion_pqs.append(lesion_pq)
            gt_agatston_scores.append(gt_agat["agatston_total"])
            pred_agatston_scores.append(pr_agat["agatston_total"])

            logger.info(f"Prediction Saved To : {target_pred_path}")
            logger.info(f"Voxel-wise Dice    : {voxel['dice']:.4f}")
            logger.info(f"Plaque-wise F1     : {plaque['f1']:.4f}")
            logger.info(f"Plaque Macro Dice  : {plaque['macro_dice']:.4f}")
            logger.info(f"Plaque PQ Score    : {lesion_pq:.4f}")
            logger.info(
                f"Agatston Score     : GT={gt_agat['agatston_total']:.2f} | Pred={pr_agat['agatston_total']:.2f}"
            )
            logger.info(
                f"Inference Time     : {t_inference:.3f} s | Eval Time: {t_eval:.3f} s\n"
            )

            results.append(patient_id)

        except Exception as e:
            logger.info(f"[ERROR] Failed processing patient {patient_id}: {e}")
            logger.info(traceback.format_exc())

        finally:
            f_handler.close()
            logger.removeHandler(f_handler)

    # 3. Output Final Dataset Aggregates
    avg_dice = np.mean(all_dices) if all_dices else 0.0
    avg_lesion_dice = np.mean(all_lesion_dices) if all_lesion_dices else 0.0
    avg_lesion_f1 = np.mean(all_lesion_f1s) if all_lesion_f1s else 0.0
    avg_lesion_pq = np.mean(all_lesion_pqs) if all_lesion_pqs else 0.0

    score_metrics = compute_basic_metrics_for_score(
        gt_agatston_scores, pred_agatston_scores
    )
    reg = score_metrics["regression"]
    cat = score_metrics["category"]
    agr = score_metrics["agreement"]

    formatted_summary = f"""Average Dice:                 \t\t {avg_dice:.4f}
Average Lesion Dice:               \t {avg_lesion_dice:.4f}\t
Average Lesion F1:            \t\t {avg_lesion_f1:.4f}
Average Lesion Panoptic Quality (PQ):    {avg_lesion_pq:.4f}

============================================================
  Score-level aggregate (n={len(results)})
============================================================
  Regression  : MAE={reg['mae']:.2f}  RMSE={reg['rmse']:.2f}  MAPE={reg['mape']:.2f}%  r={reg['pearson_r']:.4f}  ρ={reg['spearman_r']:.4f}  R²={reg['r2']:.4f}
  Category    : F1(macro)={cat['f1_macro']:.4f}  F1(weighted)={cat['f1_weighted']:.4f}  Acc={cat['accuracy']:.4f}
  Agreement   : κ_quadratic={agr['kappa_quadratic']:.4f}  κ_linear={agr['kappa_linear']:.4f}  exact={agr['exact_agreement']:.4f}"""

    # Print to console
    print(formatted_summary)

    # Save to overall_results.txt in dataset root directory
    if image_paths:
        dataset_root = Path(image_paths[0]).resolve().parent.parent
        summary_file = dataset_root / "overall_results.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(formatted_summary)
        print(f"\nSaved aggregated summary to: {summary_file}")


if __name__ == "__main__":
  # Set the base directory relative to your Linux environment
  BASE_DIR = Path("/pscratch/sd/r/rajatns/gsoc_2026_official/try1/data_resampled_(0.7_0.7_3.0)")

  # Dynamically gather image and label paths
  patient_dirs = sorted([d for d in BASE_DIR.glob("id_*") if d.is_dir()])

  image_paths_0_5 = [str(d / f"{d.name}_img.nii.gz") for d in patient_dirs]
  label_paths_0_5 = [str(d / f"{d.name}_label.nii.gz") for d in patient_dirs]

  print(f"Found {len(image_paths_0_5)} image-label pairs for evaluation.")

  # Set the checkpoint path
  CKPT_PATH = "/pscratch/sd/r/rajatns/gsoc_2026_official/try1/Segmentation_Rajat/runs/run_fno/ckpts/best_dice.pth"

  print(f"Using checkpoint: {CKPT_PATH}")

  run_batch_pipeline(
    image_paths=image_paths_0_5,
    label_paths=label_paths_0_5,
    riunet_ckpt_path=CKPT_PATH,
  )