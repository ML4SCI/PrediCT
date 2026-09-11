"""
run_batch_eval.py

Batch-evaluate voxel-wise, plaque-wise, and Agatston-score metrics over
every patient in a GROUPED root directory structured as:

    GROUPED/
        <patient_id_1>/
            image.nii.gz
            label.nii.gz
            prediction.nii.gz
        <patient_id_2>/
            image.nii.gz
            label.nii.gz
            prediction.nii.gz
        ...

Set GROUPED_DIR and OUTPUT_DIR below, then run:

    python run_batch_eval.py
"""

import csv
import time
import traceback
from pathlib import Path

from metrics import (
    voxel_wise_f1,
    plaque_wise_f1,
    compute_agatston_score,
    compute_basic_metrics_for_score,
    output_confusion_matrix,
)


# ══════════════════════════════════════════════════════════════════
#  PATHS — set these two before running
# ══════════════════════════════════════════════════════════════════

GROUPED_DIR = Path(r"E:\MyProjects\Gsoc_2026_Official\nnunet_simple_scv\GROUPED")
OUTPUT_DIR  = Path(r"E:\MyProjects\Gsoc_2026_Official\nnunet_simple_scv\Visualizations")

REQUIRED_FILES = ("image.nii.gz", "label.nii.gz", "prediction.nii.gz")


# ══════════════════════════════════════════════════════════════════
#  DISCOVERY — find every valid patient folder under GROUPED_DIR
# ══════════════════════════════════════════════════════════════════

def discover_patient_ids(grouped_dir: Path) -> list:
    """Find every subfolder of `grouped_dir` that has all 3 required files.

    Skips (with a warning) any subfolder missing one of the required
    files rather than crashing the whole batch run.
    """
    patient_ids = []
    skipped = []
    for entry in sorted(grouped_dir.iterdir()):
        if not entry.is_dir():
            continue
        missing = [f for f in REQUIRED_FILES if not (entry / f).exists()]
        if missing:
            skipped.append((entry.name, missing))
            continue
        patient_ids.append(entry.name)

    if skipped:
        print(f"  [warn] skipping {len(skipped)} folder(s) missing required files:")
        for name, missing in skipped:
            print(f"           - {name}: missing {missing}")

    return patient_ids


# ══════════════════════════════════════════════════════════════════
#  PER-PATIENT EVAL — mirrors metrics.py's _run_full_eval
# ══════════════════════════════════════════════════════════════════

def _run_full_eval(patient_id: str, root: Path) -> dict:
    """Run voxel-, plaque-, and score-level metrics on one patient.

    Same shape/keys as metrics.py's _run_full_eval, plus "plaque_pq".
    """
    patient_dir = root / patient_id
    label_path  = patient_dir / "label.nii.gz"
    pred_path   = patient_dir / "prediction.nii.gz"
    ct_path     = patient_dir / "image.nii.gz"

    voxel   = voxel_wise_f1(label_path, pred_path)
    plaque  = plaque_wise_f1(label_path, pred_path)
    gt_agat = compute_agatston_score(label_path, ct_path)
    pr_agat = compute_agatston_score(pred_path, ct_path)

    # Plaque PQ = Recognition Quality (plaque-wise F1) x Segmentation
    # Quality (plaque-wise Macro Dice), both at the PLAQUE level.
    plaque_pq = plaque["f1"] * plaque["macro_dice"]

    return {
        "scan_id":       patient_id,
        "voxel":         voxel,
        "plaque":        plaque,
        "plaque_pq":     plaque_pq,
        "gt_agatston":   gt_agat,
        "pred_agatston": pr_agat,
    }


# ══════════════════════════════════════════════════════════════════
#  PRINTING — same layout as metrics.py's _print_eval
# ══════════════════════════════════════════════════════════════════

def _print_eval(result: dict) -> None:
    scan_id = result["scan_id"]
    v       = result["voxel"]
    p       = result["plaque"]
    g       = result["gt_agatston"]
    pr      = result["pred_agatston"]

    rule = "=" * 60
    print(f"\n{rule}")
    print(f"  Scan: {scan_id}")
    print(f"{rule}")
    print(
        f"  Voxel-wise  : F1={v['f1']:.4f}  P={v['precision']:.4f}  "
        f"R={v['recall']:.4f}    (TP={v['tp']:,}  FP={v['fp']:,}  FN={v['fn']:,}  TN={v['tn']:,})"
    )
    print(
        f"  Plaque-wise : F1={p['f1']:.4f}  P={p['precision']:.4f}  "
        f"R={p['recall']:.4f} "
        f"(GT={p['n_gt_plaques']}, Pred={p['n_pred_plaques']}, "
        f"matched={p['n_matched']}, "
        f"unmatched_GT={p['n_unmatched_gt']}, "
        f"unmatched_Pred={p['n_unmatched_pred']})"
    )
    print(
        f"  Dice        : {v['dice']:.4f}\n"
        f"  MacroDice   : {p['macro_dice']:.4f}\n"
        f"  PQ Score    : {result['plaque_pq']:.4f}"
    )
    print(
        f"  Agatston    : GT={g['agatston_total']:7.2f}  "
        f"Pred={pr['agatston_total']:7.2f}  "
        f"d={(pr['agatston_total'] - g['agatston_total']):+.2f}"
    )


# ══════════════════════════════════════════════════════════════════
#  CSV — one row per patient
# ══════════════════════════════════════════════════════════════════

def write_per_patient_csv(results: list, out_path: Path) -> None:
    fieldnames = [
        "patient_id",
        "voxel_f1", "voxel_precision", "voxel_recall", "voxel_dice",
        "voxel_tp", "voxel_fp", "voxel_fn", "voxel_tn",
        "plaque_f1", "plaque_precision", "plaque_recall", "plaque_macro_dice",
        "plaque_pq",
        "n_gt_plaques", "n_pred_plaques", "n_matched",
        "n_unmatched_gt", "n_unmatched_pred",
        "gt_agatston", "pred_agatston", "agatston_diff",
        "gt_n_lesions", "pred_n_lesions",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            v, p = r["voxel"], r["plaque"]
            g, pr = r["gt_agatston"], r["pred_agatston"]
            writer.writerow({
                "patient_id":        r["scan_id"],
                "voxel_f1":          v["f1"],
                "voxel_precision":   v["precision"],
                "voxel_recall":      v["recall"],
                "voxel_dice":        v["dice"],
                "voxel_tp":          v["tp"],
                "voxel_fp":          v["fp"],
                "voxel_fn":          v["fn"],
                "voxel_tn":          v["tn"],
                "plaque_f1":         p["f1"],
                "plaque_precision":  p["precision"],
                "plaque_recall":     p["recall"],
                "plaque_macro_dice": p["macro_dice"],
                "plaque_pq":         r["plaque_pq"],
                "n_gt_plaques":      p["n_gt_plaques"],
                "n_pred_plaques":    p["n_pred_plaques"],
                "n_matched":         p["n_matched"],
                "n_unmatched_gt":    p["n_unmatched_gt"],
                "n_unmatched_pred":  p["n_unmatched_pred"],
                "gt_agatston":       g["agatston_total"],
                "pred_agatston":     pr["agatston_total"],
                "agatston_diff":     pr["agatston_total"] - g["agatston_total"],
                "gt_n_lesions":      g["n_lesions"],
                "pred_n_lesions":    pr["n_lesions"],
            })

    print(f"\n  [saved] per-patient metrics -> {out_path}")


# ══════════════════════════════════════════════════════════════════
#  MAIN — batch loop over GROUPED_DIR
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rule = "=" * 60

    print("=" * 60)
    print("  run_batch_eval.py - batch check over GROUPED/")
    print(f"  GROUPED_DIR = {GROUPED_DIR}")
    print(f"  OUTPUT_DIR  = {OUTPUT_DIR}")
    print("=" * 60)

    patient_ids = discover_patient_ids(GROUPED_DIR)
    print(f"\n  Found {len(patient_ids)} valid patient folder(s).")

    results = []
    failed = []
    t_total = time.perf_counter()

    for scan_id in patient_ids:
        try:
            t0 = time.perf_counter()
            result = _run_full_eval(scan_id, GROUPED_DIR)
            dt = time.perf_counter() - t0
            _print_eval(result)
            print(f"  (took {dt:.2f}s)")
            results.append(result)
        except Exception as e:
            print(f"\n  [ERROR] {scan_id}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed.append(scan_id)

    t_total = time.perf_counter() - t_total

    print(f"\n{rule}")
    print(f"  Completed {len(results)}/{len(patient_ids)} patients "
          f"({len(failed)} failed) in {t_total:.1f}s")
    if failed:
        print(f"  Failed IDs: {failed}")

    # ── Save per-patient CSV ───────────────────────────────────
    if results:
        write_per_patient_csv(results, OUTPUT_DIR / "per_patient_metrics.csv")

    # ── Aggregate score-level metrics across all patients ──────
    if results:
        gts   = [r["gt_agatston"]["agatston_total"]   for r in results]
        preds = [r["pred_agatston"]["agatston_total"] for r in results]
        score_metrics = compute_basic_metrics_for_score(gts, preds)

        reg = score_metrics["regression"]
        cat = score_metrics["category"]
        agr = score_metrics["agreement"]

        print(f"\n{rule}")
        print(f"  Score-level aggregate (n={len(results)})")
        print(f"{rule}")
        print(
            f"  Regression  : MAE={reg['mae']:.2f}  RMSE={reg['rmse']:.2f}  "
            f"MAPE={reg['mape']:.2f}%  r={reg['pearson_r']:.4f}  "
            f"ρ={reg['spearman_r']:.4f}  R²={reg['r2']:.4f}"
        )
        print(
            f"  Category    : F1(macro)={cat['f1_macro']:.4f}  "
            f"F1(weighted)={cat['f1_weighted']:.4f}  "
            f"Acc={cat['accuracy']:.4f}"
        )
        print(
            f"  Agreement   : κ_quadratic={agr['kappa_quadratic']:.4f}  "
            f"κ_linear={agr['kappa_linear']:.4f}  "
            f"exact={agr['exact_agreement']:.4f}"
        )

        # ── Confusion matrix (raw + normalized) ─────────────────
        if cat["confusion_matrix"] is not None:
            output_confusion_matrix(
                cm=cat["confusion_matrix"],
                categories=cat["categories"],
                out_dir=OUTPUT_DIR,
                filename="risk_category_confusion_matrix_normalized",
                normalize=True,
            )

        print(f"\n  (full eval took {t_total:.2f}s)")

        print(f"\n  PQ Score is best for clinical evaluation — it is the "
              f"product of Recognition Quality (Plaque-wise F1) and "
              f"Segmentation Quality (Plaque-wise Macro Dice)")