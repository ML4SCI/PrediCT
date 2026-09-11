"""
eval.py — Evaluate a trained RIUnet checkpoint on val or test split.

CLI:
    python Segmentation_Rajat/eval.py EXPERIMENT_NAME
    python Segmentation_Rajat/eval.py EXPERIMENT_NAME --split val
    python Segmentation_Rajat/eval.py EXPERIMENT_NAME --split test --agatston
    python Segmentation_Rajat/eval.py EXPERIMENT_NAME --ckpt ckpts/epoch_050.pth

Per-case layout written under runs/<EXPERIMENT_NAME>/eval_preds/<split>/<id>/:
    image.nii.gz
    label.nii.gz
    prediction.nii.gz

Also writes:
    eval_preds/<split>/metrics.csv         (per-id rows)
    eval_preds/<split>/metrics_summary.json (aggregate mean/std)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference

import config as cfg
from dataset import build_dataloaders
from model import RIUnet


# ══════════════════════════════════════════════════════════════════
#  Paths / logging
# ══════════════════════════════════════════════════════════════════

BASE_DIR  = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
RUNS_DIR  = BASE_DIR / "runs"
SPLITS_JSON = Path(cfg.preprocessing_config["SPLITS_JSON"])

# Reuse the project's existing metric library (voxel F1, plaque F1, Agatston, …)
METRICS_DIR = REPO_ROOT / "test_nnunet_2"
if str(METRICS_DIR) not in sys.path:
    sys.path.insert(0, str(METRICS_DIR))
import metrics as M  # noqa: E402


def get_logger(log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger("eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(fh)
    return logger


# ══════════════════════════════════════════════════════════════════
#  NIfTI I/O with original metadata preserved
# ══════════════════════════════════════════════════════════════════

def _infer_nii_path(case: dict) -> Optional[Path]:
    """Pull the original image .nii.gz path from a dataset entry."""
    p = case.get("image")
    if p is None:
        return None
    return Path(p)


def _save_nii_like(src_path: Path, arr: np.ndarray, out_path: Path) -> None:
    """Write `arr` to `out_path` copying spacing/origin/direction from `src_path`."""
    src = sitk.ReadImage(str(src_path))
    img = sitk.GetImageFromArray(arr.astype(np.uint8))
    img.SetSpacing(src.GetSpacing())
    img.SetOrigin(src.GetOrigin())
    img.SetDirection(src.GetDirection())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_path), useCompression=True)


def _load_nii_as_array(path: Path) -> np.ndarray:
    """Load a .nii.gz as a numpy array (uint8, 0/1)."""
    arr = nib.load(str(path)).get_fdata().astype(np.uint8)
    return arr


# ══════════════════════════════════════════════════════════════════
#  Inference
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_one_case(
    model:  nn.Module,
    batch:  dict,
    device: torch.device,
    roi_size: tuple,
) -> np.ndarray:
    """Run sliding-window inference on one case and return a uint8 prediction array."""
    x = batch["image"].to(device, non_blocking=True)  # (1, C, D, H, W)
    logits = sliding_window_inference(
        inputs       = x,
        roi_size     = roi_size,
        sw_batch_size= 1,
        overlap      = 0.5,
        predictor    = model,
    )
    pred = logits.argmax(dim=1, keepdim=True)
    # argmax gives 0 (bg) or 1 (calcium); binarize just to be safe
    return (pred.squeeze(0).squeeze(0).cpu().numpy() > 0).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════
#  Per-case metric computation
# ══════════════════════════════════════════════════════════════════

def compute_case_metrics(
    pred_path: Path,
    label_path: Path,
    image_path: Optional[Path],
    do_agatston: bool,
) -> Dict[str, float]:
    """Compute per-case metrics using the project's metrics module."""
    row: Dict[str, float] = {}

    # 1) Voxel-wise F1 / Dice
    try:
        vw = M.voxel_wise_f1(str(label_path), str(pred_path))
        row["voxel_f1"]   = float(vw.get("f1",          float("nan")))
        row["voxel_dice"] = float(vw.get("dice",        float("nan")))
        row["voxel_p"]    = float(vw.get("precision",   float("nan")))
        row["voxel_r"]    = float(vw.get("recall",      float("nan")))
    except Exception as e:
        row["voxel_error"] = str(e)

    # 2) Plaque-wise F1 / macro-Dice
    try:
        pw = M.plaque_wise_f1(str(label_path), str(pred_path))
        row["plaque_f1"]      = float(pw.get("f1",         float("nan")))
        row["plaque_p"]       = float(pw.get("precision",  float("nan")))
        row["plaque_r"]       = float(pw.get("recall",     float("nan")))
        row["plaque_macro_dice"] = float(pw.get("macro_dice", float("nan")))
        row["plaque_pq"]      = (
            row["plaque_f1"] * row["plaque_macro_dice"]
            if not np.isnan(row["plaque_f1"]) and not np.isnan(row["plaque_macro_dice"])
            else float("nan")
        )
    except Exception as e:
        row["plaque_error"] = str(e)

    # 3) Agatston (if asked + we have the original CT)
    if do_agatston and image_path is not None and image_path.exists():
        try:
            ag = M.compute_agatston_score(str(pred_path), str(image_path))
            row["agatston_total"] = float(ag.get("agatston_total", float("nan")))
            row["n_lesions"]      = int(ag.get("n_lesions", 0))
            row["n_slices"]       = int(ag.get("n_slices_with_calcium", 0))
        except Exception as e:
            row["agatston_error"] = str(e)

    return row


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_name", help="Run name; reads runs/<name>/ckpts/")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--ckpt",  default=None,
                    help="Override checkpoint path. Defaults to best_dice.pth, "
                         "falls back to last.pth.")
    ap.add_argument("--out-dir", default=None,
                    help="Override output dir. Defaults to runs/<name>/eval_preds/<split>/")
    ap.add_argument("--agatston", action="store_true",
                    help="Also compute Agatston score per case.")
    ap.add_argument("--max-cases", type=int, default=None,
                    help="Cap on number of cases (for debugging).")
    args = ap.parse_args()

    run_dir = RUNS_DIR / args.experiment_name
    ckpt_dir = run_dir / "ckpts"
    out_dir  = Path(args.out_dir) if args.out_dir else (run_dir / "eval_preds" / args.split)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(out_dir / "eval.log")
    logger.info(f"=== eval.py  |  experiment: {args.experiment_name}  |  split: {args.split} ===")

    # ── Pick checkpoint ─────────────────────────────────────
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        best = ckpt_dir / "best_dice.pth"
        last = ckpt_dir / "last.pth"
        ckpt_path = best if best.exists() else last
    if not ckpt_path.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt_path}")
    logger.info(f"loading checkpoint: {ckpt_path}")

    # ── Model ───────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Channel count: we need to know it before we can build RIUnet. The
    # checkpoint carries it under model_config["IN_CHANNELS"], but that
    # field was set in train.py and is a hard requirement. To stay safe,
    # we also auto-detect from a sample batch.
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_config" in state and state["model_config"].get("IN_CHANNELS"):
        cfg.model_config["IN_CHANNELS"] = int(state["model_config"]["IN_CHANNELS"])

    model = RIUnet(in_channels=cfg.model_config.get("IN_CHANNELS")).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    logger.info(f"RIUnet loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    # ── Dataloader (only the requested split) ───────────────
    train_loader, val_loader, test_loader = build_dataloaders(str(SPLITS_JSON))
    loader = val_loader if args.split == "val" else test_loader
    base_ds = loader.dataset
    if hasattr(base_ds, "data"):
        n_total = len(base_ds.data)
    else:
        n_total = len(base_ds)
    if args.max_cases is not None:
        n_total = min(n_total, args.max_cases)
    logger.info(f"split '{args.split}' has {n_total} cases (max_cases={args.max_cases})")

    # ── Per-case inference + metrics ───────────────────────
    roi_size = tuple(cfg.model_config["ROI_SIZE"])
    rows: List[Dict] = []

    for idx, batch in enumerate(loader):
        if idx >= n_total:
            break
        case = base_ds.data[idx] if hasattr(base_ds, "data") else {}
        case_id = case.get("id") or case.get("image", f"case_{idx}")
        case_id = str(Path(case_id).stem)

        t0 = time.time()
        pred_arr = predict_one_case(model, batch, device, roi_size)
        dt = time.time() - t0

        case_dir = out_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        # Save the prediction with the original image's metadata
        img_path = _infer_nii_path(case)
        if img_path is not None and img_path.exists():
            try:
                _save_nii_like(img_path, pred_arr, case_dir / "prediction.nii.gz")
                _save_nii_like(img_path, _load_nii_as_array(img_path), case_dir / "image.nii.gz")
            except Exception as e:
                logger.warning(f"  [{case_id}] could not save image copy: {e}")
        else:
            # Fallback: save pred as plain nibabel
            nib.save(nib.Nifti1Image(pred_arr.astype(np.uint8), np.eye(4)), str(case_dir / "prediction.nii.gz"))

        # Save the label too (already on disk from preprocessing)
        lbl_path = case.get("label")
        if lbl_path is not None and Path(lbl_path).exists():
            try:
                _save_nii_like(Path(lbl_path), _load_nii_as_array(Path(lbl_path)),
                               case_dir / "label.nii.gz")
            except Exception as e:
                logger.warning(f"  [{case_id}] could not save label copy: {e}")

        # Compute metrics (on the COPIES we just wrote, so paths are guaranteed
        # to have the same metadata — keeps nibabel happy)
        row = compute_case_metrics(
            pred_path  = case_dir / "prediction.nii.gz",
            label_path = case_dir / "label.nii.gz",
            image_path = Path(case.get("image", "")) if case.get("image") else None,
            do_agatston= args.agatston,
        )
        row["id"]   = case_id
        row["time_s"] = round(dt, 3)
        rows.append(row)

        logger.info(
            f"  [{idx+1}/{n_total}] {case_id}  voxel_dice={row.get('voxel_dice', float('nan')):.4f}  "
            f"plaque_f1={row.get('plaque_f1',  float('nan')):.4f}  {dt:.1f}s"
        )

    # ── Aggregate + persist ─────────────────────────────────
    if not rows:
        logger.warning("no cases processed; nothing to write")
        return

    # CSV
    keys = sorted({k for r in rows for k in r.keys()})
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    logger.info(f"per-case metrics → {out_dir / 'metrics.csv'}")

    # JSON summary
    numeric_keys = [k for k in keys if k != "id" and all(isinstance(r.get(k), (int, float)) for r in rows)]
    summary: Dict[str, Dict[str, float]] = {}
    for k in numeric_keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        finite = vals[np.isfinite(vals)]
        summary[k] = {
            "mean": float(np.mean(finite)) if finite.size else float("nan"),
            "std":  float(np.std(finite))  if finite.size else float("nan"),
            "n":    int(finite.size),
        }
    with open(out_dir / "metrics_summary.json", "w") as f:
        json.dump({"per_case": rows, "aggregate": summary}, f, indent=2, default=float)
    logger.info(f"summary  → {out_dir / 'metrics_summary.json'}")
    for k, s in summary.items():
        logger.info(f"  {k:18s}  {s['mean']:.4f} ± {s['std']:.4f}  (n={s['n']})")


if __name__ == "__main__":
    main()
