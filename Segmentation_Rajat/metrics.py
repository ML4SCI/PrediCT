"""

metrics.py consists of helper functions which will help you
evaluate different metrics over your GT and Predictions

Following Functions:
  1) Plaque wise F1 (better)
  2) F1 (Voxel wise/Class wise) (naive) (DONE)
  3) Recall, Precision (DONE)
  4) MAE over scores
  5) F1, Recall, Precision over score cateogires with confusion matrix
  6) Weighted Kappa for Scores


 """

from pathlib import Path
from typing import Union
import sys
import importlib.util

import nibabel as nib
import numpy as np
from scipy.ndimage import label as ndimage_label
from scipy.optimize import linear_sum_assignment



# ══════════════════════════════════════════════════════════════════
#  AGATSTON CALCULATOR — dynamic loader
# ══════════════════════════════════════════════════════════════════

AGATSTON_SCRIPT_PATH = Path(r"/pscratch/sd/r/rajatns/gsoc_2026_official/try1/Segmentation_Rajat/agatston_script.py")


def _load_agatston_function(script_path: Path):
    """Dynamically imports calculate_agatston() from agatston_script.py
    without requiring it to be installed as a package."""
    if not script_path.exists():
        print(f"  [warn] Agatston script not found at {script_path}; "
              f"GT/Pred Agatston scores will show N/A")
        return None
    try:
        spec = importlib.util.spec_from_file_location("agatston_script", script_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["agatston_script"] = module
        spec.loader.exec_module(module)
        return module.calculate_agatston
    except Exception as e:
        print(f"  [warn] Failed to import calculate_agatston from {script_path}: {e}")
        return None


_calculate_agatston = _load_agatston_function(AGATSTON_SCRIPT_PATH)


# ══════════════════════════════════════════════════════════════════
#  VOXEL-WISE METRICS — single entry point
# ══════════════════════════════════════════════════════════════════

def voxel_wise_f1(
    label_path: Union[str, Path],
    pred_path: Union[str, Path],
) -> dict:
    """Compute voxel-wise Precision, Recall, and F1 from two NIfTI masks.

    Loads both `label.nii.gz` and `prediction.nii.gz` from disk, binarizes
    them (any non-zero voxel -> 1), then computes the 2×2 confusion matrix
    counts over the flattened volume and derives P / R / F1.

    Edge cases (avoiding 0/0):
        - Model produced no positives  -> precision = 1.0
        - GT has no positives           -> recall    = 1.0
        - Both P and R degenerate       -> F1        = 0.0

    Args:
        label_path : path to ground-truth mask .nii.gz
        pred_path  : path to predicted  mask .nii.gz

    Returns:
        dict with keys:
            f1, precision, recall : float in [0, 1]
            tp, fp, fn, tn         : int  raw confusion counts
            n_voxels, n_gt, n_pred : int  volume / class sizes
            label_path, pred_path  : str  echoes of inputs for traceability
    """
    gt_arr   = nib.load(str(label_path)).get_fdata()
    pred_arr = nib.load(str(pred_path)).get_fdata()

    gt   = (gt_arr   > 0).astype(bool).ravel()
    pred = (pred_arr > 0).astype(bool).ravel()

    if gt.shape != pred.shape:
        raise ValueError(
            f"Shape mismatch: label {gt.shape} vs pred {pred.shape}. "
            f"Ensure both volumes are registered/resampled to the same grid."
        )

    tp = int(np.logical_and(gt, pred).sum())
    fp = int(np.logical_and(~gt, pred).sum())
    fn = int(np.logical_and(gt, ~pred).sum())
    tn = int(np.logical_and(~gt, ~pred).sum())

    precision = 1.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall    = 1.0 if (tp + fn) == 0 else tp / (tp + fn)
    f1        = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    dice = 0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)

    return {
        "f1":         f1,
        "dice":     dice,
        "precision":  precision,
        "recall":     recall,
        "tp":         tp,
        "fp":         fp,
        "fn":         fn,
        "tn":         tn,
        "n_voxels":   int(gt.size),
        "n_gt":       int(gt.sum()),
        "n_pred":     int(pred.sum()),
        "label_path": str(label_path),
        "pred_path":  str(pred_path),
    }


# ══════════════════════════════════════════════════════════════════
#  AGATSTON SCORE — wrappers
# ══════════════════════════════════════════════════════════════════

# Agatston calculator requires BOTH the CT and the calcium mask. The
# calcium mask alone is not enough — we need the HU values inside each
# lesion to assign the density weight. So callers must pass the CT path
# alongside the mask path.



def compute_agatston_score(
    mask_path: Union[str, Path],
    ct_path: Union[str, Path],
) -> dict:
    """Compute the Agatston CAC score for a single calcium mask.

    Delegates to the standard `calculate_agatston(ct, mask)` from
    `Agatston_Script/ agatston_script.py`. Both paths are required:
    the calcium mask tells the calculator WHERE the lesions are; the CT
    volume tells it the HU values (130/200/300/400 density buckets)
    inside each lesion.

    Returns:
        {
            "agatston_total"        : float,
            "n_lesions"             : int,
            "n_slices_with_calcium" : int,
            "mask_path"             : str,
            "ct_path"               : str,
        }

    Returns {"agatston_total": 0.0, ...} if the Agatston script failed to
    load at import time (the loader in `_load_agatston_function` prints a
    warning and sets `_calculate_agatston = None`).
    """
    mask_path = Path(mask_path)
    ct_path   = Path(ct_path)

    if _calculate_agatston is None:
        return {
            "agatston_total":        -196,
            "n_lesions":             -196,
            "n_slices_with_calcium": -196,
            "mask_path":             str(mask_path),
            "ct_path":               str(ct_path),
        }

    result = _calculate_agatston(ct_path=str(ct_path), mask_path=str(mask_path))
    return {
        "agatston_total":        float(result["agatston_total"]),
        "n_lesions":             int(result["n_lesions"]),
        "n_slices_with_calcium": int(result["n_slices_with_calcium"]),
        "mask_path":             str(mask_path),
        "ct_path":               str(ct_path),
    }


# ══════════════════════════════════════════════════════════════════
#  AGATSTON-SCORE-LEVEL METRICS
# ══════════════════════════════════════════════════════════════════

# Risk bins are the SAME bins used in pre_process.py for stratification,
# so the category-level metrics here line up with the model's training
# distribution. Keeping a single source of truth avoids drift.
RISK_BINS = [
    (0,    10,    "Low Risk"),
    (10,   100,   "Medium Risk"),
    (100,  400,   "High Risk"),
    (400,  1000,  "Very High Risk"),
    (1000, float("inf"), "Extreme Risk"),
]
RISK_CATEGORIES = [name for _, _, name in RISK_BINS]


def assign_risk_label(score: float) -> str:
    """Map a raw Agatston score to one of the 5 clinical risk categories."""
    score = float(score)
    for lo, hi, name in RISK_BINS:
        if lo <= score < hi:
            return name
    return RISK_CATEGORIES[-1]


def compute_basic_metrics_for_score(
    gts: list,
    preds: list,
) -> dict:
    """Evaluate segmentation quality at the SCORE level (not the voxel).

    Takes two parallel lists of Agatston scores (one per scan, in the
    same order) and computes:

        1. Regression metrics over the raw scores:
             - MAE, RMSE, mean abs % error, Pearson r, Spearman ρ, R²
        2. Risk-category metrics over the 5 ordinal bins:
             - Macro/weighted F1, Recall, Precision, Accuracy
             - 5×5 confusion matrix
        3. Agreement on the Agatson scale:
             - Quadratic-weighted Cohen's κ (preferred for ordinal scale)
             - Linear-weighted   Cohen's κ (alternative)
             - Exact-agreement rate

    Every metric is `NaN` if it cannot be computed (e.g. one of the lists
    is empty). Callers can `np.isnan(...)` to filter.

    Args:
        gts   : list[float] of ground-truth Agatston scores
        preds : list[float] of predicted  Agatston scores

    Returns:
        dict with sections:
            regression   : {mae, rmse, mape, pearson_r, spearman_r, r2, n}
            category     : {f1_macro, f1_weighted, precision_macro,
                            recall_macro, accuracy, confusion_matrix,
                            categories, y_true_cat, y_pred_cat}
            agreement    : {kappa_quadratic, kappa_linear, exact_agreement}
    """
    gts_arr   = np.asarray(gts,   dtype=float)
    preds_arr = np.asarray(preds, dtype=float)

    nan_result = {
        "regression": {"mae": np.nan, "rmse": np.nan, "mape": np.nan,
                       "pearson_r": np.nan, "spearman_r": np.nan, "r2": np.nan,
                       "n": 0},
        "category":   {"f1_macro": np.nan, "f1_weighted": np.nan,
                       "precision_macro": np.nan, "recall_macro": np.nan,
                       "accuracy": np.nan, "confusion_matrix": None,
                       "categories": RISK_CATEGORIES,
                       "y_true_cat": [], "y_pred_cat": []},
        "agreement":  {"kappa_quadratic": np.nan, "kappa_linear": np.nan,
                       "exact_agreement": np.nan},
    }

    if gts_arr.size == 0 or preds_arr.size == 0:
        return nan_result
    if gts_arr.shape != preds_arr.shape:
        raise ValueError(
            f"gts and preds must have the same length: "
            f"{gts_arr.shape} vs {preds_arr.shape}"
        )

    # ── 1. Regression metrics over raw scores ─────────────────────
    err        = preds_arr - gts_arr
    abs_err    = np.abs(err)
    mae        = float(abs_err.mean())
    rmse       = float(np.sqrt((err ** 2).mean()))
    mape       = float((abs_err / np.maximum(gts_arr, 1.0)).mean()) * 100.0
    pearson_r  = float(np.corrcoef(gts_arr, preds_arr)[0, 1]) if gts_arr.size > 1 else np.nan
    try:
        from scipy.stats import spearmanr
        spearman_r = float(spearmanr(gts_arr, preds_arr).correlation)
    except Exception:
        spearman_r = np.nan
    ss_res = float((err ** 2).sum())
    ss_tot = float(((gts_arr - gts_arr.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    regression = {
        "mae":         mae,
        "rmse":        rmse,
        "mape":        mape,
        "pearson_r":   pearson_r,
        "spearman_r":  spearman_r,
        "r2":          r2,
        "n":           int(gts_arr.size),
    }

    # ── 2. Risk-category metrics over the 5 ordinal bins ──────────
    y_true_cat = [assign_risk_label(s) for s in gts_arr]
    y_pred_cat = [assign_risk_label(s) for s in preds_arr]

    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        accuracy_score,
        precision_score,
        recall_score,
        cohen_kappa_score,
    )

    cm = confusion_matrix(y_true_cat, y_pred_cat, labels=RISK_CATEGORIES)
    category = {
        "f1_macro":        float(f1_score(y_true_cat, y_pred_cat,
                                          labels=RISK_CATEGORIES,
                                          average="macro",   zero_division=0)),
        "f1_weighted":     float(f1_score(y_true_cat, y_pred_cat,
                                          labels=RISK_CATEGORIES,
                                          average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true_cat, y_pred_cat,
                                                 labels=RISK_CATEGORIES,
                                                 average="macro", zero_division=0)),
        "recall_macro":    float(recall_score(y_true_cat, y_pred_cat,
                                              labels=RISK_CATEGORIES,
                                              average="macro", zero_division=0)),
        "accuracy":        float(accuracy_score(y_true_cat, y_pred_cat)),
        "confusion_matrix": cm,
        "categories":      RISK_CATEGORIES,
        "y_true_cat":      y_true_cat,
        "y_pred_cat":      y_pred_cat,
    }

    # ── 3. Agreement on the Agatston scale ────────────────────────
    agreement = {
        "kappa_quadratic":  float(cohen_kappa_score(
            y_true_cat, y_pred_cat,
            labels=RISK_CATEGORIES, weights="quadratic")),
        "kappa_linear":     float(cohen_kappa_score(
            y_true_cat, y_pred_cat,
            labels=RISK_CATEGORIES, weights="linear")),
        "exact_agreement":  float(np.mean(
            np.array(y_true_cat) == np.array(y_pred_cat))),
    }

    return {
        "regression": regression,
        "category":   category,
        "agreement":  agreement,
    }


# ══════════════════════════════════════════════════════════════════
#  PLAQUE-WISE METRICS — Hungarian-matched component detection
# ══════════════════════════════════════════════════════════════════

# 26-connectivity (face + edge + corner) is the standard for medical 3D
# voxel data — a calcium lesion touching on a corner is one plaque, not
# two. The agatston calculator in Agatston_Script/ uses per-slice 2D
# 4-connectivity (default ndimage.label), which is right for per-slice
# scoring but would fragment a single 3D lesion across slice boundaries.
# For DETECTION, full-volume 26-connectivity is what we want.
_3D_CONNECTIVITY = np.ones((3, 3, 3), dtype=bool)


def plaque_wise_matching(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    dice_threshold: float = 0.1,
    min_voxels: int = 1,
) -> dict:
    """Match 3D GT plaques to predicted plaques via the Hungarian algorithm.

    Treats every 3D connected component as a candidate plaque, builds
    a (n_gt × n_pred) Dice affinity matrix, runs `linear_sum_assignment`
    to find the optimal one-to-one pairing, then keeps only pairs whose
    Dice is at least `dice_threshold`. Precision / Recall / F1 are then
    defined over the resulting (matched, unmatched_gt, unmatched_pred)
    counts, and Macro Dice is the mean Dice across only the matched pairs.

    Why Hungarian and not greedy: greedy matching (largest-Dice-first)
    can let one predicted plaque "steal" two GT plaques and break recall
    counting. The Hungarian assignment is the standard in detection
    tasks (e.g. nn-Detection, Panoptic Quality) and is the same approach
    used by the MONAI PanopticQuality metric. scipy's linear_sum_assignment
    handles n_gt != n_pred natively (it saturates the smaller side) — no
    manual dummy-row/column padding is needed.

    Macro Dice vs plaque-wise F1 — these answer two different clinical
    questions:
        - Plaque-wise F1  : did we find the right SET of plaques
                            (detection quality — mix of precision/recall)
        - Macro Dice      : for the plaques we DID find, how well do the
                            boundaries overlap (segmentation quality)
      A model can have high F1 but low Macro Dice (finds every plaque,
      but draws sloppy boundaries), or vice versa (a couple of very
      precise matches, but misses/invents several plaques). Reporting
      both avoids either failure mode hiding behind a single number.

    Args:
        gt_arr          : 3D numpy array, ground-truth mask (any non-zero
                          voxel is positive)
        pred_arr        : 3D numpy array, predicted mask, same shape as
                          `gt_arr`
        dice_threshold  : minimum Dice for a GT–Pred pair to count as a
                          match. Pairs below this threshold are rejected
                          and count as unmatched. Default 0.1 (a pair
                          with <10% volume overlap is not the same lesion).
        min_voxels      : discard connected components smaller than this.
                          Default 1 (don't filter).

    Returns:
        dict with keys:
            f1, precision, recall : float in [0, 1]
            macro_dice             : float in [0, 1] — mean Dice over
                                     matched pairs only (see edge cases
                                     below). This is NOT the same as
                                     voxel-wise Dice over the whole volume.
            n_gt_plaques, n_pred_plaques, n_matched : int
            n_unmatched_gt, n_unmatched_pred       : int
            pair_dices                             : list[float]
            dice_threshold                         : float
    """
    gt_arr   = (np.asarray(gt_arr)   > 0).astype(np.uint8)
    pred_arr = (np.asarray(pred_arr) > 0).astype(np.uint8)

    if gt_arr.shape != pred_arr.shape:
        raise ValueError(
            f"Shape mismatch: GT {gt_arr.shape} vs Pred {pred_arr.shape}. "
            f"Ensure both volumes are registered/resampled to the same grid."
        )

    # ── Connected components on each side ─────────────────────────
    gt_labels,   n_gt   = ndimage_label(gt_arr,   structure=_3D_CONNECTIVITY)
    pred_labels, n_pred = ndimage_label(pred_arr, structure=_3D_CONNECTIVITY)

    # ── Filter tiny components (default 1 = no filtering) ─────────
    if min_voxels > 1:
        gt_sizes   = np.bincount(gt_labels.ravel())
        pred_sizes = np.bincount(pred_labels.ravel())
        tiny_gt   = set(np.where(gt_sizes[:n_gt + 1]   < min_voxels)[0]) - {0}
        tiny_pred = set(np.where(pred_sizes[:n_pred + 1] < min_voxels)[0]) - {0}
        n_gt   -= len(tiny_gt)
        n_pred -= len(tiny_pred)
        keep_gt   = np.zeros(gt_labels.max()   + 1, dtype=np.int32)
        keep_pred = np.zeros(pred_labels.max() + 1, dtype=np.int32)
        new_idx_gt, new_idx_pred = 1, 1
        for old in range(1, gt_labels.max() + 1):
            if old in tiny_gt:
                continue
            keep_gt[old]   = new_idx_gt;   new_idx_gt   += 1
        for old in range(1, pred_labels.max() + 1):
            if old in tiny_pred:
                continue
            keep_pred[old] = new_idx_pred; new_idx_pred += 1
        gt_labels   = keep_gt[gt_labels]
        pred_labels = keep_pred[pred_labels]

    # ── Degenerate cases: at least one side has no components ────
    # macro_dice is undefined when there are zero matched pairs to
    # average over. Convention: 1.0 when both sides are empty (nothing
    # to disagree on), 0.0 when only one side is empty (zero matches,
    # same convention as F1=0.0 below).
    if n_gt == 0 and n_pred == 0:
        return {
            "f1": 1.0, "precision": 1.0, "recall": 1.0, "macro_dice": 1.0,
            "n_gt_plaques": 0, "n_pred_plaques": 0, "n_matched": 0,
            "n_unmatched_gt": 0, "n_unmatched_pred": 0,
            "pair_dices": [], "dice_threshold": float(dice_threshold),
        }
    if n_gt == 0:
        return {
            "f1": 0.0, "precision": 0.0, "recall": 1.0, "macro_dice": 0.0,
            "n_gt_plaques": 0, "n_pred_plaques": n_pred, "n_matched": 0,
            "n_unmatched_gt": 0, "n_unmatched_pred": n_pred,
            "pair_dices": [], "dice_threshold": float(dice_threshold),
        }
    if n_pred == 0:
        return {
            "f1": 0.0, "precision": 1.0, "recall": 0.0, "macro_dice": 0.0,
            "n_gt_plaques": n_gt, "n_pred_plaques": 0, "n_matched": 0,
            "n_unmatched_gt": n_gt, "n_unmatched_pred": 0,
            "pair_dices": [], "dice_threshold": float(dice_threshold),
        }

    # ── Build (n_gt × n_pred) Dice affinity matrix ────────────────
    gt_sizes   = np.bincount(gt_labels.ravel(),   minlength=n_gt   + 1)
    pred_sizes = np.bincount(pred_labels.ravel(), minlength=n_pred + 1)
    inter = np.zeros((n_gt, n_pred), dtype=np.int64)
    for i in range(1, n_gt + 1):
        gt_mask = (gt_labels == i)
        pred_in_gt = pred_labels[gt_mask]
        counts = np.bincount(pred_in_gt.ravel(), minlength=n_pred + 1)
        inter[i - 1, :] = counts[1:n_pred + 1]

    denom = gt_sizes[1:n_gt + 1, None].astype(np.float64) \
          + pred_sizes[1:n_pred + 1][None, :].astype(np.float64)
    dice_mat = np.where(denom > 0, 2.0 * inter.astype(np.float64) / denom, 0.0)

    # Hungarian minimizes cost, so feed (1 - Dice). scipy's
    # linear_sum_assignment natively handles rectangular (n_gt != n_pred)
    # matrices by saturating the smaller side — no padding required.
    cost = 1.0 - dice_mat

    # ── Hungarian assignment ──────────────────────────────────────
    row_ind, col_ind = linear_sum_assignment(cost)

    # Keep only pairs whose Dice meets the threshold; the rest are
    # "rejected matches" — the algorithm's best pairing, but not a
    # real one.
    pair_dices = []
    kept_rows, kept_cols = [], []
    for r, c in zip(row_ind, col_ind):
        d = float(dice_mat[r, c])
        if d >= dice_threshold:
            pair_dices.append(d)
            kept_rows.append(r)
            kept_cols.append(c)
    n_matched = len(pair_dices)

    n_unmatched_gt   = n_gt   - n_matched
    n_unmatched_pred = n_pred - n_matched
    precision = n_matched / n_pred
    recall    = n_matched / n_gt
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    # Macro Dice: unweighted mean over matched-pair Dice scores only.
    # "Macro" here mirrors macro-F1 convention — every matched lesion
    # counts equally regardless of its voxel size, so one large
    # well-segmented plaque doesn't drown out several small poorly
    # segmented ones (or vice versa). If you also want a size-weighted
    # ("micro") version, weight by gt_sizes[r] before averaging.
    macro_dice = float(np.mean(pair_dices)) if n_matched > 0 else 0.0

    return {
        "f1":               f1,
        "precision":        precision,
        "recall":           recall,
        "macro_dice":       macro_dice,
        "n_gt_plaques":     int(n_gt),
        "n_pred_plaques":   int(n_pred),
        "n_matched":        int(n_matched),
        "n_unmatched_gt":   int(n_unmatched_gt),
        "n_unmatched_pred": int(n_unmatched_pred),
        "pair_dices":       pair_dices,
        "dice_threshold":   float(dice_threshold),
    }


def plaque_wise_f1(
    label_path: Union[str, Path],
    pred_path: Union[str, Path],
    dice_threshold: float = 0.1,
    min_voxels: int = 1,
) -> dict:
    """Plaque-wise Precision / Recall / F1 between two NIfTI masks.

    Loads both `label.nii.gz` and `prediction.nii.gz` from disk,
    binarizes them, extracts 3D connected components, then runs
    `plaque_wise_matching` with the same Hungarian-based logic used
    by nn-Detection / Panoptic Quality.

    Compared to `voxel_wise_f1`:
        - voxel-wise F1 is dominated by the (huge) background and can
          look fine even when the model misses every clinically
          relevant plaque.
        - plaque-wise F1 rewards *detecting* each real plaque and
          penalizes *inventing* ones that aren't there. It is the
          clinically meaningful number for CAC scoring.

    Args:
        label_path      : path to ground-truth mask .nii.gz
        pred_path       : path to predicted    mask .nii.gz
        dice_threshold  : min Dice for a GT–Pred pair to count as a
                          match (default 0.1)
        min_voxels      : discard components smaller than this
                          (default 1, i.e. don't filter)

    Returns:
        dict with keys:
            f1, precision, recall : float in [0, 1]
            n_gt_plaques, n_pred_plaques, n_matched, n_unmatched_* : int
            pair_dices          : list[float] — Dice of every kept pair
            dice_threshold      : float
            label_path, pred_path : str — echoed inputs
    """
    gt_arr   = nib.load(str(label_path)).get_fdata()
    pred_arr = nib.load(str(pred_path)).get_fdata()

    result = plaque_wise_matching(
        gt_arr, pred_arr,
        dice_threshold=dice_threshold,
        min_voxels=min_voxels,
    )
    result["label_path"] = str(label_path)
    result["pred_path"]  = str(pred_path)
    return result

# ══════════════════════════════════════════════════════════════════
#  CONFUSION MATRIX — save to disk
# ══════════════════════════════════════════════════════════════════

def output_confusion_matrix(
    cm: np.ndarray,
    categories: list,
    out_dir: Union[str, Path],
    filename: str = "risk_category_confusion_matrix",
    normalize: bool = False,
    title: str = "CAC Risk Category — Confusion Matrix",
) -> dict:
    """Save the risk-category confusion matrix as a PNG heatmap + CSV.

    Takes the raw `confusion_matrix` array (as returned inside
    `compute_basic_metrics_for_score(...)["category"]["confusion_matrix"]`)
    and the `RISK_CATEGORIES` label list, and writes two files to
    `out_dir`:
        - `{filename}.png` : annotated heatmap (matplotlib), rows = GT
                             risk category, columns = predicted category
        - `{filename}.csv` : raw counts as a labeled CSV, for anyone who
                             wants the numbers rather than the picture

    Args:
        cm         : (n_categories x n_categories) confusion matrix,
                    rows = true label, columns = predicted label
                    (this is sklearn's `confusion_matrix` convention).
        categories : ordered list of category names matching cm's axes
                    (use `RISK_CATEGORIES` for risk-bin confusion
                    matrices).
        out_dir    : directory to save into. Created if it doesn't exist.
        filename   : base filename (no extension) for both outputs.
        normalize  : if True, each row is normalized to sum to 1 (i.e.
                    shows recall-per-class as a fraction rather than raw
                    counts) — useful when class sizes are very unequal,
                    which is typical for the 5 CAC risk bins.
        title      : plot title.

    Returns:
        dict with keys:
            png_path, csv_path : str — paths of the written files
            cm                  : np.ndarray — the (possibly normalized)
                                  matrix that was actually plotted
    """
    import matplotlib
    matplotlib.use("Agg")  # headless-safe backend, no GUI required
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cm = np.asarray(cm)
    if cm.shape[0] != len(categories) or cm.shape[1] != len(categories):
        raise ValueError(
            f"cm shape {cm.shape} doesn't match len(categories)={len(categories)}"
        )

    plot_cm = cm.astype(np.float64)
    if normalize:
        row_sums = plot_cm.sum(axis=1, keepdims=True)
        # Avoid div-by-zero for GT categories with zero support in this eval
        plot_cm = np.divide(
            plot_cm, row_sums,
            out=np.zeros_like(plot_cm),
            where=row_sums > 0,
        )

    # ── Plot ────────────────────────────────────────────────────
    n = len(categories)
    fig, ax = plt.subplots(figsize=(1.3 * n + 2, 1.1 * n + 2))
    im = ax.imshow(plot_cm, cmap="Blues", vmin=0,
                    vmax=1.0 if normalize else plot_cm.max())

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_yticklabels(categories)
    ax.set_xlabel("Predicted Risk Category")
    ax.set_ylabel("Ground-Truth Risk Category")
    ax.set_title(title)

    # Annotate each cell; flip text color for readability against the
    # colormap's darker cells.
    thresh = plot_cm.max() / 2.0 if plot_cm.max() > 0 else 0.5
    for i in range(n):
        for j in range(n):
            val = plot_cm[i, j]
            text = f"{val:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=10,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Fraction" if normalize else "Count")
    fig.tight_layout()

    png_path = out_dir / f"{filename}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    print(f"\n\n\n  [saved] confusion matrix -> {png_path}")

    return {
        "png_path": str(png_path),
        "cm": plot_cm,
    }

# ══════════════════════════════════════════════════════════════════
#  STANDALONE — end-to-end sanity check on 3 real patient folders
# ══════════════════════════════════════════════════════════════════

# Each prediction folder in nnUNet_predictions contains image.nii.gz,
# label.nii.gz, and prediction.nii.gz laid out exactly as nnU-Net v2
# writes them, so all three metrics (voxel-wise, plaque-wise,
# agatston) can be exercised on the same scans.

_SCAN_IDS = ("0e5edec84ae7", "2d80e46f4a9b", "7d4d96cc137d")
_PREDICTIONS_ROOT = Path(
    r"E:\MyProjects\Gsoc_2026_Official\test_nnunet_2\nnUNet_predictions"
)


def _run_full_eval(scan_id: str, root: Path = _PREDICTIONS_ROOT) -> dict:
    """Run voxel-, plaque-, and score-level metrics on one patient.

    Returns a dict with all metrics so the __main__ block can both print
    a human report and (later) aggregate across patients.
    """
    patient_dir = root / scan_id
    label_path  = patient_dir / "label.nii.gz"
    pred_path   = patient_dir / "prediction.nii.gz"
    ct_path   = patient_dir / "image.nii.gz"


    voxel   = voxel_wise_f1(label_path, pred_path)
    plaque  = plaque_wise_f1(label_path, pred_path)
    gt_agat = compute_agatston_score(label_path, ct_path)
    pr_agat = compute_agatston_score(pred_path, ct_path)

    return {
        "scan_id":      scan_id,
        "voxel":        voxel,
        "plaque":       plaque,
        "gt_agatston":  gt_agat,
        "pred_agatston": pr_agat,
    }


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
        f"  PQ Score   : {p['macro_dice'] * v['f1']:.4f}  "
    )
    print(
        f"  Agatston    : GT={g['agatston_total']:7.2f}  "
        f"Pred={pr['agatston_total']:7.2f}  "
        f"d={(pr['agatston_total'] - g['agatston_total']):+.2f}"
    )


if __name__ == "__main__":
    import time
    rule = "=" * 60


    print("=" * 60)
    print("  metrics.py - end-to-end check on 3 nnU-Net patients")
    print("=" * 60)

    results = []
    t_total = time.perf_counter()
    for scan_id in _SCAN_IDS:
        try:
            t0 = time.perf_counter()
            result = _run_full_eval(scan_id)
            dt = time.perf_counter() - t0
            _print_eval(result)
            print(f"  (took {dt:.2f}s)")
            results.append(result)
        except Exception as e:
            print(f"\n  [ERROR] {scan_id}: {type(e).__name__}: {e}")
    t_total = time.perf_counter() - t_total

    # ── Aggregate score-level metrics across the 3 patients ───────
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

        VISUAL_OUT_DIR = Path(r"E:\MyProjects\Gsoc_2026_Official\test_nnunet_2\Visualizations")

         # ── Save confusion matrix (raw counts + normalized) ───────
        output_confusion_matrix(
            cm=cat["confusion_matrix"],
            categories=cat["categories"],
            out_dir=VISUAL_OUT_DIR,
            filename="risk_category_confusion_matrix",
            normalize=False,
        )

        print(f"\n  (full eval took {t_total:.2f}s)")
        print(f"\n PQ Score is best for clinical evalutation which is prodcut of Recognition Quality (Plaque wise F1) and Segmentation Quality (Macro Dice)")
        print(f"  All metrics callable ✓")



