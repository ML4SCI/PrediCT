"""
Qualitative segmentation visualization, paper-style.

Layout per patient row: 5 columns ->
  Prediction | Ground Truth | Overlay | Confusion Map | Info panel

Each image panel shows the FULL CT slice with small colored boxes drawn
around each lesion, and matching zoomed-in insets (like MICCAI/lesion-seg
papers: MT / MC-Net / DTC / MCF style figures).

Pipeline:
  1. Compute volumetric Dice for every patient in ROOT, print summary stats.
  2. Pick NUM_SAMPLES patients spanning best -> worst Dice (evenly spaced
     over the sorted distribution), so the figure shows a representative
     range of performance, not just N random/first cases.
  3. Render the figure.

Usage: edit CONFIG below, then run.
"""

from pathlib import Path
import sys
import importlib.util
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import ndimage

# =====================================================
# CONFIG
# =====================================================
ROOT = Path(r"E:\MyProjects\Gsoc_2026_Official\nnunet_simple_scv\Grouped")
VISUAL_OUT_DIR = Path(r"E:\MyProjects\Gsoc_2026_Official\nnunet_simple_scv\Visualizations")
DATASET_PATH = Path(r"E:\MyProjects\Gsoc_2026_Official\data_canonical\tables\scan_index.csv")
# COLUMNS: patient_id,scan_id,voxels,num_slices,lesion_count,agatston_total,agatston_rca,agatston_left_coronary,agatston_lad,agatston_lcx,folder_path

AGATSTON_SCRIPT_PATH = Path(r"E:\MyProjects\Gsoc_2026_Official\Agatston_Script\agatston_script.py")
COMPUTE_AGATSTON = True   # set False to skip GT/Pred Agatston computation (faster)

NUM_SAMPLES = 5                 # number of rows in the final figure
ALPHA = 0.45
MAX_LESIONS = 2                 # zoom insets per row (set 1 to simplify)
MIN_COMPONENT_VOXELS = 15       # ignore tiny noise blobs when picking lesions
ZOOM_MARGIN = 18                # px margin around each lesion crop
INSET_SIZE = "32%"              # inset width/height as % of parent axes
BOX_COLORS = ["#39FF14", "#00E5FF", "#FFD700"]  # green, cyan, gold
FIGSIZE_PER_ROW = 5.2
OUT_PATH = Path(VISUAL_OUT_DIR) / "qualitative_results.png"
DPI = 300
# =====================================================


def _load_agatston_function(script_path):
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


_calculate_agatston = _load_agatston_function(AGATSTON_SCRIPT_PATH) if COMPUTE_AGATSTON else None


def compute_gt_pred_agatston(patient_dir):
    """Runs the standard Agatston calculator on the GT mask and the
    prediction mask separately, both against the same CT volume.
    Returns (gt_agatston_total, pred_agatston_total), either may be None
    if the calculator isn't available or the run fails."""
    if _calculate_agatston is None:
        return None, None

    ct_path = patient_dir / "image.nii.gz"
    gt_path = patient_dir / "label.nii.gz"
    pred_path = patient_dir / "prediction.nii.gz"

    gt_total, pred_total = None, None
    try:
        gt_result = _calculate_agatston(ct_path=str(ct_path), mask_path=str(gt_path))
        gt_total = gt_result["agatston_total"]
    except Exception as e:
        print(f"  [warn] GT Agatston failed for {patient_dir.name}: {e}")

    try:
        pred_result = _calculate_agatston(ct_path=str(ct_path), mask_path=str(pred_path))
        pred_total = pred_result["agatston_total"]
    except Exception as e:
        print(f"  [warn] Pred Agatston failed for {patient_dir.name}: {e}")

    return gt_total, pred_total


def dice_score(gt, pred):
    gt, pred = gt.astype(bool), pred.astype(bool)
    inter = np.logical_and(gt, pred).sum()
    denom = gt.sum() + pred.sum()
    return 1.0 if denom == 0 else 2.0 * inter / denom


def find_lesion_boxes(gt_slice, pred_slice, max_lesions, min_voxels, margin):
    """Connected-component analysis on union(GT, Pred); returns crop boxes
    sorted by size (largest first), each as (y1, y2, x1, x2)."""
    union = gt_slice | pred_slice
    labeled, n = ndimage.label(union)
    boxes = []
    for lbl in range(1, n + 1):
        ys, xs = np.where(labeled == lbl)
        if ys.size < min_voxels:
            continue
        y1, y2 = ys.min() - margin, ys.max() + margin
        x1, x2 = xs.min() - margin, xs.max() + margin
        y1, x1 = max(0, y1), max(0, x1)
        y2 = min(union.shape[0], y2)
        x2 = min(union.shape[1], x2)
        area = (y2 - y1) * (x2 - x1)
        boxes.append((area, (y1, y2, x1, x2)))
    boxes.sort(key=lambda b: b[0], reverse=True)  # largest first
    boxes = [b[1] for b in boxes[:max_lesions]]
    return boxes


def make_overlay(mask, rgb, alpha=ALPHA):
    overlay = np.zeros((*mask.shape, 4))
    overlay[mask] = [*rgb, alpha]
    return overlay


from sklearn.metrics import confusion_matrix


# def plot_confusion_matrix_ax(ax, gt_slice, pred_slice, title="Confusion Matrix", title_color="black"):
#     """Plots a real 2x2 (TN/FP/FN/TP) confusion matrix, sklearn-style,
#     computed on the flattened pixels of one slice."""
#     y_true = gt_slice.astype(int).ravel()
#     y_pred = pred_slice.astype(int).ravel()
#     cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

#     ax.imshow(cm, cmap="Blues", interpolation="nearest")
#     ax.set_title(title, fontsize=13, fontweight="bold", color=title_color, pad=6)
#     ax.set_xticks([0, 1])
#     ax.set_yticks([0, 1])
#     ax.set_xticklabels(["Background", "Lesion"], fontsize=9)
#     ax.set_yticklabels(["Background", "Lesion"], fontsize=9)
#     ax.set_xlabel("Predicted", fontsize=10, fontweight="bold")
#     ax.set_ylabel("Ground Truth", fontsize=10, fontweight="bold")

#     cm_max = cm.max() if cm.max() > 0 else 1
#     cell_labels = [["TN", "FP"], ["FN", "TP"]]
#     for i in range(2):
#         for j in range(2):
#             value = cm[i, j]
#             color = "white" if value > cm_max * 0.5 else "black"
#             ax.text(
#                 j, i, f"{cell_labels[i][j]}\n{value:,}",
#                 ha="center", va="center", fontsize=11, fontweight="bold", color=color
#             )
#     for spine in ax.spines.values():
#         spine.set_visible(False)

def plot_confusion_matrix_ax(ax, gt, pred, title="Confusion Matrix", title_color="black"):
    """
    Plots a 2×2 confusion matrix computed over the WHOLE 3D VOLUME.
    """
    # Verify 3D volumes
    assert gt.shape == pred.shape, (
        f"Shape mismatch: GT {gt.shape}, Pred {pred.shape}"
    )

    assert gt.ndim == 3, (
        f"Expected 3D volumes, got {gt.ndim}D"
    )
    # Flatten the entire 3D volumes
    y_true = gt.astype(np.uint8).ravel()
    y_pred = pred.astype(np.uint8).ravel()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    ax.imshow(cm, cmap="Blues", interpolation="nearest")

    ax.set_title(
        title,
        fontsize=13,
        fontweight="bold",
        color=title_color,
        pad=8,
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Background", "Calcium"], fontsize=10)
    ax.set_yticklabels(["Background", "Calcium"], fontsize=10)

    ax.set_xlabel("Prediction", fontsize=11, fontweight="bold")
    ax.set_ylabel("Ground Truth", fontsize=11, fontweight="bold")

    cm_max = cm.max() if cm.max() > 0 else 1

    labels = [["TN", "FP"],
              ["FN", "TP"]]

    for i in range(2):
        for j in range(2):
            value = cm[i, j]

            ax.text(
                j,
                i,
                f"{labels[i][j]}\n{value:,}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="white" if value > cm_max / 2 else "black",
            )

    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_panel(ax, img, pred=None, gt=None, boxes=None, title="", title_color="black"):
    ax.imshow(img, cmap="gray", interpolation="nearest")
    if pred is not None:
        ax.imshow(make_overlay(pred, (0, 1, 0)))
    if gt is not None:
        ax.imshow(make_overlay(gt, (1, 0, 0)))

    ax.set_title(title, fontsize=13, fontweight="bold", color=title_color, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if not boxes:
        return

    corners = ["upper right", "lower right", "upper left", "lower left"]

    for i, (y1, y2, x1, x2) in enumerate(boxes):
        color = BOX_COLORS[i % len(BOX_COLORS)]

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=0.8, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

        ax.text(
            x1 - 3, (y1 + y2) / 2, f"L{i + 1}",
            fontsize=6, fontweight="bold", color=color,
            ha="right", va="center"
        )

        axins = inset_axes(
            ax, width=INSET_SIZE, height=INSET_SIZE,
            loc=corners[i % len(corners)],
            borderpad=0.15
        )
        axins.imshow(img[y1:y2, x1:x2], cmap="gray", interpolation="nearest")
        if pred is not None:
            axins.imshow(make_overlay(pred[y1:y2, x1:x2], (0, 1, 0)))
        if gt is not None:
            axins.imshow(make_overlay(gt[y1:y2, x1:x2], (1, 0, 0)))
        axins.set_xticks([])
        axins.set_yticks([])
        for spine in axins.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.2)


def draw_info_panel(ax, scan_id, meta, dice, gt_agatston=None, pred_agatston=None):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def fmt(key, decimals=1):
        if not meta or key not in meta or pd.isna(meta.get(key)):
            return "N/A"
        val = meta[key]
        return f"{val:.{decimals}f}" if isinstance(val, float) else str(val)

    def fmt_val(val, decimals=1):
        return f"{val:.{decimals}f}" if val is not None else "N/A"

    rows = [
        ("Scan ID", str(scan_id)),
        ("XML Agatston", fmt("agatston_total")),
        ("GT Agatston", fmt_val(gt_agatston)),
        ("Pred Agatston", fmt_val(pred_agatston)),
        ("Lesion Count", fmt("lesion_count", 0)),
    ]

    label_x, value_x = 0.06, 0.94
    y0, row_h = 0.92, 0.135
    for i, (label, value) in enumerate(rows):
        y = y0 - i * row_h
        ax.text(label_x, y, label, fontsize=9.5, color="#555555",
                 ha="left", va="center", fontweight="bold")
        ax.text(value_x, y, value, fontsize=10.5, color="black",
                 ha="right", va="center", family="monospace")
        if i < len(rows) - 1:
            ax.axhline(y - row_h / 2, xmin=0.04, xmax=0.96, color="#E0E0E0", linewidth=0.8)

    if dice >= 0.7:
        badge_color = "#2E7D32"
    elif dice >= 0.4:
        badge_color = "#F9A825"
    else:
        badge_color = "#C62828"
    ax.add_patch(patches.FancyBboxPatch(
        (0.10, 0.03), 0.80, 0.11, boxstyle="round,pad=0.02",
        linewidth=0, facecolor=badge_color, alpha=0.85
    ))
    ax.text(0.5, 0.085, f"Dice = {dice:.3f}", fontsize=10.5, color="white",
             ha="center", va="center", fontweight="bold")


def compute_all_dice(patient_dirs):
    """First pass: compute Dice (and GT/Pred Agatston, if enabled) for
    every patient, return list of dicts."""
    results = []
    for patient in patient_dirs:
        try:
            gt = nib.load(patient / "label.nii.gz").get_fdata() > 0
            pred = nib.load(patient / "prediction.nii.gz").get_fdata() > 0
        except FileNotFoundError as e:
            print(f"  [skip] {patient.name}: {e}")
            continue
        d = dice_score(gt, pred)
        gt_agat, pred_agat = compute_gt_pred_agatston(patient)
        results.append({
            "patient": patient,
            "dice": d,
            "gt_agatston": gt_agat,
            "pred_agatston": pred_agat,
        })
    return results


def select_best_to_worst(results, n):
    """Sort by Dice descending, then pick n indices evenly spaced across
    the sorted list so the figure spans best -> worst performance."""
    sorted_results = sorted(results, key=lambda r: r["dice"], reverse=True)
    total = len(sorted_results)
    if total <= n:
        return sorted_results
    idxs = np.linspace(0, total - 1, n).round().astype(int)
    idxs = sorted(set(idxs.tolist()))
    while len(idxs) < n:
        for i in range(total):
            if i not in idxs:
                idxs.append(i)
                break
        idxs = sorted(idxs)
    return [sorted_results[i] for i in idxs[:n]]


def load_scan_metadata(csv_path):
    """Returns dict: scan_id -> full CSV row (as dict)."""
    if not csv_path.exists():
        print(f"  [warn] dataset CSV not found at {csv_path}, scan metadata will show N/A")
        return {}
    df = pd.read_csv(csv_path)
    if "scan_id" not in df.columns:
        print("  [warn] 'scan_id' column not found in CSV, scan metadata will show N/A")
        return {}
    return {str(row["scan_id"]): row.to_dict() for _, row in df.iterrows()}


def visualize_single_scan(scan_id, root=ROOT, csv_path=DATASET_PATH, save=True, show=True):
    """Deep-dive figure for ONE scan: Prediction | GT | Overlay on top,
    a real sklearn confusion matrix + a text stats panel below."""
    patient = root / scan_id
    image = nib.load(patient / "image.nii.gz").get_fdata()
    gt = nib.load(patient / "label.nii.gz").get_fdata() > 0
    pred = nib.load(patient / "prediction.nii.gz").get_fdata() > 0

    dice = dice_score(gt, pred)

    areas = gt.sum(axis=(0, 1))
    z = int(np.argmax(areas))

    img_slice = np.rot90(image[:, :, z])
    gt_slice = np.rot90(gt[:, :, z])
    pred_slice = np.rot90(pred[:, :, z])
    img_disp = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)

    boxes = find_lesion_boxes(gt_slice, pred_slice, MAX_LESIONS, MIN_COMPONENT_VOXELS, ZOOM_MARGIN)

    meta_lookup = load_scan_metadata(csv_path)
    meta = meta_lookup.get(str(scan_id))
    gt_agat, pred_agat = compute_gt_pred_agatston(patient)

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 6, height_ratios=[2.0, 1.1], hspace=0.35, wspace=0.5)

    ax_pred = fig.add_subplot(gs[0, 0:2])
    ax_gt = fig.add_subplot(gs[0, 2:4])
    ax_overlay = fig.add_subplot(gs[0, 4:6])

    draw_panel(ax_pred, img_disp, pred=pred_slice, boxes=boxes,
               title="Prediction", title_color="#0B8A2A")
    draw_panel(ax_gt, img_disp, gt=gt_slice, boxes=boxes,
               title="Ground Truth", title_color="#C62828")
    draw_panel(ax_overlay, img_disp, pred=pred_slice, gt=gt_slice, boxes=boxes,
               title=f"Overlay (Dice = {dice:.3f})", title_color="#1E3A8A")

    ax_cm = fig.add_subplot(gs[1, 0:2])
    plot_confusion_matrix_ax(ax_cm, gt, pred,
                              title="Confusion Matrix", title_color="#6A1B9A")

    ax_text = fig.add_subplot(gs[1, 2:6])
    ax_text.set_xticks([])
    ax_text.set_yticks([])
    for spine in ax_text.spines.values():
        spine.set_visible(False)
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(0, 1)

    def mv(key, decimals=1):
        if not meta or key not in meta or pd.isna(meta.get(key)):
            return "N/A"
        val = meta[key]
        return f"{val:.{decimals}f}" if isinstance(val, float) else str(val)

    stats_fields = [
        ("Scan ID", str(scan_id)),
        ("Dice Score", f"{dice:.3f}"),
        ("Lesion Count", mv("lesion_count", 0)),
        ("Num Slices", mv("num_slices", 0)),
        ("XML Agatston", mv("agatston_total")),
        ("GT Agatston", f"{gt_agat:.1f}" if gt_agat is not None else "N/A"),
        ("Pred Agatston", f"{pred_agat:.1f}" if pred_agat is not None else "N/A"),
        ("Agatston RCA", mv("agatston_rca")),
        ("Agatston LAD", mv("agatston_lad")),
        ("Agatston LCX", mv("agatston_lcx")),
        ("Agatston Left Coronary", mv("agatston_left_coronary")),
    ]

    ax_text.set_title("Scan Summary", fontsize=13, fontweight="bold", color="#37474F", pad=6)

    n_fields = len(stats_fields)
    n_cols_text = 3
    for i, (label, value) in enumerate(stats_fields):
        col = i % n_cols_text
        row = i // n_cols_text
        x = 0.02 + col * 0.33
        y = 0.85 - row * 0.32
        ax_text.text(x, y, label, fontsize=10, color="#555555",
                      ha="left", va="top", fontweight="bold")
        ax_text.text(x, y - 0.13, value, fontsize=13, color="black",
                      ha="left", va="top")

    fig.suptitle(f"Scan {scan_id} — Qualitative Result", fontsize=18, fontweight="bold", y=0.99)

    if save:
        out_path = Path(VISUAL_OUT_DIR) / f"scan_{scan_id}_summary.png"
        plt.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"Saved figure to {out_path.resolve()}")
    if show:
        plt.show()
    return fig


def main():
    patient_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir()])
    print(f"Found {len(patient_dirs)} patients in {ROOT}")

    metadata_lookup = load_scan_metadata(DATASET_PATH)

    # ---- Pass 1: compute Dice for ALL patients ----
    print("Computing Dice scores for all patients...")
    results = compute_all_dice(patient_dirs)
    dices = np.array([r["dice"] for r in results])

    print("\n=== Dice score summary (all patients) ===")
    print(f"  n      = {len(dices)}")
    print(f"  mean   = {dices.mean():.4f}")
    print(f"  std    = {dices.std():.4f}")
    print(f"  median = {np.median(dices):.4f}")
    print(f"  min    = {dices.min():.4f}")
    print(f"  max    = {dices.max():.4f}")
    print(f"  p25/p75 = {np.percentile(dices, 25):.4f} / {np.percentile(dices, 75):.4f}")
    print("===========================================\n")

    print("=== Per-patient Dice & Agatston Scores ===")
    for r in sorted(results, key=lambda r: r["dice"], reverse=True):
        scan_id = r["patient"].name
        meta = metadata_lookup.get(scan_id)
        if meta and "agatston_total" in meta and not pd.isna(meta["agatston_total"]):
            xml_agat_str = f"{meta['agatston_total']:.1f}"
        else:
            xml_agat_str = "N/A"
        gt_agat_str = f"{r['gt_agatston']:.1f}" if r.get("gt_agatston") is not None else "N/A"
        pred_agat_str = f"{r['pred_agatston']:.1f}" if r.get("pred_agatston") is not None else "N/A"
        print(
            f"  {scan_id:<25} Dice = {r['dice']:.4f}   "
            f"XML Agatston = {xml_agat_str:<8} "
            f"GT Agatston = {gt_agat_str:<8} "
            f"Pred Agatston = {pred_agat_str:<8}"
        )
    print("===========================================\n")

    # ---- Pass 2: pick samples spanning best -> worst ----
    selected = select_best_to_worst(results, NUM_SAMPLES)
    print("Selected patients (best -> worst Dice):")
    for r in selected:
        print(f"  {r['patient'].name}: Dice = {r['dice']:.4f}")

    # ---- Render figure ----
    n_rows = len(selected)
    fig, axes = plt.subplots(
        n_rows, 5,
        figsize=(21, FIGSIZE_PER_ROW * n_rows)
    )
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["Prediction", "Ground Truth", "Overlay", "Confusion Map", "Summary"]
    col_colors = ["#0B8A2A", "#C62828", "#1E3A8A", "#6A1B9A", "#37474F"]

    for row, item in enumerate(selected):
        patient = item["patient"]
        dice = item["dice"]

        image = nib.load(patient / "image.nii.gz").get_fdata()
        gt = nib.load(patient / "label.nii.gz").get_fdata() > 0
        pred = nib.load(patient / "prediction.nii.gz").get_fdata() > 0

        areas = gt.sum(axis=(0, 1))
        z = int(np.argmax(areas))

        img_slice = np.rot90(image[:, :, z])
        gt_slice = np.rot90(gt[:, :, z])
        pred_slice = np.rot90(pred[:, :, z])

        img_disp = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)

        boxes = find_lesion_boxes(gt_slice, pred_slice, MAX_LESIONS, MIN_COMPONENT_VOXELS, ZOOM_MARGIN)

        draw_panel(
            axes[row, 0], img_disp, pred=pred_slice, boxes=boxes,
            title=col_titles[0] if row == 0 else "", title_color=col_colors[0]
        )
        draw_panel(
            axes[row, 1], img_disp, gt=gt_slice, boxes=boxes,
            title=col_titles[1] if row == 0 else "", title_color=col_colors[1]
        )
        draw_panel(
            axes[row, 2], img_disp, pred=pred_slice, gt=gt_slice, boxes=boxes,
            title=col_titles[2] if row == 0 else "", title_color=col_colors[2]
        )
        plot_confusion_matrix_ax(
            axes[row, 3], gt, pred,
            title=col_titles[3] if row == 0 else "", title_color=col_colors[3]
        )

        meta = metadata_lookup.get(patient.name)
        draw_info_panel(
            axes[row, 4], patient.name, meta, dice,
            gt_agatston=item.get("gt_agatston"), pred_agatston=item.get("pred_agatston")
        )
        if row == 0:
            axes[row, 4].set_title(col_titles[4], fontsize=13, fontweight="bold",
                                     color=col_colors[4], pad=6)

        axes[row, 0].set_ylabel(
            patient.name, fontsize=14, fontweight="bold", rotation=90, labelpad=14
        )

    fig.suptitle("Qualitative Segmentation Results (Best \u2192 Worst Dice)",
                  fontsize=20, fontweight="bold", y=1.01)
    plt.subplots_adjust(hspace=0.15, wspace=0.06)
    plt.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"\nSaved figure to {OUT_PATH.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
    # Deep-dive on a single scan:
    # visualize_single_scan("fd31be0151e9")