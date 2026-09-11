"""
agatston_script.py
────────────────────────────────────────────────────────────────────
Calculates the standard Agatston CAC score from a NIfTI CT scan
and its corresponding binary calcium mask.

Clinical standard:
  - Slice thickness threshold : 1.5 mm  (lesions in thicker slices scaled)
  - Minimum lesion area       : 1 mm²   (≈ 1 pixel at 1mm isotropic)
  - Density weights (HU):
        130 – 199  →  1
        200 – 299  →  2
        300 – 399  →  3
        ≥ 400      →  4

Usage:
    from agatston import calculate_agatston

    result = calculate_agatston(
        ct_path   = "path/to/scan_img.nii.gz",
        mask_path = "path/to/scan_calcium_mask.nii.gz",
    )
    print(result)
    # {
    #   "agatston_total": 142.5,
    #   "agatston_per_slice": [...],
    #   "n_lesions": 4,
    #   "n_slices_with_calcium": 6
    # }

Run standalone:
    python agatston.py --ct scan.nii.gz --mask mask.nii.gz
"""

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label


# ══════════════════════════════════════════════════════════════════
#  DENSITY WEIGHT  (standard Agatston table)
# ══════════════════════════════════════════════════════════════════

def _density_weight(peak_hu: float) -> int:
    """
    Return Agatston density weight for a lesion's peak HU value.

    HU range        Weight
    ─────────────── ──────
    130 – 199          1
    200 – 299          2
    300 – 399          3
    ≥ 400              4
    < 130              0   (not calcified — should not appear if mask is clean)
    """
    if peak_hu >= 400:
        return 4
    elif peak_hu >= 300:
        return 3
    elif peak_hu >= 200:
        return 2
    elif peak_hu >= 130:
        return 1
    else:
        return 0  # below threshold


# ══════════════════════════════════════════════════════════════════
#  CORE CALCULATOR
# ══════════════════════════════════════════════════════════════════

def calculate_agatston(ct_path: str, mask_path: str) -> dict:
    """
    Calculate the standard Agatston score.

    Algorithm (per-slice, per-lesion):
        1. Load CT + mask, orient both to RAS so axial = axis 0.
        2. For each axial slice:
            a. Get binary calcium map from mask.
            b. Label connected components (lesions).
            c. For each lesion:
                - area_mm2  = n_pixels × pixel_area_mm2
                - skip if area_mm2 < 1 mm²   (noise filter)
                - peak_hu   = max HU in lesion region on CT
                - weight    = density_weight(peak_hu)
                - slice_score = area_mm2 × weight × thickness_factor
        3. Agatston total = sum of all slice_scores.

    Thickness factor:
        Standard Agatston was defined for 3 mm slices.
        For other thicknesses: factor = slice_thickness_mm / 3.0
        (so 1 mm slices contribute 1/3 the score per area unit,
         preserving comparability across protocols)

    Args:
        ct_path   : path to CT NIfTI (.nii.gz)
        mask_path : path to binary calcium mask NIfTI (.nii.gz)
                    (1 = calcium, 0 = background)

    Returns:
        dict with keys:
            agatston_total          : float  — total score
            agatston_per_slice      : list   — score per axial slice (sparse, only non-zero)
            n_lesions               : int    — total lesions counted across all slices
            n_slices_with_calcium   : int    — slices that contributed to score
    """

    # ── Load ──────────────────────────────────────────────────────
    ct_sitk   = sitk.ReadImage(str(ct_path),   sitk.sitkFloat32)
    mask_sitk = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)

    # ── Orient to RAS so axis-0 in numpy = axial (z) slices ───────
    ct_sitk   = sitk.DICOMOrient(ct_sitk,   "RAS")
    mask_sitk = sitk.DICOMOrient(mask_sitk, "RAS")

    ct_arr   = sitk.GetArrayFromImage(ct_sitk)    # (z, y, x)
    mask_arr = sitk.GetArrayFromImage(mask_sitk)  # (z, y, x)

    # ── Spacing ───────────────────────────────────────────────────
    # sitk spacing order: (x, y, z) → flip for numpy (z, y, x)
    spacing_xyz     = ct_sitk.GetSpacing()          # (sx, sy, sz)
    slice_thickness = float(spacing_xyz[2])         # sz  (axial)
    pixel_area_mm2  = float(spacing_xyz[0]) * float(spacing_xyz[1])

    # Agatston thickness normalisation factor (defined at 3 mm)
    thickness_factor = slice_thickness / 3.0

    # ── Validate shapes match ─────────────────────────────────────
    if ct_arr.shape != mask_arr.shape:
        raise ValueError(
            f"Shape mismatch: CT {ct_arr.shape} vs mask {mask_arr.shape}. "
            f"Ensure both are registered and resampled to the same grid."
        )

    # ── Per-slice scoring ─────────────────────────────────────────
    agatston_total        = 0.0
    agatston_per_slice    = []   # list of {"slice": z, "score": s}
    n_lesions             = 0
    n_slices_with_calcium = 0

    n_slices = ct_arr.shape[0]

    for z in range(n_slices):

        ct_slice   = ct_arr[z]    # (y, x) float32
        mask_slice = mask_arr[z]  # (y, x) uint8

        assert ct_slice.shape == mask_slice.shape, f"Slice shape mismatch at z={z}: {ct_slice.shape} vs {mask_slice.shape}"

        # Skip empty slices fast
        if not mask_slice.any():
            # print(f"! Skipping slice {z} (no calcium)")
            continue

        # Label connected components in this slice
        labeled, n_components = label(mask_slice)

        if n_components == 0:
            # print(f"? Skipping slice {z} (no labeled components)")
            continue

        slice_score      = 0.0
        slice_has_calcium = False

        for comp_id in range(1, n_components + 1):

            comp_mask = labeled == comp_id          # bool (y, x)

            # ── Area filter ───────────────────────────────────────
            # < 1 mm² → noise / partial volume, skip
            area_mm2 = comp_mask.sum() * pixel_area_mm2
            if area_mm2 < 1.0:
                # print(f"# Skipping lesion in slice {z} with area {area_mm2:.4f} mm² (< 1 mm²)")
                continue

            # ── Peak HU in this lesion ────────────────────────────
            peak_hu = float(ct_slice[comp_mask].max())

            # ── Density weight ────────────────────────────────────
            weight = _density_weight(peak_hu)
            if weight == 0:
                # Mask contains sub-130 HU voxels — skip gracefully
                print(f"& Skipping lesion in slice {z} with peak HU {peak_hu:.2f} (< 130 HU)")
                continue

            # ── Lesion score ──────────────────────────────────────
            # area_mm2 × density_weight × thickness_factor
            lesion_score  = area_mm2 * weight * thickness_factor
            slice_score  += lesion_score
            n_lesions    += 1
            slice_has_calcium = True

        if slice_has_calcium:
            agatston_total        += slice_score
            n_slices_with_calcium += 1
            agatston_per_slice.append({
                "slice": int(z),
                "score": round(slice_score, 4),
            })

    return {
        "agatston_total"         : round(agatston_total, 2),
        "agatston_per_slice"     : agatston_per_slice,
        "n_lesions"              : n_lesions,
        "n_slices_with_calcium"  : n_slices_with_calcium,
    }


# ══════════════════════════════════════════════════════════════════
#  STANDALONE
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(
        description="Calculate Agatston CAC score from NIfTI CT + calcium mask."
    )
    parser.add_argument("--ct",   required=True, help="Path to CT NIfTI (.nii.gz)")
    parser.add_argument("--mask", required=True, help="Path to calcium mask NIfTI (.nii.gz)")
    args = parser.parse_args()

    print("\n─── Agatston Score Calculator ───")
    print(f"CT   : {args.ct}")
    print(f"Mask : {args.mask}\n")

    result = calculate_agatston(args.ct, args.mask)

    print(f"  Agatston Total          : {result['agatston_total']}")
    print(f"  Lesions counted         : {result['n_lesions']}")
    print(f"  Slices with calcium     : {result['n_slices_with_calcium']}")
    print(f"\n  Per-slice breakdown:")
    for s in result["agatston_per_slice"]:
        print(f"    slice {s['slice']:>4d}  →  {s['score']:.4f}")

    # print("\n─── Full JSON ───")
    # print(json.dumps(result, indent=2))