import numpy as np
import SimpleITK as sitk
from pathlib import Path
import argparse
import logging
import scipy.ndimage as ndi

def texture_calcium(ncct_img: sitk.Image, synth_mask_img: sitk.Image, vessel_img: sitk.Image) -> sitk.Image:
    """
    Overwrites the original NCCT voxels with synthetic calcium HU values where synth_mask is 1.
    HU values are drawn from a bell-curve distribution (mean ~300, std ~100, clipped 130–600).
    """
    ncct_arr = sitk.GetArrayFromImage(ncct_img)
    
    # 1. Resample Calcium Mask (Linear for smooth gradients)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ncct_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    synth_mask_img_resampled = resampler.Execute(synth_mask_img)
    synth_mask = sitk.GetArrayFromImage(synth_mask_img_resampled).astype(np.float64)
    
    # 2. Resample Vessel Mask (NearestNeighbor for strict boolean bounds)
    resampler_nn = sitk.ResampleImageFilter()
    resampler_nn.SetReferenceImage(ncct_img)
    resampler_nn.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler_nn.SetDefaultPixelValue(0)
    vessel_resampled = resampler_nn.Execute(vessel_img)
    vessel_arr = sitk.GetArrayFromImage(vessel_resampled)
    
    # 2.5 ANTI-PIXELATION: Gaussian blur on the fine NCCT grid
    # The calcium mask was generated on a coarse 1mm grid, so even after linear
    # resampling to the 0.375mm NCCT grid, the edges show staircase artifacts.
    # A gentle Gaussian blur at the fine resolution smooths out these pixel stairs
    # while preserving the overall shape and gradient profile.
    ncct_spacing = np.array(ncct_img.GetSpacing()[::-1])  # Z, Y, X
    sigma_mm = 0.6  # 0.6mm blur — enough to soften edges, small enough to stay tight
    sigma_vox = sigma_mm / ncct_spacing
    synth_mask = ndi.gaussian_filter(synth_mask, sigma=sigma_vox)
    
    # 3. STRICT BOUNDARY ENFORCEMENT & RE-NORMALIZATION
    # Erase any Gaussian blur / linear interpolation tails that bled outside the vessel wall
    synth_mask *= (vessel_arr > 0)
    
    # Now that the leaks are erased, we can safely re-normalize the peak to 1.0
    if synth_mask.max() > 0:
        synth_mask = synth_mask / synth_mask.max()
    
    # The mask from Phase 3 is now a continuous float field (0.0–1.0)
    # with smooth Gaussian edges strictly inside the vessel. We use it as an alpha channel.
    active_mask = synth_mask > 0.05
    num_active_voxels = int(np.sum(active_mask))
    
    if num_active_voxels == 0:
        return ncct_img
    
    # HU distribution: Calcium must be visibly distinct from contrast-enhanced blood (~300 HU).
    # Dense calcium core can easily reach 800-1200 HU (bone density). 
    # We increase the target HU so it is blindingly visible to the naked eye in the final CT.
    hu_mean = float(np.clip(np.random.normal(loc=850, scale=100), 600, 1200))
    hu_std  = float(np.clip(np.random.normal(loc=150, scale=30), 50, 300))
    
    hu_values = np.random.normal(loc=hu_mean, scale=hu_std, size=num_active_voxels)
    hu_values = np.clip(hu_values, 130, 1500)
    
    logger = logging.getLogger(__name__)
    logger.info(f"HU Distribution: mean={hu_mean:.0f}, std={hu_std:.0f}, "
                f"actual range=[{hu_values.min():.0f}, {hu_values.max():.0f}]")
    
    # --- Continuous Injection Strategy ---
    # We apply a perfectly continuous degrading function (alpha blend) across ALL voxels.
    # Voxels near the seed (alpha ~ 1.0) become almost pure calcium.
    # Voxels further away (alpha ~ 0.1) blend softly into the surrounding tissue.
    # This prevents the edges from looking completely white or blocky.
    
    calcium_hu_vol = np.zeros_like(ncct_arr, dtype=np.float64)
    calcium_hu_vol[active_mask] = hu_values
    
    alpha = synth_mask[active_mask]
    ncct_arr[active_mask] = (
        (1.0 - alpha) * ncct_arr[active_mask] +
        alpha * calcium_hu_vol[active_mask]
    )
    
    logger.info(f"Continuously alpha-blended {num_active_voxels} voxels using degrading function.")
    
    # --- 3. Compute Agatston Score ---
    logger.info("Computing Agatston score of synthetic calcium...")
    synthetic_agatston = 0.0
    
    # Agatston is calculated slice-by-slice in the axial plane (Z-axis)
    z_dim = ncct_arr.shape[0]
    slice_thickness = ncct_img.GetSpacing()[2]  # Z-spacing
    area_factor = ncct_img.GetSpacing()[0] * ncct_img.GetSpacing()[1] # X*Y spacing in mm^2
    
    for z in range(z_dim):
        slice_mask = synth_mask[z, :, :]
        if slice_mask.sum() == 0:
            continue
            
        slice_hu = ncct_arr[z, :, :]
        
        # Find connected components of calcium in this 2D slice
        labeled_mask, num_features = ndi.label(slice_mask > 0)
        
        for i in range(1, num_features + 1):
            component_mask = (labeled_mask == i)
            area_mm2 = component_mask.sum() * area_factor
            
            # Agatston requires area >= 1 mm^2 (typically 3-4 voxels)
            if area_mm2 < 1.0:
                continue
                
            max_hu = slice_hu[component_mask].max()
            
            # Density weighting factor
            if 130 <= max_hu < 200:
                weight = 1
            elif 200 <= max_hu < 300:
                weight = 2
            elif 300 <= max_hu < 400:
                weight = 3
            elif max_hu >= 400:
                weight = 4
            else:
                weight = 0
                
            # Normalize to standard 3mm slice thickness if acquired differently
            thickness_factor = slice_thickness / 3.0
            
            synthetic_agatston += (area_mm2 * weight * thickness_factor)
            
    logger.info(f"==> Computed Synthetic Agatston Score: {synthetic_agatston:.1f}")

    # Save back
    synth_ncct_img = sitk.GetImageFromArray(ncct_arr)
    synth_ncct_img.CopyInformation(ncct_img)
    
    return synth_ncct_img

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Physiological Texturing")
    parser.add_argument("--ncct", type=str, required=True, help="Path to original NCCT NIfTI")
    parser.add_argument("--synth_mask", type=str, required=True, help="Path to synthetic calcium mask NIfTI")
    parser.add_argument("--vessel_mask", type=str, required=True, help="Path to 3D vessel mask NIfTI (for strict bounding)")
    parser.add_argument("--output", type=str, required=True, help="Path to save synthetic NCCT scan")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Loading inputs...")
    ncct_img = sitk.ReadImage(args.ncct)
    synth_mask_img = sitk.ReadImage(args.synth_mask)
    vessel_img = sitk.ReadImage(args.vessel_mask)
    
    logger.info("Texturing calcium...")
    synth_ncct_img = texture_calcium(ncct_img, synth_mask_img, vessel_img)
    
    logger.info("Saving synthetic NCCT scan...")
    sitk.WriteImage(synth_ncct_img, args.output)
    logger.info(f"Saved to {args.output}")

if __name__ == "__main__":
    main()

