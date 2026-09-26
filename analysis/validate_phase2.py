import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import binary_dilation, label
from scipy.stats import spearmanr
from sklearn.metrics import mutual_info_score

def extract_calcium_blobs(ct_data, hu_threshold=130, min_size=3):
    calcium_mask = (ct_data > hu_threshold).astype(int)
    labeled_array, num_features = label(calcium_mask)
    valid_calcium = np.zeros_like(calcium_mask)
    for i in range(1, num_features + 1):
        blob = (labeled_array == i)
        if np.sum(blob) >= min_size:
            valid_calcium[blob] = 1
    return valid_calcium

def validate_phase2(patient_id):
    print(f"--- Phase 2 Biological Validation for Patient {patient_id} ---")
    base_dir = Path("/Users/karan/Desktop/PrediCT/approved_masks")
    export_dir = Path("/Users/karan/Desktop/PrediCT/output_v2/exports/nifti")
    
    ct_path = base_dir / f"{patient_id}_patient_ct.nii.gz"
    mask_path = base_dir / f"{patient_id}_vessels.nii.gz"
    ess_path = export_dir / "ess.nii.gz"
    wall_path = export_dir / "wall_mask.nii.gz"
    
    ct_data = nib.load(ct_path).get_fdata()
    mask_data = nib.load(mask_path).get_fdata()
    ess_data = nib.load(ess_path).get_fdata()
    wall_data = nib.load(wall_path).get_fdata()
    
    # 1. Get coronary calcium map
    dilated_vessels = binary_dilation(mask_data > 0, iterations=8)
    coronary_region_ct = np.copy(ct_data)
    coronary_region_ct[~dilated_vessels] = -1000 
    calcium_mask = extract_calcium_blobs(coronary_region_ct)
    
    # 2. Extract ESS on the wall
    wall_indices = np.where(wall_data > 0)
    ess_at_wall = ess_data[wall_indices]
    
    # 3. Determine if calcium is near each wall voxel (within 2 voxels)
    calcium_dilated = binary_dilation(calcium_mask > 0, iterations=2).astype(int)
    calcium_at_wall = calcium_dilated[wall_indices]
    
    # Filter out zeros from ess_at_wall (since scatter operation might leave some empty if resolution mismatched)
    valid_pts = ess_at_wall > 1e-8
    ess_valid = ess_at_wall[valid_pts]
    calcium_valid = calcium_at_wall[valid_pts]
    
    print(f"Valid wall points evaluated: {len(ess_valid)}")
    print(f"Wall points near calcium: {np.sum(calcium_valid)}")
    print(f"Wall points NOT near calcium: {len(ess_valid) - np.sum(calcium_valid)}")
    
    if len(ess_valid) == 0 or np.sum(calcium_valid) == 0:
        print("Not enough valid data to compute correlation.")
        return
        
    # Calculate Spearman correlation
    # We want a negative correlation: as calcium increases (1), ESS should decrease.
    corr, p_value = spearmanr(calcium_valid, ess_valid)
    print(f"Spearman Correlation (Calcium Presence vs ESS): {corr:.4f} (p-value: {p_value:.4e})")
    
    # Calculate Mutual Information
    # Bin ESS into low and high (e.g., below median vs above median, or below 1.0 Pa vs above 1.0 Pa)
    median_ess = np.median(ess_valid)
    low_ess_binary = (ess_valid < median_ess).astype(int)
    mi = mutual_info_score(calcium_valid, low_ess_binary)
    print(f"Mutual Information (Low ESS vs Calcium Presence): {mi:.4f}")
    
    # Simple means for context
    mean_ess_calcium = np.mean(ess_valid[calcium_valid == 1])
    mean_ess_healthy = np.mean(ess_valid[calcium_valid == 0])
    print(f"Mean ESS in calcified regions: {mean_ess_calcium:.6f} Pa")
    print(f"Mean ESS in healthy regions  : {mean_ess_healthy:.6f} Pa")
    
    if mean_ess_calcium < mean_ess_healthy:
        print("VALIDATION PASSED: ESS is lower in regions with calcium.")
    else:
        print("VALIDATION WARNING: ESS is NOT lower in calcified regions.")

if __name__ == "__main__":
    validate_phase2("1a19d11e263a")
