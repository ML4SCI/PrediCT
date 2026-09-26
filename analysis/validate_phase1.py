import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import binary_dilation, label

def extract_calcium_blobs(ct_data, hu_threshold=130, min_size=3):
    """Extracts calcium blobs from CT data."""
    calcium_mask = (ct_data > hu_threshold).astype(int)
    labeled_array, num_features = label(calcium_mask)
    
    valid_calcium = np.zeros_like(calcium_mask)
    for i in range(1, num_features + 1):
        blob = (labeled_array == i)
        if np.sum(blob) >= min_size:
            valid_calcium[blob] = 1
            
    return valid_calcium

def validate_phase1(patient_id):
    print(f"--- Phase 1 Validation for Patient {patient_id} ---")
    base_dir = Path("/Users/karan/Desktop/PrediCT/approved_masks")
    ct_path = base_dir / f"{patient_id}_patient_ct.nii.gz"
    mask_path = base_dir / f"{patient_id}_vessels.nii.gz"
    
    # Load NIfTIs
    ct_img = nib.load(ct_path)
    ct_data = ct_img.get_fdata()
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()
    
    print(f"Loaded CT shape: {ct_data.shape}")
    print(f"Loaded Mask shape: {mask_data.shape}")
    
    # Dilate vessel mask to get coronary region
    dilated_vessels = binary_dilation(mask_data > 0, iterations=8) 
    
    # Find calcium in the coronary region
    coronary_region_ct = np.copy(ct_data)
    coronary_region_ct[~dilated_vessels] = -1000 
    
    clinical_calcium = extract_calcium_blobs(coronary_region_ct)
    total_calcium_voxels = np.sum(clinical_calcium)
    
    print(f"Total clinical coronary calcium voxels detected: {total_calcium_voxels}")
    
    if total_calcium_voxels == 0:
        print("No calcium found to validate against.")
        return
        
    tight_wall = binary_dilation(mask_data > 0, iterations=2)
    overlap = np.logical_and(clinical_calcium, tight_wall)
    overlap_voxels = np.sum(overlap)
    
    alignment_score = (overlap_voxels / total_calcium_voxels) * 100
    print(f"Calcium voxels aligned with Phase 1 geometry: {overlap_voxels}")
    print(f"Phase 1 Calcium Alignment Score: {alignment_score:.2f}%")
    
    if alignment_score > 90:
        print("VALIDATION PASSED: Phase 1 vessel extraction correctly encompasses clinical calcium.")
    else:
        print("VALIDATION WARNING: Misalignment detected.")

if __name__ == "__main__":
    validate_phase1("1a19d11e263a")
