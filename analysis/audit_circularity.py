"""
Phase 1 Scientific Audit: Circular Alignment Hypothesis

Tests whether Phase 1 Multi-modal Atlas Registration is artificially 
warping the coronary arteries onto calcium deposits in NCCT scans.
"""
import sys
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import pandas as pd

def audit_registration_circularity():
    coca_dir = Path("/Users/karan/Desktop/PrediCT/COCA")
    output_dir = Path("/Users/karan/Desktop/PrediCT/output")
    
    print("==========================================================")
    print("  PHASE 1 SCIENTIFIC AUDIT: CALCIUM-BIASED REGISTRATION")
    print("==========================================================")
    print("Hypothesis: Registration aligns bright CCTA arteries (atlas)")
    print("onto bright NCCT calcium (patient), creating circular logic.")
    print("==========================================================\n")
    
    results = []
    
    # Iterate over processed patients in the output directory
    fused_masks = list(Path("/Users/karan/Desktop/PrediCT/approved_masks").glob("*_vessels.nii.gz"))
    
    if not fused_masks:
        print("No fused masks found. Run Phase 1 first.")
        return
        
    for mask_path in fused_masks[:5]: # Test on up to 5 patients
        patient_id = mask_path.name.split("_")[0]
        
        # Load Patient NCCT (Raw image)
        raw_img_path = coca_dir / patient_id / f"{patient_id}_raw_img.nii.gz"
        if not raw_img_path.exists():
            continue
            
        raw_img = sitk.ReadImage(str(raw_img_path), sitk.sitkFloat32)
        raw_arr = sitk.GetArrayFromImage(raw_img)
        
        # Load Patient Calcium Ground Truth (if exists)
        calcium_path = coca_dir / patient_id / f"{patient_id}_seg.nii.gz"
        has_calcium = calcium_path.exists()
        
        if has_calcium:
            calc_img = sitk.ReadImage(str(calcium_path), sitk.sitkUInt8)
            calc_arr = sitk.GetArrayFromImage(calc_img)
            total_calc_vol = np.sum(calc_arr > 0)
        else:
            calc_arr = np.zeros_like(raw_arr)
            total_calc_vol = 0
            
        # Load Registered Vessel Mask
        vessel_img = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
        
        # Resample vessel mask to raw image space just in case
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(raw_img)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        vessel_aligned = resampler.Execute(vessel_img)
        vessel_arr = sitk.GetArrayFromImage(vessel_aligned) > 0
        
        # --- METRICS ---
        
        # 1. Intensity Analysis
        # Blood in NCCT is ~30-50 HU. Myocardium is ~40-60 HU. Calcium is >130 HU.
        # If the registered vessel has a mean HU >> 60, it's locking onto calcium.
        vessel_hu = raw_arr[vessel_arr]
        mean_vessel_hu = np.mean(vessel_hu) if len(vessel_hu) > 0 else 0
        
        # 2. Calcium Capture Rate
        # What percentage of the patient's total calcium was "captured" inside the vessel mask?
        if total_calc_vol > 0:
            captured_calc = np.sum((calc_arr > 0) & vessel_arr)
            calc_capture_pct = (captured_calc / total_calc_vol) * 100
        else:
            calc_capture_pct = 0.0
            
        # 3. Calcium Density in Mask
        # What percentage of the vessel mask is composed of calcium?
        vessel_vol = np.sum(vessel_arr)
        if vessel_vol > 0:
            calc_density_pct = (np.sum((calc_arr > 0) & vessel_arr) / vessel_vol) * 100
        else:
            calc_density_pct = 0.0
            
        results.append({
            "Patient": patient_id,
            "Total Calcium (vox)": total_calc_vol,
            "Mean Vessel HU": mean_vessel_hu,
            "Calcium Capture %": calc_capture_pct,
            "Mask Calc Density %": calc_density_pct
        })
        
    # Print Report
    df = pd.DataFrame(results)
    df = df.sort_values(by="Total Calcium (vox)", ascending=False)
    
    print(df.to_string(index=False, float_format="%.1f"))
    
    print("\n--- CONCLUSION ---")
    print("If 'Mean Vessel HU' > 70-80 HU for highly calcified patients, but ~40 HU for")
    print("low-calcium patients, it proves the registration algorithm is actively stretching")
    print("the atlas arteries to snap onto the bright calcium spots.")
    print("This confirms the user's circular logic hypothesis.")

if __name__ == "__main__":
    audit_registration_circularity()
