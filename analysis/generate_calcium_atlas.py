"""
generate_calcium_atlas.py

Step 2 of the Population Cohort Approach.
This script extracts calcium deposits from the diseased COCA dataset
and registers them onto the standardized Reference Heart (ImageCAS Patient 1).

The output is a 'Population Calcium Heatmap' in the Reference space.
"""
import sys
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk

# Config
COCA_DIR = Path("/Users/karan/Desktop/PrediCT/COCA")
IMAGECAS_DIR = Path("/Users/karan/Desktop/PrediCT/1-200")
OUTPUT_DIR = Path("/Users/karan/Desktop/PrediCT/output_population")
REFERENCE_PATIENT = "1"
# Limit to 20 patients for demo speed
TARGET_SCANS = 20

def register_images(fixed_img: sitk.Image, moving_img: sitk.Image) -> sitk.Transform:
    """Rigid + Affine registration from moving to fixed space."""
    print("      Running Rigid Registration...")
    init_tx = sitk.CenteredTransformInitializer(
        fixed_img, moving_img,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(sitk.ImageRegistrationMethod.REGULAR)
    R.SetMetricSamplingPercentage(0.1)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=0.01, numberOfIterations=100)
    R.SetInitialTransform(init_tx, inPlace=False)
    rigid_tx = R.Execute(fixed_img, moving_img)
    
    print("      Running Affine Registration...")
    Ra = sitk.ImageRegistrationMethod()
    Ra.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    Ra.SetMetricSamplingStrategy(sitk.ImageRegistrationMethod.RANDOM)
    Ra.SetMetricSamplingPercentage(0.2)
    Ra.SetInterpolator(sitk.sitkLinear)
    Ra.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=0.01, numberOfIterations=100)
    
    # Pre-resample using rigid
    rigid_moving = sitk.Resample(
        moving_img, fixed_img, rigid_tx,
        sitk.sitkLinear, 0.0, sitk.sitkFloat32
    )
    
    affine_tx = sitk.AffineTransform(3)
    fixed_center = fixed_img.TransformContinuousIndexToPhysicalPoint(
        [sz / 2.0 for sz in fixed_img.GetSize()]
    )
    affine_tx.SetCenter(fixed_center)
    Ra.SetInitialTransform(affine_tx, inPlace=False)
    
    final_affine = Ra.Execute(fixed_img, rigid_moving)
    
    c = sitk.CompositeTransform(3)
    c.AddTransform(rigid_tx)
    c.AddTransform(final_affine)
    return c

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    atlas_out = OUTPUT_DIR / "calcium_atlas.csv"
    
    print("==========================================================")
    print("  STEP 2: GENERATE DISEASED POPULATION CALCIUM ATLAS")
    print("==========================================================")
    
    # Load Reference Image (CCTA)
    ref_img_path = IMAGECAS_DIR / f"{REFERENCE_PATIENT}.img.nii.gz"
    print(f"Loading Reference Heart: Patient {REFERENCE_PATIENT}")
    ref_img = sitk.ReadImage(str(ref_img_path), sitk.sitkFloat32)
    
    # Find COCA patients
    coca_patients = [p for p in COCA_DIR.iterdir() if p.is_dir()]
    
    all_calcium_points = []
    processed = 0
    
    for p_dir in coca_patients:
        if processed >= TARGET_SCANS:
            break
            
        patient_id = p_dir.name
        
        # Skip known corrupted HDF5 files
        if patient_id == "1e829fd3f94e":
            continue
            
        img_path = p_dir / f"{patient_id}_raw_img.nii.gz"
        seg_path = p_dir / f"{patient_id}_seg.nii.gz"
        
        if not img_path.exists() or not seg_path.exists():
            continue
            
        print(f"\nProcessing Diseased Patient {patient_id}...")
        moving_raw = sitk.ReadImage(str(img_path), sitk.sitkFloat32)
        
        # Downsample for speed
        def resample_volume(image: sitk.Image, spacing_mm: float) -> sitk.Image:
            old_spacing = np.array(image.GetSpacing())
            old_size    = np.array(image.GetSize())
            new_size    = np.round(old_size * old_spacing / spacing_mm).astype(int).tolist()
            if new_size[0] == 0: return image
            rs = sitk.ResampleImageFilter()
            rs.SetOutputSpacing([spacing_mm] * 3)
            rs.SetSize(new_size)
            rs.SetOutputOrigin(image.GetOrigin())
            rs.SetOutputDirection(image.GetDirection())
            rs.SetInterpolator(sitk.sitkLinear)
            return rs.Execute(image)
            
        moving_img = resample_volume(moving_raw, 5.0)
        ref_img_ds = resample_volume(ref_img, 5.0)
        
        print("  Registering to Reference space (MACROSCOPIC ALIGNMENT)...")
        # NOTE: This aligns the macro-anatomy (epicardial fat).
        # This will perfectly align the NCCT fat with the CCTA fat, dropping
        # the calcium into the exact path of the CCTA arteries!
        try:
            transform = register_images(ref_img_ds, moving_img)
        except Exception as e:
            print(f"  Registration failed: {e}")
            continue
            
        print("  Extracting native calcium geometry...")
        try:
            seg_img = sitk.ReadImage(str(seg_path), sitk.sitkUInt8)
        except Exception as e:
            print(f"  Corrupted or missing segmentation for {patient_id}. Skipping.")
            continue
            
        seg_arr = sitk.GetArrayFromImage(seg_img)
        
        z, y, x = np.where(seg_arr > 0)
        
        if len(x) == 0:
            print("  No calcium found. Skipping.")
            continue
            
        native_points = []
        # Take a subset of calcium points to represent density
        stride = max(1, len(x) // 500) 
        for i in range(0, len(x), stride):
            pt = seg_img.TransformIndexToPhysicalPoint([int(x[i]), int(y[i]), int(z[i])])
            native_points.append(pt)
            
        print(f"  Projecting {len(native_points)} calcium points to Reference space...")
        for pt in native_points:
            try:
                inv_tx = transform.GetInverse()
                ref_pt = inv_tx.TransformPoint(pt)
                
                all_calcium_points.append({
                    "patient": patient_id,
                    "x": ref_pt[0], "y": ref_pt[1], "z": ref_pt[2]
                })
            except:
                pass
                
        processed += 1
                
    # Save the Population Calcium Atlas
    df = pd.DataFrame(all_calcium_points)
    df.to_csv(atlas_out, index=False)
    print(f"\nSuccessfully saved Population Calcium Atlas ({len(df)} points) to {atlas_out}")

if __name__ == "__main__":
    main()
