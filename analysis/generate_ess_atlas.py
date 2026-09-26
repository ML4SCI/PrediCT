"""
generate_ess_atlas.py

Step 1 of the Population Cohort Approach.
This script calculates the Endothelial Shear Stress (ESS) on a set of healthy 
CCTA patients from the ImageCAS dataset, and registers the physical coordinates
onto a single Reference Heart (ImageCAS Patient 1).

The output is a 'Population ESS Heatmap' in the Reference space.
"""
import sys
import os
import time
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import SimpleITK as sitk

# Config
IMAGECAS_DIR = Path("/Users/karan/Desktop/PrediCT/1-200")
OUTPUT_DIR = Path("/Users/karan/Desktop/PrediCT/output_population")
REFERENCE_PATIENT = "1"
TARGET_PATIENTS = ["2", "3", "4", "5", "6"]

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
    atlas_out = OUTPUT_DIR / "ess_atlas.csv"
    
    print("==========================================================")
    print("  STEP 1: GENERATE HEALTHY POPULATION ESS ATLAS")
    print("==========================================================")
    
    # Load Reference Image
    ref_img_path = IMAGECAS_DIR / f"{REFERENCE_PATIENT}.img.nii.gz"
    if not ref_img_path.exists():
        print(f"Error: Reference patient {REFERENCE_PATIENT} not found at {ref_img_path}")
        return
        
    print(f"Loading Reference Heart: Patient {REFERENCE_PATIENT}")
    ref_img = sitk.ReadImage(str(ref_img_path), sitk.sitkFloat32)
    
    all_ess_points = []
    
    for patient_id in TARGET_PATIENTS:
        print(f"\nProcessing Healthy Patient {patient_id}...")
        img_path = IMAGECAS_DIR / f"{patient_id}.img.nii.gz"
        lbl_path = IMAGECAS_DIR / f"{patient_id}.label.nii.gz"
        
        if not img_path.exists() or not lbl_path.exists():
            print(f"  Missing files for patient {patient_id}. Skipping.")
            continue
            
        moving_img = sitk.ReadImage(str(img_path), sitk.sitkFloat32)
        
        # 1. Register Patient to Reference Heart
        print("  Registering to Reference space...")
        try:
            transform = register_images(ref_img, moving_img)
        except Exception as e:
            print(f"  Registration failed: {e}")
            continue
            
        # 2. Run Phase 2 (PINN) to get native ESS
        # For the sake of population throughput, we can use an analytical Poiseuille approximation
        # on the centerline radius for the healthy cohort, OR we can call the full PINN.
        # Since these are strictly healthy, non-diseased atlases, the radius completely dictates healthy ESS.
        
        print("  Extracting native vessel geometry...")
        # (This is where the PINN or analytical solver runs)
        # We will load the label, extract the surface, and compute WSS = 4 * mu * Q / (pi * R^3)
        # For now, to establish the pipeline architecture, we will extract the surface points
        # and assign a mock ESS value based on radius to verify the spatial projection works.
        
        lbl_img = sitk.ReadImage(str(lbl_path), sitk.sitkUInt8)
        lbl_arr = sitk.GetArrayFromImage(lbl_img)
        spacing = lbl_img.GetSpacing()
        
        # Get physical coordinates of the vessel wall (edge of the mask)
        import scipy.ndimage as ndi
        boundary = (lbl_arr > 0) ^ ndi.binary_erosion(lbl_arr > 0, iterations=1)
        z, y, x = np.where(boundary)
        
        # Approximate local radius using Distance Transform for analytical ESS
        dist_transform = ndi.distance_transform_edt(lbl_arr > 0, sampling=spacing[::-1])
        
        # Convert voxels to physical native coordinates
        native_points = []
        for i in range(len(x)):
            # Every 100th point to keep file size small for the demo
            if i % 100 == 0:
                pt = lbl_img.TransformIndexToPhysicalPoint([int(x[i]), int(y[i]), int(z[i])])
                # Extract distance to center (approx radius at this cross-section)
                # The max distance in the local region is the radius
                local_r = np.max(dist_transform[max(0, z[i]-2):z[i]+3, 
                                                max(0, y[i]-2):y[i]+3, 
                                                max(0, x[i]-2):x[i]+3])
                native_points.append((pt, local_r))
                
        # 3. Project coordinates to Reference space
        print(f"  Projecting {len(native_points)} wall points to Reference space...")
        for pt, local_r in native_points:
            # Transform point from moving (native) space to fixed (reference) space
            try:
                inv_tx = transform.GetInverse()
                ref_pt = inv_tx.TransformPoint(pt)
                
                # Analytical Poiseuille ESS Calculation
                # mu = 0.0035 Pa.s, U_mean ~ 0.25 m/s (healthy resting)
                # ESS = 4 * mu * U_mean / R
                # local_r is in mm, convert to meters
                r_meters = max(local_r, 0.5) / 1000.0
                ess_pa = 4 * 0.0035 * 0.25 / r_meters
                
                all_ess_points.append({
                    "patient": patient_id,
                    "x": ref_pt[0], "y": ref_pt[1], "z": ref_pt[2],
                    "ess": ess_pa
                })
            except:
                pass
                
    # Save the Population ESS Atlas
    df = pd.DataFrame(all_ess_points)
    df.to_csv(atlas_out, index=False)
    print(f"\nSuccessfully saved Population ESS Atlas ({len(df)} points) to {atlas_out}")

if __name__ == "__main__":
    main()
