import os
import nibabel as nib
import numpy as np

masks_dir = "/Users/karan/Desktop/PrediCT/approved_masks"
files = [f for f in os.listdir(masks_dir) if f.endswith("_patient_ct.nii.gz")]

for f in files:
    pid = f.split("_")[0]
    ct_path = os.path.join(masks_dir, f)
    vessel_path = os.path.join(masks_dir, f"{pid}_vessels.nii.gz")
    
    if os.path.exists(vessel_path):
        ct = nib.load(ct_path).get_fdata()
        vessel = nib.load(vessel_path).get_fdata() > 0
        
        calcium_voxels = np.sum((ct > 130) & vessel)
        print(f"Patient {pid}: {calcium_voxels} calcium voxels in vessel mask")
    else:
        print(f"Patient {pid}: No vessel mask found")
