"""
run_batch_pinn.py

Executes the Phase 2 PINN physics solver sequentially across the healthy ImageCAS 
population to compute the true 3D Endothelial Shear Stress (ESS) fields.
"""
import os
import sys
import subprocess
from pathlib import Path
import shutil

IMAGECAS_DIR = Path("/Users/karan/Desktop/PrediCT/1-200")
EXPORT_DIR = Path("/Users/karan/Desktop/PrediCT/output_v2/exports")
OUTPUT_POPULATION_DIR = Path("/Users/karan/Desktop/PrediCT/output_population/pinn_results")

def main():
    OUTPUT_POPULATION_DIR.mkdir(parents=True, exist_ok=True)
    
    # We will run the PINN on 5 healthy patients overnight
    TARGET_PATIENTS = ["2", "3", "4", "5", "6"]
    
    print("==========================================================")
    print("  BATCH PINN EXECUTION (POPULATION COHORT)")
    print("==========================================================")
    
    for patient_id in TARGET_PATIENTS:
        print(f"\n--- Starting PINN for Patient {patient_id} ---")
        lbl_path = IMAGECAS_DIR / f"{patient_id}.label.nii.gz"
        
        if not lbl_path.exists():
            print(f"Mask {lbl_path} not found. Skipping.")
            continue
            
        env = os.environ.copy()
        env["PREDICT_MASK_PATH"] = str(lbl_path)
        env["PREDICT_RUN_NAME"] = f"imagecas_{patient_id}"
        
        # Run Phase 2
        try:
            # We don't capture stdout/stderr here so it streams to the daemon log
            subprocess.run(
                [sys.executable, "src/phase2_hemodynamics/run_phase2.py"],
                env=env,
                check=True
            )
            
            # Copy the results to population folder
            csv_path = EXPORT_DIR / f"imagecas_{patient_id}_ess_predictions.csv"
            if csv_path.exists():
                shutil.copy(csv_path, OUTPUT_POPULATION_DIR / f"{patient_id}_ess_pinn.csv")
                print(f"Saved PINN results for {patient_id}.")
            else:
                print(f"Error: {csv_path} was not generated.")
                
        except subprocess.CalledProcessError as e:
            print(f"PINN failed for patient {patient_id}: {e}")
            continue

    print("\nBatch execution complete. Next step: project these PINN results to Reference space.")

if __name__ == "__main__":
    main()
