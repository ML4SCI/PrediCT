import sys
sys.path.insert(0, "src/phase1_segmentation")
import argparse
import subprocess
import os
import sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import pandas as pd
from scipy.interpolate import griddata
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Physio-Twin")

def interpolate_ess_to_grid(csv_path: Path, mask_path: Path, out_path: Path):
    """Interpolates sparse point cloud ESS values onto a dense 3D voxel grid."""
    logger.info(f"Interpolating ESS point cloud to dense grid...")
    df = pd.read_csv(csv_path)
    points = df[['x', 'y', 'z']].values
    values = df['ess_magnitude'].values
    
    mask_img = sitk.ReadImage(str(mask_path))
    mask_arr = sitk.GetArrayFromImage(mask_img)
    spacing = mask_img.GetSpacing()
    origin = mask_img.GetOrigin()
    
    # Create coordinate grid for mask
    z_dim, y_dim, x_dim = mask_arr.shape
    z, y, x = np.mgrid[0:z_dim, 0:y_dim, 0:x_dim]
    
    # Convert voxel indices to physical coordinates
    # SimpleITK uses LPS, numpy uses ZYX
    grid_coords = np.column_stack([
        x.ravel() * spacing[0] + origin[0],
        y.ravel() * spacing[1] + origin[1],
        z.ravel() * spacing[2] + origin[2]
    ])
    
    # Only interpolate inside the vessel to save time
    mask_flat = mask_arr.ravel()
    valid_idx = np.where(mask_flat > 0)[0]
    
    if len(valid_idx) == 0:
        logger.warning("Vessel mask is empty. Cannot interpolate ESS.")
        ess_arr = np.zeros_like(mask_arr, dtype=float)
    else:
        valid_coords = grid_coords[valid_idx]
        
        # Interpolate
        # Use nearest to fill gaps, then maybe smooth?
        logger.info(f"Griddata interpolating {len(points)} points to {len(valid_coords)} voxels...")
        interp_vals = griddata(points, values, valid_coords, method='nearest')
        
        ess_arr = np.zeros_like(mask_arr, dtype=float)
        ess_arr_flat = ess_arr.ravel()
        ess_arr_flat[valid_idx] = interp_vals
        ess_arr = ess_arr_flat.reshape(mask_arr.shape)
        
    ess_img = sitk.GetImageFromArray(ess_arr)
    ess_img.CopyInformation(mask_img)
    sitk.WriteImage(ess_img, str(out_path))
    logger.info(f"Dense ESS field saved to {out_path}")

def run_physio_twin(coca_patient_dir: Path, target_agatston: int, out_dir: Path, phase1_only: bool = False):
    coca_patient_dir = coca_patient_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    patient_id = coca_patient_dir.name
    logger.info(f"Starting Physio-Twin pipeline for patient {patient_id}")
    
    # ---------------------------------------------------------
    # Phase 1: Anatomical Scaffolding (Best-Atlas Voting)
    # ---------------------------------------------------------
    logger.info("=== Phase 1: Anatomical Scaffolding (Best-Atlas Voting) ===")
    vessel_mask_path = out_dir / f"{patient_id}_synthetic_vessel.nii.gz"
    
    if not vessel_mask_path.exists():
        logger.info("Running Best-Atlas Voting Registration...")
        from masking import (
            Preprocessor, RegistrationEngine, LabelTransformer,
            get_cardiac_roi, select_atlases_by_ncc, discover_atlas_cases, CFG,
            PlaqueValidator
        )
        
        # Paths
        ncct_path = coca_patient_dir / f"{patient_id}_raw_img.nii.gz"
        if not ncct_path.exists():
            ncct_path = coca_patient_dir / f"{patient_id}_img.nii.gz"
            
        coca_seg = coca_patient_dir / f"{patient_id}_seg.nii.gz"
        
        # Load Patient
        patient_pack = Preprocessor.run_memory_resample(ncct_path, coca_seg)
        fixed_win = patient_pack["reg_win"]
        eval_fixed_raw = patient_pack["eval_raw"]
        eval_coca_seg = patient_pack["eval_lbl"]
        
        # Discover and pre-load atlas pool
        atlas_pool = discover_atlas_cases(CFG.IMAGECAS_DIR)
        logger.info(f"Atlas pool: {len(atlas_pool)} candidates")
        
        # Pre-load atlas images for NCC selection
        atlas_reg_wins = {}
        atlas_lbl_evals = {}
        atlas_lbl_regs = {}
        for case in atlas_pool:
            cid = case["id"]
            pack = Preprocessor.run_memory_resample(case["img"], case["lbl"])
            atlas_reg_wins[cid] = pack["reg_win"]
            atlas_lbl_evals[cid] = pack["eval_lbl"]
            atlas_lbl_regs[cid] = Preprocessor.resample_volume(
                Preprocessor.orient_to_lps(
                    sitk.ReadImage(str(case["lbl"]), sitk.sitkUInt8)
                ),
                CFG.REG_SPACING_MM, is_label=True
            )
        
        # NCC-based atlas selection (top N)
        selected_atlases = select_atlases_by_ncc(
            fixed_win, atlas_pool, atlas_reg_wins, CFG.N_ATLASES
        )
        
        # Register each selected atlas
        transformed_labels = []
        per_atlas_results = []
        
        for aidx, atlas_case in enumerate(selected_atlases, 1):
            cid = atlas_case["id"]
            logger.info(f"  Atlas {aidx}/{CFG.N_ATLASES}: Case {cid}")
            
            atlas_win_reg = atlas_reg_wins[cid]
            atlas_lbl_reg = atlas_lbl_regs[cid]
            atlas_lbl_eval = atlas_lbl_evals[cid]
            
            # Rigid
            rigid_tx, rigid_mi = RegistrationEngine.run_rigid_stage(fixed_win, atlas_win_reg)
            rigid_atlas_lbl = sitk.Resample(
                atlas_lbl_reg, fixed_win, rigid_tx,
                sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
            )
            rigid_lbl_arr = sitk.GetArrayFromImage(rigid_atlas_lbl)
            n_rigid_vox = int(np.count_nonzero(rigid_lbl_arr))
            
            if n_rigid_vox == 0:
                logger.warning(f"  Skipping Atlas {cid} (No rigid overlap)")
                per_atlas_results.append({"atlas_id": cid, "affine_mi": None, "n_warp": 0})
                continue
                
            roi_fixed = get_cardiac_roi(rigid_lbl_arr, CFG.ROI_MARGIN_MM, CFG.REG_SPACING_MM)
            if roi_fixed is None:
                per_atlas_results.append({"atlas_id": cid, "affine_mi": None, "n_warp": 0})
                continue
                
            # Affine
            final_tx, affine_mi = RegistrationEngine.run_affine_stage(
                fixed_win, atlas_win_reg, rigid_tx, roi_fixed, rigid_mi=rigid_mi
            )
            
            # Warp Label
            warped_lbl = LabelTransformer.transform_labels(atlas_lbl_eval, eval_fixed_raw, final_tx)
            warp_arr = sitk.GetArrayFromImage(warped_lbl)
            n_warp = int(np.count_nonzero(warp_arr))
            
            if n_warp > 0:
                transformed_labels.append(warped_lbl)
            
            per_atlas_results.append({"atlas_id": cid, "affine_mi": affine_mi, "n_warp": n_warp})
            logger.info(f"    Warped voxels: {n_warp}, MI: {affine_mi:.4f}")
        
        # --- VOTE: Pick best atlas by MI (most negative = best) ---
        valid_results = [r for r in per_atlas_results if r["affine_mi"] is not None and r["n_warp"] > 0]
        if not valid_results:
            logger.error("CRITICAL: No atlas achieved valid registration.")
            return
            
        best = min(valid_results, key=lambda r: r["affine_mi"])
        
        # Map best atlas back to transformed_labels index
        warp_counter = 0
        best_label_idx = 0
        for r in per_atlas_results:
            if r["n_warp"] > 0:
                if r["atlas_id"] == best["atlas_id"]:
                    best_label_idx = warp_counter
                    break
                warp_counter += 1
        
        best_vessels = transformed_labels[best_label_idx]
        
        # Log the vote
        logger.info(f"Best-Atlas Vote: Winner = Atlas {best['atlas_id']} (MI={best['affine_mi']:.4f})")
        for r in valid_results:
            marker = " ← WINNER" if r["atlas_id"] == best["atlas_id"] else ""
            logger.info(f"  Atlas {r['atlas_id']:>4s}  MI={r['affine_mi']:.4f}  voxels={r['n_warp']}{marker}")
        
        # --- HARD GATE: Phase 1 Validation ---
        best_arr = sitk.GetArrayFromImage(best_vessels)
        best_voxel_count = np.count_nonzero(best_arr)
        logger.info(f"Phase 1 Best Vessel Voxels: {best_voxel_count}")
        if best_voxel_count < 1000:
            logger.error(f"CRITICAL: Phase 1 failed — best atlas has only {best_voxel_count} voxels.")
            return
            
        sitk.WriteImage(best_vessels, str(vessel_mask_path))
        logger.info(f"Saved best-atlas vessel scaffold to {vessel_mask_path}")
        
        logger.info("Running Metric A: Ensemble Consensus Certainty (Centerline Proximity)...")
        consensus_score = PlaqueValidator.compute_consensus_score(transformed_labels, eval_fixed_raw)
        logger.info(f"Consensus Score: {consensus_score:.2f}mm (Calibration phase)")
        
        if consensus_score > 5.0:
            logger.warning(f"Diagnostic Warning: Consensus Centerline Distance {consensus_score:.2f}mm is too high (> 5.0mm). Registration scattered.")
        else:
            logger.info(f"Validation PASS: Consensus reached (Metric A).")
        
        logger.info("Running Metric B (Diagnostic): Ostial Anchoring...")
        anchor_pt, anchor_dev = PlaqueValidator.compute_ostial_anchor(best_vessels, eval_fixed_raw)
        logger.info(f"  Center of Mass: {anchor_pt}")
        logger.info(f"  Deviation from center: {anchor_dev:.2f}mm (Calibration needed before gating)")

        logger.info("Running Metric C (Diagnostic): Ground Truth Calcium Overlap...")
        # Load calcium seg at NATIVE resolution to avoid resampling artifact
        native_ca_seg = sitk.ReadImage(str(coca_seg), sitk.sitkUInt8)
        ca_pct, ca_total = PlaqueValidator.compute_edt_overlap(
            eval_fixed_raw, best_vessels, native_ca_seg, CFG
        )
        if ca_total == 0:
            logger.info(f"  Calcium Diagnostic: Patient has no calcium (CAC=0).")
        elif ca_pct >= CFG.PASS_THRESHOLD_PCT:
            logger.info(f"  Calcium Diagnostic: {ca_pct:.1f}% calcium overlap (target > {CFG.PASS_THRESHOLD_PCT}%). ✓")
        else:
            logger.warning(f"  Calcium Diagnostic: Only {ca_pct:.1f}% calcium overlap. Scaffold may miss distal calcification.")
            
    if phase1_only:
        logger.info("Phase 1 complete. Exiting due to --phase1_only flag.")
        return

    # ---------------------------------------------------------
    # Phase 2: Hemodynamic Surrogate (PINN)
    # ---------------------------------------------------------
    logger.info("=== Phase 2: Hemodynamic Surrogate (PINN) ===")
    pinn_out_dir = out_dir / "phase2_output"
    pinn_out_dir.mkdir(exist_ok=True)
    
    # Pass mask path via environment variable
    env = os.environ.copy()
    env["PREDICT_MASK_PATH"] = str(vessel_mask_path)
    
    logger.info("Running PINN solver (Full Epochs)...")
    # Run run_phase2.py
    subprocess.run([sys.executable, "run_phase2.py"], env=env, check=True, cwd="src/phase2_hemodynamics")
    
    # Look for the output
    # By default, run_phase2 writes to out_dir / ess_predictions.csv based on CFG
    # Actually run_phase2 writes to the default output dir in config.py
    default_pinn_out = Path("/Users/karan/Desktop/PrediCT/output_v2/exports")
    ess_csv = default_pinn_out / "ess_predictions.csv"
    
    if not ess_csv.exists():
        logger.error(f"PINN failed to produce {ess_csv}")
        return
        
    # Interpolate to 3D NIfTI
    ess_nifti = out_dir / f"{patient_id}_ess_field.nii.gz"
    interpolate_ess_to_grid(ess_csv, vessel_mask_path, ess_nifti)
    
    # ---------------------------------------------------------
    # Phase 3: Stochastic Plaque Growth (SDE)
    # ---------------------------------------------------------
    logger.info("=== Phase 3: Stochastic Plaque Growth (SDE) ===")
    synth_mask_path = out_dir / f"{patient_id}_synthetic_calcium_mask.nii.gz"
    
    cmd = [
        sys.executable, "run_phase3_sde.py",
        "--vessel_mask", str(vessel_mask_path),
        "--ess_field", str(ess_nifti),
        "--output", str(synth_mask_path),
        "--agatston", str(target_agatston)
    ]
    subprocess.run(cmd, check=True, cwd="src/phase3_plaque_growth")
    
    # ---------------------------------------------------------
    # Phase 4: Physiological Texturing
    # ---------------------------------------------------------
    logger.info("=== Phase 4: Physiological Texturing ===")
    ncct_path = coca_patient_dir / f"{patient_id}_raw_img.nii.gz"
    if not ncct_path.exists():
        # Fallback to windowed
        ncct_path = coca_patient_dir / f"{patient_id}_img.nii.gz"
        
    final_output = out_dir / f"{patient_id}_synthetic_coca.nii.gz"
    
    cmd = [
        sys.executable, "run_phase4_texture.py",
        "--ncct", str(ncct_path),
        "--synth_mask", str(synth_mask_path),
        "--vessel_mask", str(vessel_mask_path),
        "--output", str(final_output)
    ]
    subprocess.run(cmd, check=True, cwd="src/phase4_texturing")
    
    logger.info(f"=== Physio-Twin Generation Complete! ===")
    logger.info(f"Final synthetic scan: {final_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Physio-Twin: Generative Coronary Calcium Synthesis")
    parser.add_argument("--patient_dir", type=str, required=True, help="Path to target COCA patient directory")
    parser.add_argument("--target_agatston", type=int, default=400, help="Target Agatston score multiplier")
    parser.add_argument("--out_dir", type=str, default="./physio_twin_output", help="Output directory")
    parser.add_argument("--phase1_only", action="store_true", help="Stop after generating and validating Phase 1 scaffold")
    
    args = parser.parse_args()
    
    # Pass args downstream
    run_physio_twin(
        Path(args.patient_dir), 
        args.target_agatston, 
        Path(args.out_dir),
        phase1_only=args.phase1_only
    )
