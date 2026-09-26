import numpy as np
import SimpleITK as sitk
from pathlib import Path
import scipy.ndimage as ndi
import scipy.stats as stats
import argparse
import logging

def compute_ess_gradient(ess_field: np.ndarray, spacing: tuple) -> tuple:
    """Computes the spatial gradient of the ESS field."""
    # ess_field shape is (Z, Y, X), spacing is (sz, sy, sx)
    grad_z, grad_y, grad_x = np.gradient(ess_field, spacing[0], spacing[1], spacing[2])
    return grad_z, grad_y, grad_x

def run_sde_growth(vessel_mask: np.ndarray, 
                   ess_field: np.ndarray, 
                   spacing: tuple,
                   target_agatston: int) -> np.ndarray:
    """
    Simulates calcium growth radially inward from the vessel wall,
    biased by low ESS, mimicking true intimal/medial calcification.
    
    All stochastic parameters are drawn from continuous distributions
    rather than discrete phenotype categories:
      - Seed count: Normal distribution (bell curve)
      - Growth depth: Right-skewed log-normal (0–3mm, mode ~1.5mm)
      - Radial decay: Normal distribution
    """
    logger = logging.getLogger(__name__)
    z_dim, y_dim, x_dim = vessel_mask.shape
    
    # 1. Morphological distance transform from the wall
    logger.info("Computing radial distance transform...")
    dist_field_mm = ndi.distance_transform_edt(vessel_mask, sampling=spacing)
    
    # ──────────────────────────────────────────────────────────────
    # 2. Stochastic Growth Parameters (replaces phenotype branching)
    # ──────────────────────────────────────────────────────────────
    
    # Seed count: Negative Binomial distribution fitted to COCA ground truth (mean=8.0, var=24)
    # Scaled by target_agatston to allow high-calcium patients to have more lesions.
    base_n = 4
    base_p = 1/3
    scale_factor = max(1.0, target_agatston / 400.0)
    # Adjust n to increase the mean while keeping the same variance/mean relationship (p)
    adjusted_n = max(1, int(base_n * scale_factor))
    num_seeds = max(1, np.random.negative_binomial(n=adjusted_n, p=base_p))
    
    # Radial decay rate is replaced by the Anisotropic penalty in the BFS loop.
    # Growth depth floor is removed as the new BFS handles wall-hugging naturally.
    max_depth_mm = float(np.clip(np.random.lognormal(mean=0.55, sigma=0.45), 0.5, 3.0))
    
    # Nodule protrusion probability: Low baseline, drawn from bell curve
    nodule_prob = float(np.clip(
        np.random.normal(loc=0.03, scale=0.02),
        0.005, 0.10
    ))
    
    logger.info(f"Stochastic Growth Params: Seeds={num_seeds}, "
                f"MaxDepth={max_depth_mm:.2f}mm, "
                f"NoduleProb={nodule_prob:.4f}")
    
    # 3. Seed Initiation
    # Seeds form strictly on the vessel wall where ESS is atherogenic (< 1.0 Pa)
    wall_mask = (dist_field_mm > 0) & (dist_field_mm <= max(spacing))
    
    # Probability inversely proportional to ESS
    ess_clipped = np.clip(ess_field, 0.1, 5.0)
    prob_field = (1.0 / ess_clipped) * wall_mask
    
    if prob_field.sum() == 0:
        prob_field = wall_mask.astype(float)
        
    if prob_field.sum() == 0:
        logger.warning("No valid wall locations for calcium.")
        return np.zeros_like(vessel_mask, dtype=np.uint8)
        
    prob_field = prob_field / prob_field.sum()
    flat_indices = np.random.choice(prob_field.size, size=num_seeds, p=prob_field.ravel())
    seed_coords = np.column_stack(np.unravel_index(flat_indices, prob_field.shape))
    
    # 4. Radial Growth (BFS constrained by distance transform)
    # We track:
    #   - calcium_mask: binary core (for Agatston)
    #   - bfs_depth_field: BFS traversal depth per voxel (for intensity gradient)
    #     This follows the vessel wall contour instead of making perfect spheres.
    calcium_mask = np.zeros_like(vessel_mask, dtype=np.uint8)
    bfs_depth_field = np.full_like(vessel_mask, dtype=np.float64, fill_value=-1.0)
    
    # Voxel Budget & Size Distribution: Sample from Log-Normal fitted to COCA
    # (mu=4.3, sigma=1.2 gives median ~74 voxels, mean ~185 voxels).
    # We cap the total budget so the Agatston score doesn't wildly overshoot.
    total_budget = int(target_agatston * 0.5)  # rough voxel budget from target
    per_seed_targets = []
    budget_used = 0
    for _ in range(num_seeds):
        size = int(np.random.lognormal(mean=4.3, sigma=1.2))
        size = max(5, min(size, 500))  # cap individual lesion at 500 voxels
        if budget_used + size > total_budget:
            size = max(5, total_budget - budget_used)
        per_seed_targets.append(size)
        budget_used += size
        if budget_used >= total_budget:
            break
    num_seeds = len(per_seed_targets)  # may have fewer seeds if budget exhausted
    per_seed_targets = np.array(per_seed_targets)
    seed_coords = seed_coords[:num_seeds]  # trim seeds to match
    
    for idx in range(num_seeds):
        sz, sy, sx = seed_coords[idx]
        
        # Per-seed depth variation
        local_depth = float(np.clip(
            np.random.lognormal(mean=0.40, sigma=0.50),
            0.3, max_depth_mm
        ))
        
        target_voxels = per_seed_targets[idx]
        
        # Breadth-first growth with 6-connected neighbors
        queue = [(sz, sy, sx, 0)]  # (z, y, x, bfs_depth)
        visited = set([(sz, sy, sx)])
        voxels_grown = 0
        max_depth_reached = 0
        
        while queue and voxels_grown < target_voxels:
            cz, cy, cx, depth = queue.pop(0)
            
            calcium_mask[cz, cy, cx] = 1
            bfs_depth_field[cz, cy, cx] = depth
            voxels_grown += 1
            max_depth_reached = max(max_depth_reached, depth)
            
            # 6-connected neighbor expansion
            for dz, dy, dx in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                nz, ny, nx = cz+dz, cy+dy, cx+dx
                
                if 0 <= nz < z_dim and 0 <= ny < y_dim and 0 <= nx < x_dim:
                    if (nz, ny, nx) not in visited and vessel_mask[nz, ny, nx] > 0:
                        neighbor_dist = dist_field_mm[nz, ny, nx]
                        current_dist = dist_field_mm[cz, cy, cx]
                        
                        # Anisotropic Wall-Hugging Growth
                        # Growth along the wall (dist change is <= 0) is easy (decay ~ 3.0).
                        # Growth inward toward lumen (dist increases) is hard (decay ~ 1.0).
                        dist_diff = neighbor_dist - current_dist
                        
                        if dist_diff <= 0.1:
                            # Growing along the wall circumferentially/longitudinally
                            prob = np.exp(-neighbor_dist / 3.0)
                        elif neighbor_dist <= local_depth:
                            # Growing radially inward
                            prob = np.exp(-neighbor_dist / 1.0)
                        else:
                            # Beyond max depth — nodule probability only
                            prob = nodule_prob
                            
                        if np.random.rand() < prob:
                            visited.add((nz, ny, nx))
                            queue.append((nz, ny, nx, depth + 1))
    
    # --- BFS-Depth Intensity Gradient ---
    # Use BFS traversal depth (NOT Euclidean distance) for the intensity gradient.
    # This makes the fade follow the irregular BFS shape along the vessel wall
    # instead of creating perfect radial spheres.
    ca_voxels = calcium_mask > 0
    calcium_intensity = np.zeros_like(vessel_mask, dtype=np.float64)
    
    if ca_voxels.sum() > 0:
        # Find max BFS depth across all deposits for normalization
        max_bfs_depth = bfs_depth_field[ca_voxels].max()
        if max_bfs_depth < 1:
            max_bfs_depth = 1.0
        
        # Core intensity: 1.0 at seed (depth=0), decaying with BFS depth
        calcium_intensity[ca_voxels] = np.exp(
            -bfs_depth_field[ca_voxels] / (max_bfs_depth * 0.5)
        )
        
        # Gentle Gaussian for sub-voxel anti-aliasing
        sigma_mm = 0.8
        sigma_vox = sigma_mm / np.array(spacing)
        calcium_intensity = ndi.gaussian_filter(calcium_intensity, sigma=sigma_vox)
        
        # Normalize peak to 1.0
        if calcium_intensity.max() > 0:
            calcium_intensity = calcium_intensity / calcium_intensity.max()
        
        # STRICT BOUNDARY ENFORCEMENT
        # Do not allow calcium to drift outside the vessel wall, even after blurring.
        # This fixes the "calcium outside vessel" issue permanently.
        calcium_intensity *= (vessel_mask > 0)
        
        # Cut noise floor
        calcium_intensity[calcium_intensity < 0.05] = 0.0
    
    core_count = int(ca_voxels.sum())
    active_count = int((calcium_intensity > 0.05).sum())
    logger.info(f"Generated calcium field: {core_count} core voxels, "
                f"{active_count} gradient voxels (distance bleed out).")
    return calcium_intensity.astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description="Phase 3: SDE Stochastic Plaque Growth")
    parser.add_argument("--vessel_mask", type=str, required=True, help="Path to 3D vessel mask NIfTI")
    parser.add_argument("--ess_field", type=str, required=True, help="Path to 3D ESS field NIfTI")
    parser.add_argument("--output", type=str, required=True, help="Path to save synthetic calcium mask")
    parser.add_argument("--agatston", type=int, default=400, help="Target Agatston score multiplier")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Loading inputs...")
    vessel_img = sitk.ReadImage(args.vessel_mask)
    ess_img = sitk.ReadImage(args.ess_field)
    
    # We must generate the calcium on the native vessel grid (1x1x1mm).
    # This guarantees perfect alignment with the 1x1x1mm vessel mask, so there is no leaking.
    vessel_arr = sitk.GetArrayFromImage(vessel_img)
    ess_arr = sitk.GetArrayFromImage(ess_img)
    spacing = vessel_img.GetSpacing()[::-1] # Z, Y, X
    
    logger.info(f"Running Stochastic Plaque Growth on native vessel grid (spacing={spacing})...")
    synth_mask_arr = run_sde_growth(
        vessel_arr, ess_arr, spacing, args.agatston
    )
    
    logger.info("Saving synthetic calcium mask (at native 1x1x1mm resolution)...")
    synth_img = sitk.GetImageFromArray(synth_mask_arr)
    synth_img.CopyInformation(vessel_img)
    sitk.WriteImage(synth_img, args.output)
    logger.info(f"Saved to {args.output}")

if __name__ == "__main__":
    main()

