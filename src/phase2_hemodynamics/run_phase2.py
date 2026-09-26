"""
Phase 2 Orchestration: PINN Training and ESS Prediction Pipeline
================================================================
"""

import logging
import random
from pathlib import Path
import torch
import numpy as np
import networkx as nx
from scipy.ndimage import label as scipy_label

import ess as ESSCalculator
from ess import evaluate_normal_quality
from ess_diagnostic import generate_diagnostics
from config import PINNConfig
from geometry import CTASegmentationLoader, FlowDomainGenerator, flow_domain_to_coronary_geometry
from sampling import AdaptiveCoronarySampler
from network import HemodynamicsPINN
from physics import SteadyNavierStokesPhysics
from losses import PINNLossEvaluator
from trainer import PINNTrainer, PINNTrainerConfig
from boundary_conditions import (
    compute_no_slip_loss,
    compute_parabolic_inlet_loss,
    compute_outlet_traction_loss,
    compute_integral_mass_loss
)

def validate_physics_config(cfg: PINNConfig) -> None:
    """
    Validates physical parameters and scaling factors to ensure the PINN solves a well-posed problem.
    """
    rho = cfg.blood.density
    mu = cfg.blood.dynamic_viscosity
    U0 = cfg.scales.char_velocity
    L0 = cfg.scales.char_length

    Re = rho * U0 * L0 / mu
    logging.getLogger(__name__).info(f"Configuration Validation: Reynolds Number Re = {Re:.2f}")

    if not (0.1 < Re < 2000.0):
        raise ValueError(f"CRITICAL: Implausible Reynolds number {Re:.2f} for coronary flow. Expected 0.1 - 2000.0.")

    if mu > 0.1 or mu < 0.001:
        raise ValueError(f"CRITICAL: Dynamic viscosity {mu} Pa.s is implausible for blood.")

    if L0 > 0.1 or L0 < 0.0001:
        raise ValueError(f"CRITICAL: Characteristic length {L0} m is outside expected mm range.")

    tau_ref = mu * U0 / L0
    logging.getLogger(__name__).info(f"Configuration Validation: Viscous stress scale = {tau_ref:.4f} Pa")
    if tau_ref < 0.01 or tau_ref > 10.0:
        logging.getLogger(__name__).warning(f"WARNING: Unusually high or low reference stress scale {tau_ref:.4f} Pa.")

def main() -> None:
    # Prevent MacBook from sleeping during long training (macOS only)
    import subprocess
    import os
    try:
        subprocess.Popen(["caffeinate", "-d", "-i", "-s", "-w", str(os.getpid())])
    except FileNotFoundError:
        pass  # Not on macOS or caffeinate not available
        
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    
    cfg = PINNConfig()

    # Set global random seeds for full reproducibility
    seed = cfg.runtime.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    validate_physics_config(cfg)
    device = cfg.runtime.resolve_device()
    logger.info(f"Starting Phase 2 Pipeline on device: {device} | seed={seed}")
    
    # --- 1. Load Geometry & Sample Points ---
    logger.info("Extracting computational domain from CTA segmentation...")
    if cfg.patient_mask_path is not None:
        mask, meta = CTASegmentationLoader.load_nifti_coronary_volume(cfg.patient_mask_path)
    else:
        mask, meta = CTASegmentationLoader.create_synthetic_coronary_volume()
    
    # FIX-1: Isolate the largest connected component.
    # Coronary CT masks contain separate arteries (LCA, RCA) that are
    # anatomically distinct vessels. Each must be simulated independently.
    # We pick the largest component (usually the LCA tree) for this run.
    labeled_mask, num_components = scipy_label(mask > 0)
    if num_components > 1:
        sizes = [(labeled_mask == i).sum() for i in range(1, num_components + 1)]
        largest_id = int(np.argmax(sizes)) + 1
        logger.info(f"Mask has {num_components} connected components (separate coronary arteries).")
        logger.info(f"  Component sizes: {sorted(sizes, reverse=True)}")
        logger.info(f"  Selecting largest component (ID={largest_id}, {sizes[largest_id-1]} voxels) for simulation.")
        mask = (labeled_mask == largest_id).astype(mask.dtype)
    
    domain_gen = FlowDomainGenerator(meta)
    # Point counts: wall increased to 5000 for better boundary-layer resolution;
    # interior increased to 8000 to support the denser gradient field.
    n_interior = min(cfg.sampling.num_interior_points, 4000)
    n_wall     = min(cfg.sampling.num_wall_points,     2000)
    n_inlet    = min(cfg.sampling.num_inlet_points,     500)
    n_outlet   = min(cfg.sampling.num_outlet_points,    500)

    flow_domain = domain_gen.generate_flow_domain(
        mask,
        num_interior_points=n_interior,
        num_wall_points=n_wall,
    )
    
    logger.info("Sampling PINN collocation points...")
    coronary_geom = flow_domain_to_coronary_geometry(flow_domain)
    sampler = AdaptiveCoronarySampler(coronary_geom, device=device)
    
    L0_mm = cfg.scales.char_length * 1000.0
    sampled = sampler.build_pinn_dataset(
        num_interior=n_interior,
        num_wall=n_wall,
        num_inlet=n_inlet,
        num_outlet=n_outlet,
        char_length_mm=L0_mm
    )
    
    # Calculate Non-Dimensional Boundary Areas for Integral Mass Loss
    graph = flow_domain.centerline_graph
    terminals = [n for n, d in graph.degree() if d == 1]
    # FIX-4: Use radius instead of Z-coordinate to identify the aortic ostium (inlet).
    # The ostium is always the widest part of the vessel, whereas Z-orientation varies across patients.
    inlet_node = max(terminals, key=lambda n: graph.nodes[n]["radius"])
    outlets = [n for n in terminals if n != inlet_node]
    
    inlet_radius_mm = graph.nodes[inlet_node]["radius"]
    area_in_mm2 = np.pi * (inlet_radius_mm ** 2)
    area_out_mm2 = sum(np.pi * (graph.nodes[out]["radius"] ** 2) for out in outlets)
    
    area_in_nd = area_in_mm2 / (L0_mm ** 2)
    area_out_nd = area_out_mm2 / (L0_mm ** 2)
    logger.info(f"Boundary areas (non-dimensional): Inlet = {area_in_nd:.4f}, Outlets = {area_out_nd:.4f}")
    
    # --- 2. Build PINN ---
    logger.info("Initializing HemodynamicsPINN...")
    model = HemodynamicsPINN.from_config(cfg.architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate_adam)
    
    # --- 3. Compute Physics (Define Closures) ---
    Re = cfg.blood.density * cfg.scales.char_velocity * cfg.scales.char_length / cfg.blood.dynamic_viscosity
    physics = SteadyNavierStokesPhysics(
        rho=cfg.blood.density,
        mu=cfg.blood.dynamic_viscosity,
        L=cfg.scales.char_length,
        U=cfg.scales.char_velocity
    )
    
    curriculum_state = {"epoch": 0}
    def epoch_closure():
        curriculum_state["epoch"] += 1
        
    def loss_closure() -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        u, v, w, p = model(sampled.x_interior)
        res_c, res_x, res_y, res_z = physics.compute_residuals(sampled.x_interior, u, v, w, p)
        
        uw, vw, ww, _ = model(sampled.x_wall)
        loss_wall = compute_no_slip_loss(uw, vw, ww)
        
        u_in, v_in, w_in, _ = model(sampled.x_inlet)
        inlet_center = torch.mean(sampled.x_inlet, dim=0)
        inlet_normal = torch.mean(sampled.n_inlet, dim=0)
        # BUG 2 FIX: coords are non-dimensional (x* = x_mm / L0_mm).
        # inlet_radius must be in the same non-dim units: R* = R_m / L0_m.
        # Previous value 0.002 (metres) caused r²/R²=62500 >> 1, clamping the
        # parabolic profile to zero everywhere and making the network learn u*=0.
        # FIX-3: Use actual inlet radius from centerline graph instead of hardcoded value.
        # The hardcoded 0.0015m was only correct for a 3mm-diameter vessel.
        # Patient-specific vessels have different inlet radii.
        actual_inlet_radius_mm = graph.nodes[inlet_node]["radius"]
        inlet_radius_nondim = actual_inlet_radius_mm / L0_mm
        loss_inlet = compute_parabolic_inlet_loss(
            sampled.x_inlet, u_in, v_in, w_in,
            inlet_center=inlet_center, inlet_normal=inlet_normal,
            inlet_radius=inlet_radius_nondim, u_max=cfg.boundary.inlet_peak_velocity_scale
        )
        
        u_out, v_out, w_out, p_out = model(sampled.x_outlet)
        loss_outlet = compute_outlet_traction_loss(
            coords=sampled.x_outlet,
            u=u_out, v=v_out, w=w_out, p=p_out,
            normals=sampled.n_outlet,
            viscosity_multiplier=1.0 / Re
        )
        
        loss_integral_mass = compute_integral_mass_loss(
            u_in, v_in, w_in, sampled.n_inlet,
            u_out, v_out, w_out, sampled.n_outlet,
            area_in_nd, area_out_nd
        )
        
        losses = PINNLossEvaluator.evaluate_all_losses(
            res_c, res_x, res_y, res_z, loss_wall, loss_inlet, loss_outlet, loss_integral_mass
        )
        
        weight_physics = 0.0 if curriculum_state["epoch"] <= cfg.curriculum.bc_pretrain_epochs else 1.0
        
        total_loss = (
            weight_physics * cfg.loss_weights.lambda_continuity * losses["mass"] +
            weight_physics * cfg.loss_weights.lambda_momentum * losses["momentum"] +
            cfg.loss_weights.lambda_wall_noslip * losses["wall"] +
            cfg.loss_weights.lambda_inlet * losses["inlet"] +
            cfg.loss_weights.lambda_outlet * losses["outlet"] +
            weight_physics * cfg.loss_weights.lambda_integral_mass * losses["integral_mass"]
        )
        return total_loss, losses, model.output_layer.weight
    
    def rar_closure():
        import gc
        logger.info("Executing Residual-Based Adaptive Refinement (RAR)...")
        num_candidates = 10000
        chunk_size = 1024
        
        x_cand_np, _ = sampler.sample_interior(
            num_interior=num_candidates,
            wall_decay_length=0.5, w_wall=1.0, w_stenosis=1.0, w_bifurcation=1.0, oversample_factor=3
        )
        x_cand_np = x_cand_np / L0_mm
        x_cand = torch.tensor(x_cand_np, dtype=sampled.x_interior.dtype, device=device)
        
        # 1. Chunked Candidate Evaluation
        res_mag_cand_list = []
        for i in range(0, num_candidates, chunk_size):
            chunk = x_cand[i:i+chunk_size].clone().requires_grad_(True)
            u_c, v_c, w_c, p_c = model(chunk)
            res_c, res_x, res_y, res_z = physics.compute_residuals(chunk, u_c, v_c, w_c, p_c)
            chunk_res = (res_c**2 + res_x**2 + res_y**2 + res_z**2).detach().squeeze()
            res_mag_cand_list.append(chunk_res)
            
            # Immediate memory cleanup
            del chunk, u_c, v_c, w_c, p_c, res_c, res_x, res_y, res_z
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
                
        res_mag_cand = torch.cat(res_mag_cand_list, dim=0)
        num_add = 1000
        _, top_idx = torch.topk(res_mag_cand, k=num_add)
        x_new = x_cand[top_idx].detach()
        
        # 2. Chunked Existing Interior Evaluation
        res_mag_o_list = []
        num_interior = sampled.x_interior.shape[0]
        for i in range(0, num_interior, chunk_size):
            chunk = sampled.x_interior[i:i+chunk_size].clone().detach().requires_grad_(True)
            u_old, v_old, w_old, p_old = model(chunk)
            res_c_o, res_x_o, res_y_o, res_z_o = physics.compute_residuals(chunk, u_old, v_old, w_old, p_old)
            chunk_res_o = (res_c_o**2 + res_x_o**2 + res_y_o**2 + res_z_o**2).detach().squeeze()
            res_mag_o_list.append(chunk_res_o)
            
            # Immediate memory cleanup
            del chunk, u_old, v_old, w_old, p_old, res_c_o, res_x_o, res_y_o, res_z_o
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
                
        res_mag_o = torch.cat(res_mag_o_list, dim=0)
        
        with torch.no_grad():
            _, bottom_idx = torch.topk(res_mag_o, k=num_add, largest=False)
            
            keep_mask = torch.ones(num_interior, dtype=torch.bool, device=device)
            keep_mask[bottom_idx] = False
            
            x_kept = sampled.x_interior[keep_mask]
            new_interior = torch.cat([x_kept, x_new], dim=0).detach()
            
        sampled.x_interior = new_interior.requires_grad_(True)
        logger.info(f"RAR added {num_add} points. New interior count: {sampled.x_interior.shape[0]}")

    # --- 4. Train ---
    logger.info("Commencing network optimization...")
    trainer_cfg = PINNTrainerConfig(
        max_epochs=cfg.training.epochs_adam,
        learning_rate=cfg.training.learning_rate_adam,
        use_mixed_precision=False,  # CI-6: AMP underflows second-order autograd
        log_frequency=10
    )
    trainer = PINNTrainer(model, optimizer, trainer_cfg, device)
    trainer.train(loss_closure, rar_closure=rar_closure, epoch_closure=epoch_closure, rar_frequency=5000)
    
    # --- 5. Post-Training Validation & Conservation Checks ---
    logger.info("Evaluating physical plausibility of the solution...")
    model.eval()
    with torch.no_grad():
        # --- 5a. Velocity Collapse Detection ---
        # Sample interior points and check mean velocity magnitude.
        # If the PINN collapsed to the trivial solution u=v=w=0, ESS will be meaningless.
        u_int, v_int, w_int, _ = model(sampled.x_interior)
        vel_mag = torch.sqrt(u_int**2 + v_int**2 + w_int**2)
        mean_vel_nondim = vel_mag.mean().item()
        max_vel_nondim = vel_mag.max().item()
        logger.info(f"Velocity field check: mean |u*| = {mean_vel_nondim:.4e}, max |u*| = {max_vel_nondim:.4e}")
        if mean_vel_nondim < 0.01:
            logger.error(
                f"CRITICAL: Velocity field has collapsed (mean |u*| = {mean_vel_nondim:.4e}). "
                f"The PINN converged to the trivial zero-velocity solution. "
                f"ESS values will be non-physiological. Increase lambda_inlet or reduce training epochs."
            )
            raise RuntimeError(
                f"PINN velocity collapse detected: mean |u*| = {mean_vel_nondim:.4e} < 0.01. "
                f"The solver failed to maintain the inlet boundary condition."
            )
        
        # Validate Inlet Profile
        u_in, v_in, w_in, _ = model(sampled.x_inlet)
        inlet_center = torch.mean(sampled.x_inlet, dim=0)
        inlet_normal = torch.mean(sampled.n_inlet, dim=0)
        inlet_radius_nondim = graph.nodes[inlet_node]["radius"] / L0_mm
        delta = sampled.x_inlet - inlet_center
        proj = torch.sum(delta * inlet_normal, dim=1, keepdim=True)
        r_sq = torch.sum((delta - proj * inlet_normal)**2, dim=1, keepdim=True)
        u_mag_exact = cfg.boundary.inlet_peak_velocity_scale * torch.clamp(1.0 - (r_sq / (inlet_radius_nondim**2)), min=0.0)
        flow_dir = -inlet_normal
        u_ex, v_ex, w_ex = u_mag_exact * flow_dir[0], u_mag_exact * flow_dir[1], u_mag_exact * flow_dir[2]
        rmse_inlet = torch.sqrt(torch.mean((u_in - u_ex)**2 + (v_in - v_ex)**2 + (w_in - w_ex)**2)).item()
        logger.info(f"Validation: Inlet velocity profile RMSE = {rmse_inlet:.4e}")

        # Mass-Flow Conservation (Deterministic per-outlet using Graph topology)
        logger.info("Computing mass conservation exact boundary fluxes...")
        graph = flow_domain.centerline_graph
        terminals = [n for n, d in graph.degree() if d == 1]
        inlet_node = max(terminals, key=lambda n: graph.nodes[n]["pos"][2])
        outlets = [n for n in terminals if n != inlet_node]
        
        L0 = cfg.scales.char_length
        U0 = cfg.scales.char_velocity
        
        def evaluate_node_flux(node_id, is_inlet=False, num_points=2000):
            # Re-generate the exact disk to get isolated points and normals
            pts, norms = domain_gen._generate_boundary_disk(graph, node_id, num_points, is_inlet)
            radius_mm = graph.nodes[node_id]["radius"]
            area_mm2 = np.pi * (radius_mm ** 2)
            
            pts_nd = pts / (L0 * 1000.0)
            x_tensor = torch.tensor(pts_nd, dtype=torch.float32, device=device)
            with torch.no_grad():
                u_n, v_n, w_n, _ = model(x_tensor)
            
            u_dim = u_n.cpu().numpy() * U0
            v_dim = v_n.cpu().numpy() * U0
            w_dim = w_n.cpu().numpy() * U0
            v_vec = np.column_stack([u_dim, v_dim, w_dim])
            
            # _generate_boundary_disk returns outward normals for the bounding box.
            # Normal velocity exiting the boundary
            normal_vel = np.sum(v_vec * norms, axis=1)
            mean_normal_vel = np.mean(normal_vel)
            
            # Flux [ml/s] = mean_v_n [m/s] * Area [mm^2]
            flux_ml_s = mean_normal_vel * area_mm2
            return flux_ml_s
            
        q_in_raw = evaluate_node_flux(inlet_node, is_inlet=True)
        # Inlet mass enters the domain, so outward flux is negative. Negate to get positive entering flux.
        q_in = -q_in_raw
        
        q_out = 0.0
        for out in outlets:
            q_out += evaluate_node_flux(out, is_inlet=False)
            
        flow_error = abs(q_in - q_out) / (abs(q_in) + 1e-8)
        logger.info(f"Validation: Inlet Flow Flux = {q_in:.4f} ml/s, Outlet Flow Flux = {q_out:.4f} ml/s")
        logger.info(f"Validation: Mass Conservation Error = {flow_error*100.0:.2f}%")

    # --- 6. Compute ESS ---
    logger.info("Projecting viscous stress tensor to compute ESS...")
    u_w, v_w, w_w, _ = model(sampled.x_wall)
    jacobian = ESSCalculator.compute_velocity_jacobian(u_w, v_w, w_w, sampled.x_wall)
    # BUG 1 FIX: The Jacobian J* = ∂u*/∂x* is dimensionless because both the
    # PINN outputs (u*) and the collocation coordinates (x*) are non-dimensional.
    # The physical viscous stress is:
    #   τ = μ × (U₀/L₀) × J*   [Pa]
    # Passing dynamic_viscosity=μ=0.0035 to a dimensionless J* gives units
    # of Pa·s (wrong by a factor of U₀/L₀ = 83.33 s⁻¹).
    # Correct: pass tau_scale = μ × U₀ / L₀ so ESS is directly in Pascals.
    tau_scale = (
        cfg.blood.dynamic_viscosity
        * cfg.scales.char_velocity
        / cfg.scales.char_length
    )  # = 0.0035 × 0.25 / 0.003 ≈ 0.2917 Pa
    logger.info(f"ESS viscous stress scale: tau = mu*U0/L0 = {tau_scale:.6f} Pa")
    wss_vector, ess_magnitude = ESSCalculator.compute_endothelial_shear_stress(
        jacobian, sampled.n_wall, dynamic_viscosity=tau_scale
    )

    mean_ess = torch.mean(ess_magnitude).item()
    logger.info(f"Mean Endothelial Shear Stress (all wall): {mean_ess:.4e} Pa")

    # --- 6b. ESS Physiological Range Gate ---
    # Literature reference values (Chatzizisis et al. 2007, Samady et al. 2011):
    #   Normal coronary ESS: 1.0–7.0 Pa
    #   Atherogenic (low): < 1.0 Pa
    #   Absolute floor in recirculation zones: ~0.1 Pa
    #   Values < 0.1 Pa indicate velocity field collapse (computational artifact)
    if mean_ess < 0.1:
        logger.error(
            f"CRITICAL: Mean ESS = {mean_ess:.4e} Pa is below physiological floor (0.1 Pa). "
            f"The PINN velocity field has likely collapsed. Results are non-physiological."
        )
        raise RuntimeError(f"ESS physiological gate FAILED: Mean ESS = {mean_ess:.4e} Pa < 0.1 Pa")
    elif mean_ess > 10.0:
        logger.error(
            f"CRITICAL: Mean ESS = {mean_ess:.4e} Pa exceeds physiological ceiling (10.0 Pa). "
            f"Possible numerical divergence or incorrect scaling."
        )
        raise RuntimeError(f"ESS physiological gate FAILED: Mean ESS = {mean_ess:.4e} Pa > 10.0 Pa")
    else:
        ess_np_all = ess_magnitude.detach().cpu().numpy().flatten()
        pct_atherogenic = float(np.mean(ess_np_all < 1.0) * 100)
        pct_normal = float(np.mean((ess_np_all >= 1.0) & (ess_np_all <= 7.0)) * 100)
        pct_high = float(np.mean(ess_np_all > 7.0) * 100)
        logger.info(f"ESS bands: Atherogenic(<1Pa)={pct_atherogenic:.1f}% | Normal(1-7Pa)={pct_normal:.1f}% | High(>7Pa)={pct_high:.1f}%")

    # --- 7. Clip outlet-edge artifact spikes before export ---
    # The outlet cut plane produces a geometric edge where the wall normal
    # transitions abruptly, causing large spurious velocity gradients.
    # Exclude the last 1 mm of the vessel + any remaining statistical outliers.
    
    x_wall_np = sampled.x_wall.detach().cpu().numpy()
    x_coords = x_wall_np[:, 0]
    x_max = x_coords.max()
    L0_mm_scale = cfg.scales.char_length * 1000.0
    cutoff_x = x_max - (1.0 / L0_mm_scale)  # 1 mm threshold in non-dimensional units
    
    ess_np = ess_magnitude.detach().cpu().numpy().flatten()
    q1_np, q3_np = float(np.percentile(ess_np, 25)), float(np.percentile(ess_np, 75))
    iqr_np = q3_np - q1_np
    fence   = q3_np + 3.0 * iqr_np
    
    valid_stat = (ess_np <= fence)
    valid_geom = (x_coords <= cutoff_x)
    valid = valid_stat & valid_geom
    
    n_clipped_stat = int((~valid_stat).sum())
    n_clipped_geom = int((~valid_geom).sum())
    logger.info(f"Clipping {n_clipped_geom} outlet-adjacent points (last 1 mm) and {n_clipped_stat} statistical outliers.")
    
    valid_t = torch.tensor(valid, device=device)
    x_wall_clean   = sampled.x_wall[valid_t]
    wss_vec_clean  = wss_vector[valid_t]
    ess_mag_clean  = ess_magnitude[valid_t]
    mean_ess_clean = torch.mean(ess_mag_clean).item()
    logger.info(f"Mean ESS (clipped): {mean_ess_clean:.4e} Pa  ({int(valid_t.sum())} points)")

    # --- 8. Export Outputs ---
    out_dir = Path(cfg.directories.export_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    out_file = out_dir / "ess_predictions.csv"
    logger.info(f"Exporting hemodynamic metrics to {out_file}...")
    ESSCalculator.export_ess_to_csv(x_wall_clean, wss_vec_clean, ess_mag_clean, out_file)
    
    # --- 9. VTP Visualization Export ---
    import sys; sys.path.insert(0, "../utils"); from visualization import export_ess_csv_to_vtp

    vtp_file = out_dir / "ess_predictions.vtp"
    logger.info(f"Generating ParaView VTP export: {vtp_file}...")
    export_ess_csv_to_vtp(out_file, vtp_file, patient_id="synthetic")
    
    # --- 10. Diagnostics & Reporting ---
    logger.info("Generating comprehensive ESS validation report and diagnostics...")
    diagnostics_dir = out_dir / "diagnostics"
    generate_diagnostics(str(out_file), str(diagnostics_dir))
    
    norm_quality = evaluate_normal_quality(sampled.n_wall, sampled.x_wall)
    logger.info(f"Validation: Wall Normal Smoothness Score = {norm_quality:.4f} (1.0 is perfect)")
    if norm_quality < 0.95:
        logger.warning("WARNING: Wall normals exhibit high variance (jaggedness). This can cause artificial ESS spikes.")

    logger.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()