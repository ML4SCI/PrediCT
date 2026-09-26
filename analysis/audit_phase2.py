"""
Phase 2 Structural Integrity Audit
===================================
Checks for leakages, gaps, incorrect geometry, and validates that
the PINN domain is watertight and physically consistent.
"""

import sys
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import binary_erosion, binary_dilation, label

sys.path.insert(0, "/Users/karan/Desktop/PrediCT/phase2_v2")
from config import PINNConfig
from geometry import CTASegmentationLoader, FlowDomainGenerator, flow_domain_to_coronary_geometry
from sampling import AdaptiveCoronarySampler
import networkx as nx

def audit():
    cfg = PINNConfig()
    print("=" * 70)
    print("  PHASE 2 STRUCTURAL INTEGRITY AUDIT")
    print("=" * 70)
    
    # 1. Load geometry
    print("\n--- 1. GEOMETRY LOADING ---")
    mask, meta = CTASegmentationLoader.load_nifti_coronary_volume(cfg.patient_mask_path)
    print(f"  Mask shape: {mask.shape}")
    print(f"  Spacing: {meta.spacing}")
    print(f"  Origin: {meta.origin}")
    print(f"  Total vessel voxels: {np.sum(mask > 0)}")
    
    # 2. Check mask connectivity — are there disconnected blobs?
    print("\n--- 2. MASK CONNECTIVITY ---")
    labeled, num_components = label(mask > 0)
    component_sizes = [np.sum(labeled == i) for i in range(1, num_components + 1)]
    component_sizes.sort(reverse=True)
    print(f"  Number of connected components: {num_components}")
    if num_components > 1:
        print(f"  ⚠️  WARNING: Mask has {num_components} disconnected blobs!")
        print(f"  Component sizes: {component_sizes[:5]}")
        print(f"  Largest component: {component_sizes[0]} voxels")
        print(f"  Disconnected fragments: {sum(component_sizes[1:])} voxels")
    else:
        print(f"  ✅ Mask is a single connected component ({component_sizes[0]} voxels)")
    
    # 3. Check wall mask — is it watertight?
    print("\n--- 3. WALL MASK INTEGRITY ---")
    eroded = binary_erosion(mask > 0)
    wall_mask = (mask > 0) & (~eroded)
    interior_mask = eroded
    print(f"  Wall voxels (1-voxel shell): {np.sum(wall_mask)}")
    print(f"  Interior voxels: {np.sum(interior_mask)}")
    
    # Check if interior is fully enclosed by wall
    dilated_interior = binary_dilation(interior_mask)
    leaking = dilated_interior & (~(mask > 0))
    n_leak = np.sum(leaking)
    if n_leak > 0:
        print(f"  ⚠️  WARNING: {n_leak} voxels leak outside the mask!")
    else:
        print(f"  ✅ Interior is fully enclosed by wall (no leaks)")
    
    # 4. Generate flow domain and check centerline graph
    print("\n--- 4. CENTERLINE GRAPH ---")
    domain_gen = FlowDomainGenerator(meta)
    flow_domain = domain_gen.generate_flow_domain(mask, num_interior_points=2000, num_wall_points=1000)
    graph = flow_domain.centerline_graph
    
    terminals = [n for n, d in graph.degree() if d == 1]
    inlet_node = max(terminals, key=lambda n: graph.nodes[n]["pos"][2])
    outlets = [n for n in terminals if n != inlet_node]
    
    print(f"  Centerline nodes: {graph.number_of_nodes()}")
    print(f"  Centerline edges: {graph.number_of_edges()}")
    print(f"  Terminal nodes: {len(terminals)}")
    print(f"  Inlet node: {inlet_node} (radius={graph.nodes[inlet_node]['radius']:.2f} mm)")
    print(f"  Outlet nodes: {outlets}")
    for o in outlets:
        print(f"    Outlet {o}: radius={graph.nodes[o]['radius']:.2f} mm")
    
    # Check graph is connected
    if nx.is_connected(graph):
        print(f"  ✅ Centerline graph is connected")
    else:
        print(f"  ⚠️  WARNING: Centerline graph is NOT connected — flow cannot reach all outlets!")
    
    # Check inlet/outlet reachability
    for o in outlets:
        try:
            path = nx.shortest_path(graph, inlet_node, o)
            path_length = sum(
                np.linalg.norm(
                    np.array(graph.nodes[path[i+1]]["pos"]) - np.array(graph.nodes[path[i]]["pos"])
                ) for i in range(len(path)-1)
            )
            print(f"    Path inlet→outlet {o}: {len(path)} nodes, {path_length:.1f} mm")
        except nx.NetworkXNoPath:
            print(f"    ⚠️  WARNING: No path from inlet to outlet {o}!")
    
    # 5. Boundary disk generation
    print("\n--- 5. BOUNDARY DISK VALIDATION ---")
    L0_mm = cfg.scales.char_length * 1000.0
    
    inlet_pts = flow_domain.inlet_points.detach().numpy()
    inlet_norms = flow_domain.inlet_normals.detach().numpy()
    print(f"  Inlet points: {len(inlet_pts)}")
    print(f"  Inlet normal (mean): {inlet_norms.mean(axis=0)}")
    print(f"  Inlet center (mm): {inlet_pts.mean(axis=0)}")
    
    outlet_pts = flow_domain.outlet_points.detach().numpy()
    outlet_norms = flow_domain.outlet_normals.detach().numpy()
    print(f"  Outlet points: {len(outlet_pts)}")
    print(f"  Outlet normal (mean): {outlet_norms.mean(axis=0)}")
    
    # Check normal consistency
    inlet_n_mean = inlet_norms.mean(axis=0)
    inlet_n_mean /= np.linalg.norm(inlet_n_mean)
    outlet_n_mean = outlet_norms.mean(axis=0)
    outlet_n_mean /= np.linalg.norm(outlet_n_mean)
    
    dot = np.dot(inlet_n_mean, outlet_n_mean)
    print(f"  Inlet-Outlet normal dot product: {dot:.3f}")
    if abs(dot) < 0.3:
        print(f"  ✅ Inlet and outlet normals are not parallel (good for bifurcating geometry)")
    
    # 6. Non-dimensionalization check
    print("\n--- 6. NON-DIMENSIONALIZATION ---")
    Re = cfg.blood.density * cfg.scales.char_velocity * cfg.scales.char_length / cfg.blood.dynamic_viscosity
    print(f"  L0 = {cfg.scales.char_length} m ({L0_mm} mm)")
    print(f"  U0 = {cfg.scales.char_velocity} m/s")
    print(f"  Re = {Re:.2f}")
    print(f"  tau_scale = mu*U0/L0 = {cfg.blood.dynamic_viscosity * cfg.scales.char_velocity / cfg.scales.char_length:.4f} Pa")
    
    # Check the non-dim coordinate ranges
    coronary_geom = flow_domain_to_coronary_geometry(flow_domain)
    sampler = AdaptiveCoronarySampler(coronary_geom, device="cpu")
    sampled = sampler.build_pinn_dataset(2000, 1000, 200, 200, char_length_mm=L0_mm)
    
    x_int = sampled.x_interior.detach().numpy()
    x_wall = sampled.x_wall.detach().numpy()
    x_in = sampled.x_inlet.detach().numpy()
    x_out = sampled.x_outlet.detach().numpy()
    
    print(f"  Interior coords range: [{x_int.min(axis=0)}, {x_int.max(axis=0)}]")
    print(f"  Wall coords range:     [{x_wall.min(axis=0)}, {x_wall.max(axis=0)}]")
    print(f"  Inlet coords range:    [{x_in.min(axis=0)}, {x_in.max(axis=0)}]")
    print(f"  Outlet coords range:   [{x_out.min(axis=0)}, {x_out.max(axis=0)}]")
    
    # Check that inlet points are INSIDE the domain bounding box
    int_min = x_int.min(axis=0)
    int_max = x_int.max(axis=0)
    
    inlet_inside = np.all(x_in >= int_min - 1.0) and np.all(x_in <= int_max + 1.0)
    outlet_inside = np.all(x_out >= int_min - 1.0) and np.all(x_out <= int_max + 1.0)
    
    if inlet_inside:
        print(f"  ✅ Inlet points are within domain bounding box")
    else:
        print(f"  ⚠️  WARNING: Inlet points are OUTSIDE the domain!")
        
    if outlet_inside:
        print(f"  ✅ Outlet points are within domain bounding box")
    else:
        print(f"  ⚠️  WARNING: Outlet points are OUTSIDE the domain!")
    
    # 7. Check inlet radius vs char_length
    print("\n--- 7. INLET RADIUS CONSISTENCY ---")
    inlet_radius_mm = graph.nodes[inlet_node]["radius"]
    inlet_radius_m = inlet_radius_mm / 1000.0
    inlet_radius_nd = inlet_radius_mm / L0_mm
    hardcoded_radius_nd = 0.0015 / cfg.scales.char_length
    print(f"  Actual inlet radius: {inlet_radius_mm:.3f} mm = {inlet_radius_m:.6f} m")
    print(f"  Actual inlet radius (non-dim): {inlet_radius_nd:.4f}")
    print(f"  Hardcoded inlet radius (non-dim): {hardcoded_radius_nd:.4f}")
    
    ratio = inlet_radius_nd / hardcoded_radius_nd
    if abs(ratio - 1.0) > 0.5:
        print(f"  ⚠️  CRITICAL: Actual inlet radius is {ratio:.2f}x the hardcoded value!")
        print(f"  This means the parabolic profile is being applied at the WRONG scale.")
        print(f"  The network will learn the wrong velocity magnitude!")
    else:
        print(f"  ✅ Inlet radius ratio: {ratio:.2f} (acceptable)")
    
    # 8. Loss weight analysis
    print("\n--- 8. CURRICULUM & LOSS WEIGHTS ---")
    print(f"  BC Pretrain epochs: {cfg.curriculum.bc_pretrain_epochs}")
    print(f"  lambda_continuity: {cfg.loss_weights.lambda_continuity}")
    print(f"  lambda_momentum: {cfg.loss_weights.lambda_momentum}")
    print(f"  lambda_wall: {cfg.loss_weights.lambda_wall_noslip}")
    print(f"  lambda_inlet: {cfg.loss_weights.lambda_inlet}")
    print(f"  lambda_outlet: {cfg.loss_weights.lambda_outlet}")
    print(f"  lambda_integral_mass: {cfg.loss_weights.lambda_integral_mass}")
    
    print("\n" + "=" * 70)
    print("  AUDIT COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    audit()
