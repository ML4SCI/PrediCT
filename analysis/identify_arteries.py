"""
Anatomical identification of the two coronary artery components.
"""
import numpy as np
import nibabel as nib
from scipy.ndimage import label as scipy_label
from pathlib import Path
import sys
sys.path.insert(0, "/Users/karan/Desktop/PrediCT/phase2_v2")
from config import PINNConfig
from geometry import CTASegmentationLoader, FlowDomainGenerator
import networkx as nx

cfg = PINNConfig()
mask, meta = CTASegmentationLoader.load_nifti_coronary_volume(cfg.patient_mask_path)

# Label connected components
labeled, num = scipy_label(mask > 0)
print(f"Number of connected components: {num}\n")

for comp_id in range(1, num + 1):
    comp_mask = (labeled == comp_id).astype(np.uint8)
    voxels = np.argwhere(comp_mask > 0)
    n_voxels = len(voxels)
    
    # Physical coordinates
    spacing = np.array(meta.spacing)
    origin = np.array(meta.origin)
    direction = np.array(meta.direction)
    
    phys_coords = voxels * spacing
    phys_coords = np.dot(phys_coords, direction.T) + origin
    
    centroid_mm = phys_coords.mean(axis=0)
    extent_mm = phys_coords.max(axis=0) - phys_coords.min(axis=0)
    
    print(f"{'='*60}")
    print(f"  COMPONENT {comp_id}")
    print(f"{'='*60}")
    print(f"  Voxels: {n_voxels}")
    print(f"  Centroid (mm): X={centroid_mm[0]:.1f}, Y={centroid_mm[1]:.1f}, Z={centroid_mm[2]:.1f}")
    print(f"  Extent  (mm):  X={extent_mm[0]:.1f}, Y={extent_mm[1]:.1f}, Z={extent_mm[2]:.1f}")
    print(f"  Bounding box:")
    print(f"    X: [{phys_coords[:,0].min():.1f}, {phys_coords[:,0].max():.1f}]")
    print(f"    Y: [{phys_coords[:,1].min():.1f}, {phys_coords[:,1].max():.1f}]")
    print(f"    Z: [{phys_coords[:,2].min():.1f}, {phys_coords[:,2].max():.1f}]")
    
    # Try to extract centerline for this component
    try:
        domain_gen = FlowDomainGenerator(meta)
        fd = domain_gen.generate_flow_domain(comp_mask, num_interior_points=500, num_wall_points=500)
        graph = fd.centerline_graph
        
        terminals = [n for n, d in graph.degree() if d == 1]
        bifurcations = [n for n, d in graph.degree() if d >= 3]
        
        # Find the highest-Z terminal (likely the ostium / aortic origin)
        if terminals:
            inlet_node = max(terminals, key=lambda n: graph.nodes[n]["pos"][2])
            inlet_pos = graph.nodes[inlet_node]["pos"]
            inlet_radius = graph.nodes[inlet_node]["radius"]
            
            outlets = [n for n in terminals if n != inlet_node]
            
            print(f"\n  Centerline:")
            print(f"    Nodes: {graph.number_of_nodes()}")
            print(f"    Terminal nodes: {len(terminals)}")
            print(f"    Bifurcation nodes: {len(bifurcations)}")
            print(f"    Inlet (highest Z): node {inlet_node}")
            print(f"      Position: X={inlet_pos[0]:.1f}, Y={inlet_pos[1]:.1f}, Z={inlet_pos[2]:.1f}")
            print(f"      Radius: {inlet_radius:.2f} mm")
            print(f"    Outlets: {len(outlets)}")
            for o in outlets:
                o_pos = graph.nodes[o]["pos"]
                o_rad = graph.nodes[o]["radius"]
                print(f"      Outlet {o}: pos=({o_pos[0]:.1f}, {o_pos[1]:.1f}, {o_pos[2]:.1f}), R={o_rad:.2f} mm")
            
            # Compute total vessel length
            total_length = sum(
                np.linalg.norm(
                    np.array(graph.nodes[u]["pos"]) - np.array(graph.nodes[v]["pos"])
                ) for u, v in graph.edges()
            )
            print(f"    Total centerline length: {total_length:.1f} mm")
    except Exception as e:
        print(f"  Centerline extraction failed: {e}")
    
    # Anatomical identification heuristic
    # In standard cardiac CT orientation:
    # - RCA typically runs along the right side of the heart (more positive X in RAS)
    # - LCA (LAD + LCx) runs along the left/anterior side
    # - The LAD descends anteriorly, the LCx wraps around posteriorly
    # - RCA runs in the right AV groove
    
    print()

# Now determine which is which based on spatial location
print("\n" + "="*60)
print("  ANATOMICAL IDENTIFICATION")
print("="*60)
print("""
In standard cardiac CT coordinates:
  - The LEFT coronary artery (LCA → LAD + LCx) typically:
    • Originates from the left coronary cusp of the aorta
    • The LAD descends anteriorly along the interventricular septum
    • The LCx wraps around the left atrioventricular groove
    
  - The RIGHT coronary artery (RCA) typically:
    • Originates from the right coronary cusp
    • Courses along the right atrioventricular groove
    • Gives off the PDA (posterior descending artery)
    
  These are anatomically separate vessels branching from different 
  sides of the aortic root. They supply different territories of
  the myocardium.
""")
print("Each component should be simulated INDEPENDENTLY with its own")
print("inlet boundary condition (its own ostium from the aorta).")
