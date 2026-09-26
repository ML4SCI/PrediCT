import sys
import numpy as np
from config import PINNConfig
from geometry import CTASegmentationLoader, FlowDomainGenerator, flow_domain_to_coronary_geometry

cfg = PINNConfig()
print(f"Loading patient mask: {cfg.patient_mask_path}")
try:
    mask, meta = CTASegmentationLoader.load_nifti_coronary_volume(cfg.patient_mask_path)
    print(f"Mask loaded. Shape: {mask.shape}, Active voxels: {mask.sum()}")
    print(f"Spacing: {meta.spacing}, Origin: {meta.origin}")
    
    domain_gen = FlowDomainGenerator(meta)
    flow_domain = domain_gen.generate_flow_domain(mask, num_interior_points=8000, num_wall_points=5000)
    print(f"FlowDomain generated.")
    print(f"Interior points: {flow_domain.interior_points.shape}")
    print(f"Wall points: {flow_domain.wall_points.shape}")
    print(f"Inlet points: {flow_domain.inlet_points.shape}")
    print(f"Outlet points: {flow_domain.outlet_points.shape}")
    
    cg = flow_domain_to_coronary_geometry(flow_domain)
    print(f"CoronaryGeometry created.")
except Exception as e:
    print(f"Error during geometry processing: {e}")
    import traceback
    traceback.print_exc()

