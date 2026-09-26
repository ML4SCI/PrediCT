import pandas as pd
import numpy as np
import pyvista as pv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def export_ess_csv_to_vtp(csv_path: str, vtp_path: str, patient_id: str = "synthetic") -> None:
    """
    Reads the ESS CSV predictions, reconstructs a 3D surface mesh from the 
    point cloud, interpolates the hemodynamics data onto the mesh, and exports a VTP.
    """
    csv_path = Path(csv_path)
    vtp_path = Path(vtp_path)
    
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return
        
    logger.info(f"Loading ESS point cloud data from {csv_path}...")
    df = pd.read_csv(csv_path)
    points = df[['x', 'y', 'z']].values
    
    # 1. Create a Point Cloud
    cloud = pv.PolyData(points)
    cloud['WSS_Vector'] = df[['wss_x', 'wss_y', 'wss_z']].values
    cloud['ESS_Magnitude'] = df['ess_magnitude'].values
    
    # 2. Reconstruct Surface Mesh
    try:
        logger.info("Running surface reconstruction (Poisson)...")
        # reconstruct_surface uses VTK's vtkSurfaceReconstructionFilter
        surf = cloud.reconstruct_surface()
        
        # Interpolate the data from the point cloud onto the surface mesh vertices
        surf_sampled = surf.sample(cloud)
        
        surf_sampled.save(str(vtp_path))
        logger.info(f"Successfully reconstructed surface mesh and saved to {vtp_path}")
    except Exception as e:
        logger.warning(f"Surface reconstruction failed: {e}. Falling back to point cloud export.")
        cloud.save(str(vtp_path))
        logger.info(f"Successfully saved point cloud to {vtp_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Allows standalone testing of the script
    import sys
    if len(sys.argv) > 2:
        export_ess_csv_to_vtp(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python visualization.py <input.csv> <output.vtp>")
