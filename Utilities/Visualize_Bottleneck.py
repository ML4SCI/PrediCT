import numpy as np
import pyvista as pv

# Load volume
vol = np.load(r"E:\MyProjects\Gsoc_2026_Official\Segmentation_Rajat\runs\run_fno\epoch_199\bottleneck\epoch_000_last_step_delta.npy")
# grid_3d = vol[0, 0]  # Shape: (16, 32, 32)
print("Shape:", vol.shape)
grid_3d = vol[0,3]

print("Shape:", grid_3d.shape)

print("Min:", grid_3d.min())
print("Max:", grid_3d.max())
print("Mean:", grid_3d.mean())
print("Std:", grid_3d.std())
print("Non-zero count:", np.count_nonzero(grid_3d))

# Wrap array directly as Point Data (1-to-1 array shape match, no +1 dimension offset needed)
grid = pv.ImageData()
grid.dimensions = grid_3d.shape
grid.point_data["delta"] = grid_3d.flatten(order="F")

max_abs = np.abs(grid_3d).max()

# Plot using Volume Rendering
plotter = pv.Plotter()

# Add Title
plotter.add_title(
    "Visualizing Feature Map Before and After FNO3D Block",
    font_size=12,
)

# Direct volume ray-casting with an opacity transfer function
plotter.add_volume(
    grid,
    scalars="delta",
    cmap="bwr",
    clim=[-max_abs, max_abs],
    opacity=[0.8, 0.0, 0.8],  # High opacity at extremes, 0 at center
)

plotter.show_grid()
plotter.show()