r"""
Endothelial Shear Stress (ESS) Computation Module
=================================================

This module computes Wall Shear Stress (WSS) and Endothelial Shear Stress (ESS)
from the velocity gradients predicted by a Physics-Informed Neural Network (PINN).
It handles the mathematical projection of the viscous stress tensor onto the 
vessel wall, unit conversions to physical dimensions, and visualization of the
hemodynamic markers for calcium plaque prediction.

Mathematical Justification
--------------------------
The Wall Shear Stress (WSS) vector $\boldsymbol{\tau}_w$ represents the tangential 
frictional force per unit area exerted by the fluid on the endothelial surface. 
For an incompressible Newtonian fluid, the viscous stress tensor $\boldsymbol{\tau}$ 
is defined as:

$$ \boldsymbol{\tau} = \mu \left( \nabla \mathbf{u} + (\nabla \mathbf{u})^T \right) $$

where $\mu$ is the dynamic viscosity and $\nabla \mathbf{u}$ is the spatial Jacobian 
of the velocity vector field. 

The traction vector $\mathbf{t}$ on a wall with unit normal $\mathbf{n}$ is:

$$ \mathbf{t} = \boldsymbol{\tau} \mathbf{n} $$

The WSS vector is the tangential component of the traction vector, obtained by 
subtracting the normal component:

$$ \boldsymbol{\tau}_w = \mathbf{t} - (\mathbf{t} \cdot \mathbf{n}) \mathbf{n} $$

The Endothelial Shear Stress (ESS) is the magnitude of the WSS vector:

$$ \text{ESS} = \| \boldsymbol{\tau}_w \|_2 $$

Author: Computational Scientist
Target Framework: PyTorch 2.x, Python 3.11+
"""

from typing import Tuple, Union
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def compute_velocity_jacobian(
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    coords: torch.Tensor
) -> torch.Tensor:
    r"""
    Compute the spatial Jacobian matrix of the velocity vector field using Automatic Differentiation.

    Mathematical Formulation
    ------------------------
    $$ \nabla \mathbf{u} = \begin{bmatrix} \frac{\partial u}{\partial x} & \frac{\partial u}{\partial y} & \frac{\partial u}{\partial z} \\ \frac{\partial v}{\partial x} & \frac{\partial v}{\partial y} & \frac{\partial v}{\partial z} \\ \frac{\partial w}{\partial x} & \frac{\partial w}{\partial y} & \frac{\partial w}{\partial z} \end{bmatrix} $$

    Parameters
    ----------
    u : torch.Tensor
        Velocity x-component $(N, 1)$.
    v : torch.Tensor
        Velocity y-component $(N, 1)$.
    w : torch.Tensor
        Velocity z-component $(N, 1)$.
    coords : torch.Tensor
        Spatial coordinates $(N, 3)$. Must have `requires_grad=True`.

    Returns
    -------
    torch.Tensor
        Jacobian tensor of shape $(N, 3, 3)$.
    """
    def _grad(field: torch.Tensor) -> torch.Tensor:
        gradient, = torch.autograd.grad(
            outputs=field,
            inputs=coords,
            grad_outputs=torch.ones_like(field),
            create_graph=True,
            retain_graph=True
        )
        return gradient

    grad_u = _grad(u)  # (N, 3)
    grad_v = _grad(v)  # (N, 3)
    grad_w = _grad(w)  # (N, 3)

    # Stack to form (N, 3, 3)
    jacobian = torch.stack([grad_u, grad_v, grad_w], dim=1)
    return jacobian


def compute_endothelial_shear_stress(
    jacobian: torch.Tensor,
    normals: torch.Tensor,
    dynamic_viscosity: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Compute the Wall Shear Stress (WSS) vector and Endothelial Shear Stress (ESS) magnitude.

    C-5 FIX — Usage contract (choose exactly ONE path):

    **Path A — Non-dimensional PINN output (recommended)**::

        # Jacobian J* = ∂u*/∂x* is dimensionless (PINN non-dim coordinates/velocity)
        wss, ess_nondim = compute_endothelial_shear_stress(J_star, normals, dynamic_viscosity=1.0)
        ess_pa = dimension_recovery(ess_nondim, U_ref, L_ref, mu)   # → Pa

    **Path B — Dimensional Jacobian** (physical u in m/s, x in m)::

        # J = ∂u/∂x has units [1/s]; pass dynamic viscosity μ directly
        wss, ess_pa = compute_endothelial_shear_stress(J_dimensional, normals, dynamic_viscosity=0.0035)
        # Do NOT call dimension_recovery — ESS is already in Pa

    The default ``dynamic_viscosity=1.0`` intentionally makes Path A the safe default.
    Passing ``mu=0.0035`` to a dimensionless Jacobian yields ESS in Pa·s (wrong by
    a factor of L/U) and would double-scale when ``dimension_recovery`` is also called.

    Parameters
    ----------
    jacobian : torch.Tensor
        Spatial velocity gradient tensor $\nabla \mathbf{u}$ of shape $(N, 3, 3)$.
        Pass the non-dimensional Jacobian J* for Path A; dimensional J for Path B.
    normals : torch.Tensor
        Unit normal vectors at the vessel wall of shape $(N, 3)$.
    dynamic_viscosity : float, optional
        For Path A (non-dim): use 1.0 (default) — returns dimensionless ESS.
        For Path B (dimensional): use μ = 0.0035 Pa·s — returns ESS in Pa.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        - wss_vector: Wall shear stress vectors $\boldsymbol{\tau}_w$ of shape $(N, 3)$.
        - ess_magnitude: Scalar endothelial shear stress magnitudes of shape $(N, 1)$.
    """
    # Number of points N
    N = jacobian.shape[0]

    # Transpose Jacobian: (\nabla \mathbf{u})^T -> shape (N, 3, 3)
    jacobian_t = jacobian.transpose(1, 2)

    # Viscous Stress Tensor: \boldsymbol{\tau} = \mu (\nabla \mathbf{u} + (\nabla \mathbf{u})^T) -> shape (N, 3, 3)
    tau = dynamic_viscosity * (jacobian + jacobian_t)

    # Expand normals for batched matrix multiplication -> shape (N, 3, 1)
    n_expanded = normals.unsqueeze(-1)

    # Traction vector: \mathbf{t} = \boldsymbol{\tau} \mathbf{n} -> shape (N, 3)
    traction = torch.bmm(tau, n_expanded).squeeze(-1)

    # Normal traction component: t_n = \mathbf{t} \cdot \mathbf{n} -> shape (N, 1)
    t_n = torch.sum(traction * normals, dim=1, keepdim=True)

    # Wall Shear Stress Vector: \boldsymbol{\tau}_w = \mathbf{t} - t_n \mathbf{n} -> shape (N, 3)
    wss_vector = traction - (t_n * normals)

    # Endothelial Shear Stress Magnitude: ESS = \| \boldsymbol{\tau}_w \|_2 -> shape (N, 1)
    ess_magnitude = torch.norm(wss_vector, p=2, dim=1, keepdim=True)

    return wss_vector, ess_magnitude


def evaluate_normal_quality(normals: torch.Tensor, coords: torch.Tensor) -> float:
    """
    Evaluate normal vector smoothness by computing the average dot product
    between neighboring normals. A value close to 1.0 implies smooth normals.
    """
    import scipy.spatial
    coords_np = coords.detach().cpu().numpy()
    normals_np = normals.detach().cpu().numpy()
    
    tree = scipy.spatial.cKDTree(coords_np)
    # Find 5 nearest neighbors (including self)
    _, idxs = tree.query(coords_np, k=5)
    
    dot_prods = []
    for i in range(len(coords_np)):
        n_i = normals_np[i]
        for j in idxs[i][1:]:  # skip self
            n_j = normals_np[j]
            dot_prods.append(np.dot(n_i, n_j))
            
    mean_dot = np.mean(dot_prods)
    return float(mean_dot)


def dimension_recovery(
    ess_non_dim: torch.Tensor,
    u_ref: float,
    l_ref: float,
    rho: float = 1060.0,
    mu: float = 0.0035
) -> torch.Tensor:
    """
    Convert non-dimensional ESS back to physical units (Pascals).

    Mathematical Formulation
    ------------------------
    The non-dimensionalization scheme in ``physics.py`` uses the viscous stress scale
    tau_0 = mu * U_ref / L_ref, **not** the dynamic pressure rho * U_ref^2.
    These differ by a factor of Re. The physical ESS is recovered via:

    ESS_physical = ESS_non_dim * (mu * U_ref / L_ref)

    This function must only be called on **dimensionless** ESS values produced by
    ``compute_endothelial_shear_stress`` with ``dynamic_viscosity=1.0`` (Path A).
    Do NOT call it after Path B (dimensional Jacobian with mu=0.0035) -- doing so
    double-applies the viscous scale and produces physically wrong units.

    Parameters
    ----------
    ess_non_dim : torch.Tensor
        Non-dimensional endothelial shear stress magnitudes $(N, 1)$.
    u_ref : float
        Reference velocity scale in m/s.
    l_ref : float
        Reference length scale in m.
    rho : float, optional
        Blood density in kg/m^3 (default: 1060.0).
    mu : float, optional
        Dynamic viscosity in Pa.s (default: 0.0035).

    Returns
    -------
    torch.Tensor
        Dimensionalized ESS tensor in Pascals (Pa).
    """
    # Assuming standard viscous scaling for stress \tau_ref = \mu * U_ref / L_ref
    stress_scale = (mu * u_ref) / l_ref
    ess_physical = ess_non_dim * stress_scale
    return ess_physical


def export_ess_to_csv(
    coords: torch.Tensor,
    wss_vector: torch.Tensor,
    ess_magnitude: torch.Tensor,
    output_path: Union[str, Path]
) -> None:
    """
    Export physical coordinates, WSS vectors, and ESS magnitudes to a CSV file.

    Parameters
    ----------
    coords : torch.Tensor
        Spatial coordinates $(N, 3)$.
    wss_vector : torch.Tensor
        Wall shear stress vectors $(N, 3)$.
    ess_magnitude : torch.Tensor
        Endothelial shear stress magnitudes $(N, 1)$.
    output_path : Union[str, Path]
        Destination file path for the CSV output.
    """
    coords_np = coords.detach().cpu().numpy()
    wss_np = wss_vector.detach().cpu().numpy()
    ess_np = ess_magnitude.detach().cpu().numpy()

    df = pd.DataFrame({
        'x': coords_np[:, 0],
        'y': coords_np[:, 1],
        'z': coords_np[:, 2],
        'wss_x': wss_np[:, 0],
        'wss_y': wss_np[:, 1],
        'wss_z': wss_np[:, 2],
        'ess_magnitude': ess_np[:, 0]
    })

    filepath = Path(output_path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Successfully exported ESS data to {filepath}")


def visualize_ess_distribution(
    coords: torch.Tensor,
    ess_magnitude: torch.Tensor,
    output_image_path: Union[str, Path]
) -> None:
    """
    Generate and save a 3D scatter plot of the Endothelial Shear Stress distribution.

    Parameters
    ----------
    coords : torch.Tensor
        Spatial coordinates of the vessel wall $(N, 3)$.
    ess_magnitude : torch.Tensor
        Endothelial shear stress magnitudes $(N, 1)$.
    output_image_path : Union[str, Path]
        Destination file path for the rendering output.
    """
    coords_np = coords.detach().cpu().numpy()
    ess_np = ess_magnitude.detach().cpu().numpy().flatten()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot colored by ESS magnitude
    scatter = ax.scatter(
        coords_np[:, 0],
        coords_np[:, 1],
        coords_np[:, 2],
        c=ess_np,
        cmap='jet',
        s=15,
        alpha=0.8,
        edgecolor='none'
    )

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Endothelial Shear Stress (Pa)', fontsize=12)

    ax.set_title('Coronary Endothelial Shear Stress Distribution', fontsize=14, pad=15)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')

    # Equal aspect ratio for proper geometry visualization
    max_range = np.array([
        coords_np[:, 0].max() - coords_np[:, 0].min(),
        coords_np[:, 1].max() - coords_np[:, 1].min(),
        coords_np[:, 2].max() - coords_np[:, 2].min()
    ]).max() / 2.0

    mid_x = (coords_np[:, 0].max() + coords_np[:, 0].min()) * 0.5
    mid_y = (coords_np[:, 1].max() + coords_np[:, 1].min()) * 0.5
    mid_z = (coords_np[:, 2].max() + coords_np[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    filepath = Path(output_image_path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Successfully saved ESS visualization to {filepath}")


if __name__ == "__main__":
    # ---------------------------------------------------------
    # Standalone Execution and Verification Block
    # ---------------------------------------------------------
    print("Initializing Endothelial Shear Stress (ESS) Module...")

    device = torch.device("cpu")
    N_pts = 500

    # 1. Generate a mock cylindrical vessel geometry (wall coordinates)
    theta = torch.linspace(0, 2 * np.pi, N_pts, device=device)
    z = torch.linspace(0, 0.05, N_pts, device=device)  # 5 cm length
    radius = 0.002  # 2 mm radius

    x_coords = radius * torch.cos(theta)
    y_coords = radius * torch.sin(theta)
    
    # Needs requires_grad=True to compute spatial derivatives
    coords = torch.stack([x_coords, y_coords, z], dim=1).requires_grad_(True)

    # 2. Compute Wall Normals (pointing inward for fluid domain)
    normals = torch.stack([-torch.cos(theta), -torch.sin(theta), torch.zeros_like(theta)], dim=1)

    # 3. Simulate predicted velocity fields (e.g., Poiseuille flow with perturbations)
    u_pred = (1e-3 * torch.randn(N_pts, 1, device=device))  # Mostly 0 at the wall
    v_pred = (1e-3 * torch.randn(N_pts, 1, device=device))  # Mostly 0 at the wall
    
    # Slight longitudinal flow derivative to simulate shear
    w_pred = (0.5 * (radius**2 - (x_coords**2 + y_coords**2))).unsqueeze(1) 
    
    # 4. Compute Jacobian
    jacobian = compute_velocity_jacobian(u_pred, v_pred, w_pred, coords)
    
    # 5. Compute WSS and ESS
    dynamic_viscosity = 0.0035 # Pa.s
    wss, ess = compute_endothelial_shear_stress(jacobian, normals, dynamic_viscosity)

    # Calculate summary statistics
    mean_ess = torch.mean(ess).item()
    max_ess = torch.max(ess).item()
    print(f"\n--- ESS Computational Results ---")
    print(f"Mean ESS: {mean_ess:.4e} Pa")
    print(f"Max ESS:  {max_ess:.4e} Pa")

    # 6. Export outputs
    output_dir = Path("./results")
    export_ess_to_csv(coords, wss, ess, output_dir / "coronary_ess_data.csv")
    visualize_ess_distribution(coords, ess, output_dir / "ess_distribution.png")

    print("\nVerification Passed: ESS mathematical derivation and processing complete.")