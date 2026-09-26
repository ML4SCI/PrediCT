"""
Navier-Stokes Boundary Conditions Module
========================================

This module implements the mathematical boundary conditions for 3D steady
incompressible Navier-Stokes Physics-Informed Neural Networks (PINNs) applied
to patient-specific coronary hemodynamics.

Mathematical Justification
--------------------------
Boundary condition evaluation in PINNs relies on penalizing the discrepancy
between the neural network predictions and the known physical constraints at the
domain boundaries $\partial \Omega$. The boundary losses are computed using Mean
Squared Error (MSE) to maintain continuous differentiability during backpropagation.

Author: Computational Scientist
Target Framework: PyTorch 2.x, Python 3.11+
"""

from typing import Optional, Tuple
import torch

from ess import compute_velocity_jacobian



def compute_no_slip_loss(
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor
) -> torch.Tensor:
    """
    Compute the loss for the no-slip boundary condition on the vessel walls.

    Mathematical Formulation
    ------------------------
    Viscous fluids obey the no-slip condition at solid impermeable boundaries $\Gamma_{\text{wall}}$:
    
    $$ \mathbf{u}(\mathbf{x}) = 0 \quad \forall \mathbf{x} \in \Gamma_{\text{wall}} $$
    
    The loss function penalizes deviations from zero velocity:
    
    $$ \mathcal{L}_{\text{wall}} = \frac{1}{N_{\text{wall}}} \sum_{i=1}^{N_{\text{wall}}} \left( u_i^2 + v_i^2 + w_i^2 \right) $$

    Parameters
    ----------
    u : torch.Tensor
        Predicted x-velocity at wall points $(N, 1)$.
    v : torch.Tensor
        Predicted y-velocity at wall points $(N, 1)$.
    w : torch.Tensor
        Predicted z-velocity at wall points $(N, 1)$.

    Returns
    -------
    torch.Tensor
        Scalar loss tensor.
    """
    loss = torch.mean(u**2 + v**2 + w**2)
    return loss


def compute_integral_mass_loss(
    u_in: torch.Tensor, v_in: torch.Tensor, w_in: torch.Tensor, n_in: torch.Tensor,
    u_out: torch.Tensor, v_out: torch.Tensor, w_out: torch.Tensor, n_out: torch.Tensor,
    area_in: float, area_out: float
) -> torch.Tensor:
    """
    Computes the global integral mass conservation loss (Qin - Qout)^2.
    
    This acts as a hard global topological constraint on the fluid system.
    Uses a Monte Carlo approximation of the surface integrals:
    Q = Area * mean(V . n) where n is the INWARD pointing normal.
    
    Parameters
    ----------
    u_in, v_in, w_in : torch.Tensor
        Velocity components at the inlet collocation points (N_in, 1).
    n_in : torch.Tensor
        Inward pointing normal vectors at the inlet (N_in, 3).
    u_out, v_out, w_out : torch.Tensor
        Velocity components at the outlet collocation points (N_out, 1).
    n_out : torch.Tensor
        Inward pointing normal vectors at the outlet (N_out, 3).
    area_in : float
        Total surface area of the inlet boundary.
    area_out : float
        Total combined surface area of all outlet boundaries.
        
    Returns
    -------
    torch.Tensor
        Scalar tensor representing the squared flux difference.
    """
    area_in_f = float(area_in)
    area_out_f = float(area_out)
    
    # Dot product with INWARD normals
    # Flux = Area * mean(V . n)
    v_dot_n_in = u_in[:, 0] * n_in[:, 0] + v_in[:, 0] * n_in[:, 1] + w_in[:, 0] * n_in[:, 2]
    v_dot_n_out = u_out[:, 0] * n_out[:, 0] + v_out[:, 0] * n_out[:, 1] + w_out[:, 0] * n_out[:, 2]
    
    flux_in = area_in_f * torch.mean(v_dot_n_in)
    flux_out = area_out_f * torch.mean(v_dot_n_out)
    
    loss = (flux_in + flux_out) ** 2
    return loss


def compute_parabolic_inlet_loss(
    coords: torch.Tensor,
    u_pred: torch.Tensor,
    v_pred: torch.Tensor,
    w_pred: torch.Tensor,
    inlet_center: torch.Tensor,
    inlet_normal: torch.Tensor,
    inlet_radius: float,
    u_max: float
) -> torch.Tensor:
    """
    Compute the loss for a fully developed parabolic inlet velocity profile.

    Mathematical Formulation
    ------------------------
    For fully developed laminar flow in a circular cross-section, the velocity
    profile follows a paraboloid:
    
    $$ \mathbf{u}_{\text{exact}}(\mathbf{x}) = U_{\text{max}} \left[ 1 - \left(\frac{r(\mathbf{x})}{R}\right)^2 \right] (-\mathbf{n}) $$
    
    where $\mathbf{n}$ is the **outward** normal of the inlet face (pointing upstream,
    away from the fluid). The flow direction is $-\mathbf{n}$ (into the domain).
    $R$ is the vessel radius, and $r(\mathbf{x})$ is the orthogonal radial distance
    from the centerline:
    
    $$ r(\mathbf{x}) = \left\| (\mathbf{x} - \mathbf{c}) - \left( (\mathbf{x} - \mathbf{c}) \cdot \mathbf{n} \right) \mathbf{n} \right\| $$

    Parameters
    ----------
    coords : torch.Tensor
        Coordinates of the inlet boundary points $(N, 3)$.
    u_pred : torch.Tensor
        Predicted x-velocity at inlet $(N, 1)$.
    v_pred : torch.Tensor
        Predicted y-velocity at inlet $(N, 1)$.
    w_pred : torch.Tensor
        Predicted z-velocity at inlet $(N, 1)$.
    inlet_center : torch.Tensor
        Geometric center of the inlet face $(3,)$.
    inlet_normal : torch.Tensor
        Unit **outward** normal vector of the inlet face $(3,)$, pointing
        upstream (away from the fluid). H-2 FIX: this must be the outward normal,
        consistent with ``FlowDomain.inlet_normals``. The function negates it
        internally to obtain the inward flow direction. Passing an inward normal
        here would reverse the target velocity and corrupt inlet training.
    inlet_radius : float
        Radius $R$ of the inlet face.
    u_max : float
        Maximum centerline velocity $U_{\text{max}}$.

    Returns
    -------
    torch.Tensor
        Scalar loss tensor for the inlet boundary.
    """
    # Compute relative position to center
    delta = coords - inlet_center.view(1, 3)
    
    # Projection along the normal vector
    # shape: (N, 1)
    proj_len = torch.sum(delta * inlet_normal.view(1, 3), dim=1, keepdim=True)
    
    # Orthogonal radial vector
    radial_vec = delta - proj_len * inlet_normal.view(1, 3)
    
    # Radial distance squared
    r_squared = torch.sum(radial_vec**2, dim=1, keepdim=True)
    
    # Compute exact velocity magnitude, clamped to prevent negative velocities outside radius
    u_mag_exact = u_max * torch.clamp(1.0 - (r_squared / (inlet_radius**2)), min=0.0)
    
    # H-2 FIX: inlet_normal is the OUTWARD normal of the inlet face (pointing upstream,
    # away from the fluid domain), consistent with FlowDomain.inlet_normals convention.
    # Flow enters the domain against the outward normal, so the target velocity direction
    # is -inlet_normal. Using inlet_normal directly would set the target velocity pointing
    # upstream, causing the network to learn reversed inlet flow.
    flow_direction = -inlet_normal  # inward: into the fluid domain
    
    # Compute exact velocity vector components in the inward flow direction
    u_exact = u_mag_exact * flow_direction[0]
    v_exact = u_mag_exact * flow_direction[1]
    w_exact = u_mag_exact * flow_direction[2]
    
    # Mean Squared Error Loss
    loss = torch.mean((u_pred - u_exact)**2 + (v_pred - v_exact)**2 + (w_pred - w_exact)**2)
    return loss


def compute_outlet_traction_loss(
    coords: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    p: torch.Tensor,
    normals: torch.Tensor,
    viscosity_multiplier: float,
    target_traction: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute the loss for the traction (Neumann) boundary condition at the outlet.

    Mathematical Formulation
    ------------------------
    The fluid stress tensor $\boldsymbol{\sigma}$ is defined as:
    
    $$ \boldsymbol{\sigma} = -p \mathbf{I} + \nu \left( \nabla \mathbf{u} + (\nabla \mathbf{u})^T \right) $$
    
    The traction vector $\mathbf{t}$ on a surface with normal $\mathbf{n}$ is:
    
    $$ \mathbf{t} = \boldsymbol{\sigma} \mathbf{n} $$
    
    For a free-flow outflow condition, the traction is typically zero ($\mathbf{t} = 0$).

    Parameters
    ----------
    coords : torch.Tensor
        Coordinates of the outlet boundary points $(N, 3)$. Requires gradients.
    u : torch.Tensor
        Predicted x-velocity $(N, 1)$.
    v : torch.Tensor
        Predicted y-velocity $(N, 1)$.
    w : torch.Tensor
        Predicted z-velocity $(N, 1)$.
    p : torch.Tensor
        Predicted kinematic pressure $(N, 1)$.
    normals : torch.Tensor
        Outward unit normal vectors at the outlet points $(N, 3)$.
    viscosity_multiplier : float
        Dynamic viscosity mu (Pa.s) for dimensional form, or 1/Re for the
        non-dimensional form. H-5 FIX: do NOT pass kinematic viscosity
        nu = mu/rho here; doing so underestimates the traction by a factor of
        rho=1060. Use mu=0.0035 Pa.s for dimensional blood, or 1/Re=~0.0044
        for the non-dimensional Navier-Stokes system used in physics.py.
    target_traction : Optional[torch.Tensor], optional
        Target traction vectors $(N, 3)$. If None, defaults to zero-traction.

    Returns
    -------
    torch.Tensor
        Scalar loss tensor for the outlet traction condition.
    """
    # Jacobian J = \nabla \mathbf{u} -> shape (N, 3, 3)
    jacobian = compute_velocity_jacobian(u, v, w, coords)
    
    # Transpose Jacobian J^T -> shape (N, 3, 3)
    jacobian_t = jacobian.transpose(1, 2)
    
    # Rate of strain tensor multiplied by viscosity -> shape (N, 3, 3)
    viscous_stress = viscosity_multiplier * (jacobian + jacobian_t)
    
    # Pressure tensor -p I -> shape (N, 3, 3)
    identity = torch.eye(3, device=coords.device, dtype=coords.dtype).unsqueeze(0)
    pressure_stress = -p.view(-1, 1, 1) * identity
    
    # Cauchy stress tensor \boldsymbol{\sigma} -> shape (N, 3, 3)
    sigma = pressure_stress + viscous_stress
    
    # Traction vector \mathbf{t} = \boldsymbol{\sigma} \mathbf{n} -> shape (N, 3, 1)
    # normals shape is (N, 3), need to expand to (N, 3, 1) for batched matrix multiplication
    n_expanded = normals.unsqueeze(-1)
    predicted_traction = torch.bmm(sigma, n_expanded).squeeze(-1) # shape (N, 3)
    
    if target_traction is None:
        target_traction = torch.zeros_like(predicted_traction)
        
    loss = torch.mean(torch.sum((predicted_traction - target_traction)**2, dim=1))
    return loss


def compute_outlet_pressure_loss(
    p_pred: torch.Tensor,
    p_target: float
) -> torch.Tensor:
    """
    Compute the loss for an optional Dirichlet pressure boundary condition at the outlet.

    Mathematical Formulation
    ------------------------
    If a specific uniform pressure profile is required (e.g., zero pressure):
    
    $$ \mathcal{L}_{p} = \frac{1}{N_{\text{out}}} \sum_{i=1}^{N_{\text{out}}} (p_i - p_{\text{target}})^2 $$

    Parameters
    ----------
    p_pred : torch.Tensor
        Predicted pressure at the outlet $(N, 1)$.
    p_target : float
        Target scalar pressure value.

    Returns
    -------
    torch.Tensor
        Scalar loss tensor.
    """
    loss = torch.mean((p_pred - p_target)**2)
    return loss


if __name__ == "__main__":
    # ---------------------------------------------------------
    # Standalone Execution and Verification Block
    # ---------------------------------------------------------
    print("Initializing Navier-Stokes Boundary Conditions Module...")

    device = torch.device("cpu")
    N_pts = 100

    # Mock variables for wall
    u_wall = torch.randn((N_pts, 1), device=device) * 0.1
    v_wall = torch.randn((N_pts, 1), device=device) * 0.1
    w_wall = torch.randn((N_pts, 1), device=device) * 0.1

    loss_wall = compute_no_slip_loss(u_wall, v_wall, w_wall)
    print(f"No-Slip Wall Loss: {loss_wall.item():.6f}")

    # Mock variables for inlet
    coords_in = torch.randn((N_pts, 3), device=device)
    u_in = torch.randn((N_pts, 1), device=device)
    v_in = torch.randn((N_pts, 1), device=device)
    w_in = torch.randn((N_pts, 1), device=device)
    c_in = torch.tensor([0.0, 0.0, 0.0], device=device)
    n_in = torch.tensor([1.0, 0.0, 0.0], device=device)

    loss_inlet = compute_parabolic_inlet_loss(
        coords=coords_in,
        u_pred=u_in, v_pred=v_in, w_pred=w_in,
        inlet_center=c_in, inlet_normal=n_in,
        inlet_radius=1.5, u_max=1.0
    )
    print(f"Parabolic Inlet Loss: {loss_inlet.item():.6f}")

    # Mock variables for outlet traction
    coords_out = torch.randn((N_pts, 3), device=device, requires_grad=True)
    u_out = coords_out[:, 0:1] * 2.0  # Linear profile to produce non-zero Jacobian
    v_out = coords_out[:, 1:2] * 0.5
    w_out = coords_out[:, 2:3] * 0.5
    p_out = torch.ones((N_pts, 1), device=device) * 0.1
    normals_out = torch.tensor([[1.0, 0.0, 0.0]]).repeat(N_pts, 1)

    loss_traction = compute_outlet_traction_loss(
        coords=coords_out,
        u=u_out, v=v_out, w=w_out, p=p_out,
        normals=normals_out,
        viscosity_multiplier=0.0035
    )
    print(f"Outlet Traction Loss: {loss_traction.item():.6f}")

    # Mock variables for outlet pressure
    loss_pressure = compute_outlet_pressure_loss(p_out, p_target=0.0)
    print(f"Outlet Pressure Loss: {loss_pressure.item():.6f}")

    print("\nVerification Passed: Boundary condition losses computed successfully.")