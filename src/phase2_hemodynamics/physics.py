"""
Navier-Stokes Physics Evaluator Module
======================================

This module implements the physical PDE residuals for the steady, incompressible
Navier-Stokes equations governing coronary hemodynamics. 

Mathematical Justification
--------------------------
To avoid numerical instability caused by the disparate magnitudes of physical
quantities (e.g., blood density, viscosity, vessel diameter, flow velocity), 
the governing equations are non-dimensionalized.

Given characteristic length $L$, velocity $U$, and fluid density $\rho$, we 
define the non-dimensional variables (denoted by asterisks):
    $$x_i^* = \frac{x_i}{L}, \quad u_i^* = \frac{u_i}{U}, \quad p^* = \frac{p}{\rho U^2}$$

Substituting these into the steady incompressible Navier-Stokes equations 
yields the non-dimensional form:
    $$\nabla^* \cdot \mathbf{u}^* = 0$$
    $$(\mathbf{u}^* \cdot \nabla^*) \mathbf{u}^* = -\nabla^* p^* + \frac{1}{Re} \nabla^{*2} \mathbf{u}^*$$

where the Reynolds number is defined as:
    $$Re = \frac{\rho U L}{\mu}$$

This module computes the exact mathematical residuals of these non-dimensional 
equations using exact automatic differentiation (autograd), eliminating the 
truncation errors inherent to finite-difference or finite-element approximations.

Author: Computational Scientist
Target Framework: PyTorch 2.x, Python 3.11+
"""

from typing import Tuple
import torch


def compute_gradient(
    scalar_field: torch.Tensor, coords: torch.Tensor
) -> torch.Tensor:
    """
    Compute the first-order spatial gradient of a scalar field with respect to coordinates.

    Mathematical Formulation
    ------------------------
    Given a scalar field $f(x, y, z)$, this function computes the Jacobian:
        $$\nabla f = \left[ \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z} \right]^T$$

    Parameters
    ----------
    scalar_field : torch.Tensor
        Predicted scalar field of shape $(N, 1)$.
    coords : torch.Tensor
        Spatial coordinates tensor of shape $(N, 3)$. Must have `requires_grad=True`.

    Returns
    -------
    torch.Tensor
        Gradient tensor of shape $(N, 3)$ representing $[f_x, f_y, f_z]$.
    """
    gradient, = torch.autograd.grad(
        outputs=scalar_field,
        inputs=coords,
        grad_outputs=torch.ones_like(scalar_field),
        create_graph=True,
        retain_graph=True,
    )
    return gradient


def compute_gradient_and_laplacian(
    scalar_field: torch.Tensor, coords: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute both the spatial gradient and the Laplacian of a scalar field efficiently.

    Mathematical Formulation
    ------------------------
    Given a scalar field $f(x, y, z)$, the gradient is $\nabla f$ and the Laplacian is
    the trace of the Hessian matrix:
        $$\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2} + \frac{\partial^2 f}{\partial z^2}$$
    
    This function avoids computing the full Hessian, specifically computing only
    the diagonal terms necessary for the viscous stress tensor divergence.

    Parameters
    ----------
    scalar_field : torch.Tensor
        Predicted scalar field of shape $(N, 1)$.
    coords : torch.Tensor
        Spatial coordinates tensor of shape $(N, 3)$. Must have `requires_grad=True`.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        - Gradient tensor of shape $(N, 3)$ representing $[f_x, f_y, f_z]$.
        - Laplacian tensor of shape $(N, 1)$ representing $\nabla^2 f$.
    """
    # 1st-order derivatives: [f_x, f_y, f_z]
    grad_f = compute_gradient(scalar_field, coords)
    
    f_x = grad_f[:, 0:1]
    f_y = grad_f[:, 1:2]
    f_z = grad_f[:, 2:3]
    
    # 2nd-order derivatives (diagonal of the Hessian)
    f_xx = compute_gradient(f_x, coords)[:, 0:1]
    f_yy = compute_gradient(f_y, coords)[:, 1:2]
    f_zz = compute_gradient(f_z, coords)[:, 2:3]
    
    laplacian_f = f_xx + f_yy + f_zz
    
    return grad_f, laplacian_f


class SteadyNavierStokesPhysics:
    """
    Evaluator for the non-dimensional steady incompressible Navier-Stokes PDE residuals.

    Parameters
    ----------
    rho : float
        Fluid density $\rho$ in physical units (e.g., $kg/m^3$).
    mu : float
        Dynamic viscosity $\mu$ in physical units (e.g., $Pa \cdot s$).
    L : float
        Characteristic length scale $L$ (e.g., inlet diameter in meters).
    U : float
        Characteristic velocity scale $U$ (e.g., peak inlet velocity in $m/s$).
    """

    def __init__(self, rho: float, mu: float, L: float, U: float) -> None:
        self.rho = rho
        self.mu = mu
        self.L = L
        self.U = U
        
        # Calculate the non-dimensional Reynolds number
        self.Re = (self.rho * self.U * self.L) / self.mu
        
    def compute_residuals(
        self,
        coords: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        p: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the PDE residuals for continuity and 3D momentum equations.

        Mathematical Formulation
        ------------------------
        Continuity Equation Residual:
            $$e_c = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z}$$
            
        Momentum Equation Residuals:
            $$e_x = u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} + w \frac{\partial u}{\partial z} + \frac{\partial p}{\partial x} - \frac{1}{Re} \nabla^2 u$$
            $$e_y = u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} + w \frac{\partial v}{\partial z} + \frac{\partial p}{\partial y} - \frac{1}{Re} \nabla^2 v$$
            $$e_z = u \frac{\partial w}{\partial x} + v \frac{\partial w}{\partial y} + w \frac{\partial w}{\partial z} + \frac{\partial p}{\partial z} - \frac{1}{Re} \nabla^2 w$$

        Parameters
        ----------
        coords : torch.Tensor
            Non-dimensional coordinates $(x, y, z)$ of shape $(N, 3)$.
        u : torch.Tensor
            Non-dimensional velocity component in x-direction of shape $(N, 1)$.
        v : torch.Tensor
            Non-dimensional velocity component in y-direction of shape $(N, 1)$.
        w : torch.Tensor
            Non-dimensional velocity component in z-direction of shape $(N, 1)$.
        p : torch.Tensor
            Non-dimensional kinematic pressure of shape $(N, 1)$.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple of four tensors, each of shape $(N, 1)$:
            - Continuity residual ($e_c$)
            - X-momentum residual ($e_x$)
            - Y-momentum residual ($e_y$)
            - Z-momentum residual ($e_z$)
        """
        # Ensure coordinates are tracking gradients for autograd
        if not coords.requires_grad:
            raise RuntimeError("Coordinates tensor must have requires_grad=True to compute PDE residuals.")

        # 1. Compute gradients and laplacians for velocity components
        grad_u, laplacian_u = compute_gradient_and_laplacian(u, coords)
        u_x, u_y, u_z = grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3]

        grad_v, laplacian_v = compute_gradient_and_laplacian(v, coords)
        v_x, v_y, v_z = grad_v[:, 0:1], grad_v[:, 1:2], grad_v[:, 2:3]

        grad_w, laplacian_w = compute_gradient_and_laplacian(w, coords)
        w_x, w_y, w_z = grad_w[:, 0:1], grad_w[:, 1:2], grad_w[:, 2:3]

        # 2. Compute first-order gradients for pressure
        grad_p = compute_gradient(p, coords)
        p_x, p_y, p_z = grad_p[:, 0:1], grad_p[:, 1:2], grad_p[:, 2:3]

        # 3. Formulate Continuity Residual (Mass Conservation)
        res_continuity = u_x + v_y + w_z

        # 4. Formulate Momentum Residuals (Momentum Conservation)
        # Advective terms: (u \cdot \nabla) u_i
        advect_u = u * u_x + v * u_y + w * u_z
        advect_v = u * v_x + v * v_y + w * v_z
        advect_w = u * w_x + v * w_y + w * w_z
        
        # Viscous terms: (1 / Re) \nabla^2 u_i
        viscous_u = (1.0 / self.Re) * laplacian_u
        viscous_v = (1.0 / self.Re) * laplacian_v
        viscous_w = (1.0 / self.Re) * laplacian_w

        # Complete Navier-Stokes momentum residuals
        res_momentum_x = advect_u + p_x - viscous_u
        res_momentum_y = advect_v + p_y - viscous_v
        res_momentum_z = advect_w + p_z - viscous_w

        return res_continuity, res_momentum_x, res_momentum_y, res_momentum_z


if __name__ == "__main__":
    # Standalone execution validation
    # Characteristic parameters for human coronary artery hemodynamics
    blood_density = 1060.0       # kg/m^3
    blood_viscosity = 0.0035     # Pa*s
    vessel_diameter = 0.003      # 3 mm
    peak_velocity = 0.25         # 25 cm/s
    
    physics_evaluator = SteadyNavierStokesPhysics(
        rho=blood_density,
        mu=blood_viscosity,
        L=vessel_diameter,
        U=peak_velocity
    )
    
    print(f"Physics Module Initialized.")
    print(f"Calculated Reynolds Number (Re): {physics_evaluator.Re:.2f}")
    
    # Generate mock tensors (batch of 100 points)
    N = 100
    mock_coords = torch.rand((N, 3), requires_grad=True)
    
    # Mock neural network outputs predicting non-dimensional variables
    mock_u = torch.rand((N, 1), requires_grad=True)
    mock_v = torch.rand((N, 1), requires_grad=True)
    mock_w = torch.rand((N, 1), requires_grad=True)
    mock_p = torch.rand((N, 1), requires_grad=True)
    
    # Compute residuals
    e_c, e_x, e_y, e_z = physics_evaluator.compute_residuals(
        coords=mock_coords, u=mock_u, v=mock_v, w=mock_w, p=mock_p
    )
    
    # Validations
    assert e_c.shape == (N, 1), "Continuity residual shape mismatch."
    assert e_x.shape == (N, 1), "X-momentum residual shape mismatch."
    assert e_y.shape == (N, 1), "Y-momentum residual shape mismatch."
    assert e_z.shape == (N, 1), "Z-momentum residual shape mismatch."
    
    print("Navier-Stokes PDE residuals computed successfully via PyTorch autograd.")