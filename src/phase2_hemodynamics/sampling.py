r"""
Coronary Artery PINN Collocation Sampler Module
====================================

This module provides spatial domain sampling capabilities for Physics-Informed
Neural Networks (PINNs) solving 3D steady incompressible Navier-Stokes equations
in patient-specific coronary artery geometries derived from Computed Tomography
Angiography (CTA).

Mathematical Justification
--------------------------
In computational hemodynamics, endothelial shear stress (ESS) is defined as the
magnitude of the tangential traction vector exerted by blood flow on the vessel wall:

    $$\boldsymbol{\tau}_w = \mu \left( \nabla \mathbf{u} + (\nabla \mathbf{u})^T \right) \mathbf{n} - \left[ \mathbf{n} \cdot \mu \left( \nabla \mathbf{u} + (\nabla \mathbf{u})^T \right) \mathbf{n} \right] \mathbf{n}$$

where $\mu$ is dynamic viscosity, $\mathbf{u}$ is velocity, and $\mathbf{n}$ is the inward unit
normal vector. Accurately resolving ESS and high-order spatial derivatives of velocity requires
dense collocation point sampling in regions characterized by sharp velocity gradients $\nabla \mathbf{u}$,
specifically:
1. Boundary layers near vessel walls ($\delta \ll R$).
2. Stenotic luminal narrowing regions with convective flow acceleration ($\mathbf{u} \cdot \nabla \mathbf{u}$).
3. Bifurcation core regions with secondary Dean vortices and flow stagnation/recirculation.

To prevent spectral bias and optimize PDE loss convergence, sampling is performed using
an adaptive probability density function $P(\mathbf{x})$ integrated with Space-Filling Latin
Hypercube Sampling (LHS) and $k$-dimensional tree (KDTree) spatial acceleration algorithms.

Author: Computational Scientist
Target Framework: PyTorch 2.x, Python 3.11+
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import qmc


@dataclass
class SampledDomain:
    """
    Container for sampled PINN collocation points and associated boundary normals.

    Attributes
    ----------
    x_interior : torch.Tensor
        Interior domain collocation points $(N_{\text{int}}, 3)$.
    x_wall : torch.Tensor
        Vessel wall boundary points $(N_{\text{wall}}, 3)$.
    n_wall : torch.Tensor
        Unit inward normal vectors at wall points $(N_{\text{wall}}, 3)$.
    x_inlet : torch.Tensor
        Inlet boundary points $(N_{\text{in}}, 3)$.
    n_inlet : torch.Tensor
        Unit inward normal vectors at inlet $(N_{\text{in}}, 3)$.
    x_outlet : torch.Tensor
        Outlet boundary points $(N_{\text{out}}, 3)$.
    n_outlet : torch.Tensor
        Unit inward normal vectors at outlets $(N_{\text{out}}, 3)$.
    wall_distance : torch.Tensor
        Euclidean distance to nearest wall point for interior points $(N_{\text{int}}, 1)$.
    """
    x_interior: torch.Tensor
    x_wall: torch.Tensor
    n_wall: torch.Tensor
    x_inlet: torch.Tensor
    n_inlet: torch.Tensor
    x_outlet: torch.Tensor
    n_outlet: torch.Tensor
    wall_distance: torch.Tensor


class CoronaryGeometry:
    """
    Representation of 3D Coronary Geometry derived from CTA Point Clouds.

    Parameters
    ----------
    wall_points : np.ndarray
        Array of shape $(M_{\text{wall}}, 3)$ defining vessel wall mesh coordinates.
    wall_normals : np.ndarray
        Array of shape $(M_{\text{wall}}, 3)$ defining inward surface unit normals.
    inlet_points : np.ndarray
        Array of shape $(M_{\text{in}}, 3)$ defining inlet boundary surface.
    inlet_normals : np.ndarray
        Array of shape $(M_{\text{in}}, 3)$ defining inlet inward unit normals.
    outlet_points : np.ndarray
        Array of shape $(M_{\text{out}}, 3)$ defining outlet boundary surface.
    outlet_normals : np.ndarray
        Array of shape $(M_{\text{out}}, 3)$ defining outlet inward unit normals.
    centerline_points : np.ndarray
        Array of shape $(M_{\text{center}}, 3)$ defining vessel centerline path.
    centerline_radii : np.ndarray
        Array of shape $(M_{\text{center}},)$ representing local vessel radii $R(s)$.
    bifurcation_nodes : Optional[np.ndarray], optional
        Array of shape $(K, 3)$ specifying bifurcation flow split coordinates.
    """

    def __init__(
        self,
        wall_points: np.ndarray,
        wall_normals: np.ndarray,
        inlet_points: np.ndarray,
        inlet_normals: np.ndarray,
        outlet_points: np.ndarray,
        outlet_normals: np.ndarray,
        centerline_points: np.ndarray,
        centerline_radii: np.ndarray,
        bifurcation_nodes: Optional[np.ndarray] = None,
    ) -> None:
        self.wall_points = np.asarray(wall_points, dtype=np.float64)
        self.wall_normals = np.asarray(wall_normals, dtype=np.float64)
        self.inlet_points = np.asarray(inlet_points, dtype=np.float64)
        self.inlet_normals = np.asarray(inlet_normals, dtype=np.float64)
        self.outlet_points = np.asarray(outlet_points, dtype=np.float64)
        self.outlet_normals = np.asarray(outlet_normals, dtype=np.float64)
        self.centerline_points = np.asarray(centerline_points, dtype=np.float64)
        self.centerline_radii = np.asarray(centerline_radii, dtype=np.float64)

        if bifurcation_nodes is not None:
            self.bifurcation_nodes = np.asarray(bifurcation_nodes, dtype=np.float64)
        else:
            self.bifurcation_nodes = np.empty((0, 3), dtype=np.float64)

        # Normalize wall surface normal vectors
        wall_norm_mags = np.linalg.norm(self.wall_normals, axis=1, keepdims=True)
        wall_norm_mags[wall_norm_mags == 0] = 1.0
        self.wall_normals /= wall_norm_mags

        # Construct spatial K-D Trees for high-performance geometric queries
        self.wall_kdtree = cKDTree(self.wall_points)
        self.centerline_kdtree = cKDTree(self.centerline_points)

        # Compute geometric bounding box
        self.min_bounds = np.min(self.wall_points, axis=0)
        self.max_bounds = np.max(self.wall_points, axis=0)

    def query_wall_distance(self, points: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distance from points to the nearest vessel wall point.

        Parameters
        ----------
        points : np.ndarray
            Evaluation points $(N, 3)$.

        Returns
        -------
        np.ndarray
            Distance array $(N,)$.
        """
        distances, _ = self.wall_kdtree.query(points, k=1)
        return distances

    def compute_stenosis_score(self, points: np.ndarray) -> np.ndarray:
        r"""
        Calculate local stenosis score $S(\mathbf{x}) \in [0, 1]$ based on spatial radius gradients.

        Mathematical Formulation
        ------------------------
        Let $R(s)$ be the centerline radius projected to query point $\mathbf{x}$. The local
        stenosis severity is quantified by comparing local radius relative to the mean
        un-constricted radius $R_{\text{ref}}$:

        $$S(\mathbf{x}) = \max\left(0, 1 - \frac{R(\mathbf{x})}{R_{\text{ref}}}\right)^2$$

        Parameters
        ----------
        points : np.ndarray
            Evaluation points $(N, 3)$.

        Returns
        -------
        np.ndarray
            Stenosis score array $(N,)$.
        """
        _, idxs = self.centerline_kdtree.query(points, k=1)
        local_radii = self.centerline_radii[idxs]
        ref_radius = np.percentile(self.centerline_radii, 90)
        stenosis_score = np.maximum(0.0, 1.0 - (local_radii / ref_radius)) ** 2
        return stenosis_score

    def compute_bifurcation_score(
        self, points: np.ndarray, kernel_bandwidth: float = 3.0
    ) -> np.ndarray:
        r"""
        Calculate spatial proximity score $B(\mathbf{x}) \in [0, 1]$ to vessel bifurcations.

        Mathematical Formulation
        ------------------------
        Using a Gaussian RBF kernel across all $K$ bifurcation junction nodes:

        $$B(\mathbf{x}) = \sum_{k=1}^K \exp\left(-\frac{\|\mathbf{x} - \mathbf{b}_k\|^2}{2 \sigma_b^2}\right)$$

        Parameters
        ----------
        points : np.ndarray
            Evaluation points $(N, 3)$.
        kernel_bandwidth : float, optional
            Bandwidth hyperparameter $\sigma_b$ in millimeters (default: 3.0).

        Returns
        -------
        np.ndarray
            Bifurcation score array $(N,)$.
        """
        if len(self.bifurcation_nodes) == 0:
            return np.zeros(points.shape[0], dtype=np.float64)

        scores = np.zeros(points.shape[0], dtype=np.float64)
        for node in self.bifurcation_nodes:
            sq_dist = np.sum((points - node) ** 2, axis=1)
            scores += np.exp(-sq_dist / (2.0 * kernel_bandwidth**2))

        return np.clip(scores, 0.0, 1.0)

    def is_inside(self, points: np.ndarray) -> np.ndarray:
        """
        Evaluate physical domain containment for sample candidate points.

        Parameters
        ----------
        points : np.ndarray
            Evaluation points $(N, 3)$.

        Returns
        -------
        np.ndarray
            Boolean array of size $(N,)$ where True indicates interior location.
        """
        dist_center, idxs = self.centerline_kdtree.query(points, k=1)
        local_radii = self.centerline_radii[idxs]
        return dist_center <= local_radii


class AdaptiveCoronarySampler:
    """
    Adaptive Physics-Informed Sampler prioritizing regions with steep velocity gradients.

    Parameters
    ----------
    geometry : CoronaryGeometry
        The targeted CTA coronary artery geometry object.
    device : Union[str, torch.device], optional
        Target PyTorch computation device (default: 'cpu').
    dtype : torch.dtype, optional
        Target PyTorch floating-point precision type (default: torch.float32).
    """

    def __init__(
        self,
        geometry: CoronaryGeometry,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.geometry = geometry
        self.device = torch.device(device)
        self.dtype = dtype

    def _sample_latin_hypercube(
        self, num_samples: int, bounds_min: np.ndarray, bounds_max: np.ndarray
    ) -> np.ndarray:
        """
        Generate space-filling samples in a 3D bounding box using Latin Hypercube Sampling.

        Parameters
        ----------
        num_samples : int
            Number of raw sampling points requested.
        bounds_min : np.ndarray
            Minimum coordinate bounds $(3,)$.
        bounds_max : np.ndarray
            Maximum coordinate bounds $(3,)$.

        Returns
        -------
        np.ndarray
            Scaled sample points $(N, 3)$.
        """
        sampler = qmc.LatinHypercube(d=3)
        sample_unit = sampler.random(n=num_samples)
        return qmc.scale(sample_unit, bounds_min, bounds_max)

    def sample_interior(
        self,
        num_interior: int,
        wall_decay_length: float = 0.5,
        w_wall: float = 4.0,
        w_stenosis: float = 3.0,
        w_bifurcation: float = 3.0,
        oversample_factor: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Sample interior PDE collocation points with adaptive PDF concentration.

        Mathematical Formulation
        ------------------------
        Generate spatial points $\mathbf{x}$ scattered throughout the 3D computational domain $\Omega$.

        Candidates $\mathbf{x} \sim \mathcal{U}(\Omega_{\text{bbox}})$ are drawn via LHS and filtered to
        interior points $\Omega$. Points are selected via Rejection Sampling with Probability Density $P(\mathbf{x})$:

        $$P(\mathbf{x}) = \frac{1}{Z} \left[ 1.0 + w_w \exp\left(-\frac{\delta_w(\mathbf{x})}{\lambda_w}\right) + w_s S(\mathbf{x}) + w_b B(\mathbf{x}) \right]$$

        where:
        - $\delta_w(\mathbf{x})$ is the distance to the nearest wall point.
        - $\lambda_w$ is the boundary layer decay parameter.
        - $S(\mathbf{x})$ is the local stenosis severity score.
        - $B(\mathbf{x})$ is the bifurcation proximity score.

        Parameters
        ----------
        num_interior : int
            Exact number of interior points to return.
        wall_decay_length : float, optional
            Boundary layer thickness parameter $\lambda_w$ in mm (default: 0.5).
        w_wall : float, optional
            Weight factor for boundary layer density boost (default: 4.0).
        w_stenosis : float, optional
            Weight factor for stenosis concentration (default: 3.0).
        w_bifurcation : float, optional
            Weight factor for bifurcation concentration (default: 3.0).
        oversample_factor : int, optional
            Candidate multiplier for Monte Carlo rejection sampling (default: 10).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - Interior domain coordinates $(N_{\text{int}}, 3)$
            - Wall distances for interior points $(N_{\text{int}}, 1)$
        """
        accepted_points: List[np.ndarray] = []
        accepted_distances: List[np.ndarray] = []
        total_accepted = 0

        max_pdf = 1.0 + w_wall + w_stenosis + w_bifurcation

        while total_accepted < num_interior:
            raw_candidates = self._sample_latin_hypercube(
                num_samples=num_interior * oversample_factor,
                bounds_min=self.geometry.min_bounds,
                bounds_max=self.geometry.max_bounds,
            )

            # Containment check
            inside_mask = self.geometry.is_inside(raw_candidates)
            candidates = raw_candidates[inside_mask]

            if len(candidates) == 0:
                continue

            # Compute metric features
            distances = self.geometry.query_wall_distance(candidates)
            stenosis_scores = self.geometry.compute_stenosis_score(candidates)
            bifurcation_scores = self.geometry.compute_bifurcation_score(candidates)

            # Evaluate probability density function
            wall_boost = np.exp(-distances / wall_decay_length)
            pdf_values = (
                1.0
                + w_wall * wall_boost
                + w_stenosis * stenosis_scores
                + w_bifurcation * bifurcation_scores
            )

            # Rejection sampling acceptance check
            acceptance_probs = pdf_values / max_pdf
            random_draws = np.random.uniform(0.0, 1.0, size=len(candidates))
            accepted_mask = random_draws <= acceptance_probs

            selected = candidates[accepted_mask]
            selected_dists = distances[accepted_mask]

            accepted_points.append(selected)
            accepted_distances.append(selected_dists)
            total_accepted += len(selected)

        final_points = np.vstack(accepted_points)[:num_interior]
        final_distances = np.concatenate(accepted_distances)[:num_interior, None]

        return final_points, final_distances

    def sample_surface_boundary(
        self,
        surface_points: np.ndarray,
        surface_normals: np.ndarray,
        num_target: int,
        weights: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample surface boundary points uniformly or adaptively using index resampling.

        Parameters
        ----------
        surface_points : np.ndarray
            Array of available boundary points $(M, 3)$.
        surface_normals : np.ndarray
            Array of available unit normals $(M, 3)$.
        num_target : int
            Desired output sample count $N$.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - Sampled boundary coordinates $(N, 3)$
            - Corresponding unit normals $(N, 3)$
        """
        num_available = len(surface_points)
        replace_flag = num_target > num_available
        
        if weights is not None:
            # Normalize weights to form a probability distribution
            probs = weights / np.sum(weights)
        else:
            probs = None
            
        selected_indices = np.random.choice(
            num_available, size=num_target, replace=replace_flag, p=probs
        )

        return surface_points[selected_indices], surface_normals[selected_indices]

    def build_pinn_dataset(
        self,
        num_interior: int,
        num_wall: int,
        num_inlet: int,
        num_outlet: int,
        wall_decay_length: float = 0.5,
        w_wall: float = 4.0,
        w_stenosis: float = 3.0,
        w_bifurcation: float = 3.0,
        char_length_mm: float = None,
    ) -> SampledDomain:
        """
        Construct and return complete set of PINN collocation point sets as PyTorch Tensors.

        Parameters
        ----------
        num_interior : int
            Number of interior PDE residual collocation points.
        num_wall : int
            Number of no-slip wall boundary points.
        num_inlet : int
            Number of inlet Dirichlet condition boundary points.
        num_outlet : int
            Number of outlet Neumann condition boundary points.
        wall_decay_length : float, optional
            Boundary layer resolution decay parameter (default: 0.5).
        w_wall : float, optional
            Weight factor for wall concentration (default: 4.0).
        w_stenosis : float, optional
            Weight factor for stenosis concentration (default: 3.0).
        w_bifurcation : float, optional
            Weight factor for bifurcation concentration (default: 3.0).
        char_length_mm : float, optional
            Reference length scale in millimetres used to non-dimensionalise
            spatial coordinates before they enter the PINN.

            H-1 FIX: geometry.py outputs coordinates in mm; physics.py computes
            Navier-Stokes residuals in non-dimensional variables x* = x / L0.
            Without this scaling the advective/viscous balance is wrong by a
            factor of L0 = 0.003 m = 3 mm. Typical value:
            ``char_length_mm = config.scales.char_length * 1e3``  (e.g. 3.0).

            Scaling is applied to the four spatial coordinate arrays only.
            Unit normals and wall_distance are NOT scaled.

        Returns
        -------
        SampledDomain
            Structured dataclass storing all PyTorch tensors for model training.
        """
        # Interior points
        x_int_np, dist_np = self.sample_interior(
            num_interior=num_interior,
            wall_decay_length=wall_decay_length,
            w_wall=w_wall,
            w_stenosis=w_stenosis,
            w_bifurcation=w_bifurcation,
        )

        # Wall points - apply geometry-aware sampling
        wall_stenosis = self.geometry.compute_stenosis_score(self.geometry.wall_points)
        wall_bifurcation = self.geometry.compute_bifurcation_score(self.geometry.wall_points)
        wall_weights = 1.0 + w_stenosis * wall_stenosis + w_bifurcation * wall_bifurcation
        
        x_wall_np, n_wall_np = self.sample_surface_boundary(
            self.geometry.wall_points,
            self.geometry.wall_normals,
            num_wall,
            weights=wall_weights
        )

        # Inlet points
        x_in_np, n_in_np = self.sample_surface_boundary(
            self.geometry.inlet_points,
            self.geometry.inlet_normals,
            num_inlet,
        )

        # Outlet points
        x_out_np, n_out_np = self.sample_surface_boundary(
            self.geometry.outlet_points,
            self.geometry.outlet_normals,
            num_outlet,
        )

        # H-1 FIX: Non-dimensionalise spatial coordinates when char_length_mm is provided.
        # All coordinate arrays are in mm from geometry.py. The physics module expects
        # x* = x_mm / char_length_mm. Normals (unit vectors) and wall_distance
        # (used only for interior sampling weights) are NOT scaled.
        if char_length_mm is not None:
            _scale = float(char_length_mm)
            x_int_np  = x_int_np  / _scale
            x_wall_np = x_wall_np / _scale
            x_in_np   = x_in_np   / _scale
            x_out_np  = x_out_np  / _scale

        # C-3 FIX: coordinate tensors must carry requires_grad=True.
        # Every physics/BC call (compute_residuals, compute_velocity_jacobian,
        # compute_parabolic_inlet_loss, compute_velocity_divergence) issues
        # torch.autograd.grad(inputs=coords, ...). Without requires_grad on the
        # coordinate tensor the autograd engine has no leaf to differentiate
        # through and raises RuntimeError.
        # Normal vectors and wall_distance are NOT differentiated through
        # and correctly remain requires_grad=False.
        return SampledDomain(
            x_interior=torch.tensor(x_int_np, dtype=self.dtype, device=self.device).requires_grad_(True),
            x_wall=torch.tensor(x_wall_np, dtype=self.dtype, device=self.device).requires_grad_(True),
            n_wall=torch.tensor(n_wall_np, dtype=self.dtype, device=self.device),
            x_inlet=torch.tensor(x_in_np, dtype=self.dtype, device=self.device).requires_grad_(True),
            n_inlet=torch.tensor(n_in_np, dtype=self.dtype, device=self.device),
            x_outlet=torch.tensor(x_out_np, dtype=self.dtype, device=self.device).requires_grad_(True),
            n_outlet=torch.tensor(n_out_np, dtype=self.dtype, device=self.device),
            wall_distance=torch.tensor(dist_np, dtype=self.dtype, device=self.device),
        )


def generate_synthetic_coronary_geometry() -> CoronaryGeometry:
    """
    Generate a benchmark synthetic bifurcated coronary artery geometry with a stenosis.

    Returns
    -------
    CoronaryGeometry
        Instantiated test vessel geometry with stenosis and bifurcation junction.
    """
    s_vals = np.linspace(0.0, 30.0, 150)
    
    # Curved centerline definition with stenosis at s = 10mm
    x_c = s_vals
    y_c = 2.0 * np.sin(s_vals / 5.0)
    z_c = np.zeros_like(s_vals)
    centerline = np.column_stack([x_c, y_c, z_c])

    # Radii profile: Baseline R=1.5mm, Stenosis stenosis at s=10mm drops to R=0.75mm
    radii = 1.5 - 0.75 * np.exp(-((s_vals - 10.0) ** 2) / 4.0)

    # Generate synthetic wall point cloud
    wall_pts = []
    wall_norms = []
    theta_vals = np.linspace(0, 2 * np.pi, 24, endpoint=False)

    for i in range(len(s_vals)):
        c = centerline[i]
        r = radii[i]
        for t in theta_vals:
            normal = np.array([0.0, np.cos(t), np.sin(t)], dtype=np.float64)
            pt = c + r * normal
            wall_pts.append(pt)
            wall_norms.append(-normal)  # Inward pointing normal

    wall_pts = np.array(wall_pts)
    wall_norms = np.array(wall_norms)

    # Inlet: s = 0
    inlet_pts = wall_pts[:24]
    inlet_norms = np.tile(np.array([1.0, 0.0, 0.0]), (24, 1))

    # Outlet: s = 30
    outlet_pts = wall_pts[-24:]
    outlet_norms = np.tile(np.array([-1.0, 0.0, 0.0]), (24, 1))

    # Bifurcation node located at s = 22mm
    bifurcation_nodes = np.array([[22.0, 2.0 * np.sin(22.0 / 5.0), 0.0]])

    return CoronaryGeometry(
        wall_points=wall_pts,
        wall_normals=wall_norms,
        inlet_points=inlet_pts,
        inlet_normals=inlet_norms,
        outlet_points=outlet_pts,
        outlet_normals=outlet_norms,
        centerline_points=centerline,
        centerline_radii=radii,
        bifurcation_nodes=bifurcation_nodes,
    )


if __name__ == "__main__":
    print("Initializing Coronary PINN Geometry & Adaptive Sampler...")

    # Instantiation of synthetic vessel geometry
    geom = generate_synthetic_coronary_geometry()
    sampler = AdaptiveCoronarySampler(geometry=geom, device="cpu", dtype=torch.float32)

    # Execute sampling
    n_int, n_w, n_in, n_out = 5000, 1000, 200, 200
    pinn_data = sampler.build_pinn_dataset(
        num_interior=n_int,
        num_wall=n_w,
        num_inlet=n_in,
        num_outlet=n_out,
        wall_decay_length=0.4,
        w_wall=5.0,
        w_stenosis=4.0,
        w_bifurcation=3.0,
    )

    print("\n--- PINN Collocation Dataset Generated Successfully ---")
    print(f"Interior Points Shape     : {pinn_data.x_interior.shape}")
    print(f"Wall Points Shape         : {pinn_data.x_wall.shape}")
    print(f"Wall Normals Shape        : {pinn_data.n_wall.shape}")
    print(f"Inlet Points Shape        : {pinn_data.x_inlet.shape}")
    print(f"Inlet Normals Shape       : {pinn_data.n_inlet.shape}")
    print(f"Outlet Points Shape       : {pinn_data.x_outlet.shape}")
    print(f"Outlet Normals Shape      : {pinn_data.n_outlet.shape}")
    print(f"Wall Distance Tensor Shape: {pinn_data.wall_distance.shape}")

    # Validation Checks
    assert pinn_data.x_interior.shape == (n_int, 3)
    assert pinn_data.x_wall.shape == (n_w, 3)
    assert pinn_data.n_wall.shape == (n_w, 3)
    assert pinn_data.wall_distance.shape == (n_int, 1)
    assert torch.all(pinn_data.wall_distance >= 0.0)

    print("\nVerification Passed: Tensor shapes and mathematical distance metrics are valid.")