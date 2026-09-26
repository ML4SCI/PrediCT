"""
Configuration Module for Coronary Hemodynamics Physics-Informed Neural Network (PINN).

This module defines immutable dataclasses holding all physical parameters,
numerical scales, network architecture settings, sampling schemes, boundary
conditions, loss weights, and optimization hyper-parameters required for 3D
steady incompressible Navier-Stokes PINN simulations in coronary arteries.

Mathematical Rationale & Non-Dimensionalization
------------------------------------------------
Fluid motion in coronary arteries is governed by the 3D steady incompressible
Navier-Stokes equations:

    u * grad(u) = - grad(p) + (1 / Re) * laplacian(u)
    div(u) = 0

To avoid numerical ill-conditioning caused by disparate physical magnitudes
(e.g., dynamic viscosity mu ~ 10^-3 Pa.s vs. blood density rho ~ 10^3 kg/m^3),
all spatial coordinates, velocities, and pressures are scaled by characteristic
dimensions:

    x* = x / L_0
    u* = u / U_0
    p* = p / (rho * U_0^2)

where:
    L_0 : Characteristic vessel lumen diameter (m)
    U_0 : Characteristic mean arterial flow velocity (m/s)
    Re  : Reynolds number = (rho * U_0 * L_0) / mu

Wall Shear Stress (WSS) and Endothelial Shear Stress (ESS) are recovered in physical
units (Pa) via the non-dimensional strain rate tensor:

    tau_w = mu * (U_0 / L_0) * [grad*(u*) + grad*(u*)^T] . n - [n^T . (...) . n] n
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Optional
import torch


@dataclass(frozen=True)
class DirectoriesConfig:
    """
    Filesystem paths for data ingestion, model checkpoints, and exports.

    Attributes
    ----------
    root_dir : Path
        Root directory of the project workspace.
    data_dir : Path
        Directory containing input CTA vessel geometry files (e.g., STL, VTK, OBJ).
    output_dir : Path
        Parent directory for all output artifacts generated during execution.
    checkpoint_dir : Path
        Directory for saving trained PINN PyTorch model checkpoints (.pt).
    export_dir : Path
        Directory for exported VTK/CSV hemodynamic prediction fields.
    log_dir : Path
        Directory for training history logs and TensorBoard metrics.
    """

    root_dir: Path = field(default_factory=lambda: Path("/Users/karan/Desktop/PrediCT"))
    data_dir: Path = field(default_factory=lambda: Path("/Users/karan/Desktop/PrediCT/approved_masks"))
    output_dir: Path = field(default_factory=lambda: Path("/Users/karan/Desktop/PrediCT/output_v2"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("/Users/karan/Desktop/PrediCT/output_v2/checkpoints"))
    export_dir: Path = field(default_factory=lambda: Path("/Users/karan/Desktop/PrediCT/output_v2/exports"))
    log_dir: Path = field(default_factory=lambda: Path("/Users/karan/Desktop/PrediCT/output_v2/logs"))


@dataclass(frozen=True)
class BloodPropertiesConfig:
    """
    Physical fluid properties of human blood under physiological conditions.

    Blood is modeled as an incompressible Newtonian fluid, which is a standard
    and validated approximation in epicardial coronary arteries (diameter > 2 mm).

    Attributes
    ----------
    density : float
        Mass density of blood, rho (kg/m^3). Standard clinical value: 1060.0 kg/m^3.
    dynamic_viscosity : float
        Dynamic viscosity of blood, mu (Pa.s or kg/(m.s)). Standard value: 0.0035 Pa.s.
    """

    density: float = 1060.0  # kg/m^3
    dynamic_viscosity: float = 0.0035  # Pa.s (3.5 cP)

    @property
    def kinematic_viscosity(self) -> float:
        """
        Calculates kinematic viscosity nu = mu / rho (m^2/s).

        Returns
        -------
        float
            Kinematic viscosity in m^2/s (~ 3.3019 e-6 m^2/s).
        """
        return self.dynamic_viscosity / self.density


@dataclass(frozen=True)
class ReferenceScalesConfig:
    """
    Characteristic reference scales used for domain non-dimensionalization.

    Attributes
    ----------
    char_length : float
        Characteristic reference length L_0 (m), typically mean lumen diameter.
    char_velocity : float
        Characteristic reference velocity U_0 (m/s), typically mean inlet velocity.
    """

    char_length: float = 0.003  # m (3.0 mm coronary artery diameter)
    char_velocity: float = 0.25  # m/s (25.0 cm/s mean coronary velocity)

    def compute_reynolds_number(self, blood: BloodPropertiesConfig) -> float:
        """
        Computes the dimensionless Reynolds number Re = (rho * U_0 * L_0) / mu.

        Parameters
        ----------
        blood : BloodPropertiesConfig
            Instance containing blood density and viscosity.

        Returns
        -------
        float
            Dimensionless Reynolds number (~ 227.14 for standard inputs).
        """
        return (blood.density * self.char_velocity * self.char_length) / blood.dynamic_viscosity

    def compute_char_pressure(self, blood: BloodPropertiesConfig) -> float:
        """
        Computes the characteristic dynamic pressure scale P_0 = rho * U_0^2 (Pa).

        Parameters
        ----------
        blood : BloodPropertiesConfig
            Instance containing blood density.

        Returns
        -------
        float
            Characteristic pressure scale in Pascals (~ 66.25 Pa).
        """
        return blood.density * (self.char_velocity ** 2)

    def compute_char_shear_stress(self, blood: BloodPropertiesConfig) -> float:
        """
        Computes characteristic viscous shear stress scale tau_0 = mu * U_0 / L_0 (Pa).

        Parameters
        ----------
        blood : BloodPropertiesConfig
            Instance containing dynamic viscosity.

        Returns
        -------
        float
            Characteristic wall shear stress in Pascals (~ 0.2917 Pa).
        """
        return blood.dynamic_viscosity * (self.char_velocity / self.char_length)


@dataclass(frozen=True)
class ArchitectureConfig:
    """
    Surrogate neural network architecture hyper-parameters.

    Attributes
    ----------
    in_features : int
        Dimensionality of spatial input domain (3 for x, y, z coordinates).
    out_features : int
        Dimensionality of state output vector (4 for u*, v*, w*, p*).
    hidden_dim : int
        Number of neurons per hidden layer.
    num_layers : int
        Number of hidden fully-connected layers.
    activation : str
        Activation function identifier ('silu' / Swish preferred for continuous C^inf derivatives).
    use_fourier_features : bool
        Whether to pass input coordinates through a Random Fourier Feature embedding layer.
    fourier_num_frequencies : int
        Number of Fourier mapping frequency components.
    fourier_scale : float
        Standard deviation of Gaussian frequency matrix projection B.
    use_residual_connections : bool
        Whether to enforce ResNet-style skip connections between adjacent hidden layers.
    """

    in_features: int = 3
    out_features: int = 4
    hidden_dim: int = 128
    num_layers: int = 6
    activation: str = "silu"
    use_fourier_features: bool = True
    fourier_num_frequencies: int = 32
    fourier_scale: float = 2.0
    use_residual_connections: bool = True


@dataclass(frozen=True)
class SamplingConfig:
    """
    Collocation and boundary point sampling configurations.

    Attributes
    ----------
    num_interior_points : int
        Number of interior domain collocation points for Navier-Stokes PDE residuals.
    num_wall_points : int
        Number of boundary points sampled along the vessel arterial wall.
    num_inlet_points : int
        Number of points sampled on the inlet cross-sectional profile.
    num_outlet_points : int
        Number of points sampled on the outlet cross-sectional profile.
    sampling_method : str
        Quasi-Monte Carlo spatial sampling algorithm ('sobol', 'halton', or 'latin_hypercube').
    adaptive_resampling : bool
        Whether to enable residual-based adaptive point resampling during training.
    resampling_interval : int
        Epoch interval for updating collocation points based on PDE error distribution.
    """

    num_interior_points: int = 25000
    num_wall_points: int = 10000
    num_inlet_points: int = 2500
    num_outlet_points: int = 2500
    sampling_method: str = "sobol"
    adaptive_resampling: bool = True
    resampling_interval: int = 1000


@dataclass(frozen=True)
class BoundaryConfig:
    """
    Boundary condition specifications and physiological profiles.

    Attributes
    ----------
    inlet_profile_type : str
        Type of prescribed inlet velocity profile ('parabolic' or 'plug').
    inlet_peak_velocity_scale : float
        Peak non-dimensional centerline velocity ratio (2.0 for 3D parabolic pipe flow).
    outlet_gauge_pressure : float
        Prescribed non-dimensional reference gauge pressure at outlet (typically 0.0).
    wall_velocity_x : float
        Non-dimensional x-velocity component on rigid arterial wall (0.0 for no-slip).
    wall_velocity_y : float
        Non-dimensional y-velocity component on rigid arterial wall (0.0 for no-slip).
    wall_velocity_z : float
        Non-dimensional z-velocity component on rigid arterial wall (0.0 for no-slip).
    """

    inlet_profile_type: str = "parabolic"
    inlet_peak_velocity_scale: float = 2.0
    outlet_gauge_pressure: float = 0.0
    wall_velocity_x: float = 0.0
    wall_velocity_y: float = 0.0
    wall_velocity_z: float = 0.0


@dataclass(frozen=True)
class LossWeightsConfig:
    """
    Multi-objective loss function weighting coefficients.

    Attributes
    ----------
    lambda_continuity : float
        Weight for incompressibility equation residual loss (div(u*) = 0).
    lambda_momentum : float
        Weight for Navier-Stokes momentum conservation equation residual loss.
    lambda_wall_noslip : float
        Weight for Dirichlet no-slip boundary condition loss on vessel walls.
    lambda_inlet : float
        Weight for Dirichlet velocity boundary condition loss at inlet.
    lambda_outlet : float
        Weight for pressure / outflow boundary condition loss at outlet.
    lambda_integral_mass : float
        Weight for integral boundary mass-flux conservation constraint.
    """

    lambda_continuity: float = 10.0
    lambda_momentum: float = 1.0
    lambda_wall_noslip: float = 10.0
    lambda_inlet: float = 500.0      # High to prevent trivial solution collapse
    lambda_outlet: float = 1.0
    lambda_integral_mass: float = 50.0  # Enforce mass conservation


@dataclass(frozen=True)
class TrainingConfig:
    """
    Optimization settings for two-phase training (Adam + L-BFGS).

    Attributes
    ----------
    epochs_adam : int
        Maximum number of iterations for primary Adam optimization phase.
    epochs_lbfgs : int
        Maximum number of iterations for secondary L-BFGS convergence phase.
    learning_rate_adam : float
        Initial learning rate for Adam optimizer.
    lr_decay_step : int
        Epoch interval for step learning rate decay scheduler.
    lr_decay_gamma : float
        Multiplicative factor for learning rate decay (lr_new = lr * gamma).
    lbfgs_lr : float
        Learning rate for L-BFGS optimizer step.
    lbfgs_max_iter : int
        Maximum function evaluations per line search step in L-BFGS.
    lbfgs_history_size : int
        Number of previous gradients stored to approximate Hessian in L-BFGS.
    gradient_clip_norm : float
        Maximum allowed norm for gradient clipping to prevent exploding gradients.
    """

    epochs_adam: int = int(os.environ.get("PREDICT_EPOCHS", 50000))
    epochs_lbfgs: int = 0
    learning_rate_adam: float = 1e-3
    lr_decay_step: int = 2500
    lr_decay_gamma: float = 0.5
    lbfgs_lr: float = 1.0
    lbfgs_max_iter: int = 20
    lbfgs_history_size: int = 50
    gradient_clip_norm: float = 1.0


@dataclass(frozen=True)
class ValidationConfig:
    """
    Model evaluation, diagnostic, and biomarker post-processing configurations.

    Attributes
    ----------
    val_interval : int
        Epoch frequency for running validation assessment and logging metrics.
    save_top_k : int
        Number of best model checkpoints to maintain based on validation loss.
    compute_ess_biomarkers : bool
        Whether to calculate Wall Shear Stress vectors and Endothelial Shear Stress maps.
    ess_low_threshold_pa : float
        Clinical low ESS threshold in Pascals (< 1.0 Pa promotes atherogenesis).
    ess_high_threshold_pa : float
        Clinical high ESS threshold in Pascals (> 7.0 Pa risks fibrous cap erosion).
    plaque_risk_k_ess : float
        Sigmoid steepness coefficient for low ESS plaque vulnerability mapping.
    plaque_risk_k_grad : float
        Sigmoid coefficient weighting ESS spatial gradients in plaque risk.
    """

    val_interval: int = 500
    save_top_k: int = 3
    compute_ess_biomarkers: bool = True
    ess_low_threshold_pa: float = 1.0
    ess_high_threshold_pa: float = 7.0
    plaque_risk_k_ess: float = 3.0
    plaque_risk_k_grad: float = 0.1


@dataclass(frozen=True)
class RuntimeConfig:
    """
    Execution environment, hardware selection, and random seed parameters.

    Attributes
    ----------
    seed : int
        Global pseudo-random generator seed for PyTorch, NumPy, and random module.
    device_name : str
        Target compute backend ('cuda', 'mps', or 'cpu'). Auto-selects if set to 'auto'.
    dtype : str
        Default floating point tensor precision ('float32' or 'float64').
    """

    seed: int = 42
    device_name: str = "auto"
    dtype: str = "float32"

    def resolve_device(self) -> torch.device:
        """
        Determines and returns the appropriate PyTorch torch.device object.

        Returns
        -------
        torch.device
            Resolved PyTorch compute device (CUDA GPU, Apple MPS, or CPU).
        """
        if self.device_name != "auto":
            return torch.device(self.device_name)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def resolve_dtype(self) -> torch.dtype:
        """
        Maps standard string representation to PyTorch floating point data type.

        Returns
        -------
        torch.dtype
            torch.float32 or torch.float64 tensor dtype.
        """
        if self.dtype == "float64":
            return torch.float64
        return torch.float32


@dataclass(frozen=True)
class CurriculumConfig:
    """
    Two-stage curriculum training configuration.
    """
    bc_pretrain_epochs: int = 5000
    max_epochs: int = int(os.environ.get("PREDICT_EPOCHS", 50000))

@dataclass(frozen=True)
class PINNConfig:
    """
    Master configuration object combining all domain sub-configurations.

    Attributes
    ----------
    directories : DirectoriesConfig
        Filesystem directory specifications.
    blood : BloodPropertiesConfig
        Fluid dynamic properties of blood.
    scales : ReferenceScalesConfig
        Non-dimensional reference characteristic scales.
    architecture : ArchitectureConfig
        PINN surrogate network topology hyper-parameters.
    sampling : SamplingConfig
        Spatial domain collocation point sampling rules.
    boundary : BoundaryConfig
        Physiological boundary conditions and profiles.
    loss_weights : LossWeightsConfig
        Multi-task penalty weights for PINN loss formulation.
    training : TrainingConfig
        Adam and L-BFGS optimization schedules.
    validation : ValidationConfig
        Validation, ESS evaluation, and diagnostic settings.
    runtime : RuntimeConfig
        Seed, device, and precision settings.
    """

    directories: DirectoriesConfig = field(default_factory=DirectoriesConfig)
    blood: BloodPropertiesConfig = field(default_factory=BloodPropertiesConfig)
    scales: ReferenceScalesConfig = field(default_factory=ReferenceScalesConfig)
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    loss_weights: LossWeightsConfig = field(default_factory=LossWeightsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    patient_mask_path: Optional[str] = field(
        default_factory=lambda: os.getenv("PREDICT_MASK_PATH", "/Users/karan/Desktop/PrediCT/approved_masks/1a19d11e263a_vessels.nii.gz")
    )
    run_name: str = field(
        default_factory=lambda: os.getenv("PREDICT_RUN_NAME", "phase2_run")
    )