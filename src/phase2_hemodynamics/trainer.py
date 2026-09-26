"""
Physics-Informed Neural Network (PINN) Trainer Module
=====================================================

This module provides a robust, high-performance training loop for Physics-Informed
Neural Networks solving the steady incompressible Navier-Stokes equations for
coronary hemodynamics.

Mathematical Justification
--------------------------
Training deep PINNs for fluid dynamics is characterized by severe ill-conditioning
arising from the disparate frequency spectra of the loss components (e.g., zero-order
boundary terms vs. second-order viscous spatial derivatives). 

To ensure numerical stability and optimal convergence, this trainer implements:

1.  **Mixed-Precision Training (AMP)**:
    Computations are cast to FP16 where numerically safe to maximize tensor core
    utilization and memory bandwidth, while accumulating gradients in FP32 to
    prevent arithmetic underflow in the small-magnitude PDE residuals.
    
2.  **Gradient Norm Clipping**:
    High-order derivatives in the viscous Laplacian ($\nabla^2 \mathbf{u}$) can produce
    exploding gradients during early training phases. We constrain the $L_2$ norm of
    the gradient vector $\mathbf{g} = \nabla_\theta \mathcal{L}$ such that:
    $$ \mathbf{g} \leftarrow \begin{cases} \frac{c}{\|\mathbf{g}\|_2} \mathbf{g} & \text{if } \|\mathbf{g}\|_2 > c \\ \mathbf{g} & \text{otherwise} \end{cases} $$
    where $c$ is the clipping threshold.

3.  **Early Stopping**:
    To prevent overfitting to the finite set of collocation points and mitigating
    spectral bias where the network fits high-frequency noise, training is halted if
    the validation loss functional fails to decrease by at least $\epsilon$ over
    $N_p$ consecutive epochs.

Author: Computational Scientist
Target Framework: PyTorch 2.x, Python 3.11+
"""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


@dataclass(frozen=True)
class PINNTrainerConfig:
    """
    Configuration hyperparameters for the PINN training loop.

    Attributes
    ----------
    max_epochs : int
        Maximum number of training epochs.
    learning_rate : float
        Initial learning rate for the main optimizer.
    clip_grad_norm : float
        Maximum allowed $L_2$ norm for network gradients.
    early_stopping_patience : int
        Number of epochs to wait for improvement before halting.
    early_stopping_delta : float
        Minimum relative decrease required to reset the patience counter.
    checkpoint_dir : str
        Directory to store model checkpoints and TensorBoard logs.
    use_mixed_precision : bool
        Flag to enable PyTorch Automatic Mixed Precision (AMP).
    log_frequency : int
        Frequency (in epochs) to print metrics to the standard output.
    """
    max_epochs: int = 5000
    learning_rate: float = 1e-3
    clip_grad_norm: float = 1.0
    early_stopping_patience: int = 5000
    early_stopping_delta: float = 1e-5
    checkpoint_dir: str = "./pinn_checkpoints"
    use_mixed_precision: bool = False  # C-4 FIX: AMP (FP16) underflows second-order autograd
                                       # derivatives used in Laplacian computation. PINN viscous
                                       # residuals at convergence are O(1e-5..1e-6), below FP16's
                                       # ~6e-8 precision floor. This silently zeros ∇²u momentum
                                       # terms. Must remain False for PINN training.
    log_frequency: int = 100


class PhysicsDiagnostics:
    """
    Diagnostic utility to monitor physical validity of PINN predictions during training.
    """

    @staticmethod
    def compute_diagnostics(
        u: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        p: torch.Tensor,
        res_continuity: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute macroscopic physical metrics for logging and validation.

        Parameters
        ----------
        u : torch.Tensor
            Predicted x-velocity $(N, 1)$.
        v : torch.Tensor
            Predicted y-velocity $(N, 1)$.
        w : torch.Tensor
            Predicted z-velocity $(N, 1)$.
        p : torch.Tensor
            Predicted kinematic pressure $(N, 1)$.
        res_continuity : torch.Tensor
            Continuity equation residual $\nabla \cdot \mathbf{u}$ $(N, 1)$.

        Returns
        -------
        Dict[str, float]
            Dictionary containing computed physical scalar metrics.
        """
        with torch.no_grad():
            velocity_magnitude = torch.sqrt(u**2 + v**2 + w**2)
            max_vel = float(torch.max(velocity_magnitude).item())
            mean_vel = float(torch.mean(velocity_magnitude).item())
            max_div = float(torch.max(torch.abs(res_continuity)).item())
            mean_pressure = float(torch.mean(p).item())
            
        return {
            "physics/max_velocity": max_vel,
            "physics/mean_velocity": mean_vel,
            "physics/max_divergence": max_div,
            "physics/mean_pressure": mean_pressure
        }


class PINNTrainer:
    """
    Orchestrator for PINN training executing the forward/backward passes,
    optimization, scheduling, logging, and checkpointing.

    Parameters
    ----------
    model : nn.Module
        The Physics-Informed Neural Network to be trained.
    optimizer : torch.optim.Optimizer
        The primary optimizer (e.g., Adam) for the network parameters.
    config : PINNTrainerConfig
        The configuration object detailing training hyperparameters.
    device : torch.device
        The computational device (CPU/CUDA) where tensors reside.
    scheduler : Optional[ReduceLROnPlateau], optional
        Learning rate scheduler to decay the learning rate upon plateau.
    weight_optimizer : Optional[torch.optim.Optimizer], optional
        Secondary optimizer dedicated to adaptive loss weighting algorithms (e.g., GradNorm).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: PINNTrainerConfig,
        device: torch.device,
        scheduler: Optional[ReduceLROnPlateau] = None,
        weight_optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.scheduler = scheduler
        self.weight_optimizer = weight_optimizer

        # State tracking
        self.current_epoch = 0
        self.best_loss = float("inf")
        self.epochs_without_improvement = 0

        # Create output directories
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging & Tensorboard
        self.logger = self._configure_logger()
        if TENSORBOARD_AVAILABLE:
            self.tensorboard = SummaryWriter(log_dir=str(self.checkpoint_dir / "logs"))
        else:
            self.tensorboard = None
            self.logger.warning("Tensorboard not found, disabling tensorboard logging.")
        
        # Automatic Mixed Precision
        self.scaler = GradScaler(enabled=self.config.use_mixed_precision)
        
        self.logger.info(f"Initialized PINNTrainer on device: {self.device}")
        self.logger.info(f"Mixed Precision Training: {self.config.use_mixed_precision}")

    def _configure_logger(self) -> logging.Logger:
        """
        Configure the standard Python logger for the trainer.

        Returns
        -------
        logging.Logger
            Configured logger instance.
        """
        logger = logging.getLogger("PINNTrainer")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            
            file_handler = logging.FileHandler(self.checkpoint_dir / "training.log")
            file_handler.setFormatter(formatter)
            
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        return logger

    def save_checkpoint(self, filename: str = "latest_checkpoint.pt") -> None:
        """
        Serialize and export the complete training state to disk.

        Parameters
        ----------
        filename : str, optional
            The destination filename within the configured checkpoint directory.
        """
        checkpoint_path = self.checkpoint_dir / filename
        state_dict: Dict[str, Any] = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
        }
        
        if self.scheduler is not None:
            state_dict["scheduler_state_dict"] = self.scheduler.state_dict()
            
        if self.weight_optimizer is not None:
            state_dict["weight_optimizer_state_dict"] = self.weight_optimizer.state_dict()
            
        if self.config.use_mixed_precision:
            state_dict["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(state_dict, checkpoint_path)
        self.logger.debug(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, filepath: str) -> None:
        """
        Restore the complete training state from a serialized checkpoint on disk.

        Parameters
        ----------
        filepath : str
            The absolute or relative path to the `.pt` checkpoint file.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"No checkpoint found at '{filepath}'")

        checkpoint = torch.load(filepath, map_location=self.device)
        self.current_epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        if self.weight_optimizer is not None and "weight_optimizer_state_dict" in checkpoint:
            self.weight_optimizer.load_state_dict(checkpoint["weight_optimizer_state_dict"])
            
        if self.config.use_mixed_precision and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.logger.info(f"Successfully resumed training from epoch {self.current_epoch}")

    def check_early_stopping(self, current_loss: float) -> bool:
        """
        Evaluate the early stopping criterion based on relative functional minimization.

        Parameters
        ----------
        current_loss : float
            The computed scalar total loss for the current epoch.

        Returns
        -------
        bool
            True if the training loop should be terminated, False otherwise.
        """
        # Handle the very first epoch
        if self.best_loss == float('inf'):
            self.best_loss = current_loss
            self.epochs_without_improvement = 0
            self.save_checkpoint("best_model.pt")
            return False

        # Calculate relative improvement requirement
        required_improvement = self.best_loss * self.config.early_stopping_delta
        
        if current_loss < (self.best_loss - required_improvement):
            self.best_loss = current_loss
            self.epochs_without_improvement = 0
            # Always checkpoint when achieving a new strict minimum
            self.save_checkpoint("best_model.pt")
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.config.early_stopping_patience:
            self.logger.info(
                f"Early stopping triggered. No improvement for "
                f"{self.config.early_stopping_patience} epochs."
            )
            return True
            
        return False

    def train(
        self,
        loss_closure: Callable[[], Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]],
        adaptive_weight_closure: Optional[Callable[[Dict[str, torch.Tensor], torch.Tensor], float]] = None,
        diagnostics_closure: Optional[Callable[[], Dict[str, float]]] = None,
        rar_closure: Optional[Callable] = None,
        epoch_closure: Optional[Callable] = None,
        rar_frequency: int = 5000,
    ) -> None:
        """
        Execute the comprehensive PINN optimization loop.

        This method abstracts the forward pass and loss computation via a closure injection,
        ensuring the trainer remains completely agnostic to the underlying PDE and spatial geometry.

        Parameters
        ----------
        loss_closure : Callable
            A strictly defined function taking no arguments and returning a 3-tuple:
            - Weighted total scalar loss `(torch.Tensor)`
            - Dictionary of individual, unweighted scalar loss components `(Dict[str, torch.Tensor])`
            - Shared layer weights tensor for GradNorm computation `(Optional[torch.Tensor])`
        adaptive_weight_closure : Optional[Callable], optional
            A function to trigger the dynamic loss weighting optimization step (e.g., GradNorm).
            Accepts the unweighted losses dictionary and the shared layer tensor. Returns the 
            meta-optimization loss value (float).
        diagnostics_closure : Optional[Callable], optional
            A function computing and returning physical validation metrics `(Dict[str, float])`.
        rar_closure : Optional[Callable], optional
            A function to trigger Residual-based Adaptive Refinement (RAR).
        epoch_closure : Optional[Callable], optional
            A function called exactly once at the beginning of each epoch (e.g., for resampling).
        rar_frequency : int, optional
            Frequency of RAR execution in epochs (default 5000).
        """
        self.logger.info("Commencing PINN Optimization Loop...")
        self.model.train()
        self.start_time = time.time()

        while self.current_epoch < self.config.max_epochs:
            self.current_epoch += 1
            
            if epoch_closure is not None:
                epoch_closure()
            
            # Step 1: Main Network Optimization Parameter Zeroing
            self.optimizer.zero_grad(set_to_none=True)

            # Step 2: Forward Pass & Loss Evaluation within Mixed Precision Context
            with autocast(enabled=self.config.use_mixed_precision):
                total_loss, losses_dict, shared_weights = loss_closure()

            # Step 3: Adaptive Loss Weighting — MUST run before backward() (C-2 FIX)
            # GradNorm internally calls torch.autograd.grad on the same computation graph
            # built during the forward pass (Step 2). loss.backward() at Step 4 frees
            # that graph. Any autograd.grad call after backward() raises RuntimeError.
            # GradNorm's retain_graph=True ensures the graph survives until Step 4.
            gradnorm_loss_val = 0.0
            if adaptive_weight_closure is not None and shared_weights is not None:
                gradnorm_loss_val = adaptive_weight_closure(losses_dict, shared_weights)

            # Step 4: Gradient Scaling and Backpropagation (graph freed here)
            self.scaler.scale(total_loss).backward()

            # Step 5: Explicit Gradient Unscaling and Clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.config.clip_grad_norm, 
                norm_type=2.0
            )

            # Step 6: Optimizer Step & Scaler Update
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss_val = float(total_loss.item())

            # Step 7: Scheduler Application
            if self.scheduler is not None:
                self.scheduler.step(total_loss_val)

            # Step 8: Logging and Telemetry Extraction
            if self.current_epoch % self.config.log_frequency == 0:
                self._log_telemetry(
                    total_loss_val, 
                    losses_dict, 
                    gradnorm_loss_val,
                    diagnostics_closure
                )

            # Step 9: Early Stopping Validation
            if self.check_early_stopping(total_loss_val):
                break

            # Step 10: Residual-Based Adaptive Refinement (RAR)
            if rar_closure is not None and self.current_epoch % rar_frequency == 0 and self.current_epoch < self.config.max_epochs - rar_frequency:
                rar_closure()
                self.model.train()  # Ensure model remains in train mode after RAR eval

        # Training complete, compile summary
        time_elapsed = time.time() - self.start_time
        early_stopped = self.epochs_without_improvement >= self.config.early_stopping_patience
        
        summary = (
            f"\nOptimization process terminated.\n"
            f"  Epochs completed : {self.current_epoch}\n"
            f"  Final loss       : {total_loss_val:.6e}\n"
            f"  Best loss        : {self.best_loss:.6e}\n"
            f"  Training time    : {time_elapsed:.2f}s\n"
            f"  Early stopping   : {'Yes' if early_stopped else 'No'}"
        )
        if early_stopped:
            summary += f"\n  Reason           : patience exhausted ({self.config.early_stopping_patience} epochs)"
            
        self.logger.info(summary)
        
        if self.tensorboard is not None:
            self.tensorboard.close()

    def _log_telemetry(
        self,
        total_loss_val: float,
        losses_dict: Dict[str, torch.Tensor],
        gradnorm_loss_val: float,
        diagnostics_closure: Optional[Callable[[], Dict[str, float]]]
    ) -> None:
        """
        Private helper to emit telemetry data to standard output and TensorBoard.
        """
        # Log learning rate
        current_lr = self.optimizer.param_groups[0]["lr"]
        if self.tensorboard is not None:
            self.tensorboard.add_scalar("Optimization/Learning_Rate", current_lr, self.current_epoch)
        
        # Log primary losses
        if self.tensorboard is not None:
            self.tensorboard.add_scalar("Loss/Total_Weighted", total_loss_val, self.current_epoch)
            if gradnorm_loss_val > 0.0:
                self.tensorboard.add_scalar("Loss/GradNorm_Meta", gradnorm_loss_val, self.current_epoch)

        log_msg = f"Epoch [{self.current_epoch:6d}/{self.config.max_epochs}] | Total Loss: {total_loss_val:.4e} | LR: {current_lr:.2e}"

        for key, value_tensor in losses_dict.items():
            val = float(value_tensor.item())
            if self.tensorboard is not None:
                self.tensorboard.add_scalar(f"Loss_Components/{key}", val, self.current_epoch)
            log_msg += f" | {key}: {val:.2e}"
            
        self.logger.info(log_msg)

        # Log physical diagnostics
        if diagnostics_closure is not None:
            metrics = diagnostics_closure()
            for metric_key, metric_val in metrics.items():
                self.tensorboard.add_scalar(metric_key, metric_val, self.current_epoch)


if __name__ == "__main__":
    # ---------------------------------------------------------
    # Standalone Execution and Verification Block
    # ---------------------------------------------------------
    print("Initializing PINN Trainer Module Verification...")
    
    device = torch.device("cpu")
    
    # Mocking a trivial neural network to serve as the PINN
    mock_model = nn.Sequential(
        nn.Linear(3, 16),
        nn.Tanh(),
        nn.Linear(16, 4)
    ).to(device)
    
    # Extract a mock "shared layer" to simulate GradNorm target
    mock_shared_weight = mock_model[0].weight

    # Initialize standard optimizer
    optimizer = torch.optim.Adam(mock_model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    
    # Initialize Configuration and Trainer
    config = PINNTrainerConfig(
        max_epochs=5, 
        log_frequency=1,
        use_mixed_precision=False # Disabled for CPU mock test clarity
    )
    
    trainer = PINNTrainer(
        model=mock_model,
        optimizer=optimizer,
        config=config,
        device=device,
        scheduler=scheduler
    )

    # Define mock closures to satisfy the modular interface without PDE code
    def mock_loss_closure() -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        # Generate dummy coordinates (batch=10, dim=3)
        dummy_coords = torch.randn((10, 3), device=device)
        dummy_preds = mock_model(dummy_coords)
        
        # Formulate mock loss constraints
        mass_loss = torch.mean(dummy_preds[:, 0]**2)
        momentum_loss = torch.mean(dummy_preds[:, 1:4]**2)
        
        losses_dict = {
            "mass": mass_loss,
            "momentum": momentum_loss
        }
        
        total_loss = mass_loss + momentum_loss
        return total_loss, losses_dict, mock_shared_weight

    def mock_adaptive_weight_closure(losses_dict: Dict[str, torch.Tensor], shared_weights: torch.Tensor) -> float:
        # Trivial mock of meta-optimization returning a dummy scalar
        return 0.05

    def mock_diagnostics_closure() -> Dict[str, float]:
        return {
            "physics/max_velocity": 0.45,
            "physics/mean_pressure": 0.01
        }

    # Execute test training loop
    trainer.train(
        loss_closure=mock_loss_closure,
        adaptive_weight_closure=mock_adaptive_weight_closure,
        diagnostics_closure=mock_diagnostics_closure
    )
    
    print("\nVerification Passed: PINN Trainer Loop executed successfully independent of external modules.")