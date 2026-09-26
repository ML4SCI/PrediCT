"""
PINN Loss Evaluation and Adaptive GradNorm Loss Balancing Module
===============================================================

This module provides independent loss evaluation functions and an adaptive
loss balancing framework using GradNorm for 3D steady incompressible
Navier-Stokes Physics-Informed Neural Networks (PINNs).

Mathematical Justification
--------------------------
The total loss functional for a Navier-Stokes PINN is expressed as a weighted sum
of individual objective terms representing PDE domain residuals and physical boundary
conditions:

    $$\mathcal{L}_{\text{total}}(\theta; \mathbf{w}) = \sum_{k \in \mathcal{K}} w_k \mathcal{L}_k(\theta)$$

where $\mathcal{K} = \{\text{mass}, \text{momentum}, \text{wall}, \text{inlet}, \text{outlet}\}$
indexes the individual loss objectives, $\theta$ represents the neural network parameters,
and $\mathbf{w} = \{w_k\}_{k \in \mathcal{K}}$ are time-dependent adaptive loss weights.

1.  **Mass Conservation Loss ($\mathcal{L}_{\text{mass}}$)**:
    Penalizes the divergence of the non-dimensional velocity field inside the vessel domain:
    $$\mathcal{L}_{\text{mass}}(\theta) = \frac{1}{N_{\text{int}}} \sum_{i=1}^{N_{\text{int}}} \left| e_c(\mathbf{x}_i) \right|^2$$

2.  **Momentum Conservation Loss ($\mathcal{L}_{\text{momentum}}$)**:
    Penalizes vector Navier-Stokes momentum residuals ($e_x, e_y, e_z$):
    $$\mathcal{L}_{\text{momentum}}(\theta) = \frac{1}{N_{\text{int}}} \sum_{i=1}^{N_{\text{int}}} \left( \left| e_x(\mathbf{x}_i) \right|^2 + \left| e_y(\mathbf{x}_i) \right|^2 + \left| e_z(\mathbf{x}_i) \right|^2 \right)$$

3.  **Boundary Losses ($\mathcal{L}_{\text{wall}}, \mathcal{L}_{\text{inlet}}, \mathcal{L}_{\text{outlet}}$)**:
    Enforce physical boundary conditions (no-slip wall, prescribed parabolic inlet, outlet traction/pressure):
    $$\mathcal{L}_{\text{wall}}(\theta) = \frac{1}{N_{\text{wall}}} \sum_{i=1}^{N_{\text{wall}}} \|\mathbf{u}(\mathbf{x}_{w,i})\|^2$$
    $$\mathcal{L}_{\text{inlet}}(\theta) = \frac{1}{N_{\text{in}}} \sum_{i=1}^{N_{\text{in}}} \|\mathbf{u}(\mathbf{x}_{\text{in},i}) - \mathbf{u}_{\text{in, target}}(\mathbf{x}_{\text{in},i})\|^2$$
    $$\mathcal{L}_{\text{outlet}}(\theta) = \frac{1}{N_{\text{out}}} \sum_{i=1}^{N_{\text{out}}} \|\mathbf{t}(\mathbf{x}_{\text{out},i}) - \mathbf{t}_{\text{out, target}}(\mathbf{x}_{\text{out},i})\|^2$$

4.  **GradNorm Adaptive Loss Balancing**:
    PINN training exhibits severe gradient pathology where boundary losses and ill-conditioned
    PDE higher-order derivatives dominate gradient magnitudes, causing training stagnation or 
    unstable convergence. GradNorm dynamically scales weights $w_k(t)$ at training step $t$
    by balancing the L2 norm of parameter gradients across task objectives:
    
    $$G_k(t) = \|\nabla_{W} (w_k(t) \mathcal{L}_k(t))\|_2$$
    
    where $W$ corresponds to the weights of the shared network architecture layer (typically the final hidden layer).
    
    The target gradient norm for loss task $k$ is formulated as:
    $$T_k(t) = \bar{G}(t) \times \left[ r_k(t) \right]^\alpha$$
    
    where:
    - $\bar{G}(t) = \frac{1}{|\mathcal{K}|} \sum_{k \in \mathcal{K}} G_k(t)$ is the mean gradient norm across all tasks.
    - $r_k(t) = \frac{\tilde{\mathcal{L}}_k(t)}{\frac{1}{|\mathcal{K}|} \sum_{j \in \mathcal{K}} \tilde{\mathcal{L}}_j(t)}$ is the relative inverse training rate.
    - $\tilde{\mathcal{L}}_k(t) = \frac{\mathcal{L}_k(t)}{\mathcal{L}_k(0)}$ is the normalized loss ratio.
    - $\alpha > 0$ is a hyperparameter regulating the strength of dynamic gradient balancing.

    The weights $w_k$ are updated by minimizing the GradNorm loss functional $\mathcal{L}_{\text{gradnorm}}$:
    $$\mathcal{L}_{\text{gradnorm}}(\mathbf{w}) = \sum_{k \in \mathcal{K}} \left| G_k(t) - T_k(t) \right|$$

Author: Computational Scientist
Target Framework: PyTorch 2.x, Python 3.11+
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn


def compute_mass_conservation_loss(res_continuity: torch.Tensor) -> torch.Tensor:
    """
    Compute scalar Mean Squared Error (MSE) loss for fluid continuity (mass conservation).

    Mathematical Formulation
    ------------------------
    $$\mathcal{L}_{\text{mass}} = \frac{1}{N} \sum_{i=1}^N \left| \nabla \cdot \mathbf{u}(\mathbf{x}_i) \right|^2$$

    Parameters
    ----------
    res_continuity : torch.Tensor
        Continuity equation residuals $e_c$ of shape $(N, 1)$.

    Returns
    -------
    torch.Tensor
        Scalar loss tensor.
    """
    return torch.mean(res_continuity**2)


def compute_momentum_conservation_loss(
    res_x: torch.Tensor, res_y: torch.Tensor, res_z: torch.Tensor
) -> torch.Tensor:
    """
    Compute scalar Mean Squared Error (MSE) loss for Navier-Stokes 3D momentum conservation.

    Mathematical Formulation
    ------------------------
    $$\mathcal{L}_{\text{momentum}} = \frac{1}{N} \sum_{i=1}^N \left( e_{x,i}^2 + e_{y,i}^2 + e_{z,i}^2 \right)$$

    Parameters
    ----------
    res_x : torch.Tensor
        Momentum residual in x-direction of shape $(N, 1)$.
    res_y : torch.Tensor
        Momentum residual in y-direction of shape $(N, 1)$.
    res_z : torch.Tensor
        Momentum residual in z-direction of shape $(N, 1)$.

    Returns
    -------
    torch.Tensor
        Scalar loss tensor.
    """
    return torch.mean(res_x**2 + res_y**2 + res_z**2)


class PINNLossEvaluator:
    """
    Evaluator for collecting individual, independent PINN loss components into a structured dictionary.
    """

    @staticmethod
    def evaluate_all_losses(
        res_continuity: torch.Tensor,
        res_x: torch.Tensor,
        res_y: torch.Tensor,
        res_z: torch.Tensor,
        loss_wall: torch.Tensor,
        loss_inlet: torch.Tensor,
        loss_outlet: torch.Tensor,
        loss_integral_mass: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Package individual loss components into an explicit dictionary.

        Parameters
        ----------
        res_continuity : torch.Tensor
            Continuity equation residual tensor $(N_{\text{int}}, 1)$.
        res_x : torch.Tensor
            X-momentum residual tensor $(N_{\text{int}}, 1)$.
        res_y : torch.Tensor
            Y-momentum residual tensor $(N_{\text{int}}, 1)$.
        res_z : torch.Tensor
            Z-momentum residual tensor $(N_{\text{int}}, 1)$.
        loss_wall : torch.Tensor
            Pre-computed wall boundary loss scalar.
        loss_inlet : torch.Tensor
            Pre-computed inlet boundary loss scalar.
        loss_outlet : torch.Tensor
            Pre-computed outlet boundary loss scalar.
        loss_integral_mass : torch.Tensor
            Pre-computed integral mass conservation loss scalar.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing unweighted scalar loss tensors mapped by key name.
        """
        loss_mass = compute_mass_conservation_loss(res_continuity)
        loss_momentum = compute_momentum_conservation_loss(res_x, res_y, res_z)

        return {
            "mass": loss_mass,
            "momentum": loss_momentum,
            "wall": loss_wall,
            "inlet": loss_inlet,
            "outlet": loss_outlet,
            "integral_mass": loss_integral_mass,
        }


class GradNormLossBalancer(nn.Module):
    """
    Adaptive loss weight manager implementing the GradNorm balancing algorithm.

    Parameters
    ----------
    task_keys : List[str]
        List of distinct loss identifiers corresponding to the loss dictionary.
    alpha : float, optional
        GradNorm hyperparameter controlling balancing strength (default: 0.12).
    device : Union[str, torch.device], optional
        Target computational device (default: 'cpu').
    """

    def __init__(
        self,
        task_keys: List[str],
        alpha: float = 0.12,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.task_keys = task_keys
        self.num_tasks = len(task_keys)
        self.alpha = alpha
        self.device = torch.device(device)

        # Initialize unnormalized loss weights w_k as learnable parameters initialized to 1.0
        self.weights = nn.Parameter(
            torch.ones(self.num_tasks, dtype=torch.float32, device=self.device)
        )

        # Buffer to store initial loss values L_k(0) for calculating relative rates
        self.register_buffer(
            "initial_losses",
            torch.zeros(self.num_tasks, dtype=torch.float32, device=self.device),
        )
        self.register_buffer("is_initialized", torch.tensor(False, dtype=torch.bool))

    def get_weight_dict(self) -> Dict[str, float]:
        """
        Retrieve normalized loss weights as a standard Python dictionary.

        Returns
        -------
        Dict[str, float]
            Mapping of task keys to current normalized loss weights $w_k$.
        """
        with torch.no_grad():
            normalized_w = (self.weights / torch.sum(self.weights)) * self.num_tasks
            return {
                key: float(normalized_w[i].item())
                for i, key in enumerate(self.task_keys)
            }

    def compute_total_loss(self, losses_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted total loss using current normalized loss weights.

        Mathematical Formulation
        ------------------------
        $$\mathcal{L}_{\text{total}} = \sum_{k=1}^K w_k \mathcal{L}_k$$
        where weights are scaled to satisfy $\sum_{k=1}^K w_k = K$.

        Parameters
        ----------
        losses_dict : Dict[str, torch.Tensor]
            Dictionary containing individual task loss scalar tensors.

        Returns
        -------
        torch.Tensor
            Weighted scalar total PINN loss for parameter backpropagation.
        """
        # Normalize weights so that sum(w_k) = num_tasks
        normalized_weights = (self.weights / torch.sum(self.weights)) * self.num_tasks

        total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        for idx, key in enumerate(self.task_keys):
            total_loss = total_loss + normalized_weights[idx] * losses_dict[key]

        return total_loss

    def update_loss_weights(
        self,
        losses_dict: Dict[str, torch.Tensor],
        shared_layer_weights: torch.Tensor,
        weight_optimizer: torch.optim.Optimizer,
    ) -> float:
        """
        Perform a single step of GradNorm optimization to update loss weights $w_k$.

        Parameters
        ----------
        losses_dict : Dict[str, torch.Tensor]
            Dictionary of unweighted scalar loss tensors for each task.
        shared_layer_weights : torch.Tensor
            Tensor of parameter weights from the shared backbone network layer $W$.
        weight_optimizer : torch.optim.Optimizer
            Dedicated optimizer (e.g., Adam) configured to optimize `self.weights`.

        Returns
        -------
        float
            Computed GradNorm loss magnitude $\mathcal{L}_{\text{gradnorm}}$.
        """
        # Step 1: Record initial losses L_k(0) on first execution
        if not self.is_initialized:
            with torch.no_grad():
                for idx, key in enumerate(self.task_keys):
                    self.initial_losses[idx] = losses_dict[key].detach()
                self.is_initialized.fill_(True)

        # Step 2: Compute L2 gradient norms G_k(t) w.r.t shared parameters W
        # Standardize weight normalization: w_k_norm = (w_k / sum(w)) * K
        norm_weights = (self.weights / torch.sum(self.weights)) * self.num_tasks
        
        grad_norms = []
        for idx, key in enumerate(self.task_keys):
            task_loss = norm_weights[idx] * losses_dict[key]
            # Compute gradient of weighted task loss w.r.t shared weights
            grads = torch.autograd.grad(
                outputs=task_loss,
                inputs=shared_layer_weights,
                retain_graph=True,
                create_graph=True,  # C-1 FIX: must be True so G_k = ‖∇_W(w_k·L_k)‖ has grad_fn
                                    # tracing back to self.weights. False makes grad_norm_loss
                                    # a constant tensor → .backward() raises RuntimeError.
            )[0]
            g_k = torch.norm(grads, p=2)
            grad_norms.append(g_k)

        grad_norms_tensor = torch.stack(grad_norms)

        # Step 3: Compute mean gradient norm G_bar(t)
        g_bar = torch.mean(grad_norms_tensor.detach())

        # Step 4: Compute relative inverse training rate r_k(t)
        with torch.no_grad():
            current_losses = torch.stack([losses_dict[key].detach() for key in self.task_keys])
            # Normalized loss ratio: L_hat_k(t) = L_k(t) / L_k(0)
            loss_ratios = current_losses / (self.initial_losses + 1e-8)
            # Relative inverse training rate: r_k(t) = L_hat_k(t) / mean(L_hat)
            r_k = loss_ratios / (torch.mean(loss_ratios) + 1e-8)
            # Target gradient norms: T_k(t) = G_bar(t) * [r_k(t)]^\alpha
            target_grad_norms = g_bar * (r_k ** self.alpha)

        # Step 5: Compute GradNorm loss L_gradnorm = sum_k | G_k(t) - T_k(t) |
        grad_norm_loss = torch.sum(torch.abs(grad_norms_tensor - target_grad_norms))

        # Step 6: Step weight optimizer and re-normalize weights
        weight_optimizer.zero_grad()
        grad_norm_loss.backward()
        weight_optimizer.step()

        # Enforce positivity constraint on learnable weights
        with torch.no_grad():
            self.weights.clamp_(min=1e-3)
            # Re-normalize weights so sum(w_k) = num_tasks
            self.weights.copy_((self.weights / torch.sum(self.weights)) * self.num_tasks)

        return float(grad_norm_loss.item())


if __name__ == "__main__":
    # ---------------------------------------------------------
    # Standalone Execution and Verification Block
    # ---------------------------------------------------------
    print("Initializing PINN Loss Evaluator & GradNorm Balancer...")

    device = torch.device("cpu")
    task_keys = ["mass", "momentum", "wall", "inlet", "outlet", "integral_mass"]

    # Instantiating synthetic backbone layer weight (simulating last layer of PINN)
    mock_shared_layer = nn.Linear(64, 4, device=device)

    # Instantiate GradNorm Balancer & Dedicated Optimizer for weights
    gradnorm_balancer = GradNormLossBalancer(
        task_keys=task_keys, alpha=0.12, device=device
    )
    weight_optimizer = torch.optim.Adam([gradnorm_balancer.weights], lr=0.02)

    # Generate mock residual outputs
    N_int = 500
    res_c = torch.randn((N_int, 1), device=device, requires_grad=True) * 0.05
    res_x = torch.randn((N_int, 1), device=device, requires_grad=True) * 0.1
    res_y = torch.randn((N_int, 1), device=device, requires_grad=True) * 0.1
    res_z = torch.randn((N_int, 1), device=device, requires_grad=True) * 0.1

    # Mock boundary losses
    loss_w = torch.tensor(0.5, device=device, requires_grad=True)
    loss_in = torch.tensor(0.8, device=device, requires_grad=True)
    loss_out = torch.tensor(0.3, device=device, requires_grad=True)
    loss_int_mass = torch.tensor(0.1, device=device, requires_grad=True)

    # Evaluate individual losses
    losses_dict = PINNLossEvaluator.evaluate_all_losses(
        res_continuity=res_c,
        res_x=res_x,
        res_y=res_y,
        res_z=res_z,
        loss_wall=loss_w,
        loss_inlet=loss_in,
        loss_outlet=loss_out,
        loss_integral_mass=loss_int_mass,
    )

    print("\n--- Evaluated Task Loss Dictionary ---")
    for key, val in losses_dict.items():
        print(f"Loss [{key:8s}]: {val.item():.6f}")

    # Dummy forward pass to establish gradient connection to mock shared parameters
    dummy_input = torch.randn((N_int, 64), device=device)
    dummy_out = mock_shared_layer(dummy_input)
    # Add dependency on shared parameters to loss dictionary
    losses_dict["mass"] = losses_dict["mass"] + 0.01 * torch.sum(dummy_out**2)

    # Compute Initial Weighted Total Loss
    total_pinn_loss = gradnorm_balancer.compute_total_loss(losses_dict)
    print(f"\nInitial Total Weighted PINN Loss: {total_pinn_loss.item():.6f}")

    # Execute single step of GradNorm weight optimization
    gn_loss_val = gradnorm_balancer.update_loss_weights(
        losses_dict=losses_dict,
        shared_layer_weights=mock_shared_layer.weight,
        weight_optimizer=weight_optimizer,
    )

    print(f"GradNorm Loss Value: {gn_loss_val:.6f}")
    
    # Print Updated Task Weights
    updated_weights = gradnorm_balancer.get_weight_dict()
    print("\n--- Dynamically Updated Task Loss Weights ---")
    for key, weight_val in updated_weights.items():
        print(f"Weight [{key:8s}]: {weight_val:.4f}")

    # Verification Checks
    assert len(losses_dict) == 6, "Loss dictionary must contain 6 task losses."
    assert total_pinn_loss.requires_grad, "Total loss tensor must retain gradient graph."
    assert abs(sum(updated_weights.values()) - 6.0) < 1e-4, "Loss weights must sum to K=6."

    print("\nVerification Passed: PINN loss evaluation & GradNorm loss balancing complete.")