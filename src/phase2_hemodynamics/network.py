"""
Physics-Informed Neural Network (PINN) Architecture Module
==========================================================

This module defines the neural network architectures used to approximate
the solution to the steady incompressible Navier-Stokes equations for 
coronary hemodynamics.

Mathematical Justification
--------------------------
1.  **Fourier Feature Encoding**:
    Standard coordinate-based multi-layer perceptrons (MLPs) suffer from 
    spectral bias, wherein they struggle to learn high-frequency mappings 
    characteristic of steep velocity gradients in boundary layers or 
    stenotic regions. We map spatial coordinates $\mathbf{x} \in \mathbb{R}^3$ 
    into a higher-dimensional Fourier feature space:
    $$ \gamma(\mathbf{x}) = [\cos(2\pi \mathbf{B}\mathbf{x}), \sin(2\pi \mathbf{B}\mathbf{x})]^T $$
    where $\mathbf{B} \in \mathbb{R}^{m \times 3}$ is sampled from a Gaussian 
    distribution $\mathcal{N}(0, \sigma^2)$.

2.  **Activation Function ($tanh$)**:
    Physics-Informed Neural Networks require the computation of high-order 
    derivatives ($\nabla \mathbf{u}, \nabla^2 \mathbf{u}$) via automatic 
    differentiation. The $\tanh$ function is infinitely continuously 
    differentiable ($C^\infty$). In contrast to ReLU (which has zero 
    second derivatives), $\tanh$ ensures non-vanishing gradients for 
    the viscous Laplacian terms.

3.  **Xavier Initialization**:
    To prevent vanishing or exploding gradients during backpropagation, 
    we apply Xavier (Glorot) normal initialization. For symmetric activation 
    functions like $\tanh$, scaling the weights by $\sqrt{2 / (n_{in} + n_{out})}$ 
    maintains the variance of activations and gradients across deep layers.

4.  **Residual Connections**:
    To facilitate deep network training and improve gradient flow from the 
    loss function back to the Fourier feature space, we employ residual 
    skip connections defining layer mappings as:
    $$ \mathbf{h}^{(l+1)} = \tanh\left(\mathbf{W}_2^{(l)} \tanh\left(\mathbf{W}_1^{(l)} \mathbf{h}^{(l)} + \mathbf{b}_1^{(l)}\right) + \mathbf{b}_2^{(l)}\right) + \mathbf{h}^{(l)} $$

Author: Computational Scientist
Target Framework: PyTorch 2.x, Python 3.11+
"""

import math
from typing import Tuple, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    # Imported only for type-checking; avoids any risk of import-time side-effects.
    from config import ArchitectureConfig


def _build_activation(name: str) -> nn.Module:
    """
    Factory that maps an activation name string to an instantiated nn.Module.

    H-4 FIX: ArchitectureConfig.activation is a string ('tanh', 'silu', etc.).
    This factory makes that string usable everywhere in the network without
    hard-coding ``nn.Tanh()``.

    Parameters
    ----------
    name : str
        Activation function identifier. Supported: 'tanh', 'silu', 'gelu',
        'relu', 'mish'.

    Returns
    -------
    nn.Module
        An instantiated, uninitialized activation module.
    """
    _registry = {
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "mish": nn.Mish,
    }
    key = name.lower()
    if key not in _registry:
        raise ValueError(
            f"Unsupported activation '{name}'. "
            f"Choose from: {sorted(_registry)}"
        )
    return _registry[key]()


class FourierFeatureEncoder(nn.Module):
    """
    Fourier feature encoding layer to project low-dimensional spatial 
    coordinates into a higher-dimensional frequency space.
    
    Parameters
    ----------
    spatial_dim : int
        The dimensionality of the input physical space (e.g., 3 for 3D).
    num_frequencies : int
        The number of frequency components $m$ to sample. The output 
        dimension will be $2m$.
    sigma : float
        The standard deviation $\sigma$ of the Gaussian distribution used 
        to sample the frequency matrix $\mathbf{B}$. Controls the maximum 
        resolvable frequency.
    """

    def __init__(self, spatial_dim: int, num_frequencies: int, sigma: float = 1.0) -> None:
        super().__init__()
        self.spatial_dim = spatial_dim
        self.num_frequencies = num_frequencies
        
        # Initialize the frequency matrix B from N(0, sigma^2).
        # B is not a learnable parameter, so we register it as a buffer.
        b_matrix = torch.normal(mean=0.0, std=sigma, size=(num_frequencies, spatial_dim))
        self.register_buffer("b_matrix", b_matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying the Fourier feature mapping.

        Parameters
        ----------
        x : torch.Tensor
            Input coordinate tensor of shape $(N, \text{spatial\_dim})$.

        Returns
        -------
        torch.Tensor
            Encoded feature tensor of shape $(N, 2 \times \text{num\_frequencies})$.
        """
        # Compute projection: 2 * pi * x * B^T
        # Shape: (N, spatial_dim) @ (spatial_dim, num_frequencies) -> (N, num_frequencies)
        projection = 2.0 * math.pi * torch.matmul(x, self.b_matrix.T)
        
        # Concatenate cosine and sine projections
        # Shape: (N, 2 * num_frequencies)
        return torch.cat([torch.cos(projection), torch.sin(projection)], dim=-1)


class ResidualBlock(nn.Module):
    """
    A fully connected residual block with two linear layers and a configurable activation.
    
    Parameters
    ----------
    hidden_dim : int
        The dimensionality of the hidden features.
    activation : str, optional
        Activation function name (default: 'tanh'). H-4 FIX: was hard-coded to
        nn.Tanh(); now configurable so ArchitectureConfig.activation is honoured.
    """

    def __init__(self, hidden_dim: int, activation: str = "tanh") -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = _build_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the residual block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape $(N, \text{hidden\_dim})$.

        Returns
        -------
        torch.Tensor
            Output tensor of shape $(N, \text{hidden\_dim})$ with skip connection added.
        """
        identity = x
        out = self.activation(self.linear1(x))
        out = self.activation(self.linear2(out))
        return out + identity


class HemodynamicsPINN(nn.Module):
    """
    Physics-Informed Neural Network architecture for 3D steady incompressible 
    Navier-Stokes equations.

    Predicts the velocity field $\mathbf{u} = [u, v, w]^T$ and scalar pressure field $p$
    from spatial coordinates $\mathbf{x} = [x, y, z]^T$.

    Parameters
    ----------
    spatial_dim : int
        Dimensionality of the physical space (default is 3).
    num_fourier_features : int
        Number of Fourier frequency vectors to generate (output size from encoder 
        is $2 \times \text{num\_fourier\_features}$).
    fourier_sigma : float
        Variance of the sampled Fourier frequency mapping.
    hidden_dim : int
        Width of the Multi-Layer Perceptron hidden layers.
    num_res_blocks : int
        Number of consecutive residual blocks in the network.
    """

    def __init__(
        self,
        spatial_dim: int = 3,
        num_fourier_features: int = 128,
        fourier_sigma: float = 1.0,
        hidden_dim: int = 256,
        num_res_blocks: int = 4,
        activation: str = "tanh",
    ) -> None:
        """H-4 FIX: ``activation`` parameter added so ArchitectureConfig.activation
        is propagated through the entire network instead of being ignored.
        The existing constructor defaults are unchanged for backward compatibility."""
        super().__init__()
        
        self.encoder = FourierFeatureEncoder(
            spatial_dim=spatial_dim,
            num_frequencies=num_fourier_features,
            sigma=fourier_sigma
        )
        
        encoder_out_dim = 2 * num_fourier_features
        
        # Initial projection from Fourier feature space to hidden dimension
        self.input_layer = nn.Sequential(
            nn.Linear(encoder_out_dim, hidden_dim),
            _build_activation(activation)
        )
        
        # Deep Residual network backbone
        res_blocks = [ResidualBlock(hidden_dim, activation=activation) for _ in range(num_res_blocks)]
        self.backbone = nn.Sequential(*res_blocks)
        
        # Final output layer mapping to state variables: u, v, w, p
        # Output dimension is 4 (velocity vector + scalar pressure)
        self.output_layer = nn.Linear(hidden_dim, 4)
        
        # Apply strict mathematically justified weight initialization
        self._initialize_weights()

    @classmethod
    def from_config(cls, cfg: "ArchitectureConfig") -> "HemodynamicsPINN":
        """
        Construct a HemodynamicsPINN from an ArchitectureConfig instance.

        H-4 FIX: Previously, ``PINNConfig``/``ArchitectureConfig`` was defined in
        ``config.py`` but never consumed; all network hyperparameters were silently
        set by hardcoded defaults that contradicted the config. This classmethod
        is the canonical construction path for production use.

        Parameters
        ----------
        cfg : ArchitectureConfig
            Frozen architecture configuration dataclass.

        Returns
        -------
        HemodynamicsPINN
            A fully configured PINN instance matching the provided config.

        Example
        -------
        ::

            from config import PINNConfig
            from network import HemodynamicsPINN

            config = PINNConfig()
            model = HemodynamicsPINN.from_config(config.architecture)
        """
        return cls(
            spatial_dim=cfg.in_features,
            num_fourier_features=cfg.fourier_num_frequencies,
            fourier_sigma=cfg.fourier_scale,
            hidden_dim=cfg.hidden_dim,
            num_res_blocks=cfg.num_layers,
            activation=cfg.activation,
        )

    def _initialize_weights(self) -> None:
        """
        Initializes network weights using Xavier (Glorot) normal distribution.
        Biases are initialized to strictly zero.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the forward pass of the network, predicting state variables
        from spatial coordinates.

        Parameters
        ----------
        x : torch.Tensor
            Spatial coordinate tensor of shape $(N, 3)$, representing 
            $[x, y, z]$.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing four distinct tensors, each of shape $(N, 1)$:
            - u: x-component of velocity.
            - v: y-component of velocity.
            - w: z-component of velocity.
            - p: relative kinematic pressure.
        """
        encoded_features = self.encoder(x)
        features = self.input_layer(encoded_features)
        features = self.backbone(features)
        
        # Output shape is (N, 4)
        outputs = self.output_layer(features)
        
        # Slicing the output strictly into independent physical variables
        # This formatting is heavily optimized for downstream PyTorch autograd 
        # operations computing individual PDE residuals.
        u = outputs[:, 0:1]
        v = outputs[:, 1:2]
        w = outputs[:, 2:3]
        p = outputs[:, 3:4]
        
        return u, v, w, p

if __name__ == "__main__":
    # Standalone execution check for the module.
    # Instantiates the network and executes a mock forward pass.
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = HemodynamicsPINN(
        spatial_dim=3,
        num_fourier_features=128,
        fourier_sigma=1.5,
        hidden_dim=256,
        num_res_blocks=5
    ).to(device)
    
    # Generate mock 3D coordinates (e.g., from Latin Hypercube Sampling module)
    mock_batch_size = 1000
    mock_x = torch.rand((mock_batch_size, 3), device=device, requires_grad=True)
    
    u_pred, v_pred, w_pred, p_pred = model(mock_x)
    
    assert u_pred.shape == (mock_batch_size, 1), "Output shape mismatch for u"
    assert v_pred.shape == (mock_batch_size, 1), "Output shape mismatch for v"
    assert w_pred.shape == (mock_batch_size, 1), "Output shape mismatch for w"
    assert p_pred.shape == (mock_batch_size, 1), "Output shape mismatch for p"
    
    print("HemodynamicsPINN module initialized and forward pass verified successfully.")