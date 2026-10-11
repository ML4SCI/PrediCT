"""
Conv building blocks used throughout Reronet.

Shaping convention: (B, C, D, H, W). InstanceNorm3d + LeakyReLU(0.01) —
matches nnU-Net style, safe at batch_size=1.
"""

import torch
import torch.nn as nn


class ConvBlockIN(nn.Module):
    """
    Conv3d -> InstanceNorm3d -> LeakyReLU(0.01).

    Used as the basic residual unit inside every encoder/decoder block.

    Shape
    -----
    Input  : (B, in_channels,  D, H, W)
    Output : (B, out_channels, D, H, W)   (same spatial if stride=1)
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int = 3,
        stride:       int = 1,
        padding:      int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size   = kernel_size,
            stride        = stride,
            padding       = padding,
            bias          = False,
        )
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.act  = nn.LeakyReLU(0.01, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class Down(nn.Module):
    """
    Strided-conv downsampler. Halves the spatial dims on (D, H, W).

    Shape
    -----
    Input  : (B, in_channels,  D, H, W)
    Output : (B, out_channels, D/2, H/2, W/2)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = ConvBlockIN(
            in_channels,
            out_channels,
            kernel_size = 3,
            stride      = 2,
            padding     = 1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """
    ConvTranspose3d upsampler. Doubles the spatial dims on (D, H, W).

    Shape
    -----
    Input  : (B, in_channels,  D, H, W)
    Output : (B, out_channels, 2D, 2H, 2W)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size = 2,
            stride      = 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)
