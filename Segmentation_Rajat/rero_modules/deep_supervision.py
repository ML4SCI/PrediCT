"""
Deep supervision head.

A 1x1x1 conv that produces class logits at the resolution of the
feature map it's attached to. Used at:
    - bottleneck (lowest spatial)
    - each decoder level (4 total in Reronet)
"""

import torch
import torch.nn as nn


class DeepSupervisionHead(nn.Module):
    """
    1x1x1 conv producing `num_classes` logits.

    Shape
    -----
    Input  : (B, in_channels, D, H, W)
    Output : (B, num_classes, D, H, W)
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
