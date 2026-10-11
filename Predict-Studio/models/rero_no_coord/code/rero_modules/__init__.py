"""
riu_modules — reusable building blocks for RIUnet.

These are blueprint-level (low-level) helpers. Not yet fully implemented —
each file contains class signatures, docstrings, and shape annotations only.

Modules:
    conv_blocks            - ConvBlockIN, Down, Up (residual conv stem/blocks)
    deform_attention       - DeformableAttention3d (Deformable DETR style, 3D)
    fno3d                  - SpectralConv3d, FNOBlock3d (full-spectrum FNO)
    deep_supervision       - DeepSupervisionHead (1x1x1 conv logits at scale)
"""

from .conv_blocks import ConvBlockIN, Down, Up
from .deform_attention import DeformableAttention3d
from .fno3d import SpectralConv3d, FNOBlock3d
from .deep_supervision import DeepSupervisionHead

__all__ = [
    "ConvBlockIN",
    "Down",
    "Up",
    "DeformableAttention3d",
    "FNOBlock3d",
    "SpectralConv3d",
    "DeepSupervisionHead",
]
