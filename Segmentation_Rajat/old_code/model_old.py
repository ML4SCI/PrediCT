"""
RIUnet — Recurrent-Inspired U-Net for Coronary Artery Calcium segmentation.

Architecture:
    Input -> Stem
        -> Encoder x4   (CNN + DeformableAttention per level, stride-2 downsample)
        -> Bottleneck    (FNO3d || CNN, summed)
        -> Decoder x4   (ConvTranspose up + DA-fuse(skip) + CNN, then DS head)
        -> Final 1x1x1 head  -> 2-channel logits (softmax: bg + calcium)

Channel schedule:  32 -> 64 -> 128 -> 256 -> 512 (bottleneck)
Spatial divisors:  every input dim is padded to a multiple of 16 inside forward().

This file is currently BLUEPRINT-LEVEL. Each helper used here lives in
Segmentation_Rajat/riu_modules/ and is also blueprint / low-level only.

Shape trace (example: input (B, 6, 96, 128, 96), padded to (96, 128, 96)):
    Stem         : (B,  32,  96, 128,  96)
    Enc skip0    : (B,  64,  96, 128,  96)   DA-refined
    Down0        : (B,  64,  48,  64,  48)
    Enc skip1    : (B, 128,  48,  64,  48)
    Down1        : (B, 128,  24,  32,  24)
    Enc skip2    : (B, 256,  24,  32,  24)
    Down2        : (B, 256,  12,  16,  12)
    Enc skip3    : (B, 512,  12,  16,  12)
    Down3        : (B, 512,   6,   8,   6)   <-- bottleneck
    FNO modes    : (6, 8, 6)  full spectrum, no truncation
    Up3          : (B, 256,  12,  16,  12)
    Up2          : (B, 128,  24,  32,  24)
    Up1          : (B,  64,  48,  64,  48)
    Up0          : (B,  32,  96, 128,  96)
    Final logits : (B,   2,  96, 128,  96)

Deep supervision heads (training mode only):
    bottleneck : (B, 2,  6,  8,  6)
    dec L3     : (B, 2, 12, 16, 12)
    dec L2     : (B, 2, 24, 32, 24)
    dec L1     : (B, 2, 48, 64, 48)
    final      : (B, 2, 96, 128, 96)
    Total heads: 5
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rero_modules.conv_blocks import ConvBlockIN, Down, Up
from rero_modules.deform_attention import DeformableAttention3d
from rero_modules.fno3d import FNOBlock3d
from rero_modules.deep_supervision import DeepSupervisionHead

import config as cfg


# ══════════════════════════════════════════════════════════════════
#  ENCODER BLOCK
# ══════════════════════════════════════════════════════════════════

class EncoderBlock(nn.Module):
    """
    One encoder level:  CNN x2 -> Deformable Attention -> (returns skip)

    The downsampler (Down) lives outside this block in the parent so
    that skips can be saved *before* downsampling.

    Shape
    -----
    Input  : (B, in_channels,  D, H, W)
    Skip   : (B, out_channels, D, H, W)    same spatial, full feature
    """

    def __init__(self, in_channels: int, out_channels: int,
                da_heads: int = 8, da_points: int = 4,
                downsample: bool = False,
                da_save_dir: Optional[str] = None):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBlockIN(in_channels, out_channels),
            ConvBlockIN(out_channels, out_channels),
        )
        # L0 (largest spatial) needs aggressive reduction; L3 (smallest) can keep full.
        if out_channels <= 64:    # L0
            spatial_reduce, anchor_stride = 4, 4
        elif out_channels <= 128: # L1
            spatial_reduce, anchor_stride = 2, 2
        else:                    # L2, L3
            spatial_reduce, anchor_stride = 1, 1
        self.da  = DeformableAttention3d(
            out_channels,
            num_heads=da_heads,
            num_points=da_points,
            spatial_reduce=spatial_reduce,
            anchor_stride=anchor_stride,
        )
        # Down lives inside the block when requested, so it moves with .to(device).
        self.down = Down(out_channels, out_channels) if downsample else None

    def forward(self, x: torch.Tensor, da_save_dir: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.cnn(x)
        # DeformableAttention3d returns a (tensor, info_dict) tuple when
        # is supplied). The model itself only needs the tensor; the info dict
        # exists purely to drive the per-step .npy attention dumps as a side
        # effect inside the module. Unpack and discard.
        out = self.da(x, save_dir=da_save_dir)
        if isinstance(out, tuple):
            out = out[0]
        x = out

        if self.down is not None:
            return x, self.down(x)
        return x


# ══════════════════════════════════════════════════════════════════
#  DECODER BLOCK
# ══════════════════════════════════════════════════════════════════

class DAFusion(nn.Module):
    """
    Channel-align then apply DA to combined upsampled + skip features.

    Shape
    -----
    Input  : (B, in_channels, D, H, W)    where in_channels = up_ch + skip_ch
    Output : (B, out_channels, D, H, W)
    """

    def __init__(self, in_channels: int, out_channels: int,
                da_heads: int = 8, da_points: int = 4):
        super().__init__()
        self.align = ConvBlockIN(in_channels, out_channels,
                                kernel_size=1, padding=0)
        if out_channels <= 64:    # decoder L0 (largest)
            spatial_reduce, anchor_stride = 4, 4
        elif out_channels <= 128: # decoder L1
            spatial_reduce, anchor_stride = 2, 2
        else:                    # L2, L3
            spatial_reduce, anchor_stride = 1, 1
        self.da    = DeformableAttention3d(
            out_channels,
            num_heads=da_heads,
            num_points=da_points,
            spatial_reduce=spatial_reduce,
            anchor_stride=anchor_stride,
        )

    def forward(self, x: torch.Tensor, da_save_dir: Optional[str] = None) -> torch.Tensor:
        x = self.align(x)
        # See EncoderBlock.forward for the (tensor, info) tuple unpack.
        out = self.da(x, save_dir=da_save_dir)
        if isinstance(out, tuple):
            out = out[0]
        return out


class DecoderBlock(nn.Module):
    """
    Decoder step:  Up -> DA-fuse(skip) -> CNN x2 -> (optional DS head).

    Shape
    -----
    Input x    : (B, in_channels,  D, H, W)        bottleneck or prev-dec level
    Skip       : (B, skip_channels, 2D, 2H, 2W)   encoder level n
    Output     : (B, out_channels, 2D, 2H, 2W)
    DS logits  : (B, num_classes, 2D, 2H, 2W)      when return_ds_logits=True
    """

    def __init__(
        self,
        in_channels:     int,
        skip_channels:   int,
        out_channels:    int,
        num_classes:     int = 2,
        return_ds_logits: bool = False,
        da_heads: int = 8, da_points: int = 4,
        da_save_dir: Optional[str] = None,
    ):
        super().__init__()
        self.up = Up(in_channels, in_channels // 2)
        self.da_fuse = DAFusion(
            in_channels // 2 + skip_channels,
            in_channels // 2,
            da_heads=da_heads, da_points=da_points,
            da_save_dir=da_save_dir,
        )
        self.conv1 = ConvBlockIN(in_channels // 2, out_channels)
        self.conv2 = ConvBlockIN(out_channels, out_channels)
        self.ds_head = (
            DeepSupervisionHead(out_channels, num_classes)
            if return_ds_logits else None
        )

    def forward(
        self,
        x:     torch.Tensor,
        skip:  torch.Tensor,
        da_save_dir: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.da_fuse(x, da_save_dir=da_save_dir)
        x = self.conv2(self.conv1(x))
        if self.ds_head is not None:
            return x, self.ds_head(x)
        return x, None


# ══════════════════════════════════════════════════════════════════
#  BOTTLENECK  (parallel FNO || CNN)
# ══════════════════════════════════════════════════════════════════

class Bottleneck(nn.Module):
    """
    Parallel:  FNO3d || CNN   -> sum -> IN -> GELU -> (deep-sup head)

    The FNO captures global periodic / spectral patterns; the parallel CNN
    captures local detail. Summing them gives both inductive biases.

    Shape
    -----
    Input  : (B, channels, D, H, W)        ch=512, DxHxW = spatial/16
    Output : (B, channels, D, H, W)
    DS log : (B, num_classes, D, H, W)
    """

    def __init__(
        self,
        channels:      int,
        fno_modes:      Tuple[int, int, int],
        return_ds_logits: bool = False,
        num_classes:    int = 2,
        fno_save_dir:   Optional[str] = None,
    ):
        super().__init__()

        if fno_save_dir is not None:
            os.makedirs(fno_save_dir, exist_ok=True)

        self.fno  = FNOBlock3d(
            channels, *fno_modes,
            save_feature_maps=(fno_save_dir is not None),
            save_dir=fno_save_dir
        )
        # Override the FNO's default save_dir so it lands in our per-run folder.

        self.cnn  = nn.Sequential(
            ConvBlockIN(channels, channels),
            ConvBlockIN(channels, channels),
        )
        self.merge_norm = nn.InstanceNorm3d(channels, affine=True)
        self.merge_act  = nn.GELU()

        self.ds_head = DeepSupervisionHead(channels, num_classes) if return_ds_logits else None

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.merge_norm(self.fno(x) + self.cnn(x))
        out = self.merge_act(out)
        if self.ds_head is None:
            return out, None
        return out, self.ds_head(out)


# ══════════════════════════════════════════════════════════════════
#  RIUnet (full model)
# ══════════════════════════════════════════════════════════════════

class RIUnet(nn.Module):
    """
    Pipeline:
        Stem -> Encoder x4 -> Bottleneck (FNO || CNN) -> Decoder x4 -> Final head

    Padding
    -------
    Spatial dims are padded *inside* forward() to the smallest multiple
    of 16 that is >= the input spatial size. After inference the padded
    region is cropped back to the original size.

    Channel schedule
    ----------------
        base_channels * (1, 2, 4, 8, 16)   # 32->64->128->256->512

    Forward outputs
    ---------------
        eval mode  : (B, num_classes, D, H, W)         single logits tensor
        train mode : List[Tensor] of length 5           [final, l1, l2, l3, l4]
                                                        one logits per scale
    """

    def __init__(
        self,
        in_channels:       Optional[int] = None,
        base_channels:     Optional[int] = None,
        num_classes:       Optional[int] = None,
        deep_supervision:  Optional[bool] = None,
        da_heads:          Optional[int] = None,
        da_points:         Optional[int] = None,
        fno_modes:         Optional[Tuple[int, int, int]] = None,
        save_dir:          Optional[str] = None,
    ):
        """
        All defaults are read from config.model_config.

        in_channels:
            None -> uses cfg.model_config["IN_CHANNELS"]; if that's also None,
            the caller should auto-detect from the first batch.
        save_dir:
            If provided, every DeformableAttention3d (8 levels: 4 enc + 4 dec)
            and the FNO SpectralConv3d will write .npy dumps into this folder
            during train-mode forwards. Caller is expected to reset counters
            at each epoch boundary.
        """
        super().__init__()

        # ── Resolve defaults from config ──────────────────────
        in_channels      = in_channels     if in_channels     is not None else cfg.model_config.get("IN_CHANNELS", 6)
        base_channels    = base_channels   if base_channels   is not None else cfg.model_config["BASE_CHANNELS"]
        num_classes      = num_classes     if num_classes     is not None else cfg.model_config["NUM_CLASSES"]
        deep_supervision = deep_supervision if deep_supervision is not None else cfg.model_config["USE_DEEP_SUPERVISION"]
        da_heads         = da_heads        if da_heads        is not None else cfg.model_config["DA_NUM_HEADS"]
        da_points        = da_points       if da_points       is not None else cfg.model_config["DA_NUM_POINTS"]
        fno_modes        = fno_modes       if fno_modes       is not None else tuple(cfg.model_config["FNO_MODES"])

        self.ch = [base_channels * (2 ** i) for i in range(5)]   # 32, 64, 128, 256, 512
        ch = self.ch
        c_in, c0, c1, c2, c3, c4 = in_channels, *ch

        # FNO modes:
        #   tuple-as-given is used as the upper bound; the runtime size is
        #   also clamped in SpectralConv3d.forward to whatever the input has,
        #   so this is safe even when the bottleneck is smaller.
        if fno_modes is None:
            fno_modes = (8, 8, 8)

        # ── Save-dir plumbing for attention / FNO dumps ─────
        # When save_dir is provided, each module writes its own .npy files
        # directly into per-level sub-folders of save_dir. We do NOT create
        # the per-level sub-folders here — each child module does that in
        # its own __init__ via os.makedirs(...).
        self.save_dir = save_dir

        # ── Stem ─────────────────────────────────────────────
        self.stem = ConvBlockIN(c_in, c0)

        # ── Encoder (4 levels) ───────────────────────────────
        self.enc_blocks = nn.ModuleList([
            EncoderBlock(ch[i],     ch[i + 1],
                        da_heads=da_heads, da_points=da_points,
                        downsample=True,
                        da_save_dir=(os.path.join(save_dir, f"attn_maps/enc_L{i}")) if save_dir else None)
            for i in range(4)
        ])

        # ── Bottleneck ───────────────────────────────────────
        # DS flag from config.model_config["USE_DEEP_SUPERVISION"].
        self.bottleneck = Bottleneck(
            channels         = c4,
            fno_modes        = fno_modes,
            return_ds_logits = deep_supervision,
            num_classes      = num_classes,
            fno_save_dir     = (os.path.join(save_dir, "fno_maps")) if save_dir else None,
        )

        # ── Decoder (4 levels) ───────────────────────────────
        # Note: skip channels = ch[4-i] (NOT ch[4-i-1]) because
        # EncoderBlock outputs ch[i+1] which matches the bottleneck/depth
        # at the mirror level. E.g. decoder i=0 mirrors encoder i=3,
        # both output ch[4]=512.
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(
                in_channels     = ch[4 - i],
                skip_channels   = ch[4 - i],
                out_channels    = ch[4 - i - 1] if i < 3 else ch[0],
                num_classes     = num_classes,
                return_ds_logits = (deep_supervision and i != 3),
                da_heads=da_heads, da_points=da_points,
                da_save_dir=(os.path.join(save_dir, f"attn_maps/dec_L{i}")) if save_dir else None,
            )
            for i in range(4)
        ])

        # ── Final head at full resolution ────────────────────
        self.final_head = nn.Conv3d(ch[0], num_classes, kernel_size=1)

        self.deep_supervision = deep_supervision

    # ----- shape utilities -----

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, multiple: int = 16) -> Tuple[torch.Tensor, Tuple[slice, ...]]:
        """
        Pad spatial dims (last 3) up to a multiple of `multiple` using replicate.

        Returns: (padded_tensor, original_crop_slice)
        """
        D, H, W = x.shape[-3:]
        pd = (multiple - D % multiple) % multiple
        ph = (multiple - H % multiple) % multiple
        pw = (multiple - W % multiple) % multiple
        x = F.pad(x, (0, pw, 0, ph, 0, pd), mode="replicate")
        sl = (slice(None),) * (x.dim() - 3) + (
            slice(0, D), slice(0, H), slice(0, W)
        )
        return x, sl

    # ----- forward -----

    def forward(self, x: torch.Tensor, da_save_dir: Optional[str] = None) -> torch.Tensor:
        """
        x: (B, in_channels, D, H, W)

        eval  : returns Tensor (B, num_classes, D, H, W)
        train : returns List[Tensor] of 5 logits at 5 scales.
        """
        _v = bool(cfg.model_config.get("VERBOSE", False))
        if _v:
            print(f"[verbose] RIUnet.forward() input shape: {x.shape}")

        # ── 1. Pad input so spatial dims are multiples of 16 ──
        x, sl = self._pad_to_multiple(x, multiple=16)

        # ── 2. Stem ──
        x = self.stem(x)                                       # (B, 32,  D',  H',  W')

        # ── 3. Encoder (save skips before downsampling) ──
        skips = []
        for i, enc in enumerate(self.enc_blocks):
            skip, x = enc(x, save_dir=os.path.join(da_save_dir, f"enc_L{i}") if da_save_dir else None)                                   # (B, ch[i+1], D', H', W')
            skips.append(skip)

        # ── 4. Bottleneck (FNO || CNN) ──
        x, ds_bottle = self.bottleneck(x)
        # ds_outs = [ds_bottle]
        ds_outs = [] #Got rid of Bottleneck DS


        # ── 5. Decoder (mirror encoder) ──
        for i, dec in enumerate(self.dec_blocks):
            skip = skips[-(i + 1)]
            x, ds = dec(x, skip, save_dir=os.path.join(da_save_dir, f"dec_L{i}") if da_save_dir else None)
            if ds is not None:
                if _v:
                    print(f"[verbose] DS head[{i}] logits shape: {ds.shape}")
                ds_outs.append(ds)

        # ── 6. Final logits ──
        logits = self.final_head(x)                            # (B, 2, D', H', W')

        # ── 7. Crop outputs back to original spatial size ──
        # eval mode: single logits tensor
        if not self.training:
            return logits[sl]

        # train mode with deep supervision: [final, bottleneck, l1, l2, l3]
        if self.deep_supervision:
            return [logits[sl]] + [d[sl] for d in ds_outs if d is not None]

        # train mode without deep supervision: just the final logits
        if _v:
            print(f"[verbose] RIUnet.forward() single-logits shape: {logits[sl].shape}")
        return logits[sl]

    