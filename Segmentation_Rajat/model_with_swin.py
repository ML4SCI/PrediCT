"""
Reronet — Recurrent-Inspired U-Net for Coronary Artery Calcium segmentation.
"""

from __future__ import annotations

import os
import shutil
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from Segmentation_Rajat.model import Reronet
from rero_modules.conv_blocks import ConvBlockIN, Down, Up
from rero_modules.deform_attention import DeformableAttention3d
from rero_modules.fno3d import FNOBlock3d
from rero_modules.deep_supervision import DeepSupervisionHead

import config as cfg


# ══════════════════════════════════════════════════════════════════
#  ENCODER BLOCK
# ══════════════════════════════════════════════════════════════════

class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                da_heads: int = 8, da_points: int = 4,
                downsample: bool = False,
                da_save_dir: Optional[str] = None):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBlockIN(in_channels, out_channels),
            ConvBlockIN(out_channels, out_channels),
        )
        if out_channels <= 64:    # L0
            spatial_reduce, anchor_stride = 4, 4
        elif out_channels <= 128: # L1
            spatial_reduce, anchor_stride = 2, 2
        else:                    # L2, L3
            spatial_reduce, anchor_stride = 1, 1
            
        self.da = DeformableAttention3d(
            out_channels,
            num_heads=da_heads,
            num_points=da_points,
            spatial_reduce=spatial_reduce,
            anchor_stride=anchor_stride,
        )
        self.down = Down(out_channels, out_channels) if downsample else None

    def forward(self, x: torch.Tensor, da_save_dir: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.cnn(x)
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
    def __init__(self, in_channels: int, out_channels: int,
                da_heads: int = 8, da_points: int = 4,
                da_save_dir: Optional[str] = None):
        super().__init__()
        self.align = ConvBlockIN(in_channels, out_channels,
                                kernel_size=1, padding=0)
        if out_channels <= 64:    # decoder L0
            spatial_reduce, anchor_stride = 4, 4
        elif out_channels <= 128: # decoder L1
            spatial_reduce, anchor_stride = 2, 2
        else:                    # L2, L3
            spatial_reduce, anchor_stride = 1, 1
            
        self.da = DeformableAttention3d(
            out_channels,
            num_heads=da_heads,
            num_points=da_points,
            spatial_reduce=spatial_reduce,
            anchor_stride=anchor_stride,
        )

    def forward(self, x: torch.Tensor, da_save_dir: Optional[str] = None) -> torch.Tensor:
        x = self.align(x)
        out = self.da(x, save_dir=da_save_dir)
        if isinstance(out, tuple):
            out = out[0]
        return out


class DecoderBlock(nn.Module):
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
#  BOTTLENECK
# ══════════════════════════════════════════════════════════════════

class Bottleneck(nn.Module):
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
        )

        self.cnn  = nn.Sequential(
            ConvBlockIN(channels, channels),
            ConvBlockIN(channels, channels),
        )
        self.merge_norm = nn.InstanceNorm3d(channels, affine=True)
        self.merge_act  = nn.GELU()

        self.ds_head = DeepSupervisionHead(channels, num_classes) if return_ds_logits else None

    # def forward(self, x: torch.Tensor, fno_save_dir: Optional[str] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    #     out = self.merge_norm(self.fno(x, save_dir=fno_save_dir) + self.cnn(x))
    #     out = self.merge_act(out)
    #     if self.ds_head is None:
    #         return out, None
    #     return out, self.ds_head(out)

    # no FNO
    def forward(self, x: torch.Tensor, fno_save_dir: Optional[str] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        out = self.merge_norm(self.cnn(x))
        out = self.merge_act(out)
        if self.ds_head is None:
            return out, None
        return out, self.ds_head(out)


# ══════════════════════════════════════════════════════════════════
#  Reronet
# ══════════════════════════════════╗
from monai.networks.nets import SwinUNETR

class Reronet(nn.Module):

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
        super().__init__()

        in_channels      = in_channels     if in_channels     is not None else cfg.model_config.get("IN_CHANNELS", 6)
        base_channels    = base_channels   if base_channels   is not None else cfg.model_config["BASE_CHANNELS"]
        num_classes      = num_classes     if num_classes     is not None else cfg.model_config["NUM_CLASSES"]
        deep_supervision = deep_supervision if deep_supervision is not None else cfg.model_config["USE_DEEP_SUPERVISION"]
        da_heads         = da_heads        if da_heads        is not None else cfg.model_config["DA_NUM_HEADS"]
        da_points        = da_points       if da_points       is not None else cfg.model_config["DA_NUM_POINTS"]
        fno_modes        = fno_modes       if fno_modes       is not None else tuple(cfg.model_config["FNO_MODES"])

        self.ch = [base_channels * (2 ** i) for i in range(5)]
        ch = self.ch
        c_in, c0, c1, c2, c3, c4 = in_channels, *ch

        self.swin_model = SwinUNETR(
            in_channels=in_channels,
            out_channels=num_classes,
            feature_size=48,
        )


    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:

        return self.swin_model(x)

    #     if fno_modes is None:
    #         fno_modes = (8, 8, 8)

    #     self.save_dir = save_dir
    #     self.stem = ConvBlockIN(c_in, c0)

    #     self.enc_blocks = nn.ModuleList([
    #         EncoderBlock(ch[i], ch[i + 1],
    #                     da_heads=da_heads, da_points=da_points,
    #                     downsample=True,
    #                     da_save_dir=(os.path.join(save_dir, f"attn_maps/enc_L{i}")) if save_dir else None)
    #         for i in range(4)
    #     ])

    #     self.bottleneck = Bottleneck(
    #         channels         = c4,
    #         fno_modes        = fno_modes,
    #         return_ds_logits = deep_supervision,
    #         num_classes      = num_classes,
    #         fno_save_dir     = (os.path.join(save_dir, "fno_maps")) if save_dir else None,
    #     )

    #     self.dec_blocks = nn.ModuleList([
    #         DecoderBlock(
    #             in_channels     = ch[4 - i],
    #             skip_channels   = ch[4 - i],
    #             out_channels    = ch[4 - i - 1] if i < 3 else ch[0],
    #             num_classes     = num_classes,
    #             return_ds_logits = (deep_supervision and i != 3),
    #             da_heads=da_heads, da_points=da_points,
    #             da_save_dir=(os.path.join(save_dir, f"attn_maps/dec_L{i}")) if save_dir else None,
    #         )
    #         for i in range(4)
    #     ])

    #     self.final_head = nn.Conv3d(ch[0], num_classes, kernel_size=1)
    #     self.deep_supervision = deep_supervision

    # @staticmethod
    # def _pad_to_multiple(x: torch.Tensor, multiple: int = 16) -> Tuple[torch.Tensor, Tuple[slice, ...]]:
    #     D, H, W = x.shape[-3:]
    #     pd = (multiple - D % multiple) % multiple
    #     ph = (multiple - H % multiple) % multiple
    #     pw = (multiple - W % multiple) % multiple
    #     x = F.pad(x, (0, pw, 0, ph, 0, pd), mode="replicate")
    #     sl = (slice(None),) * (x.dim() - 3) + (
    #         slice(0, D), slice(0, H), slice(0, W)
    #     )
    #     return x, sl

    # def forward(self, x: torch.Tensor, da_save_dir: Optional[str] = None) -> torch.Tensor | List[torch.Tensor]:
    #     _v = bool(cfg.model_config.get("VERBOSE", False))
    #     if _v:
    #         print(f"[verbose] Reronet.forward() input shape: {x.shape}")

    #     x, sl = self._pad_to_multiple(x, multiple=16)
    #     x = self.stem(x)

    #     skips = []
    #     for i, enc in enumerate(self.enc_blocks):
    #         skip, x = enc(x, da_save_dir=os.path.join(da_save_dir, f"enc_L{i}") if da_save_dir else None)
    #         skips.append(skip)

    #     x, ds_bottle = self.bottleneck(x, fno_save_dir=os.path.join(da_save_dir, "bottleneck") if da_save_dir else None)
    #     ds_outs = []

    #     for i, dec in enumerate(self.dec_blocks):
    #         skip = skips[-(i + 1)]
    #         x, ds = dec(x, skip, da_save_dir=os.path.join(da_save_dir, f"dec_L{i}") if da_save_dir else None)
    #         if ds is not None:
    #             if _v:
    #                 print(f"[verbose] DS head[{i}] logits shape: {ds.shape}")
    #             ds_outs.append(ds)

    #     logits = self.final_head(x)

    #     if not self.training:
    #         return logits[sl]

    #     if self.deep_supervision:
    #         return [logits[sl]] + [d[sl] for d in ds_outs if d is not None]

    #     if _v:
    #         print(f"[verbose] Reronet.forward() single-logits shape: {logits[sl].shape}")
    #     return logits[sl]     
    

if __name__ == "__main__":
    import os
    import time
    import shutil
    import psutil
    import torch

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config extraction with safe fallback
    roi = cfg.model_config.get("ROI_SIZE")
    D, H, W = roi if roi and not any(v is None for v in roi) else (16, 32, 32)
    in_ch = cfg.model_config.get("IN_CHANNELS") or 6
    num_classes = cfg.model_config.get("NUM_CLASSES", 2)

    B = 2
    total_epochs = 200
    selective_epochs = cfg.train_config.get("DUMP_EPOCHS")
    base_dump_dir = os.path.join(".", "temp_reronet_benchmark")

    print(f"device: {device}")
    print(f"input : (B={B}, C={in_ch}, D={D}, H={H}, W={W}), num_classes={num_classes}")

    proc = psutil.Process()

    def cleanup_dir(path: str):
        if os.path.exists(path):
            shutil.rmtree(path)

    # ─────────────────────────────────────────────────────────────
    #  STRATEGY A: Pure Forward (No Dumps)
    # ─────────────────────────────────────────────────────────────
    cleanup_dir(base_dump_dir)
    model_a = Reronet(in_channels=in_ch).to(device)
    model_a.train()
    x = torch.randn(B, in_ch, D, H, W, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    cpu_before = proc.cpu_times()
    t0 = time.perf_counter()

    for _ in range(1, total_epochs + 1):
        _ = model_a(x, da_save_dir=None)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    cpu_after = proc.cpu_times()

    time_no_dump = t1 - t0
    cpu_no_dump = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)
    gpu_mem_no_dump = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0.0

    # ─────────────────────────────────────────────────────────────
    #  STRATEGY B: Selective Dumps (Epochs 1, 10, 20)
    # ─────────────────────────────────────────────────────────────
    dump_dir_b = os.path.join(base_dump_dir, "selective")
    cleanup_dir(dump_dir_b)

    model_b = Reronet(in_channels=in_ch).to(device)
    model_b.train()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    cpu_before = proc.cpu_times()
    t0 = time.perf_counter()

    for ep in range(1, total_epochs + 1):
        save_path = os.path.join(dump_dir_b, f"epoch_{ep:03d}") if ep in selective_epochs else None
        _ = model_b(x, da_save_dir=save_path)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    cpu_after = proc.cpu_times()

    time_selective = t1 - t0
    cpu_selective = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)
    gpu_mem_selective = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0.0

    # ─────────────────────────────────────────────────────────────
    #  STRATEGY C: Dump Every Epoch (Epochs 1..20)
    # ─────────────────────────────────────────────────────────────
    dump_dir_c = os.path.join(base_dump_dir, "every_epoch")
    cleanup_dir(dump_dir_c)

    model_c = Reronet(in_channels=in_ch).to(device)
    model_c.train()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    cpu_before = proc.cpu_times()
    t0 = time.perf_counter()

    for ep in range(1, total_epochs + 1):
        save_path = os.path.join(dump_dir_c, f"epoch_{ep:03d}")
        _ = model_c(x, da_save_dir=save_path)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    cpu_after = proc.cpu_times()

    time_every = t1 - t0
    cpu_every = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)
    gpu_mem_every = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0.0

    # ─────────────────────────────────────────────────────────────
    #  SUMMARY COMPARISON REPORT
    # ─────────────────────────────────────────────────────────────
    avg_ms_no_dump = (time_no_dump / total_epochs) * 1000.0
    avg_ms_selective = (time_selective / total_epochs) * 1000.0
    avg_ms_every = (time_every / total_epochs) * 1000.0

    print("\n" + "=" * 78)
    print(f"BENCHMARK SUMMARY ({total_epochs} Epochs | Spatial: {D}x{H}x{W})")
    print("=" * 78)
    print(f"{'Strategy':<25} | {'Total Time':<10} | {'Avg/Epoch':<10} | {'CPU Time':<9} | {'Peak GPU Mem':<12} | {'Slowdown':<8}")
    print("-" * 78)
    print(f"{'1. No Dumps':<25} | {time_no_dump:6.2f} s   | {avg_ms_no_dump:7.1f} ms  | {cpu_no_dump:6.2f} s  | {gpu_mem_no_dump:8.1f} MB  | 1.00x")
    print(f"{'2. Selective Epochs':<25} | {time_selective:6.2f} s   | {avg_ms_selective:7.1f} ms  | {cpu_selective:6.2f} s  | {gpu_mem_selective:8.1f} MB  | {time_selective / time_no_dump:5.2f}x")
    print(f"{'3. Every Epoch':<25} | {time_every:6.2f} s   | {avg_ms_every:7.1f} ms  | {cpu_every:6.2f} s  | {gpu_mem_every:8.1f} MB  | {time_every / time_no_dump:5.2f}x")
    print(f"Selective Epochs: {selective_epochs}")
    print("=" * 78)

    # Cleanup test outputs
    cleanup_dir(base_dump_dir)