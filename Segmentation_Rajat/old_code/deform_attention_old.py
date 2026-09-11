"""
Deformable Attention 3D (Deformable DETR style) — classic formulation.

Classic notation used throughout this file:

    W_q       : query embedding projection
                (query drives BOTH the offset head and the weight head)
    W_delta_p : offset predictor            -> "where to sample"
    W_A       : attention-weight predictor  -> "how much to weight each sample"
                (a LINEAR function of the query, softmaxed over the K points —
                there is NO Q·K / Q·V dot-product anywhere in this module)
    W_v       : value projection            -> "what gets sampled"
    W_out     : output projection after the weighted sum

For every spatial position (anchor / query), the module:
    1. builds a query embedding via W_q
    2. predicts K offsets per head from that query via W_delta_p
    3. predicts K attention weights per head from that query via W_A
       (softmax over K, purely linear -- no similarity/dot-product)
    4. samples K points from a (possibly down-sampled) value map (W_v)
       at (anchor + offset) locations
    5. takes the W_A-weighted sum of the K sampled values
    6. projects with W_out, upsamples back to full resolution via
       TRILINEAR interpolation, and adds the residual

Spatial reduction (factor) on V is used to keep memory bounded at the
shallow levels where dims are large.
"""

import math
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path

# Get the path to folder A
# If this code is inside A/folder/a.py:
dir_A = Path(__file__).resolve().parent.parent

# Add directory A to sys.path
if str(dir_A) not in sys.path:
    sys.path.append(str(dir_A))

import config as cfg

from pathlib import Path


def _make_3d_ref_grid(D: int, H: int, W: int, device, dtype=torch.float32):
    """
    Dense grid of normalized 3D coords in [-1, 1].
    Returns (D, H, W, 3) where last dim is (z, y, x).
    """
    z = torch.linspace(-1.0, 1.0, D, device=device, dtype=dtype)
    y = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack([grid_z, grid_y, grid_x], dim=-1)  # (D, H, W, 3)


class DeformableAttention3d(nn.Module):
    """
    3D Multi-Scale Deformable Attention (Deformable DETR style, classic form).

    Projections
    -----------
        W_q        : query embedding (feeds W_delta_p and W_A)
        W_delta_p  : per-head, per-point (z, y, x) offsets
        W_A        : per-head, per-point attention weights (softmax over K,
                     linear in the query -- NOT a dot-product with K/V)
        W_v        : values to be sampled
        W_out      : output projection

    Shape
    -----
    Input  : (B, C, D, H, W)
    Output : (B, C, D, H, W)

    Residual: out = x + gamma * DA(x)    (gamma = 1.0 init)
    Upsampling from anchor resolution back to full resolution uses
    TRILINEAR interpolation.

    If `return_attn_info=True`, `forward` returns `(out, info)` where
    `info` carries the attention weights and the corresponding sample
    locations, suitable for splatting a "where was attention focused"
    heatmap back onto the input volume (see `save_attention_heatmap`).
    """

    def __init__(
        self,
        dim:              int,
        num_heads:        int = 8,
        num_points:       int = 4,
        spatial_reduce:   int = 2,
        anchor_stride:    int = 1,
        return_attn_info: bool = False,
        save_dir:         str | None = None,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim             = dim
        self.num_heads       = num_heads
        self.num_points      = num_points
        self.head_dim        = dim // num_heads
        self.spatial_reduce  = max(1, spatial_reduce)
        self.anchor_stride   = max(1, anchor_stride)
        self.return_attn_info = bool(return_attn_info)

        # ---- attention-dump sink (mirrors FNO's save_feature_maps) ----
        self.save_dir     = save_dir
        self.save_counter = 0
        # Per-epoch identifier, set by the train loop each epoch. When
        # `DUMP_LAST_STEP_ONLY` is on, the save path encodes this as
        # `epoch_{N:03d}_last_step_*` so we can tell epochs apart.
        self.epoch_id: Optional[int] = None
        self.dump_mode: str = "first_step"   # "first_step" or "last_step"
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        # ---- W_q : query embedding ----
        self.W_q = nn.Conv3d(self.dim, self.dim, kernel_size=1, bias=True)

        # ---- W_v : value projection (what gets sampled) ----
        self.W_v = nn.Conv3d(self.dim, self.dim, kernel_size=1, bias=True)

        # ---- W_delta_p : offset predictor ----
        self.W_delta_p = nn.Conv3d(
            self.dim,
            self.num_heads * self.num_points * 3,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        nn.init.zeros_(self.W_delta_p.weight)
        nn.init.zeros_(self.W_delta_p.bias)

        # ---- W_A : attention-weight predictor ----
        self.W_A = nn.Conv3d(
            self.dim,
            self.num_heads * self.num_points,
            kernel_size=1,
            bias=True,
        )
        nn.init.zeros_(self.W_A.weight)
        nn.init.zeros_(self.W_A.bias)

        # ---- W_out : output projection ----
        self.W_out = nn.Conv3d(self.dim, self.dim, kernel_size=1, bias=True)
        self.gamma = nn.Parameter(torch.ones(1))

    def reset_save_counter(self) -> None:
        """Zero the .npy-file counter so a new epoch (or new run) restarts at 0."""
        self.save_counter = 0

    def set_epoch(self, epoch_id: int, dump_mode: str = "first_step") -> None:
        """Tag this module with the current epoch and the per-epoch dump policy.

        `dump_mode` is either "first_step" (legacy: dump only the very first
        step of each epoch) or "last_step" (dump only the very last step of
        each epoch, with `epoch_{N:03d}_last_step_*` filenames so we can
        tell epochs apart).
        """
        self.epoch_id = int(epoch_id)
        self.dump_mode = str(dump_mode)

    # ---------- private helpers ----------

    def _downsample(self, x: torch.Tensor) -> torch.Tensor:
        if self.spatial_reduce == 1:
            return x
        return F.avg_pool3d(x, kernel_size=self.spatial_reduce, stride=self.spatial_reduce)

    def _sample_points(self, value, ref_points, offsets):
        """
        Sample K points per anchor from `value` in ONE fused grid_sample.

        Args:
            value:      (B, C, Dv, Hv, Wv)          downsampled value map (W_v output)
            ref_points: (B, num_heads, D, H, W, 3)   anchor grid in [-1, 1]
            offsets:    (B, num_heads*num_points*3, D, H, W)   W_delta_p output

        Returns:
            sampled: (B, num_heads, num_points, D, H, W, head_dim)
        """
        B, C, Dv, Hv, Wv = value.shape
        D, H, W = offsets.shape[-3:]
        K = self.num_points
        h = self.num_heads
        head_dim = self.head_dim

        # Reshape value -> (B*h, head_dim, Dv, Hv, Wv)
        v = value.view(B, h, head_dim, Dv, Hv, Wv).permute(0, 1, 3, 4, 5, 2).contiguous()
        v = v.view(B * h, Dv, Hv, Wv, head_dim).permute(0, 4, 1, 2, 3).contiguous()
        # v: (B*h, head_dim, Dv, Hv, Wv)

        # Reshape offsets -> (B, h, K, 3, D, H, W) -> (B, h, K, D, H, W, 3)
        offs = offsets.view(B, h, K, 3, D, H, W).permute(0, 1, 2, 4, 5, 6, 3)

        # sample_pts: (B, h, K, D, H, W, 3) in [-1, 1]
        sample_pts = ref_points.unsqueeze(2) + offs * 0.5
        # Reshape for grid_sample: treat K as part of batch dim.
        # grid: (B*h*K, D, H, W, 3)
        grid = sample_pts.reshape(B * h * K, D, H, W, 3)
        # Expand v over K: (B*h, head_dim, Dv, Hv, Wv) -> (B*h*K, head_dim, Dv, Hv, Wv)
        v_rep = v.unsqueeze(1).expand(B * h, K, head_dim, Dv, Hv, Wv)
        v_rep = v_rep.reshape(B * h * K, head_dim, Dv, Hv, Wv)

        out = F.grid_sample(
            v_rep, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )                                                                  # (B*h*K, head_dim, D, H, W)
        out = out.view(B, h, K, head_dim, D, H, W)
        out = out.permute(0, 1, 2, 4, 5, 6, 3).contiguous()
        # (B, h, K, D, H, W, head_dim)
        return out

    # ---------- forward ----------

    def forward(self, x: torch.Tensor):
        B, C, D, H, W = x.shape
        x = x.as_subclass(torch.Tensor)

        # ---- anchor grid (may be coarser than full resolution) ----
        a = self.anchor_stride
        Da, Ha, Wa = D // a, H // a, W // a
        q_in = F.avg_pool3d(x, kernel_size=a, stride=a) if a > 1 else x

        # ---- W_q : query embedding, drives W_delta_p and W_A ----

        if q_in.dtype != self.W_q.weight.dtype:
            q_in = q_in.to(self.W_q.weight.dtype)

        query = self.W_q(q_in)                                # (B, C, Da, Ha, Wa)

        # ---- W_v : value projection (downsampled, saves memory) ----
        v_in = self._downsample(x)
        v = self.W_v(v_in)                                    # (B, C, Dv, Hv, Wv)

        # ---- reference points (anchor grid) ----
        ref_points = _make_3d_ref_grid(Da, Ha, Wa, x.device, x.dtype)
        ref_points = ref_points.view(1, 1, Da, Ha, Wa, 3).expand(
            B, self.num_heads, Da, Ha, Wa, 3
        )

        # ---- W_delta_p : offsets, predicted from the query embedding ----
        offsets = self.W_delta_p(query)                       # (B, h*K*3, Da, Ha, Wa)

        # ---- W_A : attention weights, predicted from the query embedding ----
        # Linear (1x1 conv) -> reshape -> softmax over K per head.
        attn_logits = self.W_A(query)                         # (B, h*K, Da, Ha, Wa)
        attn = attn_logits.view(B, self.num_heads, self.num_points, Da, Ha, Wa)
        attn = attn.softmax(dim=2)                             # softmax over K points
        # keep a (B, h, Da, Ha, Wa, K) view for inspection
        attn_5d = attn.permute(0, 1, 3, 4, 5, 2).contiguous()  # (B, h, Da, Ha, Wa, K)
        attn = attn_5d.reshape(B, self.num_heads, Da * Ha * Wa, self.num_points)
        # (B, h, N, K)

        # ---- sample K points from V (W_v output) at anchor+offset locations ----
        sampled = self._sample_points(v, ref_points, offsets)
        # sampled: (B, h, K, Da, Ha, Wa, head_dim)
        sampled_flat = sampled.permute(0, 1, 3, 4, 5, 2, 6).reshape(
            B, self.num_heads, Da * Ha * Wa, self.num_points, self.head_dim
        )
        # (B, h, N, K, head_dim)

        # ---- weighted sum over K samples using W_A weights ----
        out = (attn.unsqueeze(-1) * sampled_flat).sum(dim=3)
        # (B, h, N, head_dim)
        out = out.transpose(-1, -2).reshape(B, C, Da, Ha, Wa)
        out = self.W_out(out)

        # ---- upsample back to full resolution via TRILINEAR interpolation ----
        if a > 1:
            out = F.interpolate(out, size=(D, H, W), mode="trilinear", align_corners=False)

        # ---- residual ----
        out = x + self.gamma * out

        if not self.return_attn_info:
            return out

        # ------------------------------------------------------------------
        # Attention introspection
        # ------------------------------------------------------------------
        # We return enough info to splat the per-sample-point attention
        # weights back onto the input volume as a 3D heatmap. The actual
        # sampling location fed to F.grid_sample was `ref + 0.5*offset`
        # (see _sample_points); we return those exact normalized coords.
        K = self.num_points
        h = self.num_heads

        # offsets: (B, h*K*3, Da, Ha, Wa) -> (B, h, K, 3, Da, Ha, Wa)
        offs = offsets.view(B, h, K, 3, Da, Ha, Wa)
        # permute the 3-dim next to the spatial dims so we can add it
        # directly to ref_points: (B, h, K, Da, Ha, Wa, 3)
        offs = offs.permute(0, 1, 2, 4, 5, 6, 3).contiguous()
        # sampling location the module actually used, normalized in [-1, 1]
        # ref_points: (B, h, Da, Ha, Wa, 3)  ->  (B, h, 1, Da, Ha, Wa, 3)
        # sample_locs: (B, h, K, Da, Ha, Wa, 3)
        sample_locs = (ref_points.unsqueeze(2) + offs * 0.5).contiguous()
        # (B, h, K, Da, Ha, Wa, 3)  -- (z, y, x) in [-1, 1]

        # ---- attention disk-dump (only first sample of the batch, train only) ----
        # Skip during eval/validation: the per-step .npy copies easily eat the
        # host RAM and the resulting OOM looks like a CUDA error. Training-
        # only dumps keep the size sane. Also bail if save_dir was cleared by
        # the train loop (dump-first-only mode).
        if self.save_dir is not None and self.training:
            # Keep only sample 0, and detach/move to CPU *before* splatting
            # so splat_attention_to_volume's index_add_ (which builds its
            # accumulator on CPU) never sees a CUDA/autograd-tracked tensor.
            attn0 = attn_5d[0:1].detach().cpu()      # (1, h, Da, Ha, Wa, K)
            locs0 = sample_locs[0:1].detach().cpu()  # (1, h, K, Da, Ha, Wa, 3)
            info0 = {"attn_weights": attn0, "sample_locs": locs0}

            # Splat-average onto the full-resolution (D, H, W) input volume.
            heatmap0 = self.splat_attention_to_volume(
                info0, out_volume=(D, H, W), avg_heads=True
            )                                          # (1, D, H, W), cpu, float32
            heatmap_np = heatmap0[0].detach().cpu().numpy()  # (D, H, W)

            # Filename: in last_step mode we tag with the epoch id so the
            # per-epoch "what happened at the end" snapshots don't collide.
            if getattr(self, "dump_mode", "first_step") == "last_step":
                ep = int(getattr(self, "epoch_id", 0) or 0)
                base = f"epoch_{ep:03d}_last_step"
            else:
                base = f"step_{self.save_counter}"
            np.save(os.path.join(self.save_dir, f"{base}_heatmap.npy"), heatmap_np)
            self.save_counter += 1

        return out

    # ---------- attention heatmap helpers ----------

    @staticmethod
    def splat_attention_to_volume(
        info:        dict,
        out_volume:   tuple,
        avg_heads:   bool = True,
    ) -> torch.Tensor:
        """
        Splat the K weighted sample locations onto a `out_volume` 3D grid
        and return the resulting heatmap (B, D, H, W).

        Each anchor contributes `attn[..., k] * delta(sample_loc[k])` to the
        output volume; since the actual grid_sample is bilinear, the
        contribution is bilinearly distributed to the 8 surrounding voxels
        of the sample location in normalized space.

        Args:
            info:      dict returned by `forward(return_attn_info=True)`
            out_volume: (D, H, W) of the volume to splat onto. The sample
                       locations are given at anchor resolution; if
                       `anchor_stride > 1` you almost always want to pass
                       the FULL-RES shape here so the heatmap aligns with
                       the input.
            avg_heads: if True, average over heads first (per-head heatmaps
                       would highlight different things for different heads).
        Returns:
            heatmap: (B, D, H, W) on CPU, float32. Not normalized -- call
                     `.normalize()` or save with a vmax you choose.
        """
        attn_5d    = info["attn_weights"]                    # (B, h, Da, Ha, Wa, K)
        sample_locs = info["sample_locs"]                    # (B, h, K, Da, Ha, Wa, 3)
        B, Da, Ha, Wa, K = attn_5d.shape[0], *attn_5d.shape[2:]
        D, H, W = out_volume

        if avg_heads:
            attn_h = attn_5d.mean(dim=1)                     # (B, Da, Ha, Wa, K)
            locs_h = sample_locs.mean(dim=1)                 # (B, K, Da, Ha, Wa, 3)
        else:
            attn_h = attn_5d.permute(0, 1, 2, 3, 4, 5)        # (B, h, Da, Ha, Wa, K)
            locs_h = sample_locs                              # (B, h, K, Da, Ha, Wa, 3)

        # Flatten anchors -> one big "source" point cloud.
        if avg_heads:
            attn_flat = attn_h.permute(0, 3, 1, 2, 4).reshape(-1, K)   # (B*Da*Ha*Wa, K)
            locs_flat = locs_h.permute(0, 2, 3, 4, 1, 5).reshape(-1, K, 3)
        else:
            attn_flat = attn_h.permute(0, 1, 4, 2, 3, 5).reshape(-1, K)
            locs_flat = locs_h.permute(0, 1, 3, 4, 5, 2, 6).reshape(-1, K, 3)

        B_ = B

        # Convert normalized coords [-1, 1] -> voxel index in [0, D) etc.
        # F.grid_sample uses align_corners=True, so the mapping is:
        #     x_voxel = (x_norm + 1) * (size - 1) / 2
        def _to_vox(norm, size):
            return (norm + 1.0) * (size - 1) / 2.0

        # We bilinearly distribute each sample point's weight to its 8
        # surrounding integer voxels. This mirrors what F.grid_sample did
        # to the value features, so the heatmap aligns with the actual
        # values that were aggregated.
        heatmap = torch.zeros(B_, D, H, W, dtype=torch.float32, device="cpu")
        # (use a 32-bit accumulator; we are on CPU only)
        # We do the splat per-batch to keep the bookkeeping simple.
        offset = 0
        for b in range(B_):
            n_anchors = Da * Ha * Wa
            a_attn = attn_flat[b * n_anchors:(b + 1) * n_anchors]   # (N, K)
            a_locs = locs_flat[b * n_anchors:(b + 1) * n_anchors]   # (N, K, 3)

            # Continuous voxel coords for every (anchor, k)
            vx = _to_vox(a_locs[..., 2], W)                          # (N, K)
            vy = _to_vox(a_locs[..., 1], H)                          # (N, K)
            vz = _to_vox(a_locs[..., 0], D)                          # (N, K)

            # Integer corners and trilinear weights
            x0 = torch.floor(vx).long().clamp(0, W - 1)
            y0 = torch.floor(vy).long().clamp(0, H - 1)
            z0 = torch.floor(vz).long().clamp(0, D - 1)
            x1 = (x0 + 1).clamp(0, W - 1)
            y1 = (y0 + 1).clamp(0, H - 1)
            z1 = (z0 + 1).clamp(0, D - 1)

            wx1 = vx - x0.float()
            wy1 = vy - y0.float()
            wz1 = vz - z0.float()
            wx0 = 1.0 - wx1
            wy0 = 1.0 - wy1
            wz0 = 1.0 - wz1

            w000 = (wx0 * wy0 * wz0) * a_attn
            w100 = (wx1 * wy0 * wz0) * a_attn
            w010 = (wx0 * wy1 * wz0) * a_attn
            w110 = (wx1 * wy1 * wz0) * a_attn
            w001 = (wx0 * wy0 * wz1) * a_attn
            w101 = (wx1 * wy0 * wz1) * a_attn
            w011 = (wx0 * wy1 * wz1) * a_attn
            w111 = (wx1 * wy1 * wz1) * a_attn

            # Distribute each weighted sample point to the 8 surrounding
            # integer voxels via index_add_ (cheap & differentiable w.r.t.
            # nothing here -- this runs at eval time only).
            def _add(w, x_idx, y_idx, z_idx):
                vals = w.reshape(-1)
                # the heatmap is indexed as heatmap[b, z, y, x]
                idx = (
                    z_idx.reshape(-1) * (H * W)
                    + y_idx.reshape(-1) * W
                    + x_idx.reshape(-1)
                )
                heatmap[b].view(-1).index_add_(0, idx, vals)

            _add(w000, x0, y0, z0)
            _add(w100, x1, y0, z0)
            _add(w010, x0, y1, z0)
            _add(w110, x1, y1, z0)
            _add(w001, x0, y0, z1)
            _add(w101, x1, y0, z1)
            _add(w011, x0, y1, z1)
            _add(w111, x1, y1, z1)

            offset += n_anchors

        return heatmap



# --------

import os
import time
import tempfile
from pathlib import Path
import psutil
import torch

# Import your module here
# from your_module import DeformableAttention3d


def run_benchmark():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Benchmarking hyperparameters
    B, C, D, H, W = 2, 32, 16, 32, 32
    num_heads, num_points = 4, 4
    n_iters = 20

    print("=" * 70)
    print("DeformableAttention3d Performance & Resource Benchmark")
    print("=" * 70)
    print(f"Device : {device}")
    print(f"Input  : (B={B}, C={C}, D={D}, H={H}, W={W})")
    print(f"Params : heads={num_heads}, points={num_points}, iters={n_iters}")
    print("-" * 70)

    x = torch.randn(B, C, D, H, W, device=device)
    proc = psutil.Process()

    # -------------------------------------------------------------------------
    # 1. Benchmark: Plain Forward Pass (Inference/No Dump)
    # -------------------------------------------------------------------------
    model_plain = DeformableAttention3d(
        dim=C,
        num_heads=num_heads,
        num_points=num_points,
        spatial_reduce=2,
        anchor_stride=1,
        return_attn_info=False,
    ).to(device)
    model_plain.eval()

    # Warmup
    for _ in range(3):
        _ = model_plain(x)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    cpu_before = proc.cpu_times()
    t0 = time.perf_counter()

    with torch.no_grad():
        for _ in range(n_iters):
            out = model_plain(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t1 = time.perf_counter()
    cpu_after = proc.cpu_times()

    plain_time_avg = ((t1 - t0) / n_iters) * 1000  # ms
    plain_cpu_used = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)
    plain_gpu_mem = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else 0.0

    assert out.shape == x.shape, "Output shape mismatch in standard forward"
    print(f"[1/2] Standard Forward Pass:")
    print(f"      Avg Latency   : {plain_time_avg:.2f} ms/iter")
    print(f"      CPU Time Used : {plain_cpu_used:.3f} s total")
    if device.type == "cuda":
        print(f"      GPU Peak Mem  : {plain_gpu_mem:.1f} MB")

    # -------------------------------------------------------------------------
    # 2. Benchmark: Attention Dump Path (Disk IO + Heatmap Splatting)
    # -------------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        model_dump = DeformableAttention3d(
            dim=C,
            num_heads=num_heads,
            num_points=num_points,
            spatial_reduce=2,
            anchor_stride=1,
            return_attn_info=True,
            save_dir=tmp_path,
        ).to(device)
        model_dump.train()
        model_dump.set_epoch(0, dump_mode="first_step")

        # Warmup
        for _ in range(3):
            _ = model_dump(x)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        cpu_before = proc.cpu_times()
        t0 = time.perf_counter()

        for _ in range(n_iters):
            _ = model_dump(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        t1 = time.perf_counter()
        cpu_after = proc.cpu_times()

        dump_time_avg = ((t1 - t0) / n_iters) * 1000  # ms
        dump_cpu_used = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)
        dump_gpu_mem = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else 0.0

        # Verify disk output
        expected_file = tmp_path / "step_0_heatmap.npy"
        assert expected_file.exists(), f"Heatmap file was not created at {expected_file}"

        print(f"\n[2/2] Forward Pass with Dump Enabled (1 dump, rest skipped):")
        print(f"      Avg Latency   : {dump_time_avg:.2f} ms/iter")
        print(f"      CPU Time Used : {dump_cpu_used:.3f} s total")
        if device.type == "cuda":
            print(f"      GPU Peak Mem  : {dump_gpu_mem:.1f} MB")

    # -------------------------------------------------------------------------
    # Summary Analysis
    # -------------------------------------------------------------------------
    slowdown = dump_time_avg / plain_time_avg if plain_time_avg > 0 else 0.0
    print("-" * 70)
    print(f"Amortized Dump Overhead Factor : {slowdown:.2f}x")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
   