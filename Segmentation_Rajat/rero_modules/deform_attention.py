"""
Deformable Attention 3D (Deformable DETR style, classic formulation).

    W_q       : query embedding (feeds W_delta_p and W_A)
    W_delta_p : per-head, per-point (z, y, x) offset predictor
    W_A       : per-head, per-point attention-weight predictor (linear
                in the query, softmaxed over K points -- no Q.K/Q.V)
    W_v       : value projection (what gets sampled)
    W_out     : output projection

Forward: build query -> predict K offsets/weights per head -> sample K
points from a (optionally down-sampled) value map -> weighted sum ->
project -> upsample to full res (trilinear) -> residual.
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_3d_ref_grid(D: int, H: int, W: int, device, dtype=torch.float32):
    """Dense grid of normalized (z, y, x) coords in [-1, 1] -> (D, H, W, 3)."""
    z = torch.linspace(-1.0, 1.0, D, device=device, dtype=dtype)
    y = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    gz, gy, gx = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack([gz, gy, gx], dim=-1)


class DeformableAttention3d(nn.Module):
    """3D Multi-Scale Deformable Attention (Deformable DETR style).

    Input/Output: (B, C, D, H, W). Residual: out = x + gamma * DA(x).

    if save_dir is given in forward then dumping happens (attention heatmap .npy) for that step, otherwise no dumping.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_points: int = 4,
        spatial_reduce: int = 2,
        anchor_stride: int = 1,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = dim // num_heads
        self.spatial_reduce = max(1, spatial_reduce)
        self.anchor_stride = max(1, anchor_stride)

        # ---- attention-dump bookkeeping ----
        # Dumping is controlled per-call via forward(..., save_dir=...),
        # not stored on the module -- pass None every step and only pass
        # a real path on the step(s) you actually want a heatmap saved.
        self.epoch_id: Optional[int] = None

        self.W_q = nn.Conv3d(dim, dim, kernel_size=1, bias=True)
        self.W_v = nn.Conv3d(dim, dim, kernel_size=1, bias=True)

        self.W_delta_p = nn.Conv3d(
            dim, num_heads * num_points * 3, kernel_size=3, padding=1, bias=True
        )
        nn.init.zeros_(self.W_delta_p.weight)
        nn.init.zeros_(self.W_delta_p.bias)

        self.W_A = nn.Conv3d(dim, num_heads * num_points, kernel_size=1, bias=True)
        nn.init.zeros_(self.W_A.weight)
        nn.init.zeros_(self.W_A.bias)

        self.W_out = nn.Conv3d(dim, dim, kernel_size=1, bias=True)
        self.gamma = nn.Parameter(torch.ones(1))

    def set_epoch(self, epoch_id: int) -> None:
        """Tag dumped filenames with the current epoch, e.g. from train.py."""
        self.epoch_id = int(epoch_id)

    # ---------- private helpers ----------

    def _downsample(self, x: torch.Tensor) -> torch.Tensor:
        if self.spatial_reduce == 1:
            return x
        return F.avg_pool3d(x, kernel_size=self.spatial_reduce, stride=self.spatial_reduce)

    def _sample_points(self, value, ref_points, offsets):
        """Sample K points per anchor from `value` in one fused grid_sample.

        value:      (B, C, Dv, Hv, Wv)
        ref_points: (B, h, D, H, W, 3)
        offsets:    (B, h*K*3, D, H, W)
        returns:    (B, h, K, D, H, W, head_dim)
        """
        B, C, Dv, Hv, Wv = value.shape
        D, H, W = offsets.shape[-3:]
        K, h, head_dim = self.num_points, self.num_heads, self.head_dim

        v = value.view(B, h, head_dim, Dv, Hv, Wv).permute(0, 1, 3, 4, 5, 2).contiguous()
        v = v.view(B * h, Dv, Hv, Wv, head_dim).permute(0, 4, 1, 2, 3).contiguous()

        offs = offsets.view(B, h, K, 3, D, H, W).permute(0, 1, 2, 4, 5, 6, 3)
        sample_pts = ref_points.unsqueeze(2) + offs * 0.5  # (B, h, K, D, H, W, 3)

        grid = sample_pts.reshape(B * h * K, D, H, W, 3)
        v_rep = v.unsqueeze(1).expand(B * h, K, head_dim, Dv, Hv, Wv)
        v_rep = v_rep.reshape(B * h * K, head_dim, Dv, Hv, Wv)

        out = F.grid_sample(v_rep, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        out = out.view(B, h, K, head_dim, D, H, W).permute(0, 1, 2, 4, 5, 6, 3).contiguous()
        return out

    # ---------- forward ----------

    def forward(self, x: torch.Tensor, save_dir: Optional[str] = None):
        """
        save_dir: if given (and the module is in train() mode), dumps an
        attention heatmap .npy for this step to that directory. Pass None
        on ordinary steps and only pass a path on the step(s) you want a
        heatmap saved (e.g. only the last step of an epoch) -- this keeps
        the CPU-bound splat/np.save work off the hot path entirely.
        """
        B, C, D, H, W = x.shape
        x = x.as_subclass(torch.Tensor)

        a = self.anchor_stride
        Da, Ha, Wa = D // a, H // a, W // a
        q_in = F.avg_pool3d(x, kernel_size=a, stride=a) if a > 1 else x
        if q_in.dtype != self.W_q.weight.dtype:
            q_in = q_in.to(self.W_q.weight.dtype)

        query = self.W_q(q_in)
        v = self.W_v(self._downsample(x))

        ref_points = _make_3d_ref_grid(Da, Ha, Wa, x.device, x.dtype)
        ref_points = ref_points.view(1, 1, Da, Ha, Wa, 3).expand(B, self.num_heads, Da, Ha, Wa, 3)

        offsets = self.W_delta_p(query)  # (B, h*K*3, Da, Ha, Wa)

        attn_logits = self.W_A(query)  # (B, h*K, Da, Ha, Wa)
        attn = attn_logits.view(B, self.num_heads, self.num_points, Da, Ha, Wa).softmax(dim=2)
        attn_5d = attn.permute(0, 1, 3, 4, 5, 2).contiguous()  # (B, h, Da, Ha, Wa, K)
        attn = attn_5d.reshape(B, self.num_heads, Da * Ha * Wa, self.num_points)

        sampled = self._sample_points(v, ref_points, offsets)  # (B, h, K, Da, Ha, Wa, head_dim)
        sampled_flat = sampled.permute(0, 1, 3, 4, 5, 2, 6).reshape(
            B, self.num_heads, Da * Ha * Wa, self.num_points, self.head_dim
        )

        out = (attn.unsqueeze(-1) * sampled_flat).sum(dim=3)  # (B, h, N, head_dim)
        out = out.transpose(-1, -2).reshape(B, C, Da, Ha, Wa)
        out = self.W_out(out)

        if a > 1:
            out = F.interpolate(out, size=(D, H, W), mode="trilinear", align_corners=False)
        out = x + self.gamma * out

        should_dump = save_dir is not None
        need_info = should_dump
        if not need_info:
            return out

        K, h = self.num_points, self.num_heads
        offs = offsets.view(B, h, K, 3, Da, Ha, Wa).permute(0, 1, 2, 4, 5, 6, 3).contiguous()
        sample_locs = (ref_points.unsqueeze(2) + offs * 0.5).contiguous()  # (B, h, K, Da, Ha, Wa, 3)

        if should_dump:
            os.makedirs(save_dir, exist_ok=True)
            info0 = {
                "attn_weights": attn_5d[0:1].detach().cpu(),
                "sample_locs": sample_locs[0:1].detach().cpu(),
            }
            heatmap0 = self.splat_attention_to_volume(info0, out_volume=(D, H, W), avg_heads=True)
            heatmap_np = heatmap0[0].numpy()

            base = f"epoch_{int(self.epoch_id or 0):03d}_last_step"
            np.save(os.path.join(save_dir, f"{base}_heatmap.npy"), heatmap_np)

        return out

    # ---------- attention heatmap helpers ----------

    @staticmethod
    def splat_attention_to_volume(info: dict, out_volume: tuple, avg_heads: bool = True) -> torch.Tensor:
        """Bilinearly splat the K weighted sample locations onto a
        (B, D, H, W) heatmap. Vectorized across the batch (no Python loop).
        """
        attn_5d = info["attn_weights"]  # (B, h, Da, Ha, Wa, K)
        sample_locs = info["sample_locs"]  # (B, h, K, Da, Ha, Wa, 3)
        B, Da, Ha, Wa, K = attn_5d.shape[0], *attn_5d.shape[2:]
        D, H, W = out_volume

        if avg_heads:
            attn_h = attn_5d.mean(dim=1)  # (B, Da, Ha, Wa, K)
            locs_h = sample_locs.mean(dim=1)  # (B, K, Da, Ha, Wa, 3)
        else:
            attn_h = attn_5d
            locs_h = sample_locs

        n_anchors = Da * Ha * Wa
        attn_flat = attn_h.reshape(B, n_anchors, K)  # (B, N, K)
        locs_flat = locs_h.permute(0, 2, 3, 4, 1, 5).reshape(B, n_anchors, K, 3) if avg_heads else locs_h

        def _to_vox(norm, size):
            return (norm + 1.0) * (size - 1) / 2.0

        vx = _to_vox(locs_flat[..., 2], W)  # (B, N, K)
        vy = _to_vox(locs_flat[..., 1], H)
        vz = _to_vox(locs_flat[..., 0], D)

        x0 = torch.floor(vx).long().clamp(0, W - 1)
        y0 = torch.floor(vy).long().clamp(0, H - 1)
        z0 = torch.floor(vz).long().clamp(0, D - 1)
        x1, y1, z1 = (x0 + 1).clamp(0, W - 1), (y0 + 1).clamp(0, H - 1), (z0 + 1).clamp(0, D - 1)

        wx1, wy1, wz1 = vx - x0.float(), vy - y0.float(), vz - z0.float()
        wx0, wy0, wz0 = 1.0 - wx1, 1.0 - wy1, 1.0 - wz1

        batch_idx = torch.arange(B, device=attn_flat.device).view(B, 1, 1).expand(B, n_anchors, K)
        heatmap = torch.zeros(B, D, H, W, dtype=torch.float32, device="cpu").view(-1)

        def _add(w, xi, yi, zi):
            idx = (
                batch_idx.reshape(-1) * (D * H * W)
                + zi.reshape(-1) * (H * W)
                + yi.reshape(-1) * W
                + xi.reshape(-1)
            )
            heatmap.index_add_(0, idx, w.reshape(-1))

        for wx, xi in ((wx0, x0), (wx1, x1)):
            for wy, yi in ((wy0, y0), (wy1, y1)):
                for wz, zi in ((wz0, z0), (wz1, z1)):
                    _add((wx * wy * wz) * attn_flat, xi, yi, zi)

        return heatmap.view(B, D, H, W)


if __name__ == "__main__":
    import time
    import psutil

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, C, D, H, W = 2, 32, 16, 32, 32
    num_heads, num_points = 4, 4
    n_iters = 20

    print(f"device: {device}")
    print(f"input : (B={B}, C={C}, D={D}, H={H}, W={W}), heads={num_heads}, points={num_points}")

    model = DeformableAttention3d(
        dim=C, num_heads=num_heads, num_points=num_points,
        spatial_reduce=2, anchor_stride=1,
    ).to(device)
    x = torch.randn(B, C, D, H, W, device=device)
    model.set_epoch(32)  # for heatmap filename tagging

    # correctness
    out = model(x)
    assert out.shape == x.shape
    print(f"[ok] forward shape {tuple(out.shape)}")

    # ---- timing + CPU/GPU usage: plain forward (no dumping) ----
    proc = psutil.Process()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    cpu_before = proc.cpu_times()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        out = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    cpu_after = proc.cpu_times()

    plain_time = (t1 - t0) / n_iters
    cpu_time = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)
    print(f"\n[plain forward]   avg {plain_time * 1000:.2f} ms/iter over {n_iters} iters")
    print(f"                  cpu time used: {cpu_time:.3f} s total")
    if device.type == "cuda":
        print(f"                  gpu peak mem: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")

    # ---- train.py-style loop: save_dir=None every step, real path only on last step ----
    dump_dir = "E:\\MyProjects\\Gsoc_2026_Official\\temp\\deform_attn_dump"
    model_dump = DeformableAttention3d(
        dim=C, num_heads=num_heads, num_points=num_points,
        spatial_reduce=2, anchor_stride=1,
    ).to(device)
    model_dump.set_epoch(32)  # for heatmap filename tagging


    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    cpu_before = proc.cpu_times()
    t0 = time.perf_counter()
    for i in range(n_iters):
        is_last = i == n_iters - 1
        _ = model_dump(x, save_dir=dump_dir if is_last else None)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    cpu_after = proc.cpu_times()

    dump_time = (t1 - t0) / n_iters
    cpu_time_dump = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)
    print(f"\n[train.py-style]  avg {dump_time * 1000:.2f} ms/iter over {n_iters} iters "
          f"(save_dir=None except the last step)")
    print(f"                  cpu time used: {cpu_time_dump:.3f} s total")
    if device.type == "cuda":
        print(f"                  gpu peak mem: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")

    assert os.path.exists(os.path.join(dump_dir, f"epoch_000_step_0_heatmap.npy"))
    print("\n[ok] heatmap dumped only on the step where save_dir was passed")
    print(f"\nslowdown factor from dump path (amortized): {dump_time / plain_time:.2f}x")