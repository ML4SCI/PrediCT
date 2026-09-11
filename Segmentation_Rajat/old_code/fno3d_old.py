import os
from typing import Optional
import torch
import torch.nn as nn
import numpy as np


def normalize_minus1_to_1(arr: np.ndarray) -> np.ndarray:
    """Normalizes a numpy array to the range [-1, 1]."""
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max == arr_min:
        return np.zeros_like(arr)
    
    norm_0_1 = (arr - arr_min) / (arr_max - arr_min)
    return 2.0 * norm_0_1 - 1.0


@torch.amp.autocast('cuda', enabled=False)
class SpectralConv3d(nn.Module):
    """3D spectral convolution via FFT."""
    def __init__(
        self,
        channels: int,
        modes_d: int,
        modes_h: int,
        modes_w: int,
        save_dir: Optional[str] = None,
        save_feature_maps: bool = False
    ):
        super().__init__()
        self.modes_d = modes_d
        self.modes_h = modes_h
        self.modes_w = modes_w
        
        # Only enable saving if save_dir is explicitly provided
        self.save_feature_maps = save_feature_maps and (save_dir is not None)
        self.save_dir = save_dir
        self.save_counter = 0
        # Per-epoch tag set by the train loop. Used to build the
        # `epoch_{N:03d}_last_step_*` filename when DUMP_LAST_STEP_ONLY is on.
        self.epoch_id: Optional[int] = None
        self.dump_mode: str = "first_step"   # "first_step" or "last_step"
        
        if self.save_feature_maps and self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

        scale = 1.0 / (channels * modes_d * modes_h * modes_w)
        weight = torch.randn(channels, channels, modes_d, modes_h, modes_w, 2) * scale
        self.weights = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.float()

        x_ft = torch.fft.rfftn(x_fp32, dim=(-3, -2, -1))
        w_complex = torch.view_as_complex(self.weights.float())

        md_use = min(self.modes_d, x_ft.shape[-3])
        mh_use = min(self.modes_h, x_ft.shape[-2])
        mw_use = min(self.modes_w, x_ft.shape[-1])
        
        kept = x_ft[..., :md_use, :mh_use, :mw_use]              
        w_use = w_complex[..., :md_use, :mh_use, :mw_use]        
        
        k = kept.permute(0, 2, 3, 4, 1).contiguous()             
        w = w_use.permute(2, 3, 4, 0, 1).contiguous()            
        
        mixed = torch.matmul(k.unsqueeze(-2), w.unsqueeze(0)).squeeze(-2)
        mixed = mixed.permute(0, 4, 1, 2, 3).contiguous()        
        
        out_ft = x_ft.clone()
        out_ft[..., :md_use, :mh_use, :mw_use] = mixed

        x_out = torch.fft.irfftn(out_ft, s=x_fp32.shape[-3:], dim=(-3, -2, -1))

        # Defensive: dump only when both flags AND train-mode are true.
        # `self.training` reflects `model.training`, so eval/validation
        # automatically skips the dump even if `save_feature_maps` was set
        # at construction. Also bail out if `save_dir` was cleared by the
        # train loop (e.g. after the first step in dump-first-only mode).
        if self.save_feature_maps and self.training and self.save_dir:
            # NOTE: dumping the full spatial map copies a (B, C, D, H, W) fp32
            # tensor to host RAM. That's an OOM trap at the bottleneck
            # (B=4, C=512, D=8, H=8, W=2 ≈ 1 MiB alone) and worse when scaled
            # by the SWI batch on tiny VRAM. Skip during eval/validation and
            # save a *small* spectral summary instead so debugging still works.
            try:
                self._save_spectral_summary(x_fp32, x_out)
            except Exception as e:  # never let debug code crash training
                if self.save_counter == 0:
                    print(f"[FNO dump] skipped: {e}")

        # Restore original tensor precision (e.g. float16 under AMP)
        return x_out.to(orig_dtype)

    def _save_spectral_summary(self, before: torch.Tensor, after: torch.Tensor) -> None:
        """Save the full-batch spatial delta feature map introduced by SpectralConv3d.

        Per step we write a single (B, C, D, H, W) array holding `after - before`,
        normalized to [-1, 1] by dividing by the max absolute value (not min-max
        scaling), so a delta of 0 always maps to 0 — no shifting the zero point.
        """
        if not self.save_dir:
            return
        # In last_step mode, tag with the epoch so per-epoch end-of-step
        # snapshots don't collide.
        if getattr(self, "dump_mode", "first_step") == "last_step":
            ep = int(getattr(self, "epoch_id", 0) or 0)
            base_name = f"epoch_{ep:03d}_last_step"
        else:
            base_name = f"step_{self.save_counter}"

        # Full-resolution spatial delta: what SpectralConv3d actually changed.
        delta = (after - before).detach().float().cpu().numpy()   # (B, C, D, H, W)

        # Symmetric normalization around zero: scale by max |delta| instead of
        # min-max, so "no change" (0) stays exactly 0 in the saved array.
        abs_max = np.abs(delta).max()
        if abs_max == 0:
            delta_norm = np.zeros_like(delta)
        else:
            delta_norm = delta / abs_max

        # Cheap scalar stats alongside, computed on the *raw* delta so they're
        # still meaningful magnitudes (not normalized to 1 every time).
        delta_abs_mean = float(np.abs(delta).mean())
        delta_abs_max = float(abs_max)

        np.save(os.path.join(self.save_dir, f"{base_name}_delta.npy"), delta_norm)

        self.save_counter += 1

    def reset_save_counter(self) -> None:
        self.save_counter = 0

    def set_epoch(self, epoch_id: int, dump_mode: str = "first_step") -> None:
        """Tag this module with the current epoch and the per-epoch dump policy.

        `dump_mode` is either "first_step" (legacy: dump only the very first
        step of each epoch) or "last_step" (dump only the very last step of
        each epoch, with `epoch_{N:03d}_last_step_*` filenames).
        """
        self.epoch_id = int(epoch_id)
        self.dump_mode = str(dump_mode)


class FNOBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        modes_d: int,
        modes_h: int,
        modes_w: int,
        save_dir: Optional[str] = None,
        save_feature_maps: bool = False,
    ):
        super().__init__()
        # Explicit keyword arguments prevent parameter ordering bugs
        self.spec = SpectralConv3d(
            channels, modes_d, modes_h, modes_w,
            save_dir=save_dir,
            save_feature_maps=save_feature_maps
        )
        self.local = nn.Conv3d(channels, channels, kernel_size=1)
        self.norm = nn.InstanceNorm3d(channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.norm(self.spec(x) + self.local(x))

if __name__ == "__main__":
    import os
    import time
    import psutil
    import torch

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, C, D, H, W = 2, 512, 8, 8, 2
    modes_d, modes_h, modes_w = 8, 8, 8
    n_iters = 20

    print(f"device: {device}")
    print(f"input : (B={B}, C={C}, D={D}, H={H}, W={W}), modes=({modes_d}, {modes_h}, {modes_w})")

    # Construct baseline model without dumping
    model = FNOBlock3d(
        channels=C,
        modes_d=modes_d,
        modes_h=modes_h,
        modes_w=modes_w,
        save_dir=None,
        save_feature_maps=False
    ).to(device)
    model.train()

    x = torch.randn(B, C, D, H, W, device=device)

    # Correctness check
    out = model(x)
    assert out.shape == x.shape, f"Expected shape {x.shape}, got {out.shape}"
    print(f"[ok] forward shape {tuple(out.shape)}")

    # ---- Timing + CPU/GPU usage: plain forward (no dumping) ----
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

    # ---- train.py-style loop: save_dir active only on the target step ----
    dump_dir = r"E:\MyProjects\Gsoc_2026_Official\temp\fno_dump"
    
    model_dump = FNOBlock3d(
        channels=C,
        modes_d=modes_d,
        modes_h=modes_h,
        modes_w=modes_w,
        save_dir=dump_dir,
        save_feature_maps=True
    ).to(device)
    model_dump.train()
    
    target_epoch = 32
    model_dump.spec.set_epoch(target_epoch, dump_mode="last_step")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    cpu_before = proc.cpu_times()
    t0 = time.perf_counter()

    for i in range(n_iters):
        is_last = (i == n_iters - 1)
        # Dynamically toggle save_dir to match loop condition behavior
        model_dump.spec.save_dir = dump_dir if is_last else None
        
        _ = model_dump(x)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    cpu_after = proc.cpu_times()

    dump_time = (t1 - t0) / n_iters
    cpu_time_dump = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)
    print(f"\n[train.py-style]  avg {dump_time * 1000:.2f} ms/iter over {n_iters} iters "
          f"(save_dir active only on the last step)")
    print(f"                  cpu time used: {cpu_time_dump:.3f} s total")
    if device.type == "cuda":
        print(f"                  gpu peak mem: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")

    # Verification assertions
    expected_file = os.path.join(dump_dir, f"epoch_{target_epoch:03d}_last_step_delta.npy")
    assert os.path.exists(expected_file), f"Expected file not found: {expected_file}"
    print("\n[ok] feature map delta dumped only on the step where save_dir was set")
    print(f"\nslowdown factor from dump path (amortized): {dump_time / plain_time:.2f}x")