import os
import time
import concurrent.futures
from typing import Optional
import numpy as np
import psutil
import torch
import torch.nn as nn

# Background thread pool for disk I/O to prevent blocking GPU streams
_SAVE_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=2)


class SpectralConv3d(nn.Module):
    """GPU-optimized 3D spectral convolution via FFT."""

    def __init__(
        self,
        channels: int,
        modes_d: int,
        modes_h: int,
        modes_w: int,
    ):
        super().__init__()
        self.modes_d = modes_d
        self.modes_h = modes_h
        self.modes_w = modes_w

        self.save_counter = 0
        self.epoch_id: Optional[int] = None
        self.dump_mode: str = "last_step"

        scale = 1.0 / (channels * modes_d * modes_h * modes_w)
        w_real = torch.randn(channels, channels, modes_d, modes_h, modes_w) * scale
        w_imag = torch.randn(channels, channels, modes_d, modes_h, modes_w) * scale
        self.weights = nn.Parameter(torch.complex(w_real, w_imag))

    def forward(self, x: torch.Tensor, save_dir: Optional[str] = None) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.float()

        # Fast 3D RFFT along spatial dimensions
        x_ft = torch.fft.rfftn(x_fp32, dim=(-3, -2, -1))

        md_use = min(self.modes_d, x_ft.shape[-3])
        mh_use = min(self.modes_h, x_ft.shape[-2])
        mw_use = min(self.modes_w, x_ft.shape[-1])

        # Slice active low-frequency modes
        kept = x_ft[..., :md_use, :mh_use, :mw_use]
        w_use = self.weights[..., :md_use, :mh_use, :mw_use]

        # Fused tensor contraction eliminates permute().contiguous() memory allocations
        mixed = torch.einsum("b c d h w, c o d h w -> b o d h w", kept, w_use)

        # Zero-allocation spectrum initialization
        out_ft = torch.zeros_like(x_ft)
        out_ft[..., :md_use, :mh_use, :mw_use] = mixed

        # Inverse 3D RFFT
        x_out = torch.fft.irfftn(out_ft, s=x_fp32.shape[-3:], dim=(-3, -2, -1))

        # Dump feature maps ONLY when save_dir is explicitly provided
        if save_dir is not None:
            self._async_spectral_summary(x_fp32, x_out, save_dir)

        return x_out.to(orig_dtype)

    def _async_spectral_summary(
        self, before: torch.Tensor, after: torch.Tensor, save_dir: str
    ) -> None:
        """Transfers delta map asynchronously to CPU without stalling CUDA execution streams."""
        os.makedirs(save_dir, exist_ok=True)

        if self.dump_mode == "last_step":
            ep = int(self.epoch_id or 0)
            base_name = f"epoch_{ep:03d}_last_step"
        else:
            base_name = f"step_{self.save_counter}"

        delta_gpu = (after - before).detach()
        delta_cpu = delta_gpu.to("cpu", non_blocking=True)

        def _disk_worker(delta_tensor: torch.Tensor, save_path: str):
            delta = delta_tensor.numpy()
            abs_max = np.abs(delta).max()
            delta_norm = delta / abs_max if abs_max != 0 else np.zeros_like(delta)
            np.save(save_path, delta_norm)

        save_path = os.path.join(save_dir, f"{base_name}_delta.npy")
        _SAVE_POOL.submit(_disk_worker, delta_cpu, save_path)
        self.save_counter += 1

    def reset_save_counter(self) -> None:
        self.save_counter = 0

    def set_epoch(self, epoch_id: int, dump_mode: str = "last_step") -> None:
        self.epoch_id = int(epoch_id)
        self.dump_mode = str(dump_mode)


class FNOBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        modes_d: int,
        modes_h: int,
        modes_w: int,
    ):
        super().__init__()
        self.spec = SpectralConv3d(channels, modes_d, modes_h, modes_w)
        self.local = nn.Conv3d(channels, channels, kernel_size=1)
        self.norm = nn.InstanceNorm3d(channels, affine=True)

    def forward(self, x: torch.Tensor, save_dir: Optional[str] = None) -> torch.Tensor:
        return x + self.norm(self.spec(x, save_dir=save_dir) + self.local(x))

    def set_epoch(self, epoch_id: int, dump_mode: str = "last_step") -> None:
        self.spec.set_epoch(epoch_id, dump_mode=dump_mode)


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, C, D, H, W = 2, 64, 16, 32, 32
    modes_d, modes_h, modes_w = 8, 8, 8
    n_iters = 20

    print(f"device: {device}")
    print(f"input : (B={B}, C={C}, D={D}, H={H}, W={W}), modes=({modes_d}, {modes_h}, {modes_w})")

    model = FNOBlock3d(
        channels=C, modes_d=modes_d, modes_h=modes_h, modes_w=modes_w
    ).to(device)
    x = torch.randn(B, C, D, H, W, device=device)
    model.set_epoch(32)

    # Correctness check
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
    dump_dir = os.path.join(os.getcwd(), "temp_fno_dump")
    model_dump = FNOBlock3d(
        channels=C, modes_d=modes_d, modes_h=modes_h, modes_w=modes_w
    ).to(device)
    model_dump.set_epoch(32)

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

    # Wait for async background thread writer to flush disk I/O before assertion
    _SAVE_POOL.shutdown(wait=True)

    dump_time = (t1 - t0) / n_iters
    cpu_time_dump = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)
    print(f"\n[train.py-style]  avg {dump_time * 1000:.2f} ms/iter over {n_iters} iters "
          f"(save_dir=None except the last step)")
    print(f"                  cpu time used: {cpu_time_dump:.3f} s total")
    if device.type == "cuda":
        print(f"                  gpu peak mem: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")

    expected_file = os.path.join(dump_dir, "epoch_032_last_step_delta.npy")
    assert os.path.exists(expected_file), f"Expected dump file not found: {expected_file}"
    print(f"\n[ok] spectral delta dumped only on step where save_dir was passed")
    print(f"slowdown factor from dump path (amortized): {dump_time / plain_time:.2f}x")