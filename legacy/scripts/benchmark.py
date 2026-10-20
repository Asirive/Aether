from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aether.utils.device import describe_device, select_device


def main() -> None:
    if not torch.cuda.is_available():
        print("FAIL: CUDA/ROCm not available")
        raise SystemExit(1)

    device = select_device()
    dtype = torch.bfloat16

    print(f"Device: {describe_device(device)}")
    print(f"VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")
    print()

    seq = 2048
    d = 768
    n_iters = 100
    warmup = 20

    a = torch.randn(d, d, device=device, dtype=dtype)
    b = torch.randn(seq, d, device=device, dtype=dtype)
    torch.cuda.synchronize()

    for _ in range(warmup):
        c = b @ a
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        c = b @ a
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    flops = 2 * seq * d * d * n_iters
    tflops = flops / elapsed / 1e12

    print(f"Matrix multiply: {seq}x{d} @ {d}x{d}")
    print(f"Sustained throughput: {tflops:.2f} TFLOPS bf16")
    print()

    if tflops < 8.0:
        print(f"FAIL: {tflops:.2f} TFLOPS below 8.0 TFLOPS threshold")
        raise SystemExit(1)
    print(f"PASS: {tflops:.2f} TFLOPS meets >= 8 TFLOPS requirement")


if __name__ == "__main__":
    main()