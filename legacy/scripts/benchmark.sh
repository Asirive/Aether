#!/usr/bin/env bash
set -euo pipefail

echo "WARNING: This script is for Linux dual-boot." 1>&2
echo "For Windows native, use scripts/benchmark.ps1" 1>&2

cd "$(dirname "$0")/.."

export PATH=/opt/rocm/bin:$PATH
export HSA_OVERRIDE_GFX_VERSION=11.0.0

python3 - <<'PY'
import time

import torch

if not torch.cuda.is_available():
    print("FAIL: CUDA/ROCm not available")
    raise SystemExit(1)

device = torch.device("cuda")
dtype = torch.bfloat16

print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Torch version: {torch.__version__}")
print(f"Dtype: {dtype}")
print()

SEQ = 2048
D = 768
N_ITERS = 100
WARMUP = 10

A = torch.randn(D, D, device=device, dtype=dtype)
B = torch.randn(SEQ, D, device=device, dtype=dtype)

for _ in range(WARMUP):
    C = B @ A
torch.cuda.synchronize()

start = time.perf_counter()
for _ in range(N_ITERS):
    C = B @ A
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

flops_per_iter = 2 * SEQ * D * D
total_flops = flops_per_iter * N_ITERS
tflops = total_flops / elapsed / 1e12

print(f"Matrix multiply: {SEQ}x{D} @ {D}x{D}")
print(f"FLOPs per iter: {flops_per_iter/1e9:.2f} GFLOPs")
print(f"Sustained throughput: {tflops:.2f} TFLOPS bf16")
print()

if tflops < 8.0:
    print(f"FAIL: throughput {tflops:.2f} TFLOPS below 8.0 TFLOPS threshold")
    raise SystemExit(1)
else:
    print(f"PASS: throughput {tflops:.2f} TFLOPS meets >= 8 TFLOPS requirement")
PY
