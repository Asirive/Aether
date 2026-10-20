$ErrorActionPreference = "Continue"

$HIP_BIN = "C:\Program Files\AMD\ROCm\7.1\bin"
$env:PATH = $HIP_BIN + ";" + $env:PATH
$env:HSA_OVERRIDE_GFX_VERSION = "11.0.0"
$env:CUDA_VISIBLE_DEVICES = "1"
$env:TORCH_DISABLE_CUDA_PROFILING = "1"
$env:HIP_ENABLE_TRACING = "0"
$env:HSA_ENABLE_PROFILING = "0"

Set-Location "C:\Users\haziq\Documents\niga"

$benchScript = @'
import time
import torch

if not torch.cuda.is_available():
    print("FAIL: CUDA/ROCm not available")
    raise SystemExit(1)

device = torch.device("cuda")
dtype = torch.bfloat16

print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Torch: {torch.__version__}")
print(f"CUDA_VISIBLE_DEVICES: {__import__('os').environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")
print()

SEQ = 2048
D = 768
N_ITERS = 100
WARMUP = 20

A_cpu = torch.randn(D, D, dtype=torch.bfloat16)
B_cpu = torch.randn(SEQ, D, dtype=torch.bfloat16)
A = A_cpu.to(device)
B = B_cpu.to(device)
torch.cuda.synchronize()

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
print(f"Sustained throughput: {tflops:.2f} TFLOPS bf16")
print()

if tflops < 8.0:
    print(f"FAIL: {tflops:.2f} TFLOPS below 8.0 TFLOPS threshold")
    raise SystemExit(1)
else:
    print(f"PASS: {tflops:.2f} TFLOPS meets >= 8 TFLOPS requirement")
'@

Push-Location "C:\Users\haziq\Documents\niga"
try {
    python -c $benchScript
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    Pop-Location
}