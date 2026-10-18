$env:CUDA_VISIBLE_DEVICES = "1"
$env:HSA_OVERRIDE_GFX_VERSION = "11.0.0"
$env:TORCH_DISABLE_CUDA_PROFILING = "1"
$env:HIP_ENABLE_TRACING = "0"
$env:HSA_ENABLE_PROFILING = "0"
$env:PATH = "C:\Program Files\AMD\ROCm\7.1\bin;" + $env:PATH

Set-Location "C:\Users\haziq\Documents\niga"

& python -m pytest tests/ -v 2>&1 | Tee-Object -FilePath "C:\Users\haziq\Documents\niga\test_log.txt"