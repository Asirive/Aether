$ErrorActionPreference = "Stop"

$HIP_BIN = "C:\Program Files\AMD\ROCm\7.1\bin"
$env:PATH = $HIP_BIN + ";" + $env:PATH
$env:HSA_OVERRIDE_GFX_VERSION = "11.0.0"
$env:CUDA_VISIBLE_DEVICES = "1"
$env:TORCH_DISABLE_CUDA_PROFILING = "1"
$env:HIP_ENABLE_TRACING = "0"
$env:HSA_ENABLE_PROFILING = "0"
$env:PYTHONUNBUFFERED = "1"
$env:TOKENIZERS_PARALLELISM = "false"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Config = "configs\aether_200m.yaml"
$Resume = $null
$Smoke = $false
$Steps = $null
$SeqLen = $null
$MicroBatch = $null

$argList = @($args)
for ($i = 0; $i -lt $argList.Count; $i++) {
    $a = $argList[$i]
    if ($a -in @("-smoke", "--smoke")) {
        $Smoke = $true
        continue
    }
    $key = $a -replace '^-+', "" -replace '=.*$', ""
    $val = $null
    if ($a -match '^-+[^=]+=([^=]+)$') {
        $val = $Matches[1]
        # Rejoin a trailing bare extension token (PowerShell splits "-x=path.pt" into "-x=path" + ".pt")
        if ($i + 1 -lt $argList.Count -and $argList[$i + 1] -match '^\.[A-Za-z0-9]+$') {
            $val += $argList[$i + 1]
            $i++
        }
    } elseif ($a -match '^-+') {
        if ($i + 1 -lt $argList.Count -and $argList[$i + 1] -notmatch '^-+') {
            $val = $argList[$i + 1]
            $i++
        }
    }
    switch ($key) {
        "resume" { $Resume = $val }
        "config" { $Config = $val }
        "steps" { $Steps = $val }
        "seq-len" { $SeqLen = $val }
        "micro-batch" { $MicroBatch = $val }
        default {
            Write-Host "[pretrain_200m.ps1] Unknown arg: $a" -ForegroundColor Yellow
        }
    }
}

$cli = @("--config", $Config)
if ($Resume) { $cli += @("--resume", $Resume) }
if ($Steps) { $cli += @("--steps", $Steps) }
if ($SeqLen) { $cli += @("--seq-len", $SeqLen) }
if ($MicroBatch) { $cli += @("--micro-batch", $MicroBatch) }
if ($Smoke) { $cli += "--smoke" }

Write-Host "[pretrain_200m.ps1] Running: python -m aether.training.pretrain $($cli -join ' ')" -ForegroundColor Cyan
python -m aether.training.pretrain @cli
if ($LASTEXITCODE -ne 0) {
    Write-Host "[pretrain_200m.ps1] FAILED exit=$LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}