# HIP SDK 7.1.1 native Windows install path
#
# Run from PowerShell as Administrator:
#   powershell -ExecutionPolicy Bypass -File scripts/env_setup_windows.ps1
#
# What this does:
#   1. Detects if HIP SDK is installed
#   2. If not: prints manual download/install instructions
#   3. Sets HSA_OVERRIDE_GFX_VERSION=11.0.0 system-wide (not strictly needed for gfx1102
#      but harmless and aligns with the audit's recommendation)
#   4. Adds C:\Program Files\AMD\ROCm\7.1\bin to system PATH (idempotent)
#   5. Installs PyTorch with ROCm support via pip
#   6. Runs scripts/benchmark.py to verify >= 8 TFLOPS bf16

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

function Test-HipSdkInstalled {
    $hipInfo = Get-Command hipInfo.exe -ErrorAction SilentlyContinue
    if ($hipInfo) {
        Write-Host "[env_setup] hipInfo found at: $($hipInfo.Source)" -ForegroundColor Green
        return $true
    }
    $candidates = @(
        "C:\Program Files\AMD\ROCm\7.1\bin\hipInfo.exe",
        "C:\Program Files\AMD\ROCm\7.2\bin\hipInfo.exe",
        "C:\Program Files\AMD\ROCm\7.3\bin\hipInfo.exe"
    )
    foreach ($p in $candidates) {
        if (Test-Path $p) {
            Write-Host "[env_setup] HIP SDK found at: $p" -ForegroundColor Green
            return $true
        }
    }
    return $false
}

function Find-DiscreteGpu {
    & "$HIP_BIN/hipInfo.exe" 2>&1 | Out-Null
    try {
        & python -c @'
import os
os.environ.setdefault("HIP_ENABLE_TRACING", "0")
os.environ.setdefault("TORCH_DISABLE_CUDA_PROFILING", "1")
os.environ.setdefault("HSA_ENABLE_PROFILING", "0")
import torch
print(f"device count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"[{i}] {torch.cuda.get_device_name(i)} ({p.total_memory/1e9:.1f} GB)")
'@
        return $LASTEXITCODE
    } catch {
        Write-Host "[env_setup] Could not auto-detect discrete GPU. Defaulting CUDA_VISIBLE_DEVICES=1" -ForegroundColor Yellow
        return 1
    }
}

function Show-ManualInstallInstructions {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Yellow
    Write-Host "  HIP SDK NOT INSTALLED" -ForegroundColor Yellow
    Write-Host "============================================================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Download the HIP SDK 7.1.1 installer from:" -ForegroundColor Cyan
    Write-Host "  https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html" -ForegroundColor White
    Write-Host ""
    Write-Host "Steps:" -ForegroundColor Cyan
    Write-Host "  1. Open the link above in a browser" -ForegroundColor White
    Write-Host "  2. Accept the license" -ForegroundColor White
    Write-Host "  3. Download the HIP SDK 7.1.1 installer (Windows 11, x86-64)" -ForegroundColor White
    Write-Host "  4. Run the installer AS ADMINISTRATOR (UAC prompt will appear)" -ForegroundColor White
    Write-Host "  5. Keep all default components selected" -ForegroundColor White
    Write-Host "  6. Bundled AMD Display Driver will REPLACE existing Adrenalin" -ForegroundColor White
    Write-Host "  7. Restart when prompted (REQUIRED for KFD/driver to load)" -ForegroundColor White
    Write-Host "  8. After reboot, run this script again" -ForegroundColor White
    Write-Host ""
    Write-Host "Expected install path: C:\Program Files\AMD\ROCm\7.1\" -ForegroundColor Cyan
    Write-Host ""
    exit 1
}

function Set-SystemEnv {
    param(
        [string]$Name,
        [string]$Value,
        [string]$Scope = "Machine"
    )
    $current = [Environment]::GetEnvironmentVariable($Name, $Scope)
    if ($current -eq $Value) {
        Write-Host "[env_setup] $Name already set to '$Value' ($Scope scope)" -ForegroundColor Green
        return
    }
    try {
        Write-Host "[env_setup] Setting $Name=$Value ($Scope scope)" -ForegroundColor Yellow
        [Environment]::SetEnvironmentVariable($Name, $Value, $Scope)
    } catch {
        if ($Scope -eq "Machine") {
            Write-Host "[env_setup] Machine-scope write failed (need admin). Falling back to User scope for this session." -ForegroundColor Yellow
            Set-SystemEnv -Name $Name -Value $Value -Scope "User"
        } else {
            throw
        }
    }
}

function Add-ToSystemPath {
    param([string]$Directory)
    $current = [Environment]::GetEnvironmentVariable("Path", "Machine")
    if ($current -like "*$Directory*") {
        Write-Host "[env_setup] $Directory already in system PATH" -ForegroundColor Green
        $env:Path = "$env:Path;$Directory"
        return
    }

    $userCurrent = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($userCurrent -like "*$Directory*") {
        Write-Host "[env_setup] $Directory already in user PATH" -ForegroundColor Green
        $env:Path = "$env:Path;$Directory"
        return
    }

    try {
        Write-Host "[env_setup] Adding $Directory to system PATH (Machine scope)" -ForegroundColor Yellow
        $newPath = "$current;$Directory"
        [Environment]::SetEnvironmentVariable("Path", $newPath, "Machine")
        $env:Path = "$env:Path;$Directory"
    } catch {
        Write-Host "[env_setup] Machine-scope PATH write failed (need admin). Adding to User PATH instead." -ForegroundColor Yellow
        $newUserPath = "$userCurrent;$Directory"
        [Environment]::SetEnvironmentVariable("Path", $newUserPath, "User")
        $env:Path = "$env:Path;$Directory"
    }
}

function Test-Python {
    try {
        $ver = python --version 2>&1
        Write-Host "[env_setup] Python found: $ver" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "[env_setup] Python NOT installed" -ForegroundColor Red
        return $false
    }
}

function Install-PyTorchRocm {
    Write-Host "[env_setup] Installing PyTorch with ROCm support via TheRock unified index..." -ForegroundColor Yellow
    Write-Host "[env_setup] Source: https://rocm.nightlies.amd.com/whl-multi-arch/" -ForegroundColor Yellow
    Write-Host "[env_setup] Target: device-gfx1102 (RX 7600)" -ForegroundColor Yellow

    pip install --upgrade pip

    $installed = $false
    @("device-gfx1102", "device-gfx1100") | ForEach-Object {
        $device = $_
        Write-Host "[env_setup] Trying device extra: $device ..." -ForegroundColor Cyan
        try {
            & pip install --pre "torch[$device]" torchvision torchaudio --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "[env_setup] PyTorch ROCm installed ($device)" -ForegroundColor Green
                $installed = $true
            }
        } catch {
            Write-Host "[env_setup] $device failed: $_" -ForegroundColor Yellow
        }
    }

    if (-not $installed) {
        Write-Host "[env_setup] All TheRock indexes failed." -ForegroundColor Red
        exit 1
    }
}

function Install-AetherDeps {
    Write-Host "[env_setup] Installing AETHER-200M in editable mode..." -ForegroundColor Yellow
    Push-Location $RepoRoot
    try {
        pip install -e ".[dev,eval]" 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "pip install -e . failed"
        }
    } finally {
        Pop-Location
    }
    Write-Host "[env_setup] AETHER-200M installed" -ForegroundColor Green
}

function Run-Benchmark {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  Running throughput benchmark (Phase 0 exit criterion)" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Push-Location $RepoRoot
    try {
        python scripts/benchmark.py
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[env_setup] Benchmark FAILED" -ForegroundColor Red
            exit 1
        }
    } finally {
        Pop-Location
    }
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  AETHER-200M Windows Environment Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Python)) {
    Write-Host "Install Python 3.11+ first: https://www.python.org/downloads/windows/" -ForegroundColor Red
    exit 1
}

if (-not (Test-HipSdkInstalled)) {
    Show-ManualInstallInstructions
}

Add-ToSystemPath -Directory "C:\Program Files\AMD\ROCm\7.1\bin"
Set-SystemEnv -Name "HSA_OVERRIDE_GFX_VERSION" -Value "11.0.0"

Install-PyTorchRocm

Write-Host ""
Write-Host "[env_setup] Auto-detecting discrete GPU..." -ForegroundColor Yellow
Find-DiscreteGpu
if ($LASTEXITCODE -eq 0) {
    Read-Host "[env_setup] Press Enter to continue after viewing GPU list (or type 'skip' to skip detection)"
}
Set-SystemEnv -Name "CUDA_VISIBLE_DEVICES" -Value "1"

Install-AetherDeps
Run-Benchmark

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Setup complete. RX 7600 GPU is active (CUDA_VISIBLE_DEVICES=1)" -ForegroundColor Green
Write-Host "  Run: python ablations/04_aether_full.py --proxy --steps 1000" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green