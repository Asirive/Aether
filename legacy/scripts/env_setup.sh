#!/usr/bin/env bash
set -euo pipefail

echo "=== AETHER-200M Ubuntu Linux Environment Setup ===" 1>&2
echo "" 1>&2
echo "WARNING: This script is for Linux dual-boot only." 1>&2
echo "For Windows native (recommended), use scripts/env_setup_windows.ps1" 1>&2
echo "On WSL2, ROCm has no /dev/kfd for gfx1102 -- WSL2 is NOT supported." 1>&2
echo "" 1>&2

if ! command -v rocminfo >/dev/null 2>&1; then
    UBUNTU_CODENAME=$(. /etc/os-release && echo "${UBUNTU_CODENAME}")
    echo "[env_setup] Detected Ubuntu codename: ${UBUNTU_CODENAME}" 1>&2
    echo "[env_setup] rocminfo not found. Installing ROCm 7.2..." 1>&2

    sudo apt update
    sudo apt install -y wget gnupg2

    wget -qO - https://repo.radeon.com/rocm/apt/7.2/rocm.gpg.key | sudo gpg --dearmor -o /usr/share/keyrings/rocm.gpg 2>/dev/null || true

    if [ ! -f /etc/apt/sources.list.d/rocm.list ]; then
        echo "deb [signed-by=/usr/share/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.2 ${UBUNTU_CODENAME} main" | \
            sudo tee /etc/apt/sources.list.d/rocm.list
    fi

    sudo apt update
    sudo apt install -y rocm-dev python3.11 python3.11-venv git

    echo 'export HSA_OVERRIDE_GFX_VERSION=11.0.0' >> ~/.bashrc
    echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
fi

export PATH=/opt/rocm/bin:$PATH
export HSA_OVERRIDE_GFX_VERSION=11.0.0

echo "[env_setup] Verifying ROCm..." 1>&2
if ! rocminfo 2>/dev/null | grep -i gfx; then
    echo "FAIL: rocminfo did not detect gfx. Check amdgpu kernel module." 1>&2
    exit 1
fi

echo "[env_setup] Verifying PyTorch with ROCm..." 1>&2
python3 -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available())" 2>/dev/null || {
    echo "[env_setup] Installing PyTorch ROCm wheel..." 1>&2
    python3 -m pip install --upgrade pip
    python3 -m pip install torch==2.7.0+rocm7.2 torchvision==0.22.0+rocm7.2 \
        --index-url https://download.pytorch.org/whl/rocm7.2
}

echo "[env_setup] Installing AETHER..." 1>&2
cd "$(dirname "$0")/.."
python3 -m pip install -e ".[dev,eval]"

echo "[env_setup] Done. Run 'bash scripts/benchmark.sh' to verify throughput." 1>&2
