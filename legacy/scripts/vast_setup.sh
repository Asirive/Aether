#!/usr/bin/env bash
# One-time setup + smoke test on a rented CUDA GPU (Vast.ai, RunPod, ...).
# Run from the repo's legacy/ directory:   bash scripts/vast_setup.sh
# Re-running is safe. Exits non-zero at the first failed check.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "== PyTorch with a kernel for this GPU =="
python3 - <<'EOF' || NEED_TORCH=1
import torch, sys
ok = torch.cuda.is_available()
cap = torch.cuda.get_device_capability() if ok else None
arch = f"sm_{cap[0]}{cap[1]}" if cap else None
print("torch", torch.__version__, "cuda", torch.version.cuda, "device", arch, "built for", torch.cuda.get_arch_list())
if not ok or arch not in torch.cuda.get_arch_list():
    sys.exit(1)
(torch.randn(256, 256, device="cuda") @ torch.randn(256, 256, device="cuda")).sum().item()
EOF
if [ "${NEED_TORCH:-0}" = 1 ]; then
  echo "Installed torch has no kernel for this GPU; installing a CUDA 12.8 build (needed for RTX 50xx)."
  pip install -q --upgrade torch --index-url https://download.pytorch.org/whl/cu128
  python3 -c "import torch; (torch.randn(8,8,device='cuda')@torch.randn(8,8,device='cuda')).sum().item(); print('torch', torch.__version__, 'OK')"
fi

echo "== Python packages =="
pip install -q "transformers>=4.51" "datasets>=3.0" pyyaml einops safetensors pytest

echo "== Unit tests =="
python3 -m pytest -q tests/test_v2_components.py tests/test_distill.py tests/test_baseline.py

echo "== Teacher and data reachable =="
python3 - <<'EOF'
from transformers import AutoTokenizer
from datasets import load_dataset
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
row = next(iter(load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)))
print("tokenizer vocab", len(tok), "| first doc tokens", len(tok(row["text"])["input_ids"]))
EOF

echo "== Smoke run: 30 steps per student (throughput + memory) =="
for s in aether_v2 vanilla; do
  rm -rf "runs/smoke_$s"
  python3 scripts/distill.py --student "$s" --out "runs/smoke_$s" --max-steps 30 --log-every 10 \
      --eval-every 30 --ckpt-every 1000000 --eval-rows 16 2>&1 | grep -E '"step": 30|EVAL|Error|error' || true
done
python3 scripts/compare_kd.py runs/smoke_aether_v2 runs/smoke_vanilla
echo "Setup OK. Next: see docs/vast_rehearsal.md for the full runs."
