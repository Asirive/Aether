#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CHECKPOINT="${CHECKPOINT:-runs/aether_200m_sft/final}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/aether_200m_dpo}"
DATASET="${DATASET:-HuggingFaceH4/ultrafeedback_binarized}"

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"

python3 - <<PY
from pathlib import Path
import torch
from aether.model import AetherConfig, AetherForCausalLM
from aether.training.tokenizer import build_tokenizer
from aether.post_train.dpo import run_dpo

path = Path("${CHECKPOINT}")
if path.is_dir():
    raise SystemExit("pass the SFT .pt checkpoint via CHECKPOINT=... (e.g. runs/aether_200m_sft/checkpoint-XXX.pt)")
ckpt = torch.load(str(path), map_location="cpu")
cfg_dict = ckpt.get("metadata", {}).get("model_config")
if cfg_dict is None:
    raise SystemExit("checkpoint has no metadata.model_config")
config = AetherConfig.from_dict({k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg_dict.items()})
model = AetherForCausalLM(config)
model.load_state_dict(ckpt["model_state_dict"])

tokenizer = build_tokenizer()
run_dpo(model, tokenizer, "${OUTPUT_DIR}", dataset_name="${DATASET}")
PY