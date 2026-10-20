#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/aether_200m.yaml}"

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
export TOKENIZERS_PARALLELISM=false

python3 -m aether.training.pretrain --config "${CONFIG}" "$@"