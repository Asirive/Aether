#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_PATH="${MODEL_PATH:-checkpoints/aether-200m/dpo/final}"
OUTPUT_PATH="${OUTPUT_PATH:-eval_results/aether-200m-instruct.json}"
TASKS="${TASKS:-hellaswag arc_easy piqa winogrande}"

python3 - <<PY
from aether.eval.harness import run_lm_eval
from pathlib import Path

tasks = "${TASKS}".split()
results = run_lm_eval(
    model_path="${MODEL_PATH}",
    tasks=tasks,
    output_path="${OUTPUT_PATH}",
)
for task, metrics in results.get("results", {}).items():
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and "acc" in k.lower():
                print(f"{task:20s} {k:30s} {v:.4f}")
PY
