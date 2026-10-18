from __future__ import annotations

from pathlib import Path

import torch


FAST_BENCHMARKS: list[str] = ["hellaswag", "arc_easy", "piqa"]


def run_lm_eval(
    model_path: str | Path,
    tasks: list[str],
    batch_size: str = "auto",
    output_path: str | Path | None = None,
    num_fewshot: int | None = None,
) -> dict:
    try:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
    except ImportError as e:
        raise ImportError("Install lm-eval: pip install lm-eval") from e

    from aether.utils.export_hf import ensure_hf_export
    from aether.utils.device import select_device

    export_dir = ensure_hf_export(model_path)
    lm = HFLM(
        pretrained=str(export_dir),
        trust_remote_code=True,
        batch_size=batch_size,
        dtype=torch.bfloat16,
        device=str(select_device()),
    )

    results = simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with output_path.open("w") as f:
            json.dump(results, f, indent=2, default=str)

    return results


def run_full_eval(model_path: str | Path, output_path: str | Path | None = None) -> dict:
    all_tasks = [
        "mmlu",
        "hellaswag",
        "arc_challenge",
        "arc_easy",
        "piqa",
        "winogrande",
        "gsm8k",
        "humaneval",
        "truthfulqa_mc2",
        "ifeval",
    ]
    return run_lm_eval(model_path, tasks=all_tasks, output_path=output_path)
