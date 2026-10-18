from __future__ import annotations

import math

import torch
import torch.nn as nn


_NO_DECAY_SUBSTRINGS = ("a_log", "mask_logits", "mix_gate", "mixer.gate", "route_proj", "embedding")


def _split_param_groups(
    model: nn.Module,
    weight_decay: float,
    embed_lr_scale: float = 1.0,
) -> list[dict]:
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    embed_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lowered = name.lower()
        if embed_lr_scale != 1.0 and "token_embedding" in lowered:
            embed_params.append(param)
        elif (
            param.ndim < 2
            or "norm" in lowered
            or lowered.endswith(".bias")
            or any(s in lowered for s in _NO_DECAY_SUBSTRINGS)
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    if embed_params:
        groups.append({"params": embed_params, "weight_decay": 0.0, "lr_scale": embed_lr_scale})
    return groups


def build_optimizer(
    model: nn.Module,
    lr: float = 5e-4,
    betas: tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.1,
    fused: bool = True,
    embed_lr_scale: float = 1.0,
) -> torch.optim.AdamW:
    param_groups = _split_param_groups(model, weight_decay, embed_lr_scale)
    for group in param_groups:
        group["lr"] = lr * group.pop("lr_scale", 1.0)
    return torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=betas,
        fused=fused,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 1000,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def linear_ramp(step: int, total_steps: int, warmup_frac: float) -> float:
    ramp_steps = max(1, int(warmup_frac * total_steps))
    return min(1.0, step / ramp_steps)