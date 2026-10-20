from __future__ import annotations

from typing import Any

try:
    import wandb
except ImportError:
    wandb = None


def init_wandb(
    project: str,
    name: str | None = None,
    config: dict[str, Any] | None = None,
    resume: bool = False,
) -> None:
    if wandb is None:
        return
    wandb.init(project=project, name=name, config=config, resume=resume)


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    if wandb is None:
        return
    wandb.log(metrics, step=step)
