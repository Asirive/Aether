from __future__ import annotations

import torch


def to_fp32(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.float32:
        return x
    return x.float()


def to_bf16(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.bfloat16:
        return x
    return x.to(torch.bfloat16)


def assert_finite(tensor: torch.Tensor, name: str = "tensor") -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")
