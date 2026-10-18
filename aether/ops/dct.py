from __future__ import annotations

from functools import lru_cache

import torch


@lru_cache(maxsize=16)
def dct_matrix(seq_len: int, device: str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    n = torch.arange(seq_len, dtype=dtype, device=device).unsqueeze(0)
    k = torch.arange(seq_len, dtype=dtype, device=device).unsqueeze(1)
    angle = torch.pi * (2 * n + 1) * k / (2 * seq_len)
    W = torch.cos(angle)
    W[0, :] = W[0, :] * (1.0 / (2.0 ** 0.5))
    W = W * (2.0 / seq_len) ** 0.5
    return W


def dct_apply(x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    if x.dim() == 3:
        x_2d = x.reshape(-1, orig_shape[-2])
    else:
        x_2d = x
    out = x_2d @ W.transpose(-2, -1).to(x_2d.dtype)
    return out.reshape(orig_shape)


def idct_apply(x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    if x.dim() == 3:
        x_2d = x.reshape(-1, orig_shape[-2])
    else:
        x_2d = x
    out = x_2d @ W.to(x_2d.dtype)
    return out.reshape(orig_shape)
