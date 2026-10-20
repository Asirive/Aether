from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_topk_mask(
    logits: torch.Tensor,
    k: int,
    temperature: float = 1.0,
    gumbel_noise: float = 0.0,
    return_soft: bool = False,
    return_indices: bool = False,
):
    if gumbel_noise > 0.0 and torch.is_grad_enabled():
        uniform = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(uniform + 1e-20) + 1e-20)
        perturbed = logits + gumbel_noise * gumbel
    else:
        perturbed = logits

    n = logits.shape[-1]
    k = max(1, min(k, n))
    indices = perturbed.topk(k, dim=-1).indices
    hard_mask = torch.zeros_like(perturbed).scatter_(-1, indices, 1.0)
    kth_value = torch.kthvalue(perturbed, n - k + 1, dim=-1).values.unsqueeze(-1)

    soft_mask = torch.sigmoid((logits - kth_value.detach()) / max(temperature, 1e-3))
    ste_mask = hard_mask + (soft_mask - soft_mask.detach())
    out: list[torch.Tensor] = [ste_mask]
    if return_soft:
        out.append(soft_mask)
    if return_indices:
        out.append(indices)
    if len(out) == 1:
        return ste_mask
    return tuple(out)
