from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from aether.ops.topk_soft import soft_topk_mask
from aether.utils.numerics import to_fp32


class MoDRouter(nn.Module):
    """Euclidean top-k depth router (Raposo et al. 2024 style).

    Returns the input unchanged plus a per-token gate in (B, L, 1). The caller
    (AetherForCausalLM) applies the gate residually around the routed block span:
        x = gate * block(x) + (1 - gate) * x
    so unselected tokens bypass the block span exactly (identity passthrough)
    instead of being zeroed.

    The load-balancing loss drives the mean SOFT selection probability per
    position toward the target keep ratio, preventing positional routing
    collapse. The hard top-k count is exact by construction, so a count-based
    loss would be identically zero (audit finding).
    """

    def __init__(
        self,
        d_model: int,
        routing_points: tuple[int, ...] = (4, 8, 12),
        keep_ratios: tuple[float, ...] = (0.50, 0.70, 1.00),
        temperature: float = 1.0,
        z_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if len(routing_points) != len(keep_ratios):
            raise ValueError("routing_points and keep_ratios must have same length")

        self.d_model = d_model
        self.routing_points = set(routing_points)
        self.keep_ratios = dict(zip(routing_points, keep_ratios))

        self.route_norm_weight = nn.Parameter(torch.ones(d_model))
        self.route_proj = nn.Linear(d_model, 1, bias=False)
        nn.init.normal_(self.route_proj.weight, mean=0.0, std=0.01)

        self._temperature = temperature
        self._z_loss_weight = z_loss_weight
        self._bal_loss: torch.Tensor = torch.zeros(())
        self._z_loss: torch.Tensor = torch.zeros(())
        self._last_score_rms: float = 0.0

    def forward(
        self,
        x: torch.Tensor,
        layer_id: int,
        state: Optional[dict] = None,
        temperature: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict, Optional[torch.Tensor]]:
        if layer_id not in self.routing_points:
            return x, torch.zeros((), device=x.device, dtype=x.dtype), state or {}, None

        temp = self._temperature if temperature is None else temperature
        bsz, seq_len, _ = x.shape
        keep_ratio = self.keep_ratios[layer_id]
        k = max(1, int(seq_len * keep_ratio))

        normed = F.rms_norm(to_fp32(x), (self.d_model,), self.route_norm_weight.float(), 1e-6)
        scores = self.route_proj(normed.to(x.dtype)).squeeze(-1).float()
        gate_ste, soft = soft_topk_mask(scores, k=k, temperature=temp, return_soft=True)
        gate = gate_ste.unsqueeze(-1).to(x.dtype)

        usage = soft.mean(dim=0)
        target = float(k) / float(seq_len)
        z_loss = scores.pow(2).mean()
        if keep_ratio >= 1.0:
            balance_loss = torch.zeros((), device=x.device, dtype=torch.float32)
        else:
            balance_loss = F.mse_loss(usage, torch.full_like(usage, target))
        combined_loss = balance_loss + self._z_loss_weight * z_loss

        new_state = dict(state or {})
        new_state[f"mod_mask_{layer_id}"] = gate_ste.detach()
        new_state[f"_last_ratio_{layer_id}"] = float(gate_ste.mean().detach().item())
        new_state[f"_last_score_rms_{layer_id}"] = float(z_loss.detach().sqrt().item())

        self._bal_loss = balance_loss.detach()
        self._z_loss = z_loss.detach()
        self._last_score_rms = float(z_loss.detach().sqrt().item())
        return x, combined_loss, new_state, gate

    def balance_loss(self) -> torch.Tensor:
        return self._bal_loss
