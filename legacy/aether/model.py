from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from aether.components.fsa import FreqSparseAttention
from aether.components.mod_router import MoDRouter
from aether.components.polymamba import PolyMamba2
from aether.utils.numerics import to_fp32


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = to_fp32(x)
        rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_fp32 * rms).to(x.dtype) * self.weight


class AetherConfig:
    def __init__(
        self,
        vocab_size: int = 32_000,
        d_model: int = 768,
        d_inner: int = 2048,
        n_layers: int = 12,
        n_heads: int = 12,
        d_k: int = 64,
        max_seq_len: int = 2048,
        rope_base: float = 10_000.0,
        tie_embeddings: bool = True,
        fsa_topk_ratio: float = 0.25,
        fsa_ortho_loss_weight: float = 0.001,
        polymamba_orders: int = 2,
        polymamba_state_dim: int = 16,
        polymamba_kbal_loss_weight: float = 0.01,
        mod_routing_points: tuple[int, ...] = (4, 8, 12),
        mod_keep_ratios: tuple[float, ...] = (0.50, 0.70, 1.00),
        mod_loss_weight: float = 0.01,
    ) -> None:
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.rope_base = rope_base
        self.tie_embeddings = tie_embeddings
        self.fsa_topk_ratio = fsa_topk_ratio
        self.fsa_ortho_loss_weight = fsa_ortho_loss_weight
        self.polymamba_orders = polymamba_orders
        self.polymamba_state_dim = polymamba_state_dim
        self.polymamba_kbal_loss_weight = polymamba_kbal_loss_weight
        self.mod_routing_points = mod_routing_points
        self.mod_keep_ratios = mod_keep_ratios
        self.mod_loss_weight = mod_loss_weight

    @classmethod
    def from_dict(cls, d: dict) -> AetherConfig:
        return cls(**d)


class AetherBlock(nn.Module):
    def __init__(self, layer_id: int, config: AetherConfig) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.config = config

        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)

        if layer_id % 2 == 0:
            self.mixer: nn.Module = PolyMamba2(
                d_model=config.d_model,
                d_inner=config.d_inner,
                n_orders=config.polymamba_orders,
                state_dim=config.polymamba_state_dim,
            )
        else:
            self.mixer = FreqSparseAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_k=config.d_k,
                max_seq_len=config.max_seq_len,
                topk_ratio=config.fsa_topk_ratio,
            )

        self.ffn = SwiGLU(d_model=config.d_model, d_inner=config.d_inner)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mix = self.mixer(self.norm1(x))
        if isinstance(self.mixer, PolyMamba2):
            kbal = self.mixer.balance_loss()
            ortho = torch.zeros((), device=x.device, dtype=x.dtype)
        else:
            kbal = torch.zeros((), device=x.device, dtype=x.dtype)
            ortho = self.mixer.orthogonality_loss()
        x = x + mix
        x = x + self.ffn(self.norm2(x))
        return x, kbal, ortho


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_inner: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_inner, bias=False)
        self.w_up = nn.Linear(d_model, d_inner, bias=False)
        self.w_down = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class AetherForCausalLM(nn.Module):
    def __init__(self, config: AetherConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [AetherBlock(i, config) for i in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.d_model)
        self.mod_router = MoDRouter(
            d_model=config.d_model,
            routing_points=config.mod_routing_points,
            keep_ratios=config.mod_keep_ratios,
        )

        if not config.tie_embeddings:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        else:
            self.lm_head = None

        self._init_weights()

    def _init_weights(self) -> None:
        std = 0.02
        residual_out = ("out_proj", "w_down", "w_o")
        block_std = std / math.sqrt(2.0 * self.config.n_layers)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=std)
        for block in self.blocks:
            for name, module in block.named_modules():
                if isinstance(module, nn.Linear):
                    if name.endswith(residual_out):
                        init_std = block_std
                    else:
                        init_std = std / (module.weight.shape[1] ** 0.5)
                    nn.init.normal_(module.weight, mean=0.0, std=init_std)
                elif isinstance(module, RMSNorm):
                    nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> dict[str, torch.Tensor]:
        x = self.token_embedding(input_ids)
        mod_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        fsa_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        pm_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        router_state: dict = {}
        gate: torch.Tensor | None = None
        routing_points = set(self.config.mod_routing_points)

        for i, block in enumerate(self.blocks):
            if i in routing_points:
                _, route_loss, router_state, gate = self.mod_router(x, layer_id=i, state=router_state)
                mod_loss = mod_loss + route_loss.to(mod_loss.dtype)
            if gate is not None:
                block_in = x
                block_out, kbal, ortho = block(block_in)
                x = gate * block_out + (1.0 - gate) * block_in
            else:
                x, kbal, ortho = block(x)
            pm_loss = pm_loss + kbal
            fsa_loss = fsa_loss + ortho

        x = self.norm(x)
        self._last_router_state = dict(router_state)

        result: dict[str, torch.Tensor] = {
            "aux_losses": {
                "mod": mod_loss,
                "fsa_ortho": fsa_loss,
                "polymamba_kbal": pm_loss,
            }
        }

        wants_logits = labels is None or return_logits
        if wants_logits:
            result["logits"] = self._compute_logits(x)

        if labels is not None:
            if wants_logits:
                shift_logits = result["logits"][..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                task_loss = self._cross_entropy_chunked(shift_logits, shift_labels)
            else:
                task_loss = self._chunked_causal_loss(x, labels)
            aux_total = (
                self.config.mod_loss_weight * mod_loss
                + self.config.fsa_ortho_loss_weight * result["aux_losses"]["fsa_ortho"]
                + self.config.polymamba_kbal_loss_weight * result["aux_losses"]["polymamba_kbal"]
            )
            result["loss"] = task_loss + aux_total
            result["task_loss"] = task_loss

        return result

    def _chunked_causal_loss(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        chunk_tokens: int = 512,
    ) -> torch.Tensor:
        """Next-token cross entropy without materialising the (N, vocab) logits.

        Materialising logits for a 128K vocabulary costs ~2 GB per micro-batch
        (logits + the contiguous shift copy + their gradients), which pushes the
        card into shared memory. Projecting one chunk of tokens at a time keeps
        the transient peak bounded by `chunk_tokens * vocab`.

        Chunking runs per sequence: flattening batch and sequence together would
        pair the last token of one document with the first token of the next,
        which the full-logits path never does.
        """
        seq_len = hidden.size(1)
        total_loss = hidden.new_zeros(())
        total_count = torch.zeros((), device=hidden.device, dtype=torch.float32)
        if seq_len < 2:
            return total_loss
        for batch_idx in range(hidden.size(0)):
            row_hidden = hidden[batch_idx]
            row_labels = labels[batch_idx]
            for start in range(0, seq_len - 1, chunk_tokens):
                end = min(start + chunk_tokens, seq_len - 1)
                chunk_logits = self._compute_logits(row_hidden[start:end])
                targets = row_labels[start + 1 : end + 1]
                total_loss = total_loss + F.cross_entropy(
                    chunk_logits.float(),
                    targets,
                    ignore_index=-100,
                    reduction="sum",
                )
                total_count = total_count + (targets != -100).sum().to(torch.float32)
        return total_loss / total_count.clamp_min(1.0)

    def _cross_entropy_chunked(
        self,
        shift_logits: torch.Tensor,
        shift_labels: torch.Tensor,
        chunk_tokens: int = 512,
    ) -> torch.Tensor:
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        total_tokens = flat_labels.numel()
        total_loss = flat_logits.new_zeros(())
        total_count = torch.zeros((), device=flat_labels.device, dtype=torch.float32)
        for start in range(0, total_tokens, chunk_tokens):
            end = min(start + chunk_tokens, total_tokens)
            chunk_logits = flat_logits[start:end]
            chunk_labels = flat_labels[start:end]
            ce = F.cross_entropy(
                chunk_logits.float(),
                chunk_labels,
                ignore_index=-100,
                reduction="sum",
            )
            total_loss = total_loss + ce
            total_count = total_count + (chunk_labels != -100).sum().to(torch.float32)
        return total_loss / total_count.clamp_min(1.0)

    def _compute_logits(self, x: torch.Tensor) -> torch.Tensor:
        if self.lm_head is None:
            return F.linear(x, self.token_embedding.weight)
        return self.lm_head(x)

    def set_component_knobs(
        self,
        fsa_temperature: float | None = None,
        fsa_gumbel: float | None = None,
        mod_temperature: float | None = None,
    ) -> None:
        for block in self.blocks:
            if isinstance(block.mixer, FreqSparseAttention):
                if fsa_temperature is not None:
                    block.mixer._temperature = fsa_temperature
                if fsa_gumbel is not None:
                    block.mixer._gumbel_noise = fsa_gumbel
        if mod_temperature is not None:
            self.mod_router._temperature = mod_temperature

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if exclude_embeddings:
            n -= self.token_embedding.weight.numel()
        return n
