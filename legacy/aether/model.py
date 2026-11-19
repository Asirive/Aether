from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from aether.components.attention import CausalSelfAttention
from aether.components.fsa import FreqSparseAttention
from aether.components.mod_router import MoDRouter
from aether.components.polymamba import PolyMamba2
from aether.components.v2 import (
    CausalDepthRouter,
    ChunkCompressedAttention,
    GatedShortConv,
    SelectiveSSM,
    matched_ssm_inner,
)

# Mixer names usable in AetherConfig.layer_pattern (arch="aether_v2").
V2_MIXERS = ("ssm", "fsa2", "attn", "conv")
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
        arch: str = "aether",
        layer_pattern: tuple[str, ...] | None = None,
        ssm_state_dim: int = 64,
        ssm_head_dim: int = 32,
        fsa2_chunk_size: int = 32,
        fsa2_n_summary: int = 8,
    ) -> None:
        if arch not in ("aether", "vanilla", "aether_v2"):
            raise ValueError(f"arch must be 'aether', 'vanilla' or 'aether_v2', got {arch!r}")
        if arch == "aether_v2":
            if layer_pattern is None:
                layer_pattern = tuple("ssm" if i % 2 == 0 else "fsa2" for i in range(n_layers))
            layer_pattern = tuple(layer_pattern)
            if len(layer_pattern) != n_layers or any(m not in V2_MIXERS for m in layer_pattern):
                raise ValueError(f"layer_pattern must list {n_layers} of {V2_MIXERS}, got {layer_pattern}")
        elif layer_pattern is not None:
            raise ValueError("layer_pattern is only used with arch='aether_v2'")
        self.layer_pattern = layer_pattern
        self.ssm_state_dim = ssm_state_dim
        self.ssm_head_dim = ssm_head_dim
        self.fsa2_chunk_size = fsa2_chunk_size
        self.fsa2_n_summary = fsa2_n_summary
        if arch == "vanilla":
            # The baseline has no depth routing; ignore any MoD settings.
            mod_routing_points, mod_keep_ratios = (), ()
        self.arch = arch
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

        kind = config.layer_pattern[layer_id] if config.arch == "aether_v2" else None
        if kind == "ssm":
            # Parameter-matched to one attention mixer (4 * d_model^2).
            self.mixer: nn.Module = SelectiveSSM(
                d_model=config.d_model,
                d_inner=matched_ssm_inner(config.d_model, config.ssm_head_dim, config.ssm_state_dim),
                head_dim=config.ssm_head_dim,
                state_dim=config.ssm_state_dim,
            )
        elif kind == "conv":
            self.mixer = GatedShortConv(config.d_model)
        elif kind == "fsa2":
            self.mixer = ChunkCompressedAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_k=config.d_k,
                max_seq_len=config.max_seq_len,
                chunk_size=config.fsa2_chunk_size,
                n_summary=config.fsa2_n_summary,
                rope_base=config.rope_base,
            )
        elif config.arch == "vanilla" or kind == "attn":
            self.mixer = CausalSelfAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_k=config.d_k,
                max_seq_len=config.max_seq_len,
                rope_base=config.rope_base,
            )
        elif layer_id % 2 == 0:
            self.mixer = PolyMamba2(
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
        elif isinstance(self.mixer, (CausalSelfAttention, SelectiveSSM, ChunkCompressedAttention, GatedShortConv)):
            kbal = torch.zeros((), device=x.device, dtype=x.dtype)
            ortho = torch.zeros((), device=x.device, dtype=x.dtype)
        else:
            kbal = torch.zeros((), device=x.device, dtype=x.dtype)
            ortho = self.mixer.orthogonality_loss()
        x = x + mix
        x = x + self.ffn(self.norm2(x))
        return x, kbal, ortho


    def step(self, x: torch.Tensor, state):
        mix, state = self.mixer.step(self.norm1(x), state)
        x = x + mix
        return x + self.ffn(self.norm2(x)), state


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
        # The vanilla baseline has no router at all, so it carries no unused parameters.
        self.mod_router: MoDRouter | CausalDepthRouter | None = None
        if config.mod_routing_points and config.arch == "aether_v2":
            # Causal per-token router (the v1 router's sequence-level top-k leaks the future).
            self.mod_router = CausalDepthRouter(
                d_model=config.d_model,
                routing_points=config.mod_routing_points,
                keep_ratios=config.mod_keep_ratios,
                capacity_weight=1.0,  # weighted by mod_loss_weight like the v1 router loss
            )
        elif config.mod_routing_points:
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
            if hasattr(block.mixer, "post_init"):
                block.mixer.post_init()

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
            if i in routing_points and isinstance(self.mod_router, CausalDepthRouter):
                gate, cap = self.mod_router.gate(x, self.mod_router.points.index(i))
                mod_loss = mod_loss + cap.to(mod_loss.dtype)
                router_state[f"_last_ratio_{i}"] = float((gate > 0.5).float().mean().detach())
            elif i in routing_points and self.mod_router is not None:
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

    def hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Final normalised hidden states (no logits). For models without a depth router."""
        if self.mod_router is not None:
            raise NotImplementedError("hidden() is for models without a depth router")
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x, _, _ = block(x)
        return self.norm(x)

    def init_state(self, batch: int, device=None, dtype=torch.float32) -> list:
        """Per-layer decode state. Supported for vanilla and aether_v2 without a depth router."""
        if self.config.arch == "aether" or self.mod_router is not None:
            raise NotImplementedError("recurrent decode needs arch vanilla/aether_v2 with no router")
        return [b.mixer.init_state(batch, device=device, dtype=dtype) for b in self.blocks]

    @torch.no_grad()
    def step(self, token: torch.Tensor, state: list) -> tuple[torch.Tensor, list]:
        """One token in, next-token logits out. token: (b,) ids."""
        x = self.token_embedding(token)
        new_state = []
        for block, st in zip(self.blocks, state):
            x, st = block.step(x, st)
            new_state.append(st)
        return self._compute_logits(self.norm(x)), new_state

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
        if mod_temperature is not None and self.mod_router is not None:
            self.mod_router._temperature = mod_temperature

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if exclude_embeddings:
            n -= self.token_embedding.weight.numel()
        return n
