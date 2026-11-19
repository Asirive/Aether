from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Plain causal multi-head attention with RoPE, used by the vanilla baseline.

    Deliberately boring: full causal attention, no chunking, no compression, no
    aux loss. RoPE is computed in fp32 and the attention itself goes through
    `scaled_dot_product_attention`, which does its softmax in fp32 internally.
    Projection names (`w_q`, `w_k`, `w_v`, `w_o`) match CC-FSA so the model's
    init treats `w_o` as a residual output projection, exactly as it does for FSA.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: int,
        max_seq_len: int = 4096,
        rope_base: float = 10_000.0,
    ) -> None:
        super().__init__()
        if d_model != n_heads * d_k:
            raise ValueError(f"d_model ({d_model}) must equal n_heads ({n_heads}) * d_k ({d_k})")
        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even for RoPE, got {d_k}")
        self.n_heads = n_heads
        self.d_k = d_k

        self.w_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_o = nn.Linear(n_heads * d_k, d_model, bias=False)

        inv_freq = 1.0 / (rope_base ** (torch.arange(0, d_k, 2, dtype=torch.float32) / d_k))
        freqs = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv_freq)
        self.register_buffer("rope_cos", freqs.cos(), persistent=False)
        self.register_buffer("rope_sin", freqs.sin(), persistent=False)

    def _rope(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        # x: [bsz, n_heads, seq, d_k]; rotates interleaved (even, odd) pairs.
        seq = x.size(-2)
        if offset + seq > self.rope_cos.size(0):
            raise ValueError(f"seq_len {offset + seq} exceeds max_seq_len {self.rope_cos.size(0)}")
        cos = self.rope_cos[offset : offset + seq]
        sin = self.rope_sin[offset : offset + seq]
        xf = x.float()
        x1, x2 = xf[..., 0::2], xf[..., 1::2]
        out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
        return out.to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, _ = x.shape

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(bsz, seq, self.n_heads, self.d_k).transpose(1, 2)

        q = self._rope(heads(self.w_q(x)))
        k = self._rope(heads(self.w_k(x)))
        v = heads(self.w_v(x))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.w_o(out.transpose(1, 2).reshape(bsz, seq, self.n_heads * self.d_k))

    def init_state(self, batch: int, device=None, dtype=torch.float32) -> dict:
        e = torch.zeros(batch, self.n_heads, 0, self.d_k, device=device, dtype=dtype)
        return {"k": e, "v": e}

    def step(self, x: torch.Tensor, state: dict):
        """One-token decode with a full KV cache (grows by one entry per token)."""
        b = x.size(0)
        t = state["k"].size(2)
        q = self._rope(self.w_q(x).view(b, self.n_heads, 1, self.d_k), t)
        k = torch.cat([state["k"], self._rope(self.w_k(x).view(b, self.n_heads, 1, self.d_k), t)], dim=2)
        v = torch.cat([state["v"], self.w_v(x).view(b, self.n_heads, 1, self.d_k)], dim=2)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.w_o(out.reshape(b, self.n_heads * self.d_k)), {"k": k, "v": v}
