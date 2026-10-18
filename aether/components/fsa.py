from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from aether.ops.dct import dct_matrix
from aether.ops.topk_soft import soft_topk_mask
from aether.utils.numerics import to_fp32


class FreqSparseAttention(nn.Module):
    """Causal chunk-compressed frequency attention (CC-FSA).

    Keys/values are split into chunks of `chunk_size` positions. Each chunk is
    transformed with a DCT-II along the chunk axis, and each head keeps a learned
    top-k set of frequency bins as that chunk's compressed summary. Queries attend
    to (a) compressed summaries of chunks that lie fully in the past (cross-chunk
    path) and (b) raw time-domain keys inside their own chunk with a standard
    causal mask (intra-chunk path). A per-head scalar gate blends the two paths.

    Strict causality: position t never sees information from positions > t.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: int,
        max_seq_len: int = 4096,
        topk_ratio: float = 0.25,
        chunk_size: int = 32,
    ) -> None:
        super().__init__()
        if d_model != n_heads * d_k:
            raise ValueError(f"d_model ({d_model}) must equal n_heads ({n_heads}) * d_k ({d_k})")
        if chunk_size < 2:
            raise ValueError(f"chunk_size must be >= 2, got {chunk_size}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.topk_ratio = topk_ratio
        self.chunk_size = chunk_size
        self.keep_bins = max(1, int(chunk_size * topk_ratio))

        self.w_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_o = nn.Linear(n_heads * d_k, d_model, bias=False)

        self.mask_logits = nn.Parameter(torch.randn(n_heads, chunk_size) * 0.02)
        self.mix_gate = nn.Parameter(torch.zeros(n_heads))

        self._temperature = 0.1
        self._gumbel_noise = 0.0
        self._cross_mask_cache: dict[tuple[int, str], torch.Tensor] = {}
        self._intra_mask_cache: dict[tuple[int, int, str], torch.Tensor] = {}

    def _cross_chunk_mask(self, padded_len: int, device: torch.device) -> torch.Tensor:
        key = (padded_len, str(device))
        if key not in self._cross_mask_cache:
            n_chunks = padded_len // self.chunk_size
            chunk_of_bin = torch.arange(n_chunks * self.keep_bins, device=device) // self.keep_bins
            chunk_end = (chunk_of_bin + 1) * self.chunk_size
            positions = torch.arange(padded_len, device=device)
            allowed = chunk_end.unsqueeze(0) <= positions.unsqueeze(1)
            self._cross_mask_cache[key] = allowed
        return self._cross_mask_cache[key]

    def _intra_chunk_mask(self, n_chunks: int, pad: int, device: torch.device) -> torch.Tensor:
        key = (n_chunks, pad, str(device))
        if key not in self._intra_mask_cache:
            c = self.chunk_size
            causal = torch.tril(torch.ones(c, c, dtype=torch.bool, device=device))
            mask = causal.view(1, c, c).repeat(n_chunks, 1, 1)
            if pad > 0:
                valid_keys = torch.arange(c, device=device) < (c - pad)
                mask[-1] = mask[-1] & valid_keys.view(1, c)
            self._intra_mask_cache[key] = mask
        return self._intra_mask_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len ({seq_len}) > max_seq_len ({self.max_seq_len})")

        c = self.chunk_size
        n_chunks = (seq_len + c - 1) // c
        pad = n_chunks * c - seq_len
        padded_len = n_chunks * c

        q = self.w_q(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        if pad > 0:
            q = F.pad(q, (0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))

        kc = k.reshape(bsz, self.n_heads, n_chunks, c, self.d_k)
        vc = v.reshape(bsz, self.n_heads, n_chunks, c, self.d_k)

        k_cent = kc - kc.mean(dim=-2, keepdim=True)
        v_cent = vc - vc.mean(dim=-2, keepdim=True)

        W = dct_matrix(c, device=x.device, dtype=torch.float32)
        kf = torch.einsum("bhnld,kl->bhnkd", to_fp32(k_cent), W)
        vf = torch.einsum("bhnld,kl->bhnkd", to_fp32(v_cent), W)

        ste_mask, _soft, kept_idx = soft_topk_mask(
            self.mask_logits,
            k=self.keep_bins,
            temperature=self._temperature,
            gumbel_noise=self._gumbel_noise,
            return_soft=True,
            return_indices=True,
        )
        bin_weights = ste_mask.gather(1, kept_idx)

        gather_idx = (
            kept_idx.view(1, self.n_heads, 1, self.keep_bins, 1)
            .expand(bsz, self.n_heads, n_chunks, self.keep_bins, self.d_k)
        )
        gather_w = bin_weights.view(1, self.n_heads, 1, self.keep_bins, 1)
        k_comp = (kf.gather(3, gather_idx) * gather_w).to(x.dtype)
        v_comp = (vf.gather(3, gather_idx) * gather_w).to(x.dtype)
        k_comp = k_comp.reshape(bsz, self.n_heads, n_chunks * self.keep_bins, self.d_k)
        v_comp = v_comp.reshape(bsz, self.n_heads, n_chunks * self.keep_bins, self.d_k)

        scores_cross = torch.matmul(q, k_comp.transpose(-2, -1)) / math.sqrt(self.d_k)
        allowed = self._cross_chunk_mask(padded_len, x.device)
        allowed_b = allowed.unsqueeze(0).unsqueeze(0)
        scores_cross = scores_cross.float().masked_fill(~allowed_b, -1e9)
        attn_cross = F.softmax(scores_cross, dim=-1) * allowed_b.float()
        attn_cross = attn_cross / attn_cross.sum(-1, keepdim=True).clamp_min(1e-9)
        out_cross = torch.matmul(attn_cross.to(x.dtype), v_comp)

        qc = q.reshape(bsz, self.n_heads, n_chunks, c, self.d_k)
        scores_intra = torch.matmul(qc, kc.transpose(-2, -1)) / math.sqrt(self.d_k)
        intra_mask = self._intra_chunk_mask(n_chunks, pad, x.device)
        scores_intra = scores_intra.float().masked_fill(
            ~intra_mask.unsqueeze(0).unsqueeze(0), -1e9
        )
        attn_intra = F.softmax(scores_intra, dim=-1)
        out_intra = torch.matmul(attn_intra.to(x.dtype), vc)
        out_intra = out_intra.reshape(bsz, self.n_heads, padded_len, self.d_k)

        g = torch.sigmoid(self.mix_gate).view(1, self.n_heads, 1, 1).to(x.dtype)
        out = g * out_cross + (1.0 - g) * out_intra

        out = out[:, :, :seq_len, :]
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.w_o(out)

    def orthogonality_loss(self) -> torch.Tensor:
        logits = self.mask_logits
        norms = logits.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        normed = logits / norms
        gram = normed @ normed.transpose(0, 1)
        identity = torch.eye(self.n_heads, device=logits.device, dtype=logits.dtype)
        loss = (gram - identity).pow(2).mean()
        self._ortho_loss = loss.detach()
        return loss

