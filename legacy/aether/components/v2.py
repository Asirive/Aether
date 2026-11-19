"""AETHER v2 components: each fixes a measured defect of the v1 component it replaces.

ChunkCompressedAttention ("fsa2") -- replaces CC-FSA
    v1 mean-centred each chunk before the DCT (so the DC bin, the chunk mean, was
    identically zero), mixed the compressed and local paths with a fixed per-head gate
    instead of one softmax, had no positional encoding, and gave tokens at the start of a
    chunk almost no raw context. v2 keeps the idea (a frequency-domain summary of distant
    chunks + exact attention nearby) but:
      * pools each chunk with `n_summary` learnable filters initialised to the lowest DCT
        basis vectors (DC included) -- the DCT is the init, not a hard constraint;
      * attends exactly to a sliding window of the previous + current chunk;
      * scores summaries and raw keys in ONE softmax so heads can trade them off per query;
      * applies RoPE before pooling (local path is position-aware; pooled keys keep the
        low-frequency position signal).

SelectiveSSM ("pm2") -- replaces PolyMamba-2
    v1 was a time-invariant linear SSM with a single 16-dim state shared by all channels
    and a spectral-radius cap of 0.95 (memory ~20 tokens): measurably no better than a
    k=8 causal conv. v2 is a selective state-space mixer in the Mamba-2 (SSD) form:
    input-dependent step size dt_t, input-dependent B_t / C_t, a per-head scalar decay
    a_t = exp(-exp(A_log) * dt_t) in (0, 1) (stable by construction, no clamp needed,
    memory up to thousands of tokens), an (N x P) matrix state per head, a short causal
    conv, and a gated RMSNorm output. "Poly" survives as multi-timescale heads: half the
    heads are initialised fast-decaying, half slow. Computed chunkwise (exact), so cost is
    linear in sequence length.

CausalDepthRouter ("mod2") -- replaces MoD-GDR
    v1 chose the top-k tokens over the whole sequence, so whether token t is processed
    depended on tokens after t (measured: future tokens move past logits). It also shared
    one projection across all routing points. v2 decides per token from that token alone
    (hard threshold forward, sigmoid straight-through backward), has one router per
    routing point, and a capacity loss pulling the processed fraction toward a target.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from aether.ops.dct import dct_matrix


def _rope_tables(d_k: int, max_len: int, base: float = 10_000.0):
    inv = 1.0 / (base ** (torch.arange(0, d_k, 2, dtype=torch.float32) / d_k))
    f = torch.outer(torch.arange(max_len, dtype=torch.float32), inv)
    return f.cos(), f.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


# =====================================================================================
# FSA v2
# =====================================================================================
class ChunkCompressedAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_k: int, max_seq_len: int = 4096,
                 chunk_size: int = 32, n_summary: int = 8, rope_base: float = 10_000.0) -> None:
        super().__init__()
        assert d_model == n_heads * d_k
        self.n_heads, self.d_k, self.c, self.m = n_heads, d_k, chunk_size, n_summary
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        # Per-head pooling filters over chunk positions, initialised to the lowest
        # n_summary DCT-II basis rows (row 0 = DC = scaled mean).
        W = dct_matrix(chunk_size)[:n_summary]
        self.pool = nn.Parameter(W.unsqueeze(0).repeat(n_heads, 1, 1).clone())  # (H, m, c)
        cos, sin = _rope_tables(d_k, max_seq_len + chunk_size, rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, L, _ = x.shape
        H, dk, c, m = self.n_heads, self.d_k, self.c, self.m
        n = (L + c - 1) // c
        pad = n * c - L
        Lp = n * c

        def heads(t):
            t = t.view(b, L, H, dk).transpose(1, 2)
            return F.pad(t, (0, 0, 0, pad)) if pad else t

        cos, sin = self.cos[:Lp], self.sin[:Lp]
        q = _apply_rope(heads(self.w_q(x)), cos, sin)
        k = _apply_rope(heads(self.w_k(x)), cos, sin)
        v = heads(self.w_v(x))
        qc = q.view(b, H, n, c, dk)
        kc = k.view(b, H, n, c, dk)
        vc = v.view(b, H, n, c, dk)

        # Summaries of every chunk: (b, H, n, m, dk)
        ks = torch.einsum("hmc,bhncd->bhnmd", self.pool, kc)
        vs = torch.einsum("hmc,bhncd->bhnmd", self.pool, vc)

        # Raw local window for chunk j: previous chunk + current chunk (2c keys).
        kprev = F.pad(kc, (0, 0, 0, 0, 1, 0))[:, :, :n]
        vprev = F.pad(vc, (0, 0, 0, 0, 1, 0))[:, :, :n]
        kl = torch.cat([kprev, kc], dim=3)                    # (b,H,n,2c,dk)
        vl = torch.cat([vprev, vc], dim=3)
        scale = 1.0 / math.sqrt(dk)
        s_loc = torch.einsum("bhnqd,bhnkd->bhnqk", qc, kl).float() * scale  # (b,H,n,c,2c)
        qi = torch.arange(c, device=x.device).view(c, 1)
        ki = torch.arange(2 * c, device=x.device).view(1, 2 * c) - c       # -c..c-1 relative to chunk start
        loc_ok = (ki <= qi).view(1, 1, 1, c, 2 * c).expand(1, 1, n, c, 2 * c).clone()
        loc_ok[:, :, 0, :, :c] = False                                      # no previous chunk for chunk 0
        s_loc = s_loc.masked_fill(~loc_ok, float("-inf"))

        # Compressed path: chunk j sees summaries of chunks 0..j-2 (j-1 is covered raw).
        ks_flat = ks.reshape(b, H, n * m, dk)
        vs_flat = vs.reshape(b, H, n * m, dk)
        s_cmp = torch.einsum("bhnqd,bhkd->bhnqk", qc, ks_flat).float() * scale  # (b,H,n,c,n*m)
        chunk_of = torch.arange(n * m, device=x.device) // m
        cmp_ok = chunk_of.view(1, n * m) <= (torch.arange(n, device=x.device).view(n, 1) - 2)
        s_cmp = s_cmp.masked_fill(~cmp_ok.view(1, 1, n, 1, n * m), float("-inf"))

        p = torch.softmax(torch.cat([s_loc, s_cmp], dim=-1), dim=-1).to(x.dtype)
        out = torch.einsum("bhnqk,bhnkd->bhnqd", p[..., : 2 * c], vl) + \
            torch.einsum("bhnqk,bhkd->bhnqd", p[..., 2 * c :], vs_flat)
        out = out.reshape(b, H, Lp, dk)[:, :, :L]
        return self.w_o(out.transpose(1, 2).reshape(b, L, H * dk))

    def init_state(self, batch: int, device=None, dtype=torch.float32) -> dict:
        e = torch.zeros(batch, self.n_heads, 0, self.d_k, device=device, dtype=dtype)
        return {"pos": 0, "k_prev": e, "v_prev": e, "k_cur": e, "v_cur": e, "k_sum": e, "v_sum": e}

    def step(self, x: torch.Tensor, state: dict):
        """One-token decode. Cache = 2 raw chunks + n_summary vectors per older chunk."""
        b = x.size(0)
        H, dk, c = self.n_heads, self.d_k, self.c
        t = state["pos"]
        cos, sin = self.cos[t : t + 1], self.sin[t : t + 1]
        q = _apply_rope(self.w_q(x).view(b, H, 1, dk), cos, sin)
        k = _apply_rope(self.w_k(x).view(b, H, 1, dk), cos, sin)
        v = self.w_v(x).view(b, H, 1, dk)
        k_cur = torch.cat([state["k_cur"], k], dim=2)
        v_cur = torch.cat([state["v_cur"], v], dim=2)
        keys = torch.cat([state["k_sum"], state["k_prev"], k_cur], dim=2)
        vals = torch.cat([state["v_sum"], state["v_prev"], v_cur], dim=2)
        sc = torch.einsum("bhqd,bhkd->bhqk", q, keys).float() / math.sqrt(dk)
        out = torch.einsum("bhqk,bhkd->bhqd", torch.softmax(sc, dim=-1).to(x.dtype), vals)
        new = dict(state, pos=t + 1, k_cur=k_cur, v_cur=v_cur)
        if (t + 1) % c == 0:
            if state["k_prev"].size(2):
                ks = torch.einsum("hmc,bhcd->bhmd", self.pool.to(x.dtype), state["k_prev"])
                vs = torch.einsum("hmc,bhcd->bhmd", self.pool.to(x.dtype), state["v_prev"])
                new["k_sum"] = torch.cat([state["k_sum"], ks], dim=2)
                new["v_sum"] = torch.cat([state["v_sum"], vs], dim=2)
            new["k_prev"], new["v_prev"] = k_cur, v_cur
            new["k_cur"], new["v_cur"] = k_cur[:, :, :0], v_cur[:, :, :0]
        return self.w_o(out.reshape(b, H * dk)), new


# =====================================================================================
# PolyMamba v2: selective SSM (Mamba-2 / SSD form), chunkwise-exact
# =====================================================================================
def _segsum_exp(a: torch.Tensor) -> torch.Tensor:
    """a: (..., Q) log-decays. Returns (..., Q, Q) with [i, j] = exp(sum_{j<k<=i} a_k) for j<=i, else 0."""
    Q = a.size(-1)
    cs = torch.cumsum(a, dim=-1)
    diff = cs.unsqueeze(-1) - cs.unsqueeze(-2)
    mask = torch.tril(torch.ones(Q, Q, dtype=torch.bool, device=a.device))
    return torch.exp(diff.masked_fill(~mask, float("-inf")))


def ssd_scan(xh: torch.Tensor, a: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
             chunk: int = 64) -> torch.Tensor:
    """Exact chunked selective scan.

    xh: (b, L, H, P) inputs already multiplied by dt;  a: (b, L, H) log decay (<= 0)
    B, C: (b, L, N) (shared across heads).  Returns y: (b, L, H, P).
    h_t = exp(a_t) h_{t-1} + B_t xh_t^T ;  y_t = C_t . h_t
    """
    b, L, H, P = xh.shape
    N = B.size(-1)
    Q = min(chunk, L)
    n = (L + Q - 1) // Q
    pad = n * Q - L
    if pad:
        xh = F.pad(xh, (0, 0, 0, 0, 0, pad))
        a = F.pad(a, (0, 0, 0, pad))
        B = F.pad(B, (0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, pad))
    xh = xh.view(b, n, Q, H, P)
    a = a.view(b, n, Q, H).permute(0, 3, 1, 2)            # (b,H,n,Q)
    B = B.view(b, n, Q, N)
    C = C.view(b, n, Q, N)

    Lmat = _segsum_exp(a)                                  # (b,H,n,Q,Q)
    G = torch.einsum("bnis,bnjs->bnij", C, B)              # (b,n,Q,Q)  C_i . B_j
    y_diag = torch.einsum("bnij,bhnij,bnjhp->bnihp", G, Lmat, xh)

    cs = torch.cumsum(a, dim=-1)                           # (b,H,n,Q)
    decay_to_end = torch.exp(cs[..., -1:] - cs)            # (b,H,n,Q)
    states = torch.einsum("bnjs,bhnj,bnjhp->bnhsp", B, decay_to_end, xh)  # (b,n,H,N,P)
    chunk_decay = torch.exp(cs[..., -1])                   # (b,H,n)

    s_in = []
    s = torch.zeros(b, H, N, P, dtype=xh.dtype, device=xh.device)
    for j in range(n):
        s_in.append(s)
        s = s * chunk_decay[:, :, j, None, None] + states[:, j]
    s_in = torch.stack(s_in, dim=1)                        # (b,n,H,N,P)
    y_off = torch.einsum("bnis,bhni,bnhsp->bnihp", C, torch.exp(cs), s_in)
    y = (y_diag + y_off).reshape(b, n * Q, H, P)
    return y[:, :L]


class SelectiveSSM(nn.Module):
    def __init__(self, d_model: int, d_inner: int, head_dim: int = 32, state_dim: int = 64,
                 conv_k: int = 4, chunk: int = 64, dt_min: float = 1e-3, dt_max: float = 0.1,
                 selective_dt: bool = True) -> None:
        super().__init__()
        # selective_dt=False makes the step size a per-head constant (ablation only).
        self.selective_dt = selective_dt
        assert d_inner % head_dim == 0
        self.d_inner, self.P, self.N, self.H = d_inner, head_dim, state_dim, d_inner // head_dim
        self.chunk, self.conv_k = chunk, conv_k
        conv_dim = d_inner + 2 * state_dim
        self.in_proj = nn.Linear(d_model, 2 * d_inner + 2 * state_dim + self.H, bias=False)
        self.conv = nn.Conv1d(conv_dim, conv_dim, conv_k, groups=conv_dim, bias=True)
        self.dt_bias = nn.Parameter(torch.zeros(self.H))
        self.A_log = nn.Parameter(torch.zeros(self.H))
        self.D = nn.Parameter(torch.ones(self.H))
        self.norm_w = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self._dt_range = (dt_min, dt_max)
        self.post_init()

    def post_init(self) -> None:
        # Multi-timescale ("poly") heads: first half fast (dt near dt_max), second half slow.
        H = self.H
        lo, hi = math.log(self._dt_range[0]), math.log(self._dt_range[1])
        half = max(1, H // 2)
        u = torch.cat([torch.rand(half) * 0.5 + 0.5, torch.rand(H - half) * 0.5])  # fast, slow
        dt = torch.exp(u * (hi - lo) + lo)
        with torch.no_grad():
            self.in_proj.weight.normal_(0.0, 1.0 / math.sqrt(3 * self.in_proj.in_features))
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))   # softplus^-1(dt)
            self.A_log.copy_(torch.log(torch.empty(H).uniform_(1.0, 16.0)))
            self.D.fill_(1.0)
            self.norm_w.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, L, _ = x.shape
        di, N, H, P = self.d_inner, self.N, self.H, self.P
        ft = torch.promote_types(x.dtype, torch.float32)
        zxbcdt = self.in_proj(x)
        z, xBC, dt = torch.split(zxbcdt, [di, di + 2 * N, H], dim=-1)
        xBC = F.pad(xBC.transpose(1, 2), (self.conv_k - 1, 0))
        xBC = F.silu(self.conv(xBC)).transpose(1, 2)
        xs, Bm, Cm = torch.split(xBC, [di, N, N], dim=-1)
        if not self.selective_dt:
            dt = torch.zeros_like(dt)
        dt = F.softplus(dt.to(ft) + self.dt_bias)                     # (b,L,H)
        a = -torch.exp(self.A_log.to(ft)) * dt                        # log decay <= 0
        xh = xs.to(ft).view(b, L, H, P)
        y = ssd_scan(xh * dt.unsqueeze(-1), a, Bm.to(ft), Cm.to(ft), self.chunk)
        y = y + xh * self.D.view(1, 1, H, 1)
        y = y.reshape(b, L, di)
        y = F.rms_norm(y * F.silu(z.to(ft)), (di,), self.norm_w.to(ft), 1e-6)
        return self.out_proj(y.to(x.dtype))

    def init_state(self, batch: int, device=None, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
        conv_dim = self.d_inner + 2 * self.N
        return (torch.zeros(batch, conv_dim, self.conv_k - 1, device=device, dtype=dtype),
                torch.zeros(batch, self.H, self.N, self.P, device=device, dtype=torch.float32))

    def step(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        """One-token recurrent update. x: (b, d_model). Constant memory and compute per token."""
        conv_buf, h = state
        b = x.size(0)
        di, N, H, P = self.d_inner, self.N, self.H, self.P
        ft = h.dtype
        z, xBC, dt = torch.split(self.in_proj(x), [di, di + 2 * N, H], dim=-1)
        window = torch.cat([conv_buf, xBC.unsqueeze(-1).to(conv_buf.dtype)], dim=-1)
        xBC = F.silu((window * self.conv.weight.squeeze(1)).sum(-1) + self.conv.bias)
        xs, Bm, Cm = torch.split(xBC, [di, N, N], dim=-1)
        if not self.selective_dt:
            dt = torch.zeros_like(dt)
        dt = F.softplus(dt.to(ft) + self.dt_bias)
        decay = torch.exp(-torch.exp(self.A_log.to(ft)) * dt)
        xh = xs.to(ft).view(b, H, P)
        h = h * decay.view(b, H, 1, 1) + torch.einsum("bs,bhp->bhsp", Bm.to(ft), xh * dt.unsqueeze(-1))
        y = torch.einsum("bs,bhsp->bhp", Cm.to(ft), h) + xh * self.D.view(1, H, 1)
        y = F.rms_norm(y.reshape(b, di) * F.silu(z.to(ft)), (di,), self.norm_w.to(ft), 1e-6)
        return self.out_proj(y.to(x.dtype)), (window[..., 1:], h)


class GatedShortConv(nn.Module):
    """Gated depthwise causal conv (LFM-style), parameter-matched to one attention mixer.

    The cheapest mixer: k multiply-adds per channel per token, a k-1 token state, no softmax
    and no scan. In the lab, swapping half the SSM layers for it kept quality and ran 1.4x faster.
    """

    def __init__(self, d_model: int, kernel: int = 4) -> None:
        super().__init__()
        inner = round(4 * d_model * d_model / (3 * d_model + kernel + 1))
        self.in_proj = nn.Linear(d_model, 2 * inner, bias=False)
        self.conv = nn.Conv1d(inner, inner, kernel, groups=inner, bias=True)
        self.out_proj = nn.Linear(inner, d_model, bias=False)
        self.k, self.inner = kernel, inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v, g = self.in_proj(x).chunk(2, dim=-1)
        v = self.conv(F.pad(v.transpose(1, 2), (self.k - 1, 0))).transpose(1, 2)
        return self.out_proj(F.silu(g) * v)

    def init_state(self, batch: int, device=None, dtype=torch.float32) -> torch.Tensor:
        return torch.zeros(batch, self.inner, self.k - 1, device=device, dtype=dtype)

    def step(self, x: torch.Tensor, state: torch.Tensor):
        v, g = self.in_proj(x).chunk(2, dim=-1)
        window = torch.cat([state, v.unsqueeze(-1).to(state.dtype)], dim=-1)
        v = (window * self.conv.weight.squeeze(1)).sum(-1) + self.conv.bias
        return self.out_proj(F.silu(g) * v.to(g.dtype)), window[..., 1:]


def matched_ssm_inner(d_model: int, head_dim: int = 32, state_dim: int = 64, conv_k: int = 4) -> int:
    """d_inner (multiple of head_dim) whose SelectiveSSM param count is closest to 4*d^2 (one attention)."""
    target = 4 * d_model * d_model
    best, best_err = head_dim, float("inf")
    for di in range(head_dim, 4 * d_model + 1, head_dim):
        H = di // head_dim
        p = d_model * (2 * di + 2 * state_dim + H) + (di + 2 * state_dim) * (conv_k + 1) + 3 * H + di + di * d_model
        if abs(p - target) < best_err:
            best, best_err = di, abs(p - target)
    return best


# =====================================================================================
# MoD v2: causal per-token depth router
# =====================================================================================
class CausalDepthRouter(nn.Module):
    """Per-token router; one projection per routing point. Gate spans until the next point."""

    def __init__(self, d_model: int, routing_points: tuple[int, ...], keep_ratios: tuple[float, ...],
                 capacity_weight: float = 0.01) -> None:
        super().__init__()
        self.points = list(routing_points)
        self.keep = dict(zip(routing_points, keep_ratios))
        self.norms = nn.ParameterList([nn.Parameter(torch.ones(d_model)) for _ in self.points])
        self.projs = nn.ModuleList([nn.Linear(d_model, 1, bias=True) for _ in self.points])
        for p, kr in zip(self.projs, keep_ratios):
            p._no_reinit = True
            nn.init.normal_(p.weight, std=0.01)
            nn.init.constant_(p.bias, math.log(kr / max(1e-3, 1 - kr)) if kr < 1 else 4.0)
        self.capacity_weight = capacity_weight
        self._aux = torch.zeros(())
        self.last_ratio: dict[int, float] = {}

    def gate(self, x: torch.Tensor, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.rms_norm(x.float(), (x.size(-1),), self.norms[idx].float(), 1e-6)
        s = self.projs[idx](h.to(x.dtype)).float().squeeze(-1)
        soft = torch.sigmoid(s)
        hard = (s > 0).float()
        g = hard + soft - soft.detach()
        target = self.keep[self.points[idx]]
        cap = (soft.mean() - target) ** 2 if target < 1.0 else soft.new_zeros(())
        return g.unsqueeze(-1).to(x.dtype), cap

    def run(self, blocks, x):
        aux = x.new_zeros((), dtype=torch.float32)
        gate = None
        for i, blk in enumerate(blocks):
            if i in self.points:
                idx = self.points.index(i)
                gate, cap = self.gate(x, idx)
                aux = aux + cap
                self.last_ratio[i] = float((gate > 0.5).float().mean())
            x = gate * blk(x) + (1.0 - gate) * x if gate is not None else blk(x)
        self._aux = aux * self.capacity_weight
        return x, x.new_zeros(())

    def aux_loss(self) -> torch.Tensor:
        return self._aux
