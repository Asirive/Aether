from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from aether.components.attention import CausalSelfAttention
from aether.components.v2 import (
    CausalDepthRouter,
    ChunkCompressedAttention,
    SelectiveSSM,
    matched_ssm_inner,
    ssd_scan,
)


def _naive_scan(xh, a, B, C):
    b, L, H, P = xh.shape
    N = B.size(-1)
    h = torch.zeros(b, H, N, P, dtype=xh.dtype)
    ys = []
    for t in range(L):
        h = h * torch.exp(a[:, t]).view(b, H, 1, 1) + torch.einsum("bs,bhp->bhsp", B[:, t], xh[:, t])
        ys.append(torch.einsum("bs,bhsp->bhp", C[:, t], h))
    return torch.stack(ys, dim=1)


def test_ssd_scan_matches_naive_recurrence():
    torch.manual_seed(0)
    for L, chunk in [(37, 8), (64, 64), (50, 16), (5, 64)]:
        b, H, P, N = 2, 3, 4, 5
        xh = torch.randn(b, L, H, P, dtype=torch.float64)
        a = -torch.rand(b, L, H, dtype=torch.float64) * 0.5
        B = torch.randn(b, L, N, dtype=torch.float64)
        C = torch.randn(b, L, N, dtype=torch.float64)
        got = ssd_scan(xh, a, B, C, chunk=chunk)
        ref = _naive_scan(xh, a, B, C)
        assert torch.allclose(got, ref, atol=1e-10), (L, chunk, (got - ref).abs().max())


def test_ssd_long_memory_is_possible():
    """Decay can approach 1: a signal at t=0 survives 1000 steps (v1 capped |A| at 0.95)."""
    xh = torch.zeros(1, 1000, 1, 1, dtype=torch.float64)
    xh[0, 0] = 1.0
    a = torch.full((1, 1000, 1), -1e-4, dtype=torch.float64)
    ones = torch.ones(1, 1000, 1, dtype=torch.float64)
    y = ssd_scan(xh, a, ones, ones, chunk=64)
    assert y[0, -1, 0, 0] > 0.9


def _causal(mod, d_model, L=96, cut=50):
    torch.manual_seed(0)
    mod = mod.double().eval()
    x = torch.randn(2, L, d_model, dtype=torch.float64)
    x2 = x.clone()
    x2[:, cut + 1 :] = torch.randn(2, L - cut - 1, d_model, dtype=torch.float64)
    with torch.no_grad():
        return (mod(x)[:, : cut + 1] - mod(x2)[:, : cut + 1]).abs().max().item()


def test_fsa2_is_causal_including_padding():
    m = ChunkCompressedAttention(64, 4, 16, max_seq_len=128, chunk_size=16, n_summary=4)
    m.cos, m.sin = m.cos.double(), m.sin.double()
    for L, cut in [(96, 50), (90, 15), (90, 33)]:
        assert _causal(m, 64, L, cut) < 1e-12


def test_fsa2_equals_full_attention_within_two_chunks():
    """Inside the first two chunks every key is in the raw window, so FSA2 == causal attention."""
    torch.manual_seed(0)
    f = ChunkCompressedAttention(64, 4, 16, max_seq_len=64, chunk_size=16, n_summary=4).double()
    a = CausalSelfAttention(64, 4, 16, max_seq_len=64).double()
    a.load_state_dict({k: v for k, v in f.state_dict().items() if k.startswith("w_")})
    f.cos, f.sin = f.cos.double(), f.sin.double()
    a.rope_cos, a.rope_sin = a.rope_cos.double(), a.rope_sin.double()
    x = torch.randn(2, 32, 64, dtype=torch.float64)
    with torch.no_grad():
        # reference attention computes RoPE in fp32, so agreement is ~1e-8, not 1e-12
        assert torch.allclose(f(x), a(x), atol=1e-6)


def test_fsa2_pool_init_keeps_dc():
    f = ChunkCompressedAttention(64, 4, 16, chunk_size=32, n_summary=8)
    dc = f.pool[0, 0]
    assert torch.allclose(dc, torch.full_like(dc, 1 / math.sqrt(32)), atol=1e-6)


def test_selective_ssm_is_causal_and_param_matched():
    d = 64
    di = matched_ssm_inner(d)
    m = SelectiveSSM(d, di, head_dim=32, state_dim=16, chunk=16)
    assert _causal(m, d, 70, 33) < 1e-10
    m2 = SelectiveSSM(256, matched_ssm_inner(256))
    n = sum(p.numel() for p in m2.parameters())
    assert abs(n - 4 * 256 * 256) / (4 * 256 * 256) < 0.03


def test_causal_router_is_causal():
    torch.manual_seed(0)
    blocks = torch.nn.ModuleList([torch.nn.Linear(32, 32) for _ in range(4)]).double()
    r = CausalDepthRouter(32, (1, 3), (0.5, 0.7)).double()
    x = torch.randn(2, 40, 32, dtype=torch.float64)
    x2 = x.clone()
    x2[:, 21:] = torch.randn(2, 19, 32, dtype=torch.float64)
    y1, _ = r.run(blocks, x)
    y2, _ = r.run(blocks, x2)
    assert (y1[:, :21] - y2[:, :21]).abs().max().item() == 0.0


# ---------------------------------------------------------------------------------------
# Integration into AetherForCausalLM (arch="aether_v2")
# ---------------------------------------------------------------------------------------
from aether.model import AetherConfig, AetherForCausalLM  # noqa: E402


def _v2_cfg(**kw):
    base = dict(vocab_size=64, d_model=64, d_inner=128, n_layers=4, n_heads=4, d_k=16,
                max_seq_len=96, arch="aether_v2", mod_routing_points=(), mod_keep_ratios=(),
                ssm_state_dim=16, fsa2_chunk_size=16, fsa2_n_summary=4)
    base.update(kw)
    return AetherConfig(**base)


def test_v2_model_default_pattern_alternates_ssm_fsa2():
    m = AetherForCausalLM(_v2_cfg())
    kinds = [type(b.mixer).__name__ for b in m.blocks]
    assert kinds == ["SelectiveSSM", "ChunkCompressedAttention"] * 2


def test_v2_model_is_causal_with_router_and_pattern():
    cfg = _v2_cfg(layer_pattern=("ssm", "ssm", "fsa2", "attn"),
                  mod_routing_points=(1, 3), mod_keep_ratios=(0.5, 1.0))
    torch.manual_seed(0)
    m = AetherForCausalLM(cfg).double().eval()
    for blk in m.blocks:
        for name in ("cos", "sin", "rope_cos", "rope_sin"):
            if hasattr(blk.mixer, name):
                setattr(blk.mixer, name, getattr(blk.mixer, name).double())
    ids = torch.randint(0, 64, (2, 80))
    ids2 = ids.clone()
    ids2[:, 41:] = torch.randint(0, 64, (2, 39))
    with torch.no_grad():
        a = m(ids)["logits"][:, :41]
        b = m(ids2)["logits"][:, :41]
    assert (a - b).abs().max().item() < 1e-9


def test_v2_model_trains_successor_task():
    torch.manual_seed(0)
    m = AetherForCausalLM(_v2_cfg(vocab_size=16, mod_routing_points=(2,), mod_keep_ratios=(0.7,)))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(150):
        ids = (torch.randint(0, 16, (8, 1)) + torch.arange(48)) % 16
        out = m(ids, labels=ids)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
    assert out["task_loss"].item() < 0.1
    assert torch.isfinite(out["aux_losses"]["mod"])


def test_v2_ssm_init_survives_model_init():
    m = AetherForCausalLM(_v2_cfg())
    ssm = m.blocks[0].mixer
    dt = F.softplus(ssm.dt_bias)
    assert dt.min() >= 1e-3 * 0.99 and dt.max() <= 0.1 * 1.01


def test_selective_ssm_step_matches_parallel_forward():
    torch.manual_seed(0)
    m = SelectiveSSM(64, matched_ssm_inner(64), head_dim=32, state_dim=16, chunk=16).double().eval()
    x = torch.randn(2, 41, 64, dtype=torch.float64)
    with torch.no_grad():
        ref = m(x)
        c, h = m.init_state(2, dtype=torch.float64)
        state = (c, h.double())
        outs = []
        for t in range(x.size(1)):
            y, state = m.step(x[:, t], state)
            outs.append(y)
    assert torch.allclose(torch.stack(outs, 1), ref, atol=1e-9)
