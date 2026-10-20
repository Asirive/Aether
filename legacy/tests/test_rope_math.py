from __future__ import annotations

import torch


def _rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Pure-RoPE helper. Accepts [bsz, seq, d_k] or [bsz, seq, n_heads, d_k]."""
    d_k = x.shape[-1]
    assert d_k % 2 == 0
    if x.dim() == 3:
        seq = x.shape[-2]
        cos_view = (1, seq, d_k // 2)
    elif x.dim() == 4:
        seq = x.shape[-3]
        cos_view = (1, seq, 1, d_k // 2)
    else:
        raise ValueError(f"Unsupported dim {x.dim()}")

    inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2, dtype=torch.float64) / d_k))
    pos = torch.arange(seq, dtype=torch.float64)
    freqs = torch.outer(pos, inv_freq)
    cos = freqs.cos().to(x.dtype).view(cos_view)
    sin = freqs.sin().to(x.dtype).view(cos_view)

    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = out1
    out[..., 1::2] = out2
    return out


def test_rope_preserves_norm():
    """RoPE rotates pairs of dims; should preserve L2 norm."""
    torch.manual_seed(0)
    for shape in [(2, 8, 16), (2, 8, 4, 16), (3, 16, 8)]:
        x = torch.randn(*shape, dtype=torch.float64)
        y = _rope(x)
        norm_before = x.norm(dim=-1)
        norm_after = y.norm(dim=-1)
        err = (norm_before - norm_after).abs().max().item()
        assert err < 1e-9, f"shape {shape}: RoPE norm change err {err}"


def test_rope_preserves_pairwise_inner_product():
    """For any two vectors, rotated inner product = original inner product."""
    torch.manual_seed(1)
    x = torch.randn(3, 16, 8, dtype=torch.float64)
    y = torch.randn(3, 16, 8, dtype=torch.float64)
    xr = _rope(x)
    yr = _rope(y)
    before = (x * y).sum(dim=-1)
    after = (xr * yr).sum(dim=-1)
    err = (before - after).abs().max().item()
    assert err < 1e-9, f"inner product err {err}"


def test_rope_zero_freq_is_unchanged():
    """Position 0 should be unchanged (cos(0)=1, sin(0)=0)."""
    torch.manual_seed(0)
    for shape in [(2, 1, 16), (2, 1, 4, 16), (3, 1, 8)]:
        x = torch.randn(*shape, dtype=torch.float64)
        y = _rope(x)
        err = (x - y).abs().max().item()
        assert err < 1e-12, f"shape {shape}: err {err}"


def test_rope_relative_position_property():
    """For RoPE, the attention score S[m, n] = q_r[m] · k_r[n] depends only on (m - n).

    The relative position property holds when the SAME q and k vectors are used
    at every position (this is the Su et al. property about translation invariance
    of the score for a fixed token pair). For each offset d, S[m, m+d] must be
    constant across m.

    Note: with random per-position vectors, the property does not hold — each
    diagonal entry then involves a different pair of vectors, so scores vary.
    A separate variant below covers translation invariance for fixed token pairs
    using per-position vectors.
    """
    torch.manual_seed(7)
    d_k = 16
    seq = 16
    u = torch.randn(d_k, dtype=torch.float64)
    v = torch.randn(d_k, dtype=torch.float64)
    q = u.expand(1, seq, 1, d_k)
    k = v.expand(1, seq, 1, d_k)
    q_r = _rope(q)
    k_r = _rope(k)
    S = q_r[0, :, 0] @ k_r[0, :, 0].T
    for d in range(1, seq):
        diag = torch.tensor([S[i, i + d].item() for i in range(seq - d)])
        err = (diag.max() - diag.min()).item()
        assert err < 1e-9, f"diag offset {d}: relative position err {err}"


def test_rope_translation_invariance_token_pair():
    """Translation invariance: for two fixed vectors u, v, place u at position i and
    v at position j. The score q_r[i] · k_r[j] depends only on (j - i). Shift the
    pair by t: place u at i+t, v at j+t; the score is identical to the original.
    """
    torch.manual_seed(11)
    d_k = 16
    seq = 32
    u = torch.randn(1, 1, 1, d_k, dtype=torch.float64)
    v = torch.randn(1, 1, 1, d_k, dtype=torch.float64)
    for d in (1, 3, 7, 15):
        base = torch.zeros(1, seq, 1, d_k, dtype=torch.float64)
        q_base = base.clone()
        k_base = base.clone()
        q_base[0, 0, 0, :] = u[0, 0, 0, :]
        k_base[0, d, 0, :] = v[0, 0, 0, :]
        base_score = ((_rope(q_base) * _rope(k_base)).sum(-1))[0, 0, 0].item()

        for shift in range(1, 4):
            q_shift = base.clone()
            k_shift = base.clone()
            q_shift[0, shift, 0, :] = u[0, 0, 0, :]
            k_shift[0, shift + d, 0, :] = v[0, 0, 0, :]
            shifted_score = ((_rope(q_shift) * _rope(k_shift)).sum(-1))[0, 0, 0].item()
            assert abs(base_score - shifted_score) < 1e-9, (
                f"d={d} shift={shift}: base={base_score:.6f} shifted={shifted_score:.6f}"
            )
