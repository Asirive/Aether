from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from aether.components.fsa import FreqSparseAttention
from aether.ops.dct import dct_matrix
from aether.ops.topk_soft import soft_topk_mask


def test_soft_topk_k_exactly_correct_at_low_temp():
    torch.manual_seed(0)
    logits = torch.randn(8, 100)
    k = 25
    mask = soft_topk_mask(logits, k=k, temperature=0.0001)
    ones_per_row = mask.sum(dim=-1)
    err = (ones_per_row - k).abs().max().item()
    assert err == 0, f"Expected exactly {k} ones, got {ones_per_row.tolist()}"


def test_soft_topk_selects_top_k():
    torch.manual_seed(1)
    logits = torch.randn(4, 50)
    k = 10
    mask = soft_topk_mask(logits, k=k, temperature=0.0001)
    for row in range(4):
        expected = logits[row].topk(k).indices
        actual = (mask[row] > 0.5).nonzero(as_tuple=True)[0]
        assert set(actual.tolist()) == set(expected.tolist())


def test_soft_topk_return_soft_matches_ste_forward():
    torch.manual_seed(2)
    logits = torch.randn(3, 40)
    ste, soft = soft_topk_mask(logits, k=8, temperature=0.5, return_soft=True)
    assert ((ste > 0.5).sum(dim=-1) == 8).all()
    assert ((soft > 0) & (soft < 1)).any()
    fwd_vals = ste.detach()
    hard = (fwd_vals > 0.5).float()
    assert torch.allclose(fwd_vals, hard), "STE forward value must equal the hard mask"


def test_fsa_mask_logits_shape_and_keep_bins():
    fsa = FreqSparseAttention(d_model=64, n_heads=4, d_k=16, max_seq_len=128, chunk_size=32, topk_ratio=0.25)
    assert fsa.mask_logits.shape == (4, 32)
    assert fsa.keep_bins == 8
    idx = fsa.mask_logits.topk(fsa.keep_bins, dim=-1).indices
    assert idx.shape == (4, 8)


def test_fsa_cross_chunk_mask_is_strictly_causal():
    fsa = FreqSparseAttention(d_model=64, n_heads=4, d_k=16, max_seq_len=128, chunk_size=16)
    L = 64
    allowed = fsa._cross_chunk_mask(L, torch.device("cpu"))
    assert allowed.shape == (L, 4 * fsa.keep_bins)
    assert not allowed[:16].any(), "first chunk positions must not attend any compressed bin"
    assert allowed[16, : fsa.keep_bins].all(), "position 16 should attend chunk-0 bins"
    assert not allowed[16, fsa.keep_bins :].all(), "position 16 must not attend chunk-1+ bins"
    assert allowed[63, : 3 * fsa.keep_bins].all(), "position 63 should attend chunks 0-2 bins"
    assert not allowed[63, 3 * fsa.keep_bins :].any(), "position 63 must not attend its own chunk bins"


def test_fsa_intra_chunk_mask_tril_and_pad():
    fsa = FreqSparseAttention(d_model=64, n_heads=4, d_k=16, max_seq_len=128, chunk_size=16)
    mask = fsa._intra_chunk_mask(n_chunks=2, pad=5, device=torch.device("cpu"))
    assert mask.shape == (2, 16, 16)
    tril = torch.tril(torch.ones(16, 16, dtype=torch.bool))
    assert torch.equal(mask[0], tril)
    expected_last = tril & (torch.arange(16) < 11).view(1, 16)
    assert torch.equal(mask[1], expected_last)


def test_fsa_chunk_dct_roundtrip():
    n = 32
    torch.manual_seed(0)
    x = torch.randn(2, n, 8, dtype=torch.float64)
    W = dct_matrix(n, device="cpu", dtype=torch.float64)
    X = torch.einsum("bld,kl->bkd", x, W)
    x_back = torch.einsum("bkd,kl->bld", X, W)
    err = (x - x_back).abs().max().item()
    assert err < 1e-9, f"chunk DCT roundtrip err {err}"


def test_fsa_attention_finite_and_nonzero():
    fsa = FreqSparseAttention(d_model=128, n_heads=4, d_k=32)
    x = torch.randn(2, 32, 128)
    out = fsa(x)
    assert torch.isfinite(out).all(), "Output contains NaN/Inf"
    out_std = out.std().item()
    assert out_std > 1e-7, f"Output collapsed to zero: std={out_std}"


def test_fsa_causality_mid_sequence_perturbation():
    torch.manual_seed(7)
    fsa = FreqSparseAttention(d_model=64, n_heads=4, d_k=16, max_seq_len=128, chunk_size=8).eval()
    x1 = torch.randn(1, 40, 64)
    for cut in (8, 9, 17, 31, 39):
        x2 = x1.clone()
        x2[:, cut:] = torch.randn(1, 40 - cut, 64)
        with torch.no_grad():
            o1 = fsa(x1)
            o2 = fsa(x2)
        diff = (o1[:, :cut] - o2[:, :cut]).abs().max().item()
        assert diff < 1e-5, f"cut={cut}: past outputs changed by {diff}"