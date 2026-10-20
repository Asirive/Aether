from __future__ import annotations

import torch

from aether.components.fsa import FreqSparseAttention


def test_fsa_forward_shape():
    fsa = FreqSparseAttention(d_model=64, n_heads=4, d_k=16, max_seq_len=128)
    x = torch.randn(2, 64, 64)
    out = fsa(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_fsa_forward_shape_non_multiple_of_chunk():
    fsa = FreqSparseAttention(d_model=64, n_heads=4, d_k=16, max_seq_len=128, chunk_size=32)
    x = torch.randn(2, 50, 64)
    out = fsa(x)
    assert out.shape == x.shape


def test_fsa_causality_no_future_leak():
    torch.manual_seed(0)
    fsa = FreqSparseAttention(d_model=64, n_heads=4, d_k=16, max_seq_len=128, chunk_size=16).eval()
    x1 = torch.randn(1, 48, 64)
    x2 = x1.clone()
    x2[:, 24:] = torch.randn(1, 24, 64)
    with torch.no_grad():
        out1 = fsa(x1)
        out2 = fsa(x2)
    diff = (out1[:, :24] - out2[:, :24]).abs().max().item()
    assert diff < 1e-5, f"Future leak: outputs at 0..23 changed by {diff} when tokens 24..47 changed"


def test_fsa_first_chunk_has_no_cross_context():
    torch.manual_seed(0)
    fsa = FreqSparseAttention(d_model=64, n_heads=4, d_k=16, max_seq_len=128, chunk_size=16).eval()
    with torch.no_grad():
        fsa.mix_gate.fill_(30.0)
    x = torch.randn(1, 32, 64)
    with torch.no_grad():
        out = fsa(x)
    first_chunk_norm = out[:, :16].abs().max().item()
    assert first_chunk_norm < 1e-4, (
        f"First-chunk output should vanish when gate selects pure cross path, got {first_chunk_norm}"
    )


def test_per_head_mask_diversity():
    torch.manual_seed(42)
    fsa = FreqSparseAttention(d_model=64, n_heads=4, d_k=16, max_seq_len=128)
    with torch.no_grad():
        idx = fsa.mask_logits.topk(fsa.keep_bins, dim=-1).indices
        sets = [set(row.tolist()) for row in idx]
        max_overlap = 0.0
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                overlap = len(sets[i] & sets[j]) / fsa.keep_bins
                max_overlap = max(max_overlap, overlap)
    assert max_overlap < 0.75, f"Heads pick too-similar bin sets: max overlap {max_overlap}"


def test_fsa_orthogonality_loss_bounded():
    fsa = FreqSparseAttention(d_model=64, n_heads=4, d_k=16, max_seq_len=128)
    loss = fsa.orthogonality_loss()
    assert 0.0 <= loss.item() < 16.0, f"orthogonality loss {loss.item()} out of range"


def test_fsa_gradient_flow():
    fsa = FreqSparseAttention(d_model=64, n_heads=4, d_k=16, max_seq_len=128)
    x = torch.randn(2, 64, 64, requires_grad=True)
    out = fsa(x)
    out.sum().backward()
    grads = [p.grad for p in fsa.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_fsa_mask_logits_receive_gradient():
    torch.manual_seed(0)
    fsa = FreqSparseAttention(d_model=64, n_heads=4, d_k=16, max_seq_len=128, chunk_size=16)
    x = torch.randn(2, 96, 64)
    out = fsa(x)
    out.sum().backward()
    assert fsa.mask_logits.grad is not None
    assert fsa.mask_logits.grad.abs().sum() > 0, "mask_logits has zero gradient"
    assert fsa.mix_gate.grad is not None
    assert fsa.mix_gate.grad.abs().sum() > 0, "mix_gate has zero gradient"
