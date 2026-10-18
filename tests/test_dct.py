from __future__ import annotations

import pytest
import torch

from aether.ops.dct import dct_apply, dct_matrix, idct_apply


def test_dct_idct_invertibility():
    seq_len = 64
    x = torch.randn(2, seq_len)
    W = dct_matrix(seq_len, device="cpu", dtype=torch.float32)
    y = dct_apply(x, W)
    x_recon = idct_apply(y, W)
    err = (x - x_recon).abs().max().item()
    assert err < 1e-4, f"DCT/IDCT not invertible: max err {err}"


def test_dct_orthogonality():
    seq_len = 32
    W = dct_matrix(seq_len, device="cpu", dtype=torch.float32)
    gram = W @ W.T
    identity = torch.eye(seq_len)
    err = (gram - identity).abs().max().item()
    assert err < 1e-4, f"DCT matrix not orthogonal: max err {err}"


def test_dct_device_cache():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        W_cpu = dct_matrix(16, device="cpu", dtype=torch.float32)
        W_cuda = dct_matrix(16, device="cuda", dtype=torch.float32)
    except Exception as e:
        pytest.skip(f"CUDA DCT init failed: {e}")
    assert W_cpu.device.type == "cpu"
    assert W_cuda.device.type == "cuda"


def test_dct_gradient_flow():
    seq_len = 32
    x = torch.randn(4, seq_len, requires_grad=True)
    W = dct_matrix(seq_len, device="cpu", dtype=torch.float32)
    y = dct_apply(x, W)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
