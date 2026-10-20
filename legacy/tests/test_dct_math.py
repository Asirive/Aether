from __future__ import annotations

import math

import torch

from aether.ops.dct import dct_apply, dct_matrix, idct_apply


def test_dct_orthonormal():
    """W must be orthonormal: W @ W.T = I."""
    for n in [16, 32, 64, 128, 100, 257]:
        W = dct_matrix(n, device="cpu", dtype=torch.float64)
        gram = W @ W.T
        identity = torch.eye(n, dtype=torch.float64)
        err = (gram - identity).abs().max().item()
        assert err < 1e-9, f"n={n}: W @ W.T - I max err {err}"


def test_dct_inverse_relation():
    """DCT-II and IDCT-II are inverses up to normalization."""
    for n in [16, 64, 256]:
        torch.manual_seed(n)
        x = torch.randn(4, n, dtype=torch.float64)
        W = dct_matrix(n, device="cpu", dtype=torch.float64)
        X = dct_apply(x, W)
        x_recon = idct_apply(X, W)
        err = (x - x_recon).abs().max().item()
        assert err < 1e-9, f"n={n}: roundtrip max err {err}"


def test_dct_known_constant():
    """Constant vector DCT should be zero except DC bin."""
    for n in [16, 64, 256]:
        c = torch.full((1, n), 7.0, dtype=torch.float64)
        W = dct_matrix(n, device="cpu", dtype=torch.float64)
        X = dct_apply(c, W)
        x_dc = X[0, 0].item()
        x_other_max = X[0, 1:].abs().max().item()
        assert abs(x_dc - 7.0 * math.sqrt(n)) < 1e-9
        assert x_other_max < 1e-9


def test_dct_known_basis():
    """DCT-II basis vectors are columns of W: DCT(e_k) = W[:, k]."""
    n = 32
    W = dct_matrix(n, device="cpu", dtype=torch.float64)
    for k in [0, 5, 17, 31]:
        e = torch.zeros(1, n, dtype=torch.float64)
        e[0, k] = 1.0
        X = dct_apply(e, W)
        err = (X[0] - W[:, k]).abs().max().item()
        assert err < 1e-12, f"DCT basis vector k={k}: err {err}"


def test_dct_linearity():
    """DCT(a*x + b*y) == a*DCT(x) + b*DCT(y)."""
    torch.manual_seed(0)
    n = 64
    x = torch.randn(3, n, dtype=torch.float64)
    y = torch.randn(3, n, dtype=torch.float64)
    a, b = 2.5, -1.3
    W = dct_matrix(n, device="cpu", dtype=torch.float64)
    lhs = dct_apply(a * x + b * y, W)
    rhs = a * dct_apply(x, W) + b * dct_apply(y, W)
    err = (lhs - rhs).abs().max().item()
    assert err < 1e-12, f"linearity err {err}"


def test_dct_idct_orthonormal():
    """W (DCT) and W.T (IDCT) together form an involution up to scale."""
    n = 48
    W = dct_matrix(n, device="cpu", dtype=torch.float64)
    WT = W.T
    product = W @ WT
    identity = torch.eye(n, dtype=torch.float64)
    err = (product - identity).abs().max().item()
    assert err < 1e-9, f"W @ W.T - I err {err}"
