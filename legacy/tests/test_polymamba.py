from __future__ import annotations

import torch

from aether.components.polymamba import PolyMamba2


def test_polymamba_forward_shape():
    pm = PolyMamba2(d_model=64, d_inner=128, n_orders=2, state_dim=16)
    x = torch.randn(2, 32, 64)
    out = pm(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_hurwitz_init():
    pm = PolyMamba2(d_model=64, d_inner=128, n_orders=2, state_dim=16)
    A = pm._A()
    for k in range(pm.n_orders):
        eigvals = torch.linalg.eigvals(A[k])
        max_radius = eigvals.abs().max().item()
        assert max_radius < 1.0, f"Order {k} A has spectral radius {max_radius} >= 1.0"


def test_spectral_radius_bounded_after_step():
    pm = PolyMamba2(d_model=64, d_inner=128, n_orders=2, state_dim=16)
    optim = torch.optim.Adam(pm.parameters(), lr=1e-2)
    x = torch.randn(2, 16, 64)
    for _ in range(10):
        out = pm(x)
        out.sum().backward()
        optim.step()
        optim.zero_grad()
    radii = pm.spectral_radius()
    assert (radii < 1.0).all(), f"Spectral radii exceeded 1.0: {radii}"


def test_balance_loss_bounds():
    pm = PolyMamba2(d_model=64, d_inner=128, n_orders=2, state_dim=16)
    x = torch.randn(2, 16, 64)
    pm(x)
    loss = pm.balance_loss()
    assert 0.0 <= loss.item() < 5.0, f"Balance loss out of range: {loss.item()}"


def test_balance_loss_positive_when_gate_collapsed_on_data():
    pm = PolyMamba2(d_model=64, d_inner=128, n_orders=2, state_dim=16)
    with torch.no_grad():
        pm.gate.weight.data[0, :] = 5.0
        pm.gate.weight.data[1, :] = -5.0
    x = torch.ones(2, 16, 64)
    pm(x)
    loss = pm.balance_loss()
    assert loss.item() > 0.1, f"globally collapsed gate should give loss > 0.1, got {loss.item()}"


def test_balance_loss_zero_when_gate_uniform_on_data():
    pm = PolyMamba2(d_model=64, d_inner=128, n_orders=2, state_dim=16)
    with torch.no_grad():
        pm.gate.weight.data.zero_()
    x = torch.randn(2, 16, 64)
    pm(x)
    loss = pm.balance_loss()
    assert loss.item() < 0.05, f"uniform gate should give ~0 loss, got {loss.item()}"


def test_gradient_flow_through_A():
    pm = PolyMamba2(d_model=64, d_inner=128, n_orders=2, state_dim=16)
    x = torch.randn(2, 16, 64)
    out = pm(x)
    out.sum().backward()
    assert pm.A_log.grad is not None
    assert torch.isfinite(pm.A_log.grad).all()
    assert pm.A_log.grad.abs().sum() > 0
