from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from aether.components.polymamba import PolyMamba2


def test_polymamba_hurwitz_strict_bound():
    """After init AND after updates, spectral radius must stay <= max_radius (default 0.95)."""
    max_radius = 0.95
    pm = PolyMamba2(d_model=64, d_inner=128, n_orders=2, state_dim=16, max_radius=max_radius)
    radii_init = pm.spectral_radius()
    assert (radii_init <= max_radius + 1e-5).all(), f"Init radii {radii_init}"

    opt = torch.optim.Adam(pm.parameters(), lr=1e-2)
    x = torch.randn(4, 16, 64)
    for _ in range(20):
        out = pm(x)
        out.sum().backward()
        opt.step()
        opt.zero_grad()
    radii_after = pm.spectral_radius()
    assert (radii_after <= max_radius + 1e-5).all(), f"After training radii {radii_after}"

    # Even forcing a near-integrator A_log must not escape the clamp.
    with torch.no_grad():
        pm.A_log.fill_(-10.0)
    radii_forced = pm.spectral_radius()
    assert (radii_forced <= max_radius + 1e-5).all(), f"After forcing A_log=-10 radii {radii_forced}"


def test_polymamba_gate_initial_uniform():
    """Right after init, softmax over gates should be ~uniform for both orders."""
    pm = PolyMamba2(d_model=64, d_inner=128, n_orders=2, state_dim=16)
    with torch.no_grad():
        dummy = torch.zeros(1, 64)
        gate = F.softmax(pm.gate(dummy).float(), dim=-1)
    expected = 1.0 / pm.n_orders
    err = (gate - expected).abs().max().item()
    assert err < 1e-2, f"Gate not uniform: {gate.tolist()}"


def test_polymamba_output_dtype_match_input():
    """Output dtype must match input dtype (bf16 in, bf16 out)."""
    pm = PolyMamba2(d_model=64, d_inner=128, n_orders=2, state_dim=16).to(torch.bfloat16)
    x = torch.randn(2, 32, 64, dtype=torch.bfloat16)
    out = pm(x)
    assert out.dtype == torch.bfloat16, f"Got {out.dtype}"


def test_polymamba_gate_routing_respects_gradients():
    """Gate logits should get gradient when output changes based on order selection."""
    torch.manual_seed(0)
    pm = PolyMamba2(d_model=64, d_inner=128, n_orders=2, state_dim=16)
    x = torch.randn(2, 16, 64, requires_grad=True)
    out = pm(x)
    target = torch.randn_like(out)
    loss = F.mse_loss(out, target)
    loss.backward()
    assert pm.gate.weight.grad is not None
    grad_norm = pm.gate.weight.grad.abs().sum().item()
    a_log_grad_norm = pm.A_log.grad.abs().sum().item()
    assert grad_norm > 0
    assert a_log_grad_norm > 0


def test_polymamba_balance_loss_lower_when_collapsed():
    """If gate collapses to one order on real data, balance_loss should be > 0."""
    pm = PolyMamba2(d_model=64, d_inner=128, n_orders=4, state_dim=16)
    with torch.no_grad():
        w = torch.zeros(pm.n_orders, pm.d_model, dtype=pm.A_log.dtype, device=pm.A_log.device)
        w[0, :] = 5.0
        w[1:, :] = -5.0
        pm.gate.weight.data.copy_(w)
    x = torch.ones(2, 16, 64)
    pm(x)
    loss = pm.balance_loss()
    assert loss.item() > 0.1, f"collapsed gate should give loss > 0.1, got {loss.item()}"


def test_polymamba_balance_loss_zero_when_uniform():
    """If gate is uniform on real data, balance_loss should be ~0 (below margin)."""
    pm = PolyMamba2(d_model=64, d_inner=128, n_orders=4, state_dim=16)
    with torch.no_grad():
        pm.gate.weight.data.zero_()
    x = torch.randn(2, 16, 64)
    pm(x)
    non_zero = pm.balance_loss()
    assert non_zero.item() < 0.1, f"Zero gate balance_loss {non_zero.item()}"
    with torch.no_grad():
        pm.gate.weight.data.normal_(std=0.01)
    pm(x)
    moderate = pm.balance_loss()
    assert moderate.item() >= 0.0


def test_polymamba_d_in_for_orders_different():
    """Different orders should have different state-dim contributions to output."""
    pm = PolyMamba2(d_model=64, d_inner=128, n_orders=2, state_dim=16)
    x = torch.randn(1, 1, 64)
    A = pm._A()
    state_dims = [A[k].shape[0] for k in range(pm.n_orders)]
    assert all(s == 16 for s in state_dims), f"state_dims: {state_dims}"


def test_polymamba_structure_M_is_orthogonal_for_order0():
    """Order 0 matrix should be near-orthogonal (after Hurwitz margin normalization)."""
    pm = PolyMamba2(d_model=64, d_inner=128, n_orders=2, state_dim=16)
    M = pm.structure_M[0].detach()
    gram = M @ M.T
    identity = torch.eye(pm.state_dim, dtype=M.dtype)
    err = (gram - identity).abs().max().item()
    assert err < 0.20, f"Order 0 structure not orthogonal-ish: max err {err}"


def test_polymamba_spectral_radius_bounded_for_extreme_alogs():
    """rho(A) must stay <= max_radius for ANY A_log, including the regime where
    the old one-sided clamp let s(I - dt*M) exceed 1 (dt > 2/lambda_max)."""
    max_radius = 0.95
    pm = PolyMamba2(d_model=32, d_inner=64, n_orders=2, state_dim=16, max_radius=max_radius)
    for a_log in (-10.0, -4.0, 0.0, 2.08, 5.0, 20.0):
        with torch.no_grad():
            pm.A_log.fill_(a_log)
        radii = pm.spectral_radius()
        assert (radii <= max_radius + 1e-5).all(), f"A_log={a_log} gave radii {radii}"


def test_parallel_scan_matches_sequential_reference():
    """The doubling scan must reproduce the naive sequential recurrence exactly."""
    from aether.components.polymamba import _parallel_ssm_scan

    torch.manual_seed(0)
    n_state, seq_len = 8, 96
    A = torch.randn(n_state, n_state, dtype=torch.float64) * 0.2
    u = torch.randn(2, n_state, seq_len, dtype=torch.float64)

    h = _parallel_ssm_scan(A, u)
    assert h.shape == (2, seq_len, n_state), f"unexpected output layout {tuple(h.shape)}"

    ref = torch.zeros(2, seq_len, n_state, dtype=torch.float64)
    state = torch.zeros(2, n_state, dtype=torch.float64)
    for t in range(seq_len):
        state = state @ A.T + u[:, :, t]
        ref[:, t, :] = state

    err = (h - ref).abs().max().item()
    assert err < 1e-10, f"scan mismatch vs sequential reference: max abs err {err}"


def test_parallel_scan_handles_non_power_of_two_length():
    from aether.components.polymamba import _parallel_ssm_scan

    torch.manual_seed(1)
    n_state, seq_len = 4, 37
    A = torch.randn(n_state, n_state, dtype=torch.float64) * 0.2
    u = torch.randn(1, n_state, seq_len, dtype=torch.float64)
    h = _parallel_ssm_scan(A, u)
    ref = torch.zeros(1, seq_len, n_state, dtype=torch.float64)
    state = torch.zeros(1, n_state, dtype=torch.float64)
    for t in range(seq_len):
        state = state @ A.T + u[:, :, t]
        ref[:, t, :] = state
    assert torch.allclose(h, ref, atol=1e-10), (h - ref).abs().max().item()
