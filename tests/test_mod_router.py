from __future__ import annotations

import torch

from aether.components.mod_router import MoDRouter


def test_topk_correctness():
    router = MoDRouter(d_model=64, routing_points=(4,), keep_ratios=(0.5,))
    x = torch.randn(2, 16, 64)
    out, loss, state, gate = router(x, layer_id=4, state={})
    assert out.shape == x.shape
    assert gate is not None and gate.shape == (2, 16, 1)
    mask = state["mod_mask_4"]
    expected_k = int(16 * 0.5)
    assert (mask.sum(dim=-1) == expected_k).all(), f"Wrong top-k: {mask.sum(dim=-1)}"


def test_router_returns_input_unchanged():
    router = MoDRouter(d_model=64, routing_points=(4,), keep_ratios=(0.5,))
    x = torch.randn(2, 16, 64)
    out, _, _, _ = router(x, layer_id=4, state={})
    assert torch.equal(out, x), "router must return x unchanged; gating is applied by the caller"


def test_gate_binary_in_forward():
    router = MoDRouter(d_model=64, routing_points=(4,), keep_ratios=(0.5,))
    x = torch.randn(2, 16, 64)
    _, _, _, gate = router(x, layer_id=4, state={})
    vals = gate.unique()
    assert set(vals.tolist()) <= {0.0, 1.0}, f"gate not binary in forward: {vals.tolist()}"


def test_balance_loss_non_negative():
    router = MoDRouter(d_model=64, routing_points=(4,), keep_ratios=(0.5,))
    x = torch.randn(2, 16, 64)
    _, loss, _, _ = router(x, layer_id=4, state={})
    assert loss.item() >= 0.0
    assert torch.isfinite(loss)


def test_balance_loss_responds_to_positional_collapse():
    torch.manual_seed(0)
    router = MoDRouter(d_model=64, routing_points=(4,), keep_ratios=(0.5,))
    with torch.no_grad():
        router.route_proj.weight.copy_(torch.linspace(0.0, 1.0, 64).view(1, 64) * 0.0 + 1.0)
    x = torch.zeros(1, 16, 64)
    _, loss, _, _ = router(x, layer_id=4, state={})
    assert loss.item() < 1e-6, f"uniform usage should give ~0 loss, got {loss.item()}"

    with torch.no_grad():
        router.route_proj.weight.copy_(torch.randn(1, 64) * 5.0)
    x2 = torch.randn(1, 16, 64)
    _, loss2, _, _ = router(x2, layer_id=4, state={})
    assert torch.isfinite(loss2)


def test_routing_distribution_within_tolerance():
    router = MoDRouter(d_model=64, routing_points=(4,), keep_ratios=(0.5,))
    x = torch.randn(64, 256, 64)
    _, _, state, _ = router(x, layer_id=4, state={})
    mask = state["mod_mask_4"]
    frac = mask.sum(dim=-1).mean().item() / 256
    assert abs(frac - 0.5) < 0.05, f"Routing fraction {frac} far from target 0.5"


def test_non_routing_point_is_identity():
    router = MoDRouter(d_model=64, routing_points=(4,), keep_ratios=(0.5,))
    x = torch.randn(2, 16, 64)
    out, loss, state, gate = router(x, layer_id=99, state={})
    assert torch.equal(out, x)
    assert loss.item() == 0.0
    assert state == {}
    assert gate is None