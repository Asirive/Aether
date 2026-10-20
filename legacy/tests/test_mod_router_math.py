from __future__ import annotations

import torch

from aether.components.mod_router import MoDRouter


def test_mod_router_gate_count_matches_ratio():
    torch.manual_seed(1)
    router = MoDRouter(d_model=64, routing_points=(4,), keep_ratios=(0.5,))
    for seq_len in [8, 16, 32, 100]:
        x = torch.randn(4, seq_len, 64)
        _, _, state, gate = router(x, layer_id=4, state={})
        kept = state["mod_mask_4"].sum(dim=-1)
        expected = int(seq_len * 0.5)
        err = (kept - expected).abs().max().item()
        assert err == 0, f"seq_len={seq_len}: expected {expected} kept, got {kept.tolist()}"


def test_mod_router_gradient_flows_to_route_proj():
    torch.manual_seed(2)
    router = MoDRouter(d_model=64, routing_points=(4,), keep_ratios=(0.5,))
    x = torch.randn(1, 32, 64, requires_grad=True)
    _, loss, _, gate = router(x, layer_id=4, state={})
    (gate.sum() + loss).backward()
    assert router.route_proj.weight.grad is not None
    assert torch.isfinite(router.route_proj.weight.grad).all()
    assert router.route_proj.weight.grad.abs().sum().item() > 0


def test_mod_router_residual_gate_math():
    torch.manual_seed(0)
    router = MoDRouter(d_model=64, routing_points=(4,), keep_ratios=(0.5,))
    x = torch.randn(2, 16, 64)
    block_out = torch.randn(2, 16, 64)
    _, _, state, gate = router(x, layer_id=4, state={})
    blended = gate * block_out + (1.0 - gate) * x
    mask = state["mod_mask_4"].unsqueeze(-1)
    unselected = mask == 0
    assert torch.allclose(blended[unselected.expand_as(blended)], x[unselected.expand_as(x)]), (
        "unselected tokens must see identity passthrough"
    )
    selected = mask.bool().expand_as(blended)
    assert torch.allclose(blended[selected], block_out[selected]), (
        "selected tokens must see block output"
    )


def test_mod_router_routing_points_decreasing_density():
    torch.manual_seed(4)
    router = MoDRouter(d_model=64, routing_points=(4, 8, 12), keep_ratios=(0.5, 0.7, 1.0))
    x = torch.randn(2, 100, 64)
    sizes = []
    for layer_id in [4, 8, 12]:
        _, _, state, _ = router(x, layer_id=layer_id, state={})
        sizes.append(state[f"mod_mask_{layer_id}"].sum(dim=-1)[0].item())
    assert sizes[0] == 50
    assert sizes[1] == 70
    assert sizes[2] == 100


def test_mod_router_balance_loss_is_non_neg_finite():
    router = MoDRouter(d_model=64, routing_points=(4, 8, 12), keep_ratios=(0.5, 0.7, 1.0))
    for trial in range(5):
        torch.manual_seed(trial)
        x = torch.randn(2, 100, 64)
        for layer_id in [4, 8, 12]:
            _, loss, _, _ = router(x, layer_id=layer_id, state={})
            assert loss.item() >= 0.0
            assert torch.isfinite(loss).item()


def test_mod_router_keep_ratio_one_selects_all():
    router = MoDRouter(d_model=64, routing_points=(4,), keep_ratios=(1.0,))
    x = torch.randn(2, 32, 64)
    _, _, _, gate = router(x, layer_id=4, state={})
    assert (gate == 1.0).all(), "keep_ratio=1.0 must select every token"


def test_router_gradient_is_invariant_to_residual_scale():
    """Root cause of the training collapse (13-15 Sep): the router read the raw
    residual stream, so its gradient grew with ||x||^2 / temperature until it
    alone pinned the global grad norm. Normalising the router input must make
    the parameter gradient scale-invariant."""
    torch.manual_seed(0)
    d_model = 32
    base = torch.randn(2, 128, d_model)
    grads = []
    for scale in (1.0, 4.0, 16.0):
        router = MoDRouter(d_model=d_model, routing_points=(0,), keep_ratios=(0.5,))
        _, loss, _, _ = router(base * scale, layer_id=0)
        loss.backward()
        grads.append(float(router.route_proj.weight.grad.norm()))
    assert grads[-1] / max(grads[0], 1e-12) < 3.0, f"router grad grew with ||x||: {grads}"


def test_router_output_is_finite_with_large_input():
    router = MoDRouter(d_model=32, routing_points=(0,), keep_ratios=(0.5,))
    _, loss, _, gate = router(torch.randn(2, 64, 32) * 500.0, layer_id=0)
    assert torch.isfinite(loss)
    assert torch.isfinite(gate).all()