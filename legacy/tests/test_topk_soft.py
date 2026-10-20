from __future__ import annotations

import torch

from aether.ops.topk_soft import soft_topk_mask


def test_topk_correctness():
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
    mask = soft_topk_mask(logits, k=2, temperature=0.0)
    selected = (mask[0] > 0.5).nonzero(as_tuple=True)[0].tolist()
    assert sorted(selected) == [1, 4], f"Expected indices [1, 4], got {selected}"


def test_topk_soft_to_hard_anneal():
    logits = torch.randn(1, 100)
    mask_hard = soft_topk_mask(logits, k=10, temperature=0.001)
    mask_soft = soft_topk_mask(logits, k=10, temperature=5.0)
    hard_density = (mask_hard > 0.9).float().sum().item()
    soft_density = (mask_soft > 0.1).float().sum().item()
    assert soft_density >= hard_density, f"Soft mask should be denser: hard={hard_density}, soft={soft_density}"


def test_topk_gradient_flow():
    logits = torch.randn(2, 50, requires_grad=True)
    mask = soft_topk_mask(logits, k=5, temperature=0.1)
    mask.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
