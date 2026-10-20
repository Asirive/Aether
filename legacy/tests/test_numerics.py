from __future__ import annotations

import torch

from aether.utils.numerics import assert_finite, to_bf16, to_fp32


def test_to_fp32_promotes_bf16():
    x = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    y = to_fp32(x)
    assert y.dtype == torch.float32


def test_to_fp32_preserves_fp32():
    x = torch.tensor([1.0, 2.0], dtype=torch.float32)
    y = to_fp32(x)
    assert y.dtype == torch.float32


def test_to_bf16_promotes_fp32():
    x = torch.tensor([1.0, 2.0], dtype=torch.float32)
    y = to_bf16(x)
    assert y.dtype == torch.bfloat16


def test_assert_finite_passes_on_finite():
    x = torch.tensor([1.0, 2.0, 3.0])
    assert_finite(x, name="test_tensor")


def test_assert_finite_raises_on_nan():
    x = torch.tensor([1.0, float("nan"), 3.0])
    try:
        assert_finite(x, name="test_nan")
        raise AssertionError("Expected FloatingPointError")
    except FloatingPointError:
        pass


def test_assert_finite_raises_on_inf():
    x = torch.tensor([1.0, float("inf"), 3.0])
    try:
        assert_finite(x, name="test_inf")
        raise AssertionError("Expected FloatingPointError")
    except FloatingPointError:
        pass
