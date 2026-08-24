# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for PConvC3k and PConvC3k2 modules."""

import torch

from ultralytics.nn.modules import PConvBottleneck, PConvC3k, PConvC3k2


def _assert_finite_gradients(module: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> None:
    """Check that backward propagation reaches the input and every trainable parameter."""
    y.square().mean().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all(), "Input gradient is missing or non-finite"
    missing_gradients = [
        name for name, parameter in module.named_parameters() if parameter.requires_grad and parameter.grad is None
    ]
    nonfinite_gradients = [
        name
        for name, parameter in module.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    assert not missing_gradients, f"Parameters without gradients: {missing_gradients}"
    assert not nonfinite_gradients, f"Parameters with non-finite gradients: {nonfinite_gradients}"


def test_pconvc3k() -> None:
    """Verify PConvC3k structure, output shape, and gradient propagation."""
    module = PConvC3k(16, 32, n=2, shortcut=True, e=0.5, k=5, n_div=4).train()
    x = torch.randn(2, 16, 32, 32, requires_grad=True)
    y = module(x)

    assert y.shape == (2, 32, 32, 32)
    assert len(module.m) == 2
    assert all(isinstance(block, PConvBottleneck) for block in module.m)
    assert all(block.cv2.partial_conv3.kernel_size == (5, 5) for block in module.m)
    _assert_finite_gradients(module, x, y)


def test_pconvc3k2() -> None:
    """Verify every PConvC3k2 branch, output shape, and gradient propagation."""
    cases = (
        (PConvC3k2(16, 32, n=2, c3k=False, n_div=4), PConvBottleneck),
        (PConvC3k2(16, 32, n=2, c3k=True, n_div=4), PConvC3k),
        (PConvC3k2(16, 32, n=2, attn=True, n_div=4), torch.nn.Sequential),
    )

    for module, expected_block_type in cases:
        module.train()
        x = torch.randn(2, 16, 32, 32, requires_grad=True)
        y = module(x)

        assert y.shape == (2, 32, 32, 32)
        assert len(module.m) == 2
        assert all(isinstance(block, expected_block_type) for block in module.m)
        if expected_block_type is PConvC3k:
            assert all(isinstance(block.m[0], PConvBottleneck) for block in module.m)
        elif expected_block_type is torch.nn.Sequential:
            assert all(isinstance(block[0], PConvBottleneck) for block in module.m)
        _assert_finite_gradients(module, x, y)
