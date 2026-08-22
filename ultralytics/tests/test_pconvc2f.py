# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Standalone correctness checks for PConv, PConvBottleneck, and PConvC2f."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Prefer this checkout over an Ultralytics package installed in the active environment.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.nn.modules import PConv, PConvBottleneck, PConvC2f
from ultralytics.nn.tasks import DetectionModel


def check_pconv_kernels(device: torch.device, image_size: int) -> None:
    """Check that supported odd kernels preserve tensor shape."""
    x = torch.randn(2, 16, image_size, image_size, device=device)
    for kernel in (1, 3, 5, (5, 5)):
        y = PConv(16, k=kernel, n_div=4).to(device)(x)
        assert y.shape == x.shape, f"PConv(k={kernel}) returned {tuple(y.shape)}, expected {tuple(x.shape)}"


def check_pconv_division_validation() -> None:
    """Check that an invalid channel division fails with a useful error."""
    try:
        PConv(3, n_div=4)
    except ValueError as error:
        assert "n_div" in str(error)
    else:
        raise AssertionError("PConv accepted n_div greater than its input channel count")


def check_pconv_bottleneck(device: torch.device, image_size: int) -> None:
    """Check bottleneck output shapes with and without residual shortcuts."""
    x = torch.randn(2, 16, image_size, image_size, device=device)
    for shortcut in (False, True):
        y = PConvBottleneck(16, 16, shortcut=shortcut).to(device)(x)
        assert y.shape == x.shape, f"PConvBottleneck(shortcut={shortcut}) changed shape to {tuple(y.shape)}"


def check_pconvc2f_forward_backward(device: torch.device, image_size: int) -> None:
    """Check output shape and gradient propagation through every trainable parameter."""
    module = PConvC2f(16, 32, n=2, shortcut=True, g=1, e=0.5, n_div=4).to(device).train()
    x = torch.randn(2, 16, image_size, image_size, device=device, requires_grad=True)
    y = module(x)

    expected_shape = (2, 32, image_size, image_size)
    assert y.shape == expected_shape, f"PConvC2f returned {tuple(y.shape)}, expected {expected_shape}"

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


def check_state_dict_round_trip(device: torch.device, image_size: int) -> None:
    """Check that saved weights load into an equivalent PConvC2f module."""
    source = PConvC2f(16, 32, n=2, n_div=4).to(device).eval()
    restored = PConvC2f(16, 32, n=2, n_div=4).to(device).eval()
    restored.load_state_dict(source.state_dict())
    x = torch.randn(1, 16, image_size, image_size, device=device)

    with torch.no_grad():
        source_output = source(x)
        restored_output = restored(x)
    torch.testing.assert_close(restored_output, source_output)


def check_yaml_integration(device: torch.device, image_size: int) -> None:
    """Check parser channel injection, standard C2f arguments, and internal repeats."""
    cfg = {
        "nc": 1,
        "backbone": [
            [-1, 1, "Conv", [16, 3, 1]],
            [-1, 2, "PConvC2f", [32, False, 1, 0.5, 4]],
            [-1, 1, "Conv", [8, 1, 1]],
        ],
        "head": [],
    }
    model = DetectionModel(cfg, ch=3, verbose=False).to(device).eval()
    layer = model.model[1]
    assert isinstance(layer, PConvC2f), f"Parser created {type(layer).__name__}, expected PConvC2f"
    assert len(layer.m) == 2, f"Parser created {len(layer.m)} internal blocks, expected 2"

    with torch.no_grad():
        y = model(torch.randn(1, 3, image_size, image_size, device=device))
    expected_shape = (1, 8, image_size, image_size)
    assert y.shape == expected_shape, f"Parsed model returned {tuple(y.shape)}, expected {expected_shape}"


def check_routed_pconv_channels(device: torch.device, image_size: int) -> None:
    """Check parser bookkeeping when PConv consumes a non-previous layer."""
    cfg = {
        "nc": 1,
        "backbone": [
            [-1, 1, "Conv", [16, 3, 1]],
            [-1, 1, "Conv", [32, 3, 1]],
            [0, 1, "PConv", []],
            [-1, 1, "Conv", [8, 1, 1]],
        ],
        "head": [],
    }
    model = DetectionModel(cfg, ch=3, verbose=False).to(device).eval()
    assert model.model[3].conv.in_channels == 16, "Parser did not preserve the routed PConv channel count"

    with torch.no_grad():
        y = model(torch.randn(1, 3, image_size, image_size, device=device))
    assert y.shape == (1, 8, image_size, image_size)


def run_checks(device: torch.device, image_size: int, verbose: bool = True) -> None:
    """Run all PConvC2f checks."""
    checks = (
        ("PConv odd kernels", lambda: check_pconv_kernels(device, image_size)),
        ("PConv division validation", check_pconv_division_validation),
        ("PConvBottleneck", lambda: check_pconv_bottleneck(device, image_size)),
        ("PConvC2f forward/backward", lambda: check_pconvc2f_forward_backward(device, image_size)),
        ("state-dict round trip", lambda: check_state_dict_round_trip(device, image_size)),
        ("YAML integration", lambda: check_yaml_integration(device, image_size)),
        ("routed PConv channels", lambda: check_routed_pconv_channels(device, image_size)),
    )
    for name, check in checks:
        check()
        if verbose:
            print(f"PASS: {name}")


def test_pconvc2f_smoke() -> None:
    """Run the complete smoke test on CPU under pytest."""
    torch.manual_seed(0)
    run_checks(torch.device("cpu"), image_size=32, verbose=False)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for standalone execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="Torch device, for example cpu, cuda, or cuda:0")
    parser.add_argument("--image-size", type=int, default=32, help="Square test image size")
    return parser.parse_args()


def main() -> None:
    """Run the standalone test suite."""
    args = parse_args()
    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)
    torch.manual_seed(0)
    run_checks(device, args.image_size)
    print(f"All PConvC2f checks passed on {device}.")


if __name__ == "__main__":
    main()
