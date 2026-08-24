# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Compare parameter counts and FLOPs between two YOLO detection model YAML files."""

from __future__ import annotations

import argparse

from print_model_complexity import calculate_complexity, parse_image_size, resolve_model_path


def percentage_change(original: int | float, updated: int | float) -> float:
    """Return the percentage change from an original value to an updated value."""
    return (updated - original) / original * 100


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_a", help="Path to the first model YAML (comparison baseline)")
    parser.add_argument("model_b", help="Path to the second model YAML")
    parser.add_argument(
        "--scales",
        nargs="+",
        choices=("n", "s", "m", "l", "x"),
        default=("n", "s", "m", "l", "x"),
        help="Model scales to compare",
    )
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640], help="Image size: SIZE or HEIGHT WIDTH")
    return parser.parse_args()


def main() -> None:
    """Compare two model configurations across the selected scales."""
    args = parse_args()
    model_a = resolve_model_path(args.model_a)
    model_b = resolve_model_path(args.model_b)
    image_size = parse_image_size(args.imgsz)

    print(f"Model A: {model_a}")
    print(f"Model B: {model_b}")
    print(f"Input size: {image_size[0]}x{image_size[1]}")
    print("Deltas: Model B - Model A\n")
    print(
        f"{'Scale':<5} {'A params':>14} {'B params':>14} {'Param delta':>14} {'Param %':>9} "
        f"{'A GFLOPs':>12} {'B GFLOPs':>12} {'FLOPs delta':>13} {'FLOPs %':>9}"
    )
    print(
        f"{'-' * 5} {'-' * 14} {'-' * 14} {'-' * 14} {'-' * 9} "
        f"{'-' * 12} {'-' * 12} {'-' * 13} {'-' * 9}"
    )

    for scale in args.scales:
        params_a, _, gflops_a = calculate_complexity(model_a, scale, image_size)
        params_b, _, gflops_b = calculate_complexity(model_b, scale, image_size)
        param_delta = params_b - params_a
        flops_delta = gflops_b - gflops_a
        print(
            f"{scale:<5} {params_a:>14,} {params_b:>14,} {param_delta:>+14,} "
            f"{percentage_change(params_a, params_b):>+8.2f}% {gflops_a:>12.3f} "
            f"{gflops_b:>12.3f} {flops_delta:>+13.3f} {percentage_change(gflops_a, gflops_b):>+8.2f}%"
        )


if __name__ == "__main__":
    main()
