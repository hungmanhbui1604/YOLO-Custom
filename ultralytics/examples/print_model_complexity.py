# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Print parameter counts and FLOPs for one or more YOLO detection model YAML files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import get_flops, get_num_gradients, get_num_params

DEFAULT_MODEL = PROJECT_ROOT / "ultralytics/cfg/models/26/yolo26-custom.yaml"


def resolve_model_path(model: str | Path) -> Path:
    """Resolve a model YAML from the current directory or the Ultralytics project root."""
    path = Path(model).expanduser()
    for candidate in (path, PROJECT_ROOT / path):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Model YAML not found: {model}")


def parse_image_size(values: list[int]) -> tuple[int, int]:
    """Convert one square size or a height-width pair to a two-dimensional size."""
    if len(values) == 1:
        height = width = values[0]
    elif len(values) == 2:
        height, width = values
    else:
        raise ValueError("--imgsz requires one value (square) or two values (height width)")
    if height <= 0 or width <= 0:
        raise ValueError("--imgsz values must be positive")
    return height, width


def calculate_complexity(model_path: str | Path, scale: str, image_size: tuple[int, int]) -> tuple[int, int, float]:
    """Return total parameters, trainable parameters, and GFLOPs for a model configuration."""
    model_path = resolve_model_path(model_path)
    config = YAML.load(model_path)
    config["scale"] = scale
    model = DetectionModel(config, verbose=False)
    parameters = get_num_params(model)
    trainable_parameters = get_num_gradients(model)
    gflops = get_flops(model, list(image_size))
    if not gflops:
        raise RuntimeError("FLOPs calculation failed. Ensure the 'ultralytics-thop' package is installed.")
    return parameters, trainable_parameters, gflops


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", default=[str(DEFAULT_MODEL)], help="Paths to model YAML files")
    parser.add_argument("--scale", choices=("n", "s", "m", "l", "x"), default="n", help="Model scale")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640], help="Image size: SIZE or HEIGHT WIDTH")
    return parser.parse_args()


def main() -> None:
    """Calculate and print model complexity from the command line."""
    args = parse_args()
    image_size = parse_image_size(args.imgsz)
    input_size = f"{image_size[0]}x{image_size[1]}"
    rows = []
    for model_name in args.models:
        model_path = resolve_model_path(model_name)
        parameters, trainable, gflops = calculate_complexity(model_path, args.scale, image_size)
        rows.append((model_path.stem, args.scale, input_size, parameters, trainable, gflops))

    print(f"{'Model':<26} {'Scale':<5} {'Input':<11} {'Parameters':>14} {'Trainable':>14} {'GFLOPs':>12}")
    print(f"{'-' * 26} {'-' * 5} {'-' * 11} {'-' * 14} {'-' * 14} {'-' * 12}")
    for model_name, scale, size, parameters, trainable, gflops in rows:
        print(f"{model_name:<26} {scale:<5} {size:<11} {parameters:>14,} {trainable:>14,} {gflops:>12.3f}")


if __name__ == "__main__":
    main()
