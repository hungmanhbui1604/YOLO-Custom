# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Print the input and output tensor sizes of every top-level layer in a YOLO detection model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML

DEFAULT_MODEL = PROJECT_ROOT / "ultralytics/cfg/models/26/yolo26-custom.yaml"


def tensor_sizes(value: Any) -> str:
    """Return a compact recursive representation containing only tensor sizes."""
    if isinstance(value, torch.Tensor):
        return "x".join(map(str, value.shape))
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key}: {tensor_sizes(item)}" for key, item in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        opening, closing = ("[", "]") if isinstance(value, list) else ("(", ")")
        return opening + ", ".join(tensor_sizes(item) for item in value) + closing
    return type(value).__name__


def resolve_model_path(model: str) -> Path:
    """Resolve a model path from the current directory or the Ultralytics project root."""
    path = Path(model).expanduser()
    candidates = (path, PROJECT_ROOT / path)
    for candidate in candidates:
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


def inspect_model(model_path: Path, scale: str, image_size: tuple[int, int], batch: int, device: str) -> None:
    """Build a model, run a dummy input, and print layer input/output sizes."""
    config = YAML.load(model_path)
    config["scale"] = scale
    model = DetectionModel(config, verbose=False).to(device).eval()
    rows: list[tuple[int, str, str, str, str]] = []
    handles = []

    def make_hook(index: int, source: Any, module_name: str):
        def record_shapes(_module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            layer_input = inputs[0] if len(inputs) == 1 else inputs
            rows.append((index, str(source), module_name, tensor_sizes(layer_input), tensor_sizes(output)))

        return record_shapes

    for layer in model.model:
        handles.append(layer.register_forward_hook(make_hook(layer.i, layer.f, type(layer).__name__)))

    height, width = image_size
    try:
        with torch.no_grad():
            model(torch.zeros(batch, 3, height, width, device=device))
    finally:
        for handle in handles:
            handle.remove()

    print(f"Model: {model_path}")
    print(f"Scale: {scale} | Input: {batch}x3x{height}x{width} | Device: {device}")
    print(f"{'Idx':>3}  {'From':<12}  {'Module':<18}  {'Input size':<48}  Output size")
    print(f"{'-' * 3}  {'-' * 12}  {'-' * 18}  {'-' * 48}  {'-' * 48}")
    for index, source, module_name, input_size, output_size in rows:
        print(f"{index:>3}  {source:<12}  {module_name:<18}  {input_size:<48}  {output_size}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", default=str(DEFAULT_MODEL), help="Path to a YOLO detection model YAML")
    parser.add_argument("--scale", choices=("n", "s", "m", "l", "x"), default="n", help="Model scale")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640], help="Image size: SIZE or HEIGHT WIDTH")
    parser.add_argument("--batch", type=int, default=1, help="Dummy input batch size")
    parser.add_argument("--device", default="cpu", help="Torch device, such as cpu, cuda, or cuda:0")
    return parser.parse_args()


def main() -> None:
    """Run layer-shape inspection from the command line."""
    args = parse_args()
    if args.batch <= 0:
        raise ValueError("--batch must be positive")
    inspect_model(
        model_path=resolve_model_path(args.model),
        scale=args.scale,
        image_size=parse_image_size(args.imgsz),
        batch=args.batch,
        device=args.device,
    )


if __name__ == "__main__":
    main()
