# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

import torch

from ultralytics.nn.modules import DySample, PConvC2f, SPDConv
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML

MODEL_CONFIG_DIR = Path(__file__).resolve().parents[1] / "ultralytics/cfg/models/26"


def _build_and_forward(config_name: str) -> tuple[DetectionModel, dict]:
    """Build the nano variant of a custom YOLO26 model and run one training-mode forward pass."""
    config = YAML.load(MODEL_CONFIG_DIR / config_name)
    config["scale"] = "n"
    model = DetectionModel(config, verbose=False).train()

    with torch.no_grad():
        output = model(torch.zeros(1, 3, 64, 64))["one2many"]
    return model, output


def test_yolo26_custom_model() -> None:
    """Verify the custom P3-P5 model composition, strides, and detection feature maps."""
    model, output = _build_and_forward("yolo26-custom.yaml")

    assert sum(isinstance(module, SPDConv) for module in model.modules()) == 2
    assert sum(isinstance(module, DySample) for module in model.modules()) == 2
    assert sum(isinstance(module, PConvC2f) for module in model.modules()) == 8
    assert model.stride.tolist() == [8.0, 16.0, 32.0]
    assert [tuple(feature.shape) for feature in output["feats"]] == [
        (1, 64, 8, 8),
        (1, 128, 4, 4),
        (1, 256, 2, 2),
    ]
    assert output["boxes"].shape == (1, 4, 84)
    assert output["scores"].shape == (1, 80, 84)


def test_yolo26_p2_custom_model() -> None:
    """Verify the custom P2-P5 model composition, strides, and additional P2 detection feature map."""
    model, output = _build_and_forward("yolo26-p2-custom.yaml")

    assert sum(isinstance(module, SPDConv) for module in model.modules()) == 2
    assert sum(isinstance(module, DySample) for module in model.modules()) == 3
    assert sum(isinstance(module, PConvC2f) for module in model.modules()) == 10
    assert model.stride.tolist() == [4.0, 8.0, 16.0, 32.0]
    assert [tuple(feature.shape) for feature in output["feats"]] == [
        (1, 32, 16, 16),
        (1, 64, 8, 8),
        (1, 128, 4, 4),
        (1, 256, 2, 2),
    ]
    assert output["boxes"].shape == (1, 4, 340)
    assert output["scores"].shape == (1, 80, 340)
