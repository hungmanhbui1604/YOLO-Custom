# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import unittest
from copy import deepcopy

import torch
import torch.nn.functional as F

from ultralytics.nn.modules import DySample
from ultralytics.nn.tasks import parse_model


class TestDySample(unittest.TestCase):
    """Tests for the DySample upsampling module and its model-parser integration."""

    def test_forward_backward_all_variants(self):
        """All paper variants should upsample without changing the channel count."""
        variants = (
            ("lp", 4, False),
            ("lp", 4, True),
            ("pl", 8, False),
            ("pl", 8, True),
        )

        for style, groups, dyscope in variants:
            with self.subTest(style=style, groups=groups, dyscope=dyscope):
                torch.manual_seed(0)
                module = DySample(
                    c1=64,
                    scale=2,
                    style=style,
                    groups=groups,
                    dyscope=dyscope,
                )
                x = torch.randn(2, 64, 7, 11, requires_grad=True)

                output = module(x)

                self.assertEqual(output.shape, (2, 64, 14, 22))
                self.assertTrue(torch.isfinite(output).all())

                output.mean().backward()
                self.assertIsNotNone(x.grad)
                self.assertTrue(torch.isfinite(x.grad).all())

    def test_zero_offsets_match_bilinear_interpolation(self):
        """The initial sampling grid should reduce to bilinear upsampling at zero offset."""
        torch.manual_seed(0)
        module = DySample(c1=16, scale=2, style="lp", groups=4, dyscope=False).eval()
        with torch.no_grad():
            module.offset.weight.zero_()
            module.offset.bias.zero_()

        x = torch.randn(2, 16, 7, 11)
        output = module(x)
        expected = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        torch.testing.assert_close(output, expected, rtol=1e-5, atol=3e-6)

    def test_invalid_arguments(self):
        """Invalid scale, style, and channel-group combinations should fail clearly."""
        cases = (
            ({"c1": 64, "scale": 0}, "scale"),
            ({"c1": 64, "groups": 0}, "groups"),
            ({"c1": 64, "style": "invalid"}, "style"),
            ({"c1": 62, "groups": 4}, "divisible by groups"),
            ({"c1": 66, "scale": 2, "style": "pl", "groups": 3}, "divisible by scale"),
        )

        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    DySample(**kwargs)

    def test_parse_model_preserves_channels_and_yaml_arguments(self):
        """parse_model should inject c1 while preserving DySample's YAML arguments."""
        config = {
            "nc": 1,
            "depth_multiple": 1.0,
            "width_multiple": 1.0,
            "backbone": [
                [-1, 1, "DySample", [2, "lp", 4, False]],
                [-1, 1, "Conv", [32, 3, 1]],
            ],
            "head": [],
        }

        model, _ = parse_model(deepcopy(config), ch=64, verbose=False)
        dysample = model[0]

        self.assertIsInstance(dysample, DySample)
        self.assertEqual(dysample.scale, 2)
        self.assertEqual(dysample.style, "lp")
        self.assertEqual(dysample.groups, 4)

        output = model(torch.randn(1, 64, 8, 12))
        self.assertEqual(output.shape, (1, 32, 16, 24))


if __name__ == "__main__":
    unittest.main()
