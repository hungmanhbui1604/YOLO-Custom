import torch

from ultralytics.nn.modules import PConv, SPDConv

try:
    import pytest
except ModuleNotFoundError:
    pytest = None

SPD_CASES = (
    (2, (2, 16, 64, 64), 32, False, None),
    (3, (2, 8, 63, 96), 24, False, None),
    (2, (2, 16, 64, 64), 32, True, None),
    (3, (2, 8, 63, 96), 24, True, 8),
)
SPD_CASE_IDS = (
    "conv-scale2",
    "conv-scale3",
    "pconv-default-division",
    "pconv-explicit-division",
)
INVALID_SCALES = (1, 0, -2, 2.0)


def reference_space_to_depth(x: torch.Tensor, scale: int) -> torch.Tensor:
    """Apply the explicit space-to-depth operation used by SPDConv."""
    return torch.cat(
        [x[..., row::scale, col::scale] for col in range(scale) for row in range(scale)],
        dim=1,
    )


def test_spdconv(
    scale: int,
    input_shape: tuple[int, ...],
    output_channels: int,
    pconv: bool,
    n_div: int | None,
) -> None:
    """Verify SPDConv output, space-to-depth ordering, and backward propagation."""
    torch.manual_seed(0)

    input_channels = input_shape[1]
    model = SPDConv(
        c1=input_channels,
        c2=output_channels,
        k=3,
        scale=scale,
        pconv=pconv,
        n_div=n_div,
    ).eval()
    x = torch.randn(*input_shape, requires_grad=True)

    output = model(x)
    expected_spatial_shape = (input_shape[2] // scale, input_shape[3] // scale)
    assert output.shape == (input_shape[0], output_channels, *expected_spatial_shape)

    spd_output = reference_space_to_depth(x, scale)
    assert spd_output.shape == (
        input_shape[0],
        input_channels * scale**2,
        *expected_spatial_shape,
    )
    torch.testing.assert_close(output, model.conv(spd_output))

    if pconv:
        partial_conv = model.conv[0]
        expected_n_div = 4 if n_div is None else n_div
        assert isinstance(partial_conv, PConv)
        assert partial_conv.c_partial == input_channels * scale**2 // expected_n_div

    output.mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_spdconv_rejects_invalid_scale(scale: int | float) -> None:
    """Verify that scale must be an integer greater than one."""
    try:
        SPDConv(c1=8, c2=16, scale=scale)
    except ValueError as error:
        assert "scale must be an integer >= 2" in str(error)
    else:
        raise AssertionError(f"SPDConv accepted invalid scale {scale!r}")


if pytest is not None:
    test_spdconv = pytest.mark.parametrize(
        ("scale", "input_shape", "output_channels", "pconv", "n_div"),
        SPD_CASES,
        ids=SPD_CASE_IDS,
    )(test_spdconv)
    test_spdconv_rejects_invalid_scale = pytest.mark.parametrize("scale", INVALID_SCALES)(
        test_spdconv_rejects_invalid_scale
    )


def main() -> None:
    """Run the tests directly without pytest."""
    for case_id, case in zip(SPD_CASE_IDS, SPD_CASES):
        test_spdconv(*case)
        print(f"PASS: {case_id}")

    for scale in INVALID_SCALES:
        test_spdconv_rejects_invalid_scale(scale)
        print(f"PASS: invalid-scale-{scale}")

    print("All SPDConv tests passed.")


if __name__ == "__main__":
    main()
