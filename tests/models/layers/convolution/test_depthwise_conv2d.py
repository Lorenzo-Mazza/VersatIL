"""Tests for versatil.models.layers.convolution.depthwise_conv2d module."""

from collections.abc import Callable

import pytest
import torch
import torch.nn as nn

from versatil.models.layers.convolution.depthwise_conv2d import DepthwiseConv2D


@pytest.fixture
def depthwise_conv2d_factory() -> Callable[..., DepthwiseConv2D]:
    """Factory for DepthwiseConv2D instances."""

    def factory(
        dimension: int = 16,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> DepthwiseConv2D:
        return DepthwiseConv2D(
            dimension=dimension,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    return factory


class TestDepthwiseConv2DInitialization:
    @pytest.mark.parametrize("dimension", [16, 32])
    @pytest.mark.parametrize("kernel_size", [3, 5])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("padding", [0, 1])
    def test_stores_configuration(
        self,
        depthwise_conv2d_factory: Callable[..., DepthwiseConv2D],
        dimension: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        module = depthwise_conv2d_factory(
            dimension=dimension,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        assert module.convolution.in_channels == dimension
        assert module.convolution.out_channels == dimension
        assert module.convolution.kernel_size == (kernel_size, kernel_size)
        assert module.convolution.stride == (stride, stride)
        assert module.convolution.padding == (padding, padding)
        assert module.convolution.groups == dimension


class TestDepthwiseConv2DForward:
    @pytest.mark.parametrize("dimension", [16, 32])
    @pytest.mark.parametrize("height, width", [(8, 8), (12, 16)])
    def test_output_shape_with_same_padding(
        self,
        depthwise_conv2d_factory: Callable[..., DepthwiseConv2D],
        nhwc_tensor_factory: Callable[..., torch.Tensor],
        dimension: int,
        height: int,
        width: int,
    ):
        module = depthwise_conv2d_factory(
            dimension=dimension,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        tensor = nhwc_tensor_factory(
            batch_size=2,
            height=height,
            width=width,
            channels=dimension,
        )
        output = module(tensor)
        assert output.shape == (2, height, width, dimension)

    @pytest.mark.parametrize(
        "stride, padding, expected_height, expected_width",
        [
            (1, 1, 8, 8),
            (2, 0, 3, 3),
        ],
    )
    def test_stride_reduces_spatial_dimensions(
        self,
        depthwise_conv2d_factory: Callable[..., DepthwiseConv2D],
        nhwc_tensor_factory: Callable[..., torch.Tensor],
        stride: int,
        padding: int,
        expected_height: int,
        expected_width: int,
    ):
        dimension = 16
        module = depthwise_conv2d_factory(
            dimension=dimension,
            kernel_size=3,
            stride=stride,
            padding=padding,
        )
        tensor = nhwc_tensor_factory(
            batch_size=2,
            height=8,
            width=8,
            channels=dimension,
        )
        output = module(tensor)
        assert output.shape == (2, expected_height, expected_width, dimension)

    def test_channels_are_processed_independently(
        self,
        depthwise_conv2d_factory: Callable[..., DepthwiseConv2D],
        nhwc_tensor_factory: Callable[..., torch.Tensor],
    ):
        dimension = 4
        module = depthwise_conv2d_factory(
            dimension=dimension,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        nn.init.zeros_(module.convolution.weight)
        nn.init.zeros_(module.convolution.bias)
        # Activate only channel 0's filter
        module.convolution.weight.data[0].fill_(1.0)

        tensor = nhwc_tensor_factory(
            batch_size=2,
            height=8,
            width=8,
            channels=dimension,
        )
        with torch.no_grad():
            output = module(tensor)

        # Channel 0 has active filter, should produce non-zero output
        assert not torch.all(output[:, :, :, 0] == 0)
        # Channels 1-3 have zero weights and bias, must produce exactly zero
        assert torch.all(output[:, :, :, 1:] == 0)
