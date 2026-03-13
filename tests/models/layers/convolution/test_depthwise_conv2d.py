"""Tests for versatil.models.layers.convolution.depthwise_conv2d module."""
from collections.abc import Callable

import numpy as np
import pytest
import torch
import torch.nn as nn

from versatil.models.layers.convolution.depthwise_conv2d import DepthwiseConv2D


@pytest.fixture
def channels_last_input_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for channels-last input tensors (B, H, W, C)."""
    def factory(
        batch_size: int = 2,
        height: int = 8,
        width: int = 8,
        channels: int = 16,
    ) -> torch.Tensor:
        data = rng.standard_normal(
            (batch_size, height, width, channels)
        ).astype(np.float32)
        return torch.from_numpy(data)
    return factory


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

    def test_inherits_nn_module(
        self,
        depthwise_conv2d_factory: Callable[..., DepthwiseConv2D],
    ):
        module = depthwise_conv2d_factory(dimension=16)
        assert isinstance(module, nn.Module)

    @pytest.mark.parametrize("dimension", [16, 32])
    def test_convolution_uses_groups_equal_to_dimension(
        self,
        depthwise_conv2d_factory: Callable[..., DepthwiseConv2D],
        dimension: int,
    ):
        module = depthwise_conv2d_factory(dimension=dimension)
        assert module.convolution.groups == dimension


class TestDepthwiseConv2DForward:

    @pytest.mark.parametrize("dimension", [16, 32])
    @pytest.mark.parametrize("height, width", [(8, 8), (12, 16)])
    def test_output_shape_matches_input(
        self,
        depthwise_conv2d_factory: Callable[..., DepthwiseConv2D],
        channels_last_input_factory: Callable[..., torch.Tensor],
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
        tensor = channels_last_input_factory(
            batch_size=2,
            height=height,
            width=width,
            channels=dimension,
        )
        output = module(tensor)
        assert output.shape == (2, height, width, dimension)

    def test_channels_last_input_and_output_convention(
        self,
        depthwise_conv2d_factory: Callable[..., DepthwiseConv2D],
        channels_last_input_factory: Callable[..., torch.Tensor],
    ):
        dimension = 16
        module = depthwise_conv2d_factory(dimension=dimension)
        tensor = channels_last_input_factory(
            batch_size=2,
            height=8,
            width=8,
            channels=dimension,
        )
        output = module(tensor)
        # Last dimension is channels (B, H, W, C)
        assert output.shape[-1] == dimension
        assert output.ndim == 4

    @pytest.mark.parametrize("stride, expected_height, expected_width", [
        (2, 4, 4),
        (1, 8, 8),
    ])
    def test_stride_reduces_spatial_dimensions(
        self,
        channels_last_input_factory: Callable[..., torch.Tensor],
        stride: int,
        expected_height: int,
        expected_width: int,
    ):
        dimension = 16
        # padding=0 for stride>1 to get clean division
        padding = 1 if stride == 1 else 0
        module = DepthwiseConv2D(
            dimension=dimension,
            kernel_size=3,
            stride=stride,
            padding=padding,
        )
        tensor = channels_last_input_factory(
            batch_size=2,
            height=8,
            width=8,
            channels=dimension,
        )
        output = module(tensor)
        assert output.shape == (2, expected_height, expected_width, dimension)
