"""Tests for versatil.models.layers.pooling.spatial_softmax module."""
from collections.abc import Callable

import pytest
import torch
import torch.nn as nn

from versatil.models.layers.pooling.spatial_softmax import SpatialSoftmax


@pytest.fixture
def spatial_softmax_factory() -> Callable[..., SpatialSoftmax]:
    """Factory for SpatialSoftmax instances."""
    def factory(
        height: int = 8,
        width: int = 8,
        channel: int = 16,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
    ) -> SpatialSoftmax:
        return SpatialSoftmax(
            height=height,
            width=width,
            channel=channel,
            temperature=temperature,
            learnable_temperature=learnable_temperature,
        )
    return factory


class TestSpatialSoftmaxInitialization:

    @pytest.mark.parametrize("height", [8, 16])
    @pytest.mark.parametrize("width", [8, 12])
    @pytest.mark.parametrize("channel", [16, 32])
    def test_stores_configuration(
        self,
        spatial_softmax_factory: Callable[..., SpatialSoftmax],
        height: int,
        width: int,
        channel: int,
    ):
        module = spatial_softmax_factory(
            height=height,
            width=width,
            channel=channel,
        )
        assert module.height == height
        assert module.width == width
        assert module.channel == channel

    def test_pos_x_registered_as_buffer(
        self,
        spatial_softmax_factory: Callable[..., SpatialSoftmax],
    ):
        module = spatial_softmax_factory(height=8, width=8, channel=16)
        buffers = dict(module.named_buffers())
        assert "pos_x" in buffers
        assert buffers["pos_x"].shape == (1, 8 * 8)

    def test_pos_y_registered_as_buffer(
        self,
        spatial_softmax_factory: Callable[..., SpatialSoftmax],
    ):
        module = spatial_softmax_factory(height=8, width=8, channel=16)
        buffers = dict(module.named_buffers())
        assert "pos_y" in buffers
        assert buffers["pos_y"].shape == (1, 8 * 8)

    def test_learnable_temperature_is_parameter(
        self,
        spatial_softmax_factory: Callable[..., SpatialSoftmax],
    ):
        module = spatial_softmax_factory(
            height=8,
            width=8,
            channel=16,
            temperature=0.5,
            learnable_temperature=True,
        )
        assert isinstance(module.temperature, nn.Parameter)
        assert module.temperature.requires_grad is True

    def test_non_learnable_temperature_is_buffer(
        self,
        spatial_softmax_factory: Callable[..., SpatialSoftmax],
    ):
        module = spatial_softmax_factory(
            height=8,
            width=8,
            channel=16,
            temperature=2.0,
            learnable_temperature=False,
        )
        assert not isinstance(module.temperature, nn.Parameter)
        buffers = dict(module.named_buffers())
        assert "temperature" in buffers

    def test_inherits_nn_module(
        self,
        spatial_softmax_factory: Callable[..., SpatialSoftmax],
    ):
        module = spatial_softmax_factory(height=8, width=8, channel=16)
        assert isinstance(module, nn.Module)


class TestSpatialSoftmaxForward:

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("channel", [8, 32])
    def test_output_shape(
        self,
        spatial_softmax_factory: Callable[..., SpatialSoftmax],
        feature_map_factory: Callable[..., torch.Tensor],
        batch_size: int,
        channel: int,
    ):
        height, width = 8, 8
        module = spatial_softmax_factory(
            height=height,
            width=width,
            channel=channel,
        )
        tensor = feature_map_factory(
            batch_size=batch_size,
            channels=channel,
            height=height,
            width=width,
        )
        output = module(tensor)
        assert output.shape == (batch_size, channel * 2)

    def test_output_values_in_coordinate_range(
        self,
        spatial_softmax_factory: Callable[..., SpatialSoftmax],
        feature_map_factory: Callable[..., torch.Tensor],
    ):
        module = spatial_softmax_factory(height=8, width=8, channel=16)
        tensor = feature_map_factory(
            batch_size=2,
            channels=16,
            height=8,
            width=8,
        )
        output = module(tensor)
        # Expected coordinates are weighted averages of pos_x/pos_y in [-1, 1]
        assert output.min() >= -1.0
        assert output.max() <= 1.0
