"""Tests for versatil.models.layers.pooling.spatial_softmax module."""
from collections.abc import Callable

import pytest
import torch

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
        temperature = 0.5
        module = spatial_softmax_factory(
            height=8,
            width=8,
            channel=16,
            temperature=temperature,
            learnable_temperature=True,
        )
        # Verify temperature is discoverable in module parameters (functional consequence)
        param_names = {name for name, _ in module.named_parameters()}
        assert "temperature" in param_names
        assert module.temperature.requires_grad is True
        assert torch.allclose(
            module.temperature, torch.tensor([temperature])
        )

    def test_non_learnable_temperature_is_buffer(
        self,
        spatial_softmax_factory: Callable[..., SpatialSoftmax],
    ):
        temperature = 2.0
        module = spatial_softmax_factory(
            height=8,
            width=8,
            channel=16,
            temperature=temperature,
            learnable_temperature=False,
        )
        # Temperature should be a buffer, not in parameters
        param_names = {name for name, _ in module.named_parameters()}
        assert "temperature" not in param_names
        buffers = dict(module.named_buffers())
        assert "temperature" in buffers
        assert torch.allclose(
            module.temperature, torch.tensor([temperature])
        )


class TestSpatialSoftmaxForward:

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("channel", [8, 32])
    def test_output_shape(
        self,
        spatial_softmax_factory: Callable[..., SpatialSoftmax],
        nchw_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
        channel: int,
    ):
        height, width = 8, 8
        module = spatial_softmax_factory(
            height=height,
            width=width,
            channel=channel,
        )
        tensor = nchw_tensor_factory(
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
        nchw_tensor_factory: Callable[..., torch.Tensor],
    ):
        module = spatial_softmax_factory(height=8, width=8, channel=16)
        tensor = nchw_tensor_factory(
            batch_size=2,
            channels=16,
            height=8,
            width=8,
        )
        output = module(tensor)
        # Expected coordinates are weighted averages of pos_x/pos_y in [-1, 1]
        assert output.min() >= -1.0
        assert output.max() <= 1.0

    def test_different_temperatures_produce_different_outputs(
        self,
        spatial_softmax_factory: Callable[..., SpatialSoftmax],
        nchw_tensor_factory: Callable[..., torch.Tensor],
    ):
        height, width, channel = 8, 8, 16
        low_temperature = spatial_softmax_factory(
            height=height, width=width, channel=channel, temperature=0.1,
        )
        high_temperature = spatial_softmax_factory(
            height=height, width=width, channel=channel, temperature=10.0,
        )
        tensor = nchw_tensor_factory(
            batch_size=2, channels=channel, height=height, width=width,
        )
        output_low = low_temperature(tensor)
        output_high = high_temperature(tensor)
        assert not torch.allclose(output_low, output_high, atol=1e-5)

    def test_lower_temperature_produces_sharper_keypoints(
        self,
        spatial_softmax_factory: Callable[..., SpatialSoftmax],
    ):
        height, width, channel = 4, 4, 1
        # Create a feature map with a single peak at position (0, 0)
        tensor = torch.zeros(1, channel, height, width)
        tensor[0, 0, 0, 0] = 10.0
        low_temperature = spatial_softmax_factory(
            height=height, width=width, channel=channel, temperature=0.01,
        )
        high_temperature = spatial_softmax_factory(
            height=height, width=width, channel=channel, temperature=100.0,
        )
        output_low = low_temperature(tensor)
        output_high = high_temperature(tensor)
        # Low temperature should concentrate attention on the peak, producing
        # coordinates closer to the corner (-1, -1); high temperature should
        # spread attention, pulling coordinates toward center (0, 0)
        expected_x_low = output_low[0, 0]
        expected_x_high = output_high[0, 0]
        assert expected_x_low < expected_x_high
