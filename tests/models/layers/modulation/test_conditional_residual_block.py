"""Tests for versatil.models.layers.modulation.conditional_residual_block module."""
from collections.abc import Callable

import pytest
import torch
import torch.nn as nn

from versatil.models.layers.modulation.conditional_residual_block import (
    ConditionalResidualBlock1D,
)


@pytest.fixture
def residual_block_factory() -> Callable[..., ConditionalResidualBlock1D]:
    """Factory for ConditionalResidualBlock1D instances."""

    def factory(
        input_channels: int = 16,
        output_channels: int = 32,
        condition_dimension: int = 32,
        kernel_size: int = 3,
        num_groups: int = 8,
        condition_predict_scale: bool = False,
    ) -> ConditionalResidualBlock1D:
        return ConditionalResidualBlock1D(
            input_channels=input_channels,
            output_channels=output_channels,
            condition_dimension=condition_dimension,
            kernel_size=kernel_size,
            num_groups=num_groups,
            condition_predict_scale=condition_predict_scale,
        )

    return factory


class TestConditionalResidualBlock1DInitialization:

    @pytest.mark.parametrize("input_channels", [16, 32])
    @pytest.mark.parametrize("output_channels", [32, 64])
    @pytest.mark.parametrize("condition_dimension", [32, 64])
    def test_stores_configuration(
        self,
        residual_block_factory: Callable[..., ConditionalResidualBlock1D],
        input_channels: int,
        output_channels: int,
        condition_dimension: int,
    ):
        module = residual_block_factory(
            input_channels=input_channels,
            output_channels=output_channels,
            condition_dimension=condition_dimension,
        )
        assert len(module.blocks) == 2
        assert module.modulator.feature_dim == output_channels

    def test_residual_path_adapts_channels_when_mismatch(
        self,
        residual_block_factory: Callable[..., ConditionalResidualBlock1D],
        conv1d_tensor_factory: Callable[..., torch.Tensor],
    ):
        input_channels = 16
        output_channels = 32
        module = residual_block_factory(
            input_channels=input_channels,
            output_channels=output_channels,
        )
        tensor = conv1d_tensor_factory(
            batch_size=2, channels=input_channels, sequence_length=20,
        )
        with torch.no_grad():
            residual_output = module.residual_convolution(tensor)
        assert residual_output.shape == (2, output_channels, 20)

    def test_residual_path_preserves_values_when_channels_match(
        self,
        residual_block_factory: Callable[..., ConditionalResidualBlock1D],
        conv1d_tensor_factory: Callable[..., torch.Tensor],
    ):
        channels = 16
        module = residual_block_factory(
            input_channels=channels, output_channels=channels,
        )
        tensor = conv1d_tensor_factory(
            batch_size=2, channels=channels, sequence_length=20,
        )
        with torch.no_grad():
            residual_output = module.residual_convolution(tensor)
        # Identity: output values equal input values
        assert torch.equal(residual_output, tensor)


class TestConditionalResidualBlock1DForward:

    @pytest.mark.parametrize(
        "input_channels, output_channels",
        [
            (16, 32),
            (16, 16),
        ],
    )
    def test_output_shape(
        self,
        residual_block_factory: Callable[..., ConditionalResidualBlock1D],
        conv1d_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
        input_channels: int,
        output_channels: int,
    ):
        prediction_horizon = 20
        module = residual_block_factory(
            input_channels=input_channels,
            output_channels=output_channels,
            condition_dimension=32,
        )
        tensor = conv1d_tensor_factory(
            batch_size=2,
            channels=input_channels,
            sequence_length=prediction_horizon,
        )
        condition = condition_factory(batch_size=2, condition_dim=32)
        with torch.no_grad():
            output = module(x=tensor, condition=condition)
        assert output.shape == (2, output_channels, prediction_horizon)

    def test_different_conditions_produce_different_outputs(
        self,
        residual_block_factory: Callable[..., ConditionalResidualBlock1D],
        conv1d_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        module = residual_block_factory(
            input_channels=16, output_channels=16, condition_dimension=32,
        )
        # Set modulator weights to nonzero so conditioning has effect
        for layer in module.modulator.projection.modules():
            if hasattr(layer, "weight"):
                nn.init.xavier_uniform_(layer.weight)
        tensor = conv1d_tensor_factory(
            batch_size=2, channels=16, sequence_length=20,
        )
        condition_a = condition_factory(batch_size=2, condition_dim=32)
        condition_b = condition_factory(batch_size=2, condition_dim=32)
        with torch.no_grad():
            output_a = module(x=tensor, condition=condition_a)
            output_b = module(x=tensor, condition=condition_b)
        assert not torch.allclose(output_a, output_b)

    def test_residual_connection_is_active(
        self,
        residual_block_factory: Callable[..., ConditionalResidualBlock1D],
        conv1d_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        channels = 16
        module = residual_block_factory(
            input_channels=channels, output_channels=channels, condition_dimension=32,
        )
        tensor = conv1d_tensor_factory(
            batch_size=2, channels=channels, sequence_length=20,
        )
        condition = condition_factory(batch_size=2, condition_dim=32)
        with torch.no_grad():
            output = module(x=tensor, condition=condition)
        # Output should be nonzero even at identity init because of the residual path
        assert torch.any(output != 0)

    def test_condition_predict_scale_changes_modulator_output_dimension(
        self,
        residual_block_factory: Callable[..., ConditionalResidualBlock1D],
    ):
        output_channels = 32
        without_scale = residual_block_factory(
            input_channels=16,
            output_channels=output_channels,
            condition_dimension=32,
            condition_predict_scale=False,
        )
        with_scale = residual_block_factory(
            input_channels=16,
            output_channels=output_channels,
            condition_dimension=32,
            condition_predict_scale=True,
        )
        # condition_predict_scale=True maps to use_shift=True in ConditionalModulation,
        # which adds a shift (beta) channel, increasing output_dim
        assert with_scale.modulator.output_dim > without_scale.modulator.output_dim

    def test_condition_predict_scale_produces_different_outputs(
        self,
        residual_block_factory: Callable[..., ConditionalResidualBlock1D],
        conv1d_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        channels = 16
        module_without_scale = residual_block_factory(
            input_channels=channels,
            output_channels=channels,
            condition_dimension=32,
            condition_predict_scale=False,
        )
        module_with_scale = residual_block_factory(
            input_channels=channels,
            output_channels=channels,
            condition_dimension=32,
            condition_predict_scale=True,
        )
        # Initialize modulator weights to nonzero so conditioning has effect
        for module in [module_without_scale, module_with_scale]:
            for layer in module.modulator.projection.modules():
                if hasattr(layer, "weight"):
                    nn.init.xavier_uniform_(layer.weight)
        tensor = conv1d_tensor_factory(
            batch_size=2, channels=channels, sequence_length=20,
        )
        condition = condition_factory(batch_size=2, condition_dim=32)
        with torch.no_grad():
            output_without_scale = module_without_scale(x=tensor, condition=condition)
            output_with_scale = module_with_scale(x=tensor, condition=condition)
        assert not torch.allclose(output_without_scale, output_with_scale)