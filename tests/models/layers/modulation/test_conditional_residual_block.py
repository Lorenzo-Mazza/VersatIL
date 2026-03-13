"""Tests for versatil.models.layers.modulation.conditional_residual_block module."""
from collections.abc import Callable

import numpy as np
import pytest
import torch
import torch.nn as nn

from versatil.models.layers.modulation.conditional_residual_block import (
    ConditionalResidualBlock1D,
)


@pytest.fixture
def conv1d_input_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for 1D convolution inputs (B, C, T)."""
    def factory(
        batch_size: int = 2,
        channels: int = 16,
        prediction_horizon: int = 20,
    ) -> torch.Tensor:
        data = rng.standard_normal(
            (batch_size, channels, prediction_horizon)
        ).astype(np.float32)
        return torch.from_numpy(data)
    return factory


@pytest.fixture
def residual_block_factory() -> Callable[..., ConditionalResidualBlock1D]:
    """Factory for ConditionalResidualBlock1D instances."""
    def factory(
        input_channels: int = 16,
        output_channels: int = 32,
        condition_dimension: int = 32,
        kernel_size: int = 3,
        num_groups: int = 8,
    ) -> ConditionalResidualBlock1D:
        return ConditionalResidualBlock1D(
            input_channels=input_channels,
            output_channels=output_channels,
            condition_dimension=condition_dimension,
            kernel_size=kernel_size,
            num_groups=num_groups,
        )
    return factory


class TestConditionalResidualBlock1DInitialization:

    def test_inherits_nn_module(
        self,
        residual_block_factory: Callable[..., ConditionalResidualBlock1D],
    ):
        module = residual_block_factory(
            input_channels=16,
            output_channels=32,
        )
        assert isinstance(module, nn.Module)

    def test_residual_uses_conv1d_when_channels_differ(
        self,
        residual_block_factory: Callable[..., ConditionalResidualBlock1D],
    ):
        module = residual_block_factory(
            input_channels=16,
            output_channels=32,
        )
        assert isinstance(module.residual_convolution, nn.Conv1d)

    def test_residual_uses_identity_when_channels_match(
        self,
        residual_block_factory: Callable[..., ConditionalResidualBlock1D],
    ):
        module = residual_block_factory(
            input_channels=16,
            output_channels=16,
        )
        assert isinstance(module.residual_convolution, nn.Identity)


class TestConditionalResidualBlock1DForward:

    @pytest.mark.parametrize("input_channels, output_channels", [
        (16, 32),
        (32, 64),
    ])
    def test_output_shape_different_channels(
        self,
        residual_block_factory: Callable[..., ConditionalResidualBlock1D],
        conv1d_input_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
        input_channels: int,
        output_channels: int,
    ):
        condition_dimension = 32
        prediction_horizon = 20
        module = residual_block_factory(
            input_channels=input_channels,
            output_channels=output_channels,
            condition_dimension=condition_dimension,
        )
        tensor = conv1d_input_factory(
            batch_size=2,
            channels=input_channels,
            prediction_horizon=prediction_horizon,
        )
        condition = condition_factory(
            batch_size=2,
            condition_dim=condition_dimension,
        )
        output = module(tensor, condition)
        assert output.shape == (2, output_channels, prediction_horizon)

    def test_output_shape_same_channels(
        self,
        residual_block_factory: Callable[..., ConditionalResidualBlock1D],
        conv1d_input_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        channels = 16
        condition_dimension = 32
        prediction_horizon = 20
        module = residual_block_factory(
            input_channels=channels,
            output_channels=channels,
            condition_dimension=condition_dimension,
        )
        tensor = conv1d_input_factory(
            batch_size=2,
            channels=channels,
            prediction_horizon=prediction_horizon,
        )
        condition = condition_factory(
            batch_size=2,
            condition_dim=condition_dimension,
        )
        output = module(tensor, condition)
        assert output.shape == (2, channels, prediction_horizon)
