"""Tests for versatil.models.layers.convolution.conv1d module."""
from collections.abc import Callable

import numpy as np
import pytest
import torch
import torch.nn as nn

from versatil.models.layers.convolution.conv1d import (
    Conv1dBlock,
    Downsample1d,
    Upsample1d,
)


@pytest.fixture
def conv1d_input_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for 1D convolution input tensors (B, C, T)."""
    def factory(
        batch_size: int = 2,
        channels: int = 16,
        sequence_length: int = 32,
    ) -> torch.Tensor:
        data = rng.standard_normal(
            (batch_size, channels, sequence_length)
        ).astype(np.float32)
        return torch.from_numpy(data)
    return factory


@pytest.fixture
def downsample_factory() -> Callable[..., Downsample1d]:
    """Factory for Downsample1d instances."""
    def factory(dim: int = 16) -> Downsample1d:
        return Downsample1d(dim=dim)
    return factory


@pytest.fixture
def upsample_factory() -> Callable[..., Upsample1d]:
    """Factory for Upsample1d instances."""
    def factory(dim: int = 16) -> Upsample1d:
        return Upsample1d(dim=dim)
    return factory


@pytest.fixture
def conv1d_block_factory() -> Callable[..., Conv1dBlock]:
    """Factory for Conv1dBlock instances."""
    def factory(
        input_channels: int = 16,
        output_channels: int = 32,
        kernel_size: int = 3,
        num_groups: int = 8,
    ) -> Conv1dBlock:
        return Conv1dBlock(
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=kernel_size,
            num_groups=num_groups,
        )
    return factory


class TestDownsample1d:

    def test_inherits_nn_module(
        self,
        downsample_factory: Callable[..., Downsample1d],
    ):
        module = downsample_factory(dim=16)
        assert isinstance(module, nn.Module)

    @pytest.mark.parametrize("dim", [16, 32])
    @pytest.mark.parametrize("sequence_length", [32, 64])
    def test_halves_sequence_length(
        self,
        downsample_factory: Callable[..., Downsample1d],
        conv1d_input_factory: Callable[..., torch.Tensor],
        dim: int,
        sequence_length: int,
    ):
        module = downsample_factory(dim=dim)
        tensor = conv1d_input_factory(
            batch_size=2,
            channels=dim,
            sequence_length=sequence_length,
        )
        output = module(tensor)
        assert output.shape == (2, dim, sequence_length // 2)

    def test_preserves_channel_dimension(
        self,
        downsample_factory: Callable[..., Downsample1d],
        conv1d_input_factory: Callable[..., torch.Tensor],
    ):
        dim = 24
        module = downsample_factory(dim=dim)
        tensor = conv1d_input_factory(batch_size=3, channels=dim, sequence_length=40)
        output = module(tensor)
        assert output.shape[1] == dim


class TestUpsample1d:

    def test_inherits_nn_module(
        self,
        upsample_factory: Callable[..., Upsample1d],
    ):
        module = upsample_factory(dim=16)
        assert isinstance(module, nn.Module)

    @pytest.mark.parametrize("dim", [16, 32])
    @pytest.mark.parametrize("sequence_length", [16, 32])
    def test_doubles_sequence_length(
        self,
        upsample_factory: Callable[..., Upsample1d],
        conv1d_input_factory: Callable[..., torch.Tensor],
        dim: int,
        sequence_length: int,
    ):
        module = upsample_factory(dim=dim)
        tensor = conv1d_input_factory(
            batch_size=2,
            channels=dim,
            sequence_length=sequence_length,
        )
        output = module(tensor)
        assert output.shape == (2, dim, sequence_length * 2)

    def test_preserves_channel_dimension(
        self,
        upsample_factory: Callable[..., Upsample1d],
        conv1d_input_factory: Callable[..., torch.Tensor],
    ):
        dim = 24
        module = upsample_factory(dim=dim)
        tensor = conv1d_input_factory(batch_size=3, channels=dim, sequence_length=20)
        output = module(tensor)
        assert output.shape[1] == dim


class TestConv1dBlock:

    def test_inherits_nn_module(
        self,
        conv1d_block_factory: Callable[..., Conv1dBlock],
    ):
        module = conv1d_block_factory(input_channels=16, output_channels=32)
        assert isinstance(module, nn.Module)

    @pytest.mark.parametrize("input_channels, output_channels", [
        (16, 32),
        (32, 64),
    ])
    @pytest.mark.parametrize("kernel_size", [3, 5])
    @pytest.mark.parametrize("sequence_length", [20, 40])
    def test_output_shape(
        self,
        conv1d_block_factory: Callable[..., Conv1dBlock],
        conv1d_input_factory: Callable[..., torch.Tensor],
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        sequence_length: int,
    ):
        module = conv1d_block_factory(
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=kernel_size,
        )
        tensor = conv1d_input_factory(
            batch_size=2,
            channels=input_channels,
            sequence_length=sequence_length,
        )
        output = module(tensor)
        assert output.shape == (2, output_channels, sequence_length)

    def test_block_contains_conv_groupnorm_mish(
        self,
        conv1d_block_factory: Callable[..., Conv1dBlock],
    ):
        module = conv1d_block_factory(
            input_channels=16,
            output_channels=32,
            num_groups=8,
        )
        layers = list(module.block)
        assert isinstance(layers[0], nn.Conv1d)
        assert isinstance(layers[1], nn.GroupNorm)
        assert isinstance(layers[2], nn.Mish)
