"""Tests for versatil.models.layers.convolution.conv1d module."""
from collections.abc import Callable

import pytest
import torch
import torch.nn as nn

from versatil.models.layers.convolution.conv1d import (
    Conv1dBlock,
    Downsample1d,
    Upsample1d,
)


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

    @pytest.mark.parametrize("dim", [16, 32])
    def test_stores_configuration(
        self,
        downsample_factory: Callable[..., Downsample1d],
        dim: int,
    ):
        module = downsample_factory(dim=dim)
        assert module.conv.in_channels == dim
        assert module.conv.out_channels == dim
        assert module.conv.kernel_size == (3,)
        assert module.conv.stride == (2,)
        assert module.conv.padding == (1,)

    @pytest.mark.parametrize("dim", [16, 32])
    @pytest.mark.parametrize("sequence_length", [32, 64])
    def test_halves_sequence_length(
        self,
        downsample_factory: Callable[..., Downsample1d],
        conv1d_tensor_factory: Callable[..., torch.Tensor],
        dim: int,
        sequence_length: int,
    ):
        module = downsample_factory(dim=dim)
        tensor = conv1d_tensor_factory(
            batch_size=2,
            channels=dim,
            sequence_length=sequence_length,
        )
        output = module(tensor)
        assert output.shape == (2, dim, sequence_length // 2)


class TestUpsample1d:

    @pytest.mark.parametrize("dim", [16, 32])
    def test_stores_configuration(
        self,
        upsample_factory: Callable[..., Upsample1d],
        dim: int,
    ):
        module = upsample_factory(dim=dim)
        assert module.conv.in_channels == dim
        assert module.conv.out_channels == dim
        assert module.conv.kernel_size == (4,)
        assert module.conv.stride == (2,)
        assert module.conv.padding == (1,)

    @pytest.mark.parametrize("dim", [16, 32])
    @pytest.mark.parametrize("sequence_length", [16, 32])
    def test_doubles_sequence_length(
        self,
        upsample_factory: Callable[..., Upsample1d],
        conv1d_tensor_factory: Callable[..., torch.Tensor],
        dim: int,
        sequence_length: int,
    ):
        module = upsample_factory(dim=dim)
        tensor = conv1d_tensor_factory(
            batch_size=2,
            channels=dim,
            sequence_length=sequence_length,
        )
        output = module(tensor)
        assert output.shape == (2, dim, sequence_length * 2)


class TestDownsampleUpsampleComplementarity:

    @pytest.mark.parametrize("dim", [16, 32])
    @pytest.mark.parametrize("sequence_length", [32, 64])
    def test_downsample_then_upsample_restores_shape(
        self,
        downsample_factory: Callable[..., Downsample1d],
        upsample_factory: Callable[..., Upsample1d],
        conv1d_tensor_factory: Callable[..., torch.Tensor],
        dim: int,
        sequence_length: int,
    ):
        downsample = downsample_factory(dim=dim)
        upsample = upsample_factory(dim=dim)
        tensor = conv1d_tensor_factory(
            batch_size=2,
            channels=dim,
            sequence_length=sequence_length,
        )
        with torch.no_grad():
            output = upsample(downsample(tensor))
        assert output.shape == tensor.shape


class TestConv1dBlock:

    @pytest.mark.parametrize("input_channels", [16, 32])
    @pytest.mark.parametrize("output_channels", [32, 64])
    @pytest.mark.parametrize("kernel_size", [3, 5])
    @pytest.mark.parametrize("num_groups", [4, 8])
    def test_stores_configuration(
        self,
        conv1d_block_factory: Callable[..., Conv1dBlock],
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        num_groups: int,
    ):
        module = conv1d_block_factory(
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=kernel_size,
            num_groups=num_groups,
        )
        convolution = module.block[0]
        assert convolution.in_channels == input_channels
        assert convolution.out_channels == output_channels
        assert convolution.kernel_size == (kernel_size,)
        assert convolution.padding == (kernel_size // 2,)
        group_norm = module.block[1]
        assert group_norm.num_groups == num_groups
        assert group_norm.num_channels == output_channels

    @pytest.mark.parametrize(
        "input_channels, output_channels",
        [
            (16, 32),
            (32, 64),
        ],
    )
    @pytest.mark.parametrize("kernel_size", [3, 5])
    def test_output_shape(
        self,
        conv1d_block_factory: Callable[..., Conv1dBlock],
        conv1d_tensor_factory: Callable[..., torch.Tensor],
        input_channels: int,
        output_channels: int,
        kernel_size: int,
    ):
        sequence_length = 32
        module = conv1d_block_factory(
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=kernel_size,
        )
        tensor = conv1d_tensor_factory(
            batch_size=2,
            channels=input_channels,
            sequence_length=sequence_length,
        )
        output = module(tensor)
        assert output.shape == (2, output_channels, sequence_length)

    def test_applies_normalization_and_nonlinearity(
        self,
        conv1d_block_factory: Callable[..., Conv1dBlock],
        conv1d_tensor_factory: Callable[..., torch.Tensor],
    ):
        input_channels = 16
        output_channels = 32
        module = conv1d_block_factory(
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=3,
            num_groups=8,
        )
        # Standalone Conv1d with identical weights to isolate normalization + activation
        raw_convolution = nn.Conv1d(
            input_channels, output_channels, 3, padding=1
        )
        raw_convolution.weight.data.copy_(module.block[0].weight.data)
        raw_convolution.bias.data.copy_(module.block[0].bias.data)

        tensor = conv1d_tensor_factory(
            batch_size=2,
            channels=input_channels,
            sequence_length=32,
        )
        with torch.no_grad():
            block_output = module(tensor)
            conv_only_output = raw_convolution(tensor)

        # GroupNorm + Mish transform the raw convolution output
        assert not torch.allclose(block_output, conv_only_output)

    @pytest.mark.parametrize("kernel_size", [3, 5, 7])
    def test_preserves_sequence_length_across_kernel_sizes(
        self,
        conv1d_block_factory: Callable[..., Conv1dBlock],
        conv1d_tensor_factory: Callable[..., torch.Tensor],
        kernel_size: int,
    ):
        sequence_length = 32
        module = conv1d_block_factory(
            input_channels=16,
            output_channels=16,
            kernel_size=kernel_size,
            num_groups=8,
        )
        tensor = conv1d_tensor_factory(
            batch_size=2,
            channels=16,
            sequence_length=sequence_length,
        )
        output = module(tensor)
        assert output.shape[2] == sequence_length
