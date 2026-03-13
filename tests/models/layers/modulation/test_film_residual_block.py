"""Tests for versatil.models.layers.modulation.film_residual_block module."""
from collections.abc import Callable

import numpy as np
import pytest
import torch
import torch.nn as nn

from versatil.models.layers.modulation.conditional_modulation import (
    ConditionalModulation,
)
from versatil.models.layers.modulation.film_residual_block import FiLMedResBlock


@pytest.fixture
def feature_map_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for 2D feature map inputs (B, C, H, W)."""
    def factory(
        batch_size: int = 2,
        channels: int = 16,
        height: int = 8,
        width: int = 8,
    ) -> torch.Tensor:
        data = rng.standard_normal(
            (batch_size, channels, height, width)
        ).astype(np.float32)
        return torch.from_numpy(data)
    return factory


@pytest.fixture
def filmed_resblock_factory() -> Callable[..., FiLMedResBlock]:
    """Factory for FiLMedResBlock instances."""
    def factory(
        in_channels: int = 16,
        out_channels: int = 16,
        condition_dim: int = 32,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> FiLMedResBlock:
        return FiLMedResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            condition_dim=condition_dim,
            stride=stride,
            downsample=downsample,
        )
    return factory


class TestFiLMedResBlockInitialization:

    def test_inherits_nn_module(
        self,
        filmed_resblock_factory: Callable[..., FiLMedResBlock],
    ):
        module = filmed_resblock_factory(in_channels=16, out_channels=16)
        assert isinstance(module, nn.Module)

    def test_has_film1_layer(
        self,
        filmed_resblock_factory: Callable[..., FiLMedResBlock],
    ):
        module = filmed_resblock_factory(
            in_channels=16,
            out_channels=32,
            condition_dim=64,
        )
        assert isinstance(module.film1, ConditionalModulation)

    def test_has_film2_layer(
        self,
        filmed_resblock_factory: Callable[..., FiLMedResBlock],
    ):
        module = filmed_resblock_factory(
            in_channels=16,
            out_channels=32,
            condition_dim=64,
        )
        assert isinstance(module.film2, ConditionalModulation)

    def test_downsample_is_none_by_default(
        self,
        filmed_resblock_factory: Callable[..., FiLMedResBlock],
    ):
        module = filmed_resblock_factory(in_channels=16, out_channels=16)
        assert module.downsample is None


class TestFiLMedResBlockForward:

    @pytest.mark.parametrize("in_channels, out_channels", [
        (16, 16),
        (16, 32),
    ])
    def test_output_shape(
        self,
        filmed_resblock_factory: Callable[..., FiLMedResBlock],
        feature_map_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
        in_channels: int,
        out_channels: int,
    ):
        condition_dim = 32
        # When in_channels != out_channels, need a downsample for the residual
        downsample = None
        if in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        module = filmed_resblock_factory(
            in_channels=in_channels,
            out_channels=out_channels,
            condition_dim=condition_dim,
            downsample=downsample,
        )
        tensor = feature_map_factory(
            batch_size=2,
            channels=in_channels,
            height=8,
            width=8,
        )
        condition = condition_factory(batch_size=2, condition_dim=condition_dim)
        output = module(tensor, condition)
        assert output.shape == (2, out_channels, 8, 8)

    def test_with_stride_downsample(
        self,
        filmed_resblock_factory: Callable[..., FiLMedResBlock],
        feature_map_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        in_channels = 16
        out_channels = 32
        condition_dim = 32
        stride = 2
        downsample = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            ),
            nn.BatchNorm2d(out_channels),
        )
        module = filmed_resblock_factory(
            in_channels=in_channels,
            out_channels=out_channels,
            condition_dim=condition_dim,
            stride=stride,
            downsample=downsample,
        )
        tensor = feature_map_factory(
            batch_size=2,
            channels=in_channels,
            height=8,
            width=8,
        )
        condition = condition_factory(batch_size=2, condition_dim=condition_dim)
        output = module(tensor, condition)
        assert output.shape == (2, out_channels, 4, 4)

    def test_without_downsample_same_channels(
        self,
        filmed_resblock_factory: Callable[..., FiLMedResBlock],
        feature_map_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        channels = 16
        condition_dim = 32
        module = filmed_resblock_factory(
            in_channels=channels,
            out_channels=channels,
            condition_dim=condition_dim,
        )
        tensor = feature_map_factory(
            batch_size=2,
            channels=channels,
            height=8,
            width=8,
        )
        condition = condition_factory(batch_size=2, condition_dim=condition_dim)
        output = module(tensor, condition)
        assert output.shape == (2, channels, 8, 8)
