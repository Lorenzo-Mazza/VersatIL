"""Tests for versatil.models.layers.modulation.film_residual_block module."""

from collections.abc import Callable

import pytest
import torch
import torch.nn as nn

from versatil.models.layers.modulation.film_residual_block import FiLMedResBlock


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
    @pytest.mark.parametrize("in_channels", [16, 32])
    @pytest.mark.parametrize("out_channels", [16, 64])
    @pytest.mark.parametrize("condition_dim", [32, 64])
    def test_stores_configuration(
        self,
        filmed_resblock_factory: Callable[..., FiLMedResBlock],
        in_channels: int,
        out_channels: int,
        condition_dim: int,
    ):
        module = filmed_resblock_factory(
            in_channels=in_channels,
            out_channels=out_channels,
            condition_dim=condition_dim,
        )
        assert module.conv1.in_channels == in_channels
        assert module.conv1.out_channels == out_channels
        assert module.conv2.in_channels == out_channels
        assert module.conv2.out_channels == out_channels
        assert module.film1.feature_dim == out_channels
        assert module.film2.feature_dim == out_channels
        assert module.bn1.num_features == out_channels
        assert module.bn2.num_features == out_channels

    def test_film_layers_have_no_effect_at_init(
        self,
        filmed_resblock_factory: Callable[..., FiLMedResBlock],
        nchw_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        feature_dim = 16
        module = filmed_resblock_factory(
            in_channels=feature_dim,
            out_channels=feature_dim,
            condition_dim=32,
        )
        module.eval()
        tensor = nchw_tensor_factory(batch_size=2, channels=feature_dim)
        condition = condition_factory(batch_size=2, condition_dim=32)
        # Identity init: gamma=0, beta=0 → film(x, cond) = x * (1+0) + 0 = x
        with torch.no_grad():
            bn_output = module.bn1(module.conv1(tensor))
            film_output = module.film1(x=bn_output, condition=condition)
        assert torch.allclose(film_output, bn_output, atol=1e-6)


class TestFiLMedResBlockForward:
    def test_output_shape_same_channels(
        self,
        filmed_resblock_factory: Callable[..., FiLMedResBlock],
        nchw_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        channels = 16
        module = filmed_resblock_factory(
            in_channels=channels,
            out_channels=channels,
            condition_dim=32,
        )
        module.eval()
        tensor = nchw_tensor_factory(batch_size=2, channels=channels, height=8, width=8)
        condition = condition_factory(batch_size=2, condition_dim=32)
        with torch.no_grad():
            output = module(x=tensor, condition=condition)
        assert output.shape == (2, channels, 8, 8)

    def test_output_shape_with_channel_change_and_downsample(
        self,
        filmed_resblock_factory: Callable[..., FiLMedResBlock],
        nchw_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        in_channels = 16
        out_channels = 32
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        module = filmed_resblock_factory(
            in_channels=in_channels,
            out_channels=out_channels,
            condition_dim=32,
            downsample=downsample,
        )
        module.eval()
        tensor = nchw_tensor_factory(
            batch_size=2, channels=in_channels, height=8, width=8
        )
        condition = condition_factory(batch_size=2, condition_dim=32)
        with torch.no_grad():
            output = module(x=tensor, condition=condition)
        assert output.shape == (2, out_channels, 8, 8)

    def test_stride_reduces_spatial_dimensions(
        self,
        filmed_resblock_factory: Callable[..., FiLMedResBlock],
        nchw_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        in_channels = 16
        out_channels = 32
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
            condition_dim=32,
            stride=stride,
            downsample=downsample,
        )
        module.eval()
        tensor = nchw_tensor_factory(
            batch_size=2, channels=in_channels, height=8, width=8
        )
        condition = condition_factory(batch_size=2, condition_dim=32)
        with torch.no_grad():
            output = module(x=tensor, condition=condition)
        assert output.shape == (2, out_channels, 4, 4)

    def test_different_conditions_produce_different_outputs(
        self,
        filmed_resblock_factory: Callable[..., FiLMedResBlock],
        nchw_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        channels = 16
        module = filmed_resblock_factory(
            in_channels=channels,
            out_channels=channels,
            condition_dim=32,
        )
        module.eval()
        # Set FiLM projection weights to nonzero so conditioning has an effect
        for layer in module.film1.projection.modules():
            if hasattr(layer, "weight"):
                nn.init.xavier_uniform_(layer.weight)
        tensor = nchw_tensor_factory(batch_size=2, channels=channels, height=8, width=8)
        condition_a = condition_factory(batch_size=2, condition_dim=32)
        condition_b = condition_factory(batch_size=2, condition_dim=32)
        with torch.no_grad():
            output_a = module(x=tensor, condition=condition_a)
            output_b = module(x=tensor, condition=condition_b)
        assert not torch.allclose(output_a, output_b)

    def test_residual_connection_incorporates_input(
        self,
        filmed_resblock_factory: Callable[..., FiLMedResBlock],
        nchw_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        channels = 16
        module = filmed_resblock_factory(
            in_channels=channels,
            out_channels=channels,
            condition_dim=32,
        )
        module.eval()
        # Zero conv weights so main branch produces ~zero output
        nn.init.zeros_(module.conv1.weight)
        nn.init.zeros_(module.conv2.weight)
        tensor = nchw_tensor_factory(batch_size=2, channels=channels, height=8, width=8)
        condition = condition_factory(batch_size=2, condition_dim=32)
        with torch.no_grad():
            output = module(x=tensor, condition=condition)
        # With conv weights zeroed, main branch ≈ 0, so output ≈ relu(0 + x) = relu(x)
        expected = torch.relu(tensor)
        assert torch.allclose(output, expected, atol=1e-4)
