"""Tests for versatil.models.layers.convert_layers module."""
from collections.abc import Callable

import pytest
import torch.nn as nn

from versatil.models.layers.convert_layers import (
    _compute_num_groups,
    convert_layers,
    replace_batchnorm_with_groupnorm,
)


@pytest.fixture
def simple_model_factory() -> Callable[..., nn.Sequential]:
    """Factory for simple models with BatchNorm layers."""
    def factory(
        channels: int = 64,
        include_non_bn_layers: bool = True,
    ) -> nn.Sequential:
        layers = [nn.BatchNorm2d(channels)]
        if include_non_bn_layers:
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        return nn.Sequential(*layers)
    return factory


@pytest.fixture
def nested_model_factory() -> Callable[..., nn.Module]:
    """Factory for nested models with BatchNorm at multiple levels."""
    def factory(
        channels: int = 64,
    ) -> nn.Module:
        return nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
            ),
        )
    return factory


class TestComputeNumGroups:

    @pytest.mark.parametrize("num_channels, expected_groups", [
        (64, 4),
        (128, 8),
        (256, 16),
        (32, 2),
        (1, 1),
        (16, 1),
        (48, 3),
    ])
    def test_computes_correct_num_groups(
        self,
        num_channels: int,
        expected_groups: int,
    ):
        result = _compute_num_groups(num_channels)
        assert result == expected_groups
        # Groups must evenly divide channels
        assert num_channels % result == 0


class TestConvertLayers:

    def test_converts_batchnorm_to_groupnorm(
        self,
        simple_model_factory: Callable[..., nn.Sequential],
    ):
        model = simple_model_factory(channels=64)
        assert isinstance(model[0], nn.BatchNorm2d)
        convert_layers(
            model=model,
            layer_type_old=nn.BatchNorm2d,
            layer_type_new=nn.GroupNorm,
        )
        assert isinstance(model[0], nn.GroupNorm)

    def test_preserves_non_target_layers(
        self,
        simple_model_factory: Callable[..., nn.Sequential],
    ):
        model = simple_model_factory(channels=64, include_non_bn_layers=True)
        convert_layers(
            model=model,
            layer_type_old=nn.BatchNorm2d,
            layer_type_new=nn.GroupNorm,
        )
        assert isinstance(model[1], nn.ReLU)
        assert isinstance(model[2], nn.Conv2d)

    def test_converts_nested_layers(
        self,
        nested_model_factory: Callable[..., nn.Module],
    ):
        model = nested_model_factory(channels=64)
        convert_layers(
            model=model,
            layer_type_old=nn.BatchNorm2d,
            layer_type_new=nn.GroupNorm,
        )
        # Both nested BatchNorm layers should be converted
        assert isinstance(model[0][1], nn.GroupNorm)
        assert isinstance(model[1][1], nn.GroupNorm)

    def test_copies_weights_when_convert_weights_enabled(
        self,
        simple_model_factory: Callable[..., nn.Sequential],
    ):
        model = simple_model_factory(channels=64, include_non_bn_layers=False)
        original_weight = model[0].weight.data.clone()
        original_bias = model[0].bias.data.clone()
        convert_layers(
            model=model,
            layer_type_old=nn.BatchNorm2d,
            layer_type_new=nn.GroupNorm,
            convert_weights=True,
        )
        assert isinstance(model[0], nn.GroupNorm)
        assert (model[0].weight.data == original_weight).all()
        assert (model[0].bias.data == original_bias).all()

    def test_sets_correct_num_channels(
        self,
        simple_model_factory: Callable[..., nn.Sequential],
    ):
        channels = 128
        model = simple_model_factory(channels=channels, include_non_bn_layers=False)
        convert_layers(
            model=model,
            layer_type_old=nn.BatchNorm2d,
            layer_type_new=nn.GroupNorm,
        )
        assert model[0].num_channels == channels


class TestReplaceBatchnormWithGroupnorm:

    def test_replaces_all_batchnorm_layers(
        self,
        nested_model_factory: Callable[..., nn.Module],
    ):
        model = nested_model_factory(channels=64)
        result = replace_batchnorm_with_groupnorm(model=model)
        # No BatchNorm2d should remain
        for module in result.modules():
            assert not isinstance(module, nn.BatchNorm2d)

    def test_resulting_model_contains_groupnorm(
        self,
        nested_model_factory: Callable[..., nn.Module],
    ):
        model = nested_model_factory(channels=64)
        result = replace_batchnorm_with_groupnorm(model=model)
        groupnorm_count = sum(
            1 for module in result.modules() if isinstance(module, nn.GroupNorm)
        )
        assert groupnorm_count == 2

    def test_preserves_conv_layers(
        self,
        nested_model_factory: Callable[..., nn.Module],
    ):
        model = nested_model_factory(channels=64)
        result = replace_batchnorm_with_groupnorm(model=model)
        conv_count = sum(
            1 for module in result.modules() if isinstance(module, nn.Conv2d)
        )
        assert conv_count == 2