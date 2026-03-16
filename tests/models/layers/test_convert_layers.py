"""Tests for versatil.models.layers.convert_layers module."""

from collections.abc import Callable

import pytest
import torch
import torch.nn as nn

from versatil.models.layers.convert_layers import (
    _compute_num_groups,
    convert_layers,
    replace_batchnorm_with_groupnorm,
)


class _CustomNorm(nn.Module):
    """Test helper that accepts the same kwargs as GroupNorm."""

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float,
        affine: bool,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine


@pytest.fixture
def simple_model_factory() -> Callable[..., nn.Sequential]:
    """Factory for simple models with a BatchNorm layer."""

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


@pytest.fixture
def model_with_none_module_factory() -> Callable[..., nn.Sequential]:
    """Factory for models with a None entry in _modules."""

    def factory(
        channels: int = 64,
    ) -> nn.Sequential:
        model = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        model._modules["none_slot"] = None
        return model

    return factory


class TestComputeNumGroups:
    @pytest.mark.parametrize(
        "num_channels, expected_groups",
        [
            (64, 4),
            (128, 8),
            (256, 16),
            (32, 2),
            (1, 1),
            (16, 1),
            (48, 3),
        ],
    )
    def test_computes_correct_num_groups(
        self,
        num_channels: int,
        expected_groups: int,
    ):
        result = _compute_num_groups(num_channels)
        assert result == expected_groups
        assert num_channels % result == 0

    @pytest.mark.parametrize("num_channels", [1, 2, 3, 7, 13, 17, 31])
    def test_result_always_divides_num_channels(
        self,
        num_channels: int,
    ):
        result = _compute_num_groups(num_channels)
        assert num_channels % result == 0
        assert result >= 1
        assert result <= max(1, num_channels // 16)


class TestConvertLayers:
    @pytest.mark.parametrize("channels", [64, 128])
    def test_converted_model_produces_valid_output(
        self,
        simple_model_factory: Callable[..., nn.Sequential],
        nchw_tensor_factory: Callable[..., torch.Tensor],
        channels: int,
    ):
        model = simple_model_factory(channels=channels)
        convert_layers(
            model=model,
            layer_type_old=nn.BatchNorm2d,
            layer_type_new=nn.GroupNorm,
        )
        tensor = nchw_tensor_factory(batch_size=2, channels=channels)
        output = model(tensor)
        assert output.shape == (2, channels, 8, 8)
        assert model[0].num_channels == channels
        assert model[0].num_groups == _compute_num_groups(channels)

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
        assert torch.all(model[0].weight.data == original_weight)
        assert torch.all(model[0].bias.data == original_bias)

    def test_does_not_copy_weights_when_disabled(
        self,
        simple_model_factory: Callable[..., nn.Sequential],
    ):
        model = simple_model_factory(channels=64, include_non_bn_layers=False)
        model[0].weight.data.fill_(42.0)
        convert_layers(
            model=model,
            layer_type_old=nn.BatchNorm2d,
            layer_type_new=nn.GroupNorm,
            convert_weights=False,
        )
        assert not torch.all(model[0].weight.data == 42.0)

    def test_skips_weight_copy_when_affine_is_false(self):
        model = nn.Sequential(nn.BatchNorm2d(64, affine=False))
        convert_layers(
            model=model,
            layer_type_old=nn.BatchNorm2d,
            layer_type_new=nn.GroupNorm,
            convert_weights=True,
        )
        assert model[0].affine is False
        assert len(list(model[0].parameters())) == 0

    def test_converts_nested_layers_recursively(
        self,
        nested_model_factory: Callable[..., nn.Module],
        nchw_tensor_factory: Callable[..., torch.Tensor],
    ):
        channels = 64
        model = nested_model_factory(channels=channels)
        convert_layers(
            model=model,
            layer_type_old=nn.BatchNorm2d,
            layer_type_new=nn.GroupNorm,
        )
        tensor = nchw_tensor_factory(batch_size=2, channels=3)
        output = model(tensor)
        assert output.shape == (2, channels, 8, 8)
        expected_groups = _compute_num_groups(channels)
        assert model[0][1].num_groups == expected_groups
        assert model[0][1].num_channels == channels
        assert model[1][1].num_groups == expected_groups
        assert model[1][1].num_channels == channels

    def test_returns_same_model_object(
        self,
        simple_model_factory: Callable[..., nn.Sequential],
    ):
        model = simple_model_factory(channels=64)
        result = convert_layers(
            model=model,
            layer_type_old=nn.BatchNorm2d,
            layer_type_new=nn.GroupNorm,
        )
        assert result is model

    def test_skips_none_modules(
        self,
        model_with_none_module_factory: Callable[..., nn.Sequential],
    ):
        channels = 64
        model = model_with_none_module_factory(channels=channels)
        convert_layers(
            model=model,
            layer_type_old=nn.BatchNorm2d,
            layer_type_new=nn.GroupNorm,
        )
        assert model[0].num_channels == channels
        assert model[0].num_groups == _compute_num_groups(channels)

    def test_non_groupnorm_target_uses_num_channels_as_num_groups(
        self,
        simple_model_factory: Callable[..., nn.Sequential],
    ):
        channels = 64
        model = simple_model_factory(channels=channels, include_non_bn_layers=False)
        convert_layers(
            model=model,
            layer_type_old=nn.BatchNorm2d,
            layer_type_new=_CustomNorm,
        )
        # When target is not GroupNorm, num_groups falls back to num_channels
        assert model[0].num_groups == channels
        assert model[0].num_channels == channels
        assert model[0].eps == 1e-5
        assert model[0].affine is True


class TestReplaceBatchnormWithGroupnorm:
    def test_converts_all_batchnorm_variants(
        self,
        nchw_tensor_factory: Callable[..., torch.Tensor],
    ):
        channels = 64
        model = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SyncBatchNorm(channels),
        )
        replace_batchnorm_with_groupnorm(model=model)
        tensor = nchw_tensor_factory(batch_size=2, channels=3)
        output = model(tensor)
        assert output.shape == (2, channels, 8, 8)
        # No module should retain the BatchNorm-specific num_features attribute
        for module in model.modules():
            assert not hasattr(module, "num_features")
        expected_groups = _compute_num_groups(channels)
        assert model[1].num_groups == expected_groups
        assert model[1].num_channels == channels
        assert model[4].num_groups == expected_groups
        assert model[4].num_channels == channels

    def test_does_not_modify_non_batchnorm_layers(
        self,
        nested_model_factory: Callable[..., nn.Module],
    ):
        channels = 64
        model = nested_model_factory(channels=channels)
        conv1_weight = model[0][0].weight.data.clone()
        conv2_weight = model[1][0].weight.data.clone()
        replace_batchnorm_with_groupnorm(model=model)
        assert torch.all(model[0][0].weight.data == conv1_weight)
        assert torch.all(model[1][0].weight.data == conv2_weight)
