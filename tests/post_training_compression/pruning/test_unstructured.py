"""Tests for versatil.post_training_compression.pruning.unstructured module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise

import pytest
import torch
from torch import nn

from versatil.post_training_compression.constants import PrunableLayerType
from versatil.post_training_compression.preparation import (
    fuse_all_conv_batchnorm_pairs,
    prepare_batchnorms_for_quantization,
)
from versatil.post_training_compression.pruning.unstructured import UnstructuredPruner


@pytest.fixture
def conv_batchnorm_model_factory() -> Callable[..., nn.Module]:
    """Factory for a model with Conv+BN pairs that can be fused."""

    def factory() -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 4),
        )

    return factory


@pytest.mark.unit
class TestUnstructuredPruner:
    @pytest.mark.parametrize("amount", [0.3, 0.7])
    @pytest.mark.parametrize(
        "layer_types",
        [
            [PrunableLayerType.CONV2D.value, PrunableLayerType.LINEAR.value],
            [PrunableLayerType.CONV2D.value],
            None,
        ],
    )
    def test_stores_configuration(
        self,
        amount: float,
        layer_types: list[str] | None,
    ):
        pruner = UnstructuredPruner(amount=amount, layer_types=layer_types)

        assert pruner.amount == amount
        if layer_types is None:
            # None targets convolution and linear layers, never norm scales
            # or embedding tables.
            assert pruner.layer_types == (nn.Conv1d, nn.Conv2d, nn.Linear)
        else:
            expected_types = tuple(
                PrunableLayerType(name).to_module_type() for name in layer_types
            )
            assert pruner.layer_types == expected_types

    @pytest.mark.parametrize("amount", [0.3, 0.5, 0.8])
    def test_achieves_target_sparsity(
        self,
        pruning_model_factory: Callable[..., nn.Module],
        amount: float,
    ):
        model = pruning_model_factory(hidden_channels=32, linear_features=64)
        pruner = UnstructuredPruner(amount=amount)

        total_parameters, zero_parameters = pruner.prune(module=model)

        actual_sparsity = zero_parameters / total_parameters
        assert abs(actual_sparsity - amount) < 0.05

    @pytest.mark.parametrize(
        "input_channels, image_size",
        [
            (3, 8),
            (1, 16),
        ],
    )
    def test_output_changes_while_shape_preserved(
        self,
        pruning_model_factory: Callable[..., nn.Module],
        spatial_input_factory: Callable[..., torch.Tensor],
        input_channels: int,
        image_size: int,
    ):
        model = pruning_model_factory(input_channels=input_channels)
        input_data = spatial_input_factory(
            channels=input_channels,
            height=image_size,
            width=image_size,
        )
        with torch.no_grad():
            output_before = model(input_data)

        UnstructuredPruner(amount=0.5).prune(module=model)

        with torch.no_grad():
            output_after = model(input_data)
        assert output_after.shape == output_before.shape
        assert not torch.equal(output_after, output_before)

    def test_pruning_removes_reparametrization_hooks(
        self,
        pruning_model_factory: Callable[..., nn.Module],
    ):
        model = pruning_model_factory()
        UnstructuredPruner(amount=0.5).prune(module=model)

        for child in model.modules():
            if hasattr(child, "weight"):
                assert isinstance(child.weight, nn.Parameter)
                assert len(child._forward_pre_hooks) == 0

    @pytest.mark.parametrize(
        "targeted, untouched_type",
        [
            ([PrunableLayerType.CONV2D.value], nn.Linear),
            ([PrunableLayerType.LINEAR.value], nn.Conv2d),
        ],
    )
    def test_layer_types_filtering(
        self,
        pruning_model_factory: Callable[..., nn.Module],
        targeted: list[str],
        untouched_type: type[nn.Module],
    ):
        model = pruning_model_factory()
        original_weights = {
            id(mod): mod.weight.data.clone()
            for mod in model.modules()
            if isinstance(mod, untouched_type)
        }

        UnstructuredPruner(
            amount=0.5,
            layer_types=targeted,
        ).prune(module=model)

        for mod in model.modules():
            if isinstance(mod, untouched_type):
                assert torch.equal(
                    mod.weight.data,
                    original_weights[id(mod)],
                )

    def test_none_layer_types_prunes_all_weighted_modules(
        self,
        pruning_model_factory: Callable[..., nn.Module],
    ):
        model = pruning_model_factory(hidden_channels=32, linear_features=64)
        pruner = UnstructuredPruner(amount=0.5, layer_types=None)

        pruner.prune(module=model)

        for mod in model.modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                zeros = (mod.weight == 0).sum().item()
                assert zeros > 0, f"{type(mod).__name__} was not pruned"

    @pytest.mark.parametrize(
        "amount, expectation",
        [
            (0.5, does_not_raise()),
            (
                0.0,
                pytest.raises(
                    ValueError,
                    match=re.escape("Pruning amount must be in (0, 1), got 0.0"),
                ),
            ),
            (
                1.0,
                pytest.raises(
                    ValueError,
                    match=re.escape("Pruning amount must be in (0, 1), got 1.0"),
                ),
            ),
            (
                -0.1,
                pytest.raises(
                    ValueError,
                    match=re.escape("Pruning amount must be in (0, 1), got -0.1"),
                ),
            ),
        ],
    )
    def test_amount_validation(self, amount: float, expectation):
        with expectation:
            UnstructuredPruner(amount=amount)

    def test_prunes_after_batchnorm_fusion(
        self,
        conv_batchnorm_model_factory: Callable[..., nn.Module],
    ):
        model = conv_batchnorm_model_factory()
        model.eval()
        # BN fusion can leave residual modules with float weight attributes
        prepare_batchnorms_for_quantization(model)
        fuse_all_conv_batchnorm_pairs(model)

        total, zeroed = UnstructuredPruner(amount=0.5).prune(module=model)

        assert zeroed > 0
        assert zeroed / total > 0.3

    def test_prune_raises_when_no_layers_match(self):
        pruner = UnstructuredPruner(
            amount=0.5, layer_types=[PrunableLayerType.CONV2D.value]
        )
        module = nn.Sequential(nn.LayerNorm(4), nn.ReLU())

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Unstructured pruning selected no modules; the target module "
                "contains no ['Conv2d'] layers."
            ),
        ):
            pruner.prune(module=module)
