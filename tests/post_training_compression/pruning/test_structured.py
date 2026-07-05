"""Tests for versatil.post_training_compression.pruning.structured module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise

import pytest
import torch
from torch import nn

from versatil.post_training_compression.constants import PrunableLayerType
from versatil.post_training_compression.pruning.structured import StructuredPruner


@pytest.mark.unit
class TestStructuredPruner:
    @pytest.mark.parametrize("amount", [0.3, 0.7])
    @pytest.mark.parametrize("norm_order", [1, 2])
    @pytest.mark.parametrize("dimension", [0, 1])
    @pytest.mark.parametrize(
        "layer_types",
        [
            [PrunableLayerType.CONV2D.value],
            [PrunableLayerType.CONV2D.value, PrunableLayerType.LINEAR.value],
            None,
        ],
    )
    def test_stores_configuration(
        self,
        amount: float,
        norm_order: int,
        dimension: int,
        layer_types: list[str] | None,
    ):
        pruner = StructuredPruner(
            amount=amount,
            norm_order=norm_order,
            dimension=dimension,
            layer_types=layer_types,
        )

        assert pruner.amount == amount
        assert pruner.norm_order == norm_order
        assert pruner.dimension == dimension
        if layer_types is None:
            expected = tuple(
                PrunableLayerType(name).to_module_type()
                for name in [
                    PrunableLayerType.CONV1D.value,
                    PrunableLayerType.CONV2D.value,
                    PrunableLayerType.LINEAR.value,
                ]
            )
        else:
            expected = tuple(
                PrunableLayerType(name).to_module_type() for name in layer_types
            )
        assert pruner.layer_types == expected

    @pytest.mark.parametrize("amount", [0.25, 0.5])
    def test_achieves_target_sparsity(
        self,
        pruning_model_factory: Callable[..., nn.Module],
        amount: float,
    ):
        model = pruning_model_factory(hidden_channels=16)
        pruner = StructuredPruner(
            amount=amount,
            layer_types=[PrunableLayerType.CONV2D.value],
        )

        total_parameters, zero_parameters = pruner.prune(module=model)

        actual_sparsity = zero_parameters / total_parameters
        assert actual_sparsity > 0.05
        assert actual_sparsity < 0.8

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

        StructuredPruner(
            amount=0.25,
            layer_types=[PrunableLayerType.CONV2D.value],
        ).prune(module=model)

        with torch.no_grad():
            output_after = model(input_data)
        assert output_after.shape == output_before.shape
        assert not torch.equal(output_after, output_before)

    def test_pruning_removes_reparametrization_hooks(
        self,
        pruning_model_factory: Callable[..., nn.Module],
    ):
        model = pruning_model_factory(hidden_channels=16)
        StructuredPruner(
            amount=0.25,
            layer_types=[PrunableLayerType.CONV2D.value],
        ).prune(module=model)

        for child in model.modules():
            if isinstance(child, nn.Conv2d):
                assert isinstance(child.weight, nn.Parameter)
                assert len(child._forward_pre_hooks) == 0

    def test_different_norm_orders_produce_different_patterns(
        self,
        pruning_model_factory: Callable[..., nn.Module],
    ):
        model_l1 = pruning_model_factory(hidden_channels=16)
        model_l2 = pruning_model_factory(hidden_channels=16)

        StructuredPruner(
            amount=0.25,
            norm_order=1,
            layer_types=[PrunableLayerType.CONV2D.value],
        ).prune(module=model_l1)
        StructuredPruner(
            amount=0.25,
            norm_order=2,
            layer_types=[PrunableLayerType.CONV2D.value],
        ).prune(module=model_l2)

        l1_zeros = {
            name: (param == 0).flatten()
            for name, param in model_l1.named_parameters()
            if "weight" in name
        }
        l2_zeros = {
            name: (param == 0).flatten()
            for name, param in model_l2.named_parameters()
            if "weight" in name
        }
        patterns_differ = any(
            not torch.equal(l1_zeros[name], l2_zeros[name])
            for name in l1_zeros
            if name in l2_zeros
        )
        assert patterns_differ, "L1 and L2 norms produced identical zero patterns"

    def test_dimension_zeros_correct_axis(
        self,
        pruning_model_factory: Callable[..., nn.Module],
    ):
        model = pruning_model_factory(hidden_channels=16)
        conv = None
        for mod in model.modules():
            if isinstance(mod, nn.Conv2d):
                conv = mod
                break
        assert conv is not None

        StructuredPruner(
            amount=0.5,
            dimension=0,
            layer_types=[PrunableLayerType.CONV2D.value],
        ).prune(module=model)

        # dimension=0 zeros entire output filters
        filter_norms = conv.weight.data.flatten(1).norm(dim=1)
        zeroed_filters = (filter_norms == 0).sum().item()
        assert zeroed_filters > 0

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
        model = pruning_model_factory(hidden_channels=16)
        original_weights = {
            id(mod): mod.weight.data.clone()
            for mod in model.modules()
            if isinstance(mod, untouched_type)
        }

        StructuredPruner(
            amount=0.5,
            layer_types=targeted,
        ).prune(module=model)

        for mod in model.modules():
            if isinstance(mod, untouched_type):
                assert torch.equal(
                    mod.weight.data,
                    original_weights[id(mod)],
                )

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
                -0.5,
                pytest.raises(
                    ValueError,
                    match=re.escape("Pruning amount must be in (0, 1), got -0.5"),
                ),
            ),
        ],
    )
    def test_amount_validation(self, amount: float, expectation):
        with expectation:
            StructuredPruner(amount=amount)

    def test_prune_raises_when_no_layers_match(self):
        pruner = StructuredPruner(
            amount=0.5, layer_types=[PrunableLayerType.CONV2D.value]
        )
        module = nn.Sequential(nn.LayerNorm(4), nn.ReLU())

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Structured pruning selected no modules; the target module "
                "contains no ['Conv2d'] layers."
            ),
        ):
            pruner.prune(module=module)
