"""Tests for versatil.models.layers.normalization.ada_norm module."""

from collections.abc import Callable

import pytest
import torch
from torch import nn

from versatil.models.layers.normalization.ada_norm import AdaNorm
from versatil.models.layers.normalization.rms_norm import RMSNorm


class TestAdaNormInitialization:
    @pytest.mark.parametrize("condition_dim", [16, 64])
    @pytest.mark.parametrize("feature_dim", [32, 128])
    @pytest.mark.parametrize("use_gate", [True, False])
    def test_stores_configuration(
        self,
        ada_norm_factory: Callable[..., AdaNorm],
        condition_dim: int,
        feature_dim: int,
        use_gate: bool,
    ):
        norm = ada_norm_factory(
            condition_dim=condition_dim,
            feature_dim=feature_dim,
            use_gate=use_gate,
        )
        assert norm.condition_dim == condition_dim
        assert norm.feature_dim == feature_dim
        assert norm.modulation.feature_dim == feature_dim

    @pytest.mark.parametrize("use_gate", [True, False])
    def test_zero_init_modulation_has_no_effect(
        self,
        ada_norm_factory: Callable[..., AdaNorm],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
        use_gate: bool,
    ):
        feature_dim = 64
        norm = ada_norm_factory(
            condition_dim=32,
            feature_dim=feature_dim,
            use_gate=use_gate,
        )
        tensor = sequence_tensor_factory(
            batch_size=2,
            sequence_length=8,
            embedding_dimension=feature_dim,
        )
        condition = condition_factory(batch_size=2, condition_dim=32)
        base_norm = nn.LayerNorm(feature_dim, elementwise_affine=False)
        normalized = base_norm(tensor)
        with torch.no_grad():
            output, gate = norm(tensor, condition)
        # Zero init: modulation is identity, output equals plain norm
        assert torch.allclose(output, normalized, atol=1e-6)
        if use_gate:
            assert torch.allclose(gate, torch.zeros_like(gate))
        else:
            assert torch.equal(gate, torch.ones(1, dtype=tensor.dtype))

    @pytest.mark.parametrize("init_strategy", ["zero", "xavier"])
    def test_init_strategy_applied_to_modulation(
        self,
        ada_norm_factory: Callable[..., AdaNorm],
        init_strategy: str,
    ):
        norm = ada_norm_factory(
            condition_dim=32,
            feature_dim=64,
            init_strategy=init_strategy,
        )
        linear_weights = [
            layer.weight
            for layer in norm.modulation.projection.modules()
            if isinstance(layer, nn.Linear)
        ]
        if init_strategy == "zero":
            for weight in linear_weights:
                assert torch.all(weight == 0)
        else:
            has_nonzero = any(torch.any(w != 0).item() for w in linear_weights)
            assert has_nonzero


class TestAdaNormForward:
    def test_gate_shape_with_gating(
        self,
        ada_norm_factory: Callable[..., AdaNorm],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        feature_dim = 64
        batch_size = 2
        norm = ada_norm_factory(
            condition_dim=32,
            feature_dim=feature_dim,
            use_gate=True,
        )
        features = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=8,
            embedding_dimension=feature_dim,
        )
        condition = condition_factory(batch_size=batch_size, condition_dim=32)
        _, gate = norm(features, condition)
        assert gate.shape == (batch_size, 1, feature_dim)

    def test_different_conditions_produce_different_outputs(
        self,
        ada_norm_factory: Callable[..., AdaNorm],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        feature_dim = 64
        norm = ada_norm_factory(
            condition_dim=32,
            feature_dim=feature_dim,
            use_gate=False,
            init_strategy="xavier",
        )
        features = sequence_tensor_factory(
            batch_size=2,
            sequence_length=8,
            embedding_dimension=feature_dim,
        )
        condition_a = condition_factory(batch_size=2, condition_dim=32)
        condition_b = condition_factory(batch_size=2, condition_dim=32)
        with torch.no_grad():
            output_a, _ = norm(features, condition_a)
            output_b, _ = norm(features, condition_b)
        assert not torch.allclose(output_a, output_b)

    @pytest.mark.parametrize(
        "base_norm_class",
        [
            lambda dim: nn.LayerNorm(dim, elementwise_affine=False),
            lambda dim: RMSNorm(normalized_shape=dim, elementwise_affine=False),
        ],
        ids=["layernorm", "rmsnorm"],
    )
    def test_zero_init_output_equals_base_norm_output(
        self,
        ada_norm_factory: Callable[..., AdaNorm],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
        base_norm_class: Callable,
    ):
        feature_dim = 64
        base_norm = base_norm_class(feature_dim)
        norm = ada_norm_factory(
            condition_dim=32,
            feature_dim=feature_dim,
            use_gate=False,
            base_norm=base_norm,
        )
        features = sequence_tensor_factory(
            batch_size=2,
            sequence_length=8,
            embedding_dimension=feature_dim,
        )
        condition = condition_factory(batch_size=2, condition_dim=32)
        with torch.no_grad():
            ada_output, _ = norm(features, condition)
            base_output = base_norm(features)
        assert torch.allclose(ada_output, base_output, atol=1e-6)
