"""Tests for versatil.models.layers.normalization.ada_norm module."""

from collections.abc import Callable

import pytest
import torch
from torch import nn

from versatil.models.layers.normalization.ada_norm import AdaNorm
from versatil.models.layers.normalization.rms_norm import RMSNorm


@pytest.fixture
def ada_norm_factory() -> Callable[..., AdaNorm]:
    """Factory for AdaNorm instances with configurable parameters."""

    def factory(
        condition_dim: int = 32,
        feature_dim: int = 64,
        use_gate: bool = False,
        base_norm: nn.Module | None = None,
    ) -> AdaNorm:
        if base_norm is None:
            base_norm = nn.LayerNorm(feature_dim, elementwise_affine=False)
        return AdaNorm(
            base_norm=base_norm,
            condition_dim=condition_dim,
            feature_dim=feature_dim,
            use_gate=use_gate,
        )

    return factory


class TestAdaNormInitialization:
    @pytest.mark.parametrize("condition_dim", [16, 64])
    @pytest.mark.parametrize("feature_dim", [32, 128])
    @pytest.mark.parametrize("use_gate", [True, False])
    def test_stores_configuration(
        self,
        condition_dim: int,
        feature_dim: int,
        use_gate: bool,
    ):
        base_norm = nn.LayerNorm(feature_dim, elementwise_affine=False)
        norm = AdaNorm(
            base_norm=base_norm,
            condition_dim=condition_dim,
            feature_dim=feature_dim,
            use_gate=use_gate,
        )
        assert norm.condition_dim == condition_dim
        assert norm.feature_dim == feature_dim
        assert norm.modulation.feature_dim == feature_dim

    @pytest.mark.parametrize("use_gate", [True, False])
    def test_modulation_has_no_effect_at_init(
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
        # Both "identity" (no gate) and "zero" (gate) init zero out weights
        normalized = nn.LayerNorm(feature_dim, elementwise_affine=False)(tensor)
        with torch.no_grad():
            result = norm.modulation(normalized, condition)
        modulated, gate = result
        assert torch.allclose(modulated, normalized, atol=1e-6)
        if use_gate:
            assert torch.allclose(gate, torch.zeros_like(gate))
        else:
            assert torch.equal(gate, torch.ones(1, dtype=tensor.dtype))


class TestAdaNormForward:
    @pytest.mark.parametrize("batch_size, sequence_length", [(2, 6), (4, 10)])
    def test_output_shape_without_gate(
        self,
        ada_norm_factory: Callable[..., AdaNorm],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
        batch_size: int,
        sequence_length: int,
    ):
        feature_dim = 64
        norm = ada_norm_factory(
            condition_dim=32,
            feature_dim=feature_dim,
            use_gate=False,
        )
        features = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=feature_dim,
        )
        condition = condition_factory(batch_size=batch_size, condition_dim=32)
        output, _ = norm(features, condition)
        assert output.shape == features.shape

    def test_gate_returns_tuple_with_correct_shapes(
        self,
        ada_norm_factory: Callable[..., AdaNorm],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        feature_dim = 64
        batch_size = 2
        sequence_length = 8
        norm = ada_norm_factory(
            condition_dim=32,
            feature_dim=feature_dim,
            use_gate=True,
        )
        features = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=feature_dim,
        )
        condition = condition_factory(batch_size=batch_size, condition_dim=32)
        result = norm(features, condition)
        assert len(result) == 2
        modulated, gate = result
        assert modulated.shape == features.shape
        assert gate.shape == (batch_size, 1, feature_dim)

    def test_identity_init_output_equals_base_norm_output(
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
        )
        features = sequence_tensor_factory(
            batch_size=2,
            sequence_length=8,
            embedding_dimension=feature_dim,
        )
        condition = condition_factory(batch_size=2, condition_dim=32)
        base_norm = nn.LayerNorm(feature_dim, elementwise_affine=False)
        with torch.no_grad():
            ada_output, _ = norm(features, condition)
            base_output = base_norm(features)
        # At init, modulation is identity → ada output == base norm output
        assert torch.allclose(ada_output, base_output, atol=1e-6)

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
        )
        # Set modulation weights to nonzero so conditioning has effect
        for layer in norm.modulation.projection.modules():
            if hasattr(layer, "weight"):
                nn.init.xavier_uniform_(layer.weight)
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

    def test_works_with_rms_norm_base(
        self,
        ada_norm_factory: Callable[..., AdaNorm],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        feature_dim = 64
        rms_base = RMSNorm(normalized_shape=feature_dim, elementwise_affine=False)
        norm = ada_norm_factory(
            condition_dim=32,
            feature_dim=feature_dim,
            use_gate=False,
            base_norm=rms_base,
        )
        features = sequence_tensor_factory(
            batch_size=2,
            sequence_length=8,
            embedding_dimension=feature_dim,
        )
        condition = condition_factory(batch_size=2, condition_dim=32)
        with torch.no_grad():
            ada_output, _ = norm(features, condition)
            rms_output = rms_base(features)
        # At init, modulation is identity → output equals RMSNorm output
        assert torch.allclose(ada_output, rms_output, atol=1e-6)
