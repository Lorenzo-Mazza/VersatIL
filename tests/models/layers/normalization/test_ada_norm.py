"""Tests for versatil.models.layers.normalization.ada_norm module."""
from collections.abc import Callable

import numpy as np
import pytest
import torch
from torch import nn

from versatil.models.layers.modulation.conditional_modulation import (
    ConditionalModulation,
)
from versatil.models.layers.normalization.ada_norm import AdaNorm
from versatil.models.layers.normalization.rms_norm import RMSNorm


@pytest.fixture
def ada_norm_factory() -> Callable[..., AdaNorm]:
    """Factory for AdaNorm instances with configurable parameters."""
    def factory(
        condition_dim: int = 32,
        feature_dim: int = 64,
        use_gate: bool = False,
    ) -> AdaNorm:
        base_norm = nn.LayerNorm(feature_dim, elementwise_affine=False)
        return AdaNorm(
            base_norm=base_norm,
            condition_dim=condition_dim,
            feature_dim=feature_dim,
            use_gate=use_gate,
        )
    return factory


@pytest.fixture
def condition_tensor_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for condition tensors (B, condition_dim)."""
    def factory(
        batch_size: int = 2,
        condition_dim: int = 32,
    ) -> torch.Tensor:
        shape = (batch_size, condition_dim)
        return torch.from_numpy(
            rng.standard_normal(shape).astype(np.float32)
        )
    return factory


@pytest.fixture
def feature_tensor_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for 3D feature tensors (B, S, D) as expected by ConditionalModulation."""
    def factory(
        batch_size: int = 2,
        sequence_length: int = 8,
        feature_dim: int = 64,
    ) -> torch.Tensor:
        shape = (batch_size, sequence_length, feature_dim)
        return torch.from_numpy(
            rng.standard_normal(shape).astype(np.float32)
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
        assert norm.norm is base_norm

    def test_creates_modulation_layer(
        self,
        ada_norm_factory: Callable[..., AdaNorm],
    ):
        norm = ada_norm_factory(
            condition_dim=32,
            feature_dim=64,
        )
        assert isinstance(norm.modulation, ConditionalModulation)

    def test_inherits_from_nn_module(
        self,
        ada_norm_factory: Callable[..., AdaNorm],
    ):
        norm = ada_norm_factory(condition_dim=32, feature_dim=64, use_gate=False)
        assert isinstance(norm, nn.Module)

    def test_accepts_rms_norm_as_base(self):
        feature_dim = 64
        base_norm = RMSNorm(
            normalized_shape=feature_dim,
            elementwise_affine=False,
        )
        norm = AdaNorm(
            base_norm=base_norm,
            condition_dim=32,
            feature_dim=feature_dim,
        )
        assert isinstance(norm.norm, RMSNorm)


class TestAdaNormForward:

    @pytest.mark.parametrize("batch_size, sequence_length", [
        (2, 6),
        (4, 10),
    ])
    def test_output_shape_matches_input(
        self,
        ada_norm_factory: Callable[..., AdaNorm],
        feature_tensor_factory: Callable[..., torch.Tensor],
        condition_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
        sequence_length: int,
    ):
        condition_dim = 32
        feature_dim = 64
        norm = ada_norm_factory(
            condition_dim=condition_dim,
            feature_dim=feature_dim,
            use_gate=False,
        )
        features = feature_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            feature_dim=feature_dim,
        )
        condition = condition_tensor_factory(
            batch_size=batch_size,
            condition_dim=condition_dim,
        )
        output = norm(features, condition)
        assert output.shape == features.shape

    def test_returns_tuple_with_gate(
        self,
        ada_norm_factory: Callable[..., AdaNorm],
        feature_tensor_factory: Callable[..., torch.Tensor],
        condition_tensor_factory: Callable[..., torch.Tensor],
    ):
        condition_dim = 32
        feature_dim = 64
        batch_size = 2
        sequence_length = 8
        norm = ada_norm_factory(
            condition_dim=condition_dim,
            feature_dim=feature_dim,
            use_gate=True,
        )
        features = feature_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            feature_dim=feature_dim,
        )
        condition = condition_tensor_factory(
            batch_size=batch_size,
            condition_dim=condition_dim,
        )
        output = norm(features, condition)
        assert isinstance(output, tuple)
        assert len(output) == 2
        modulated, gate = output
        assert modulated.shape == features.shape
        # Gate is broadcast along sequence dimension: (B, 1, D)
        assert gate.shape == (batch_size, 1, feature_dim)

    def test_returns_tensor_without_gate(
        self,
        ada_norm_factory: Callable[..., AdaNorm],
        feature_tensor_factory: Callable[..., torch.Tensor],
        condition_tensor_factory: Callable[..., torch.Tensor],
    ):
        condition_dim = 32
        feature_dim = 64
        batch_size = 2
        sequence_length = 8
        norm = ada_norm_factory(
            condition_dim=condition_dim,
            feature_dim=feature_dim,
            use_gate=False,
        )
        features = feature_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            feature_dim=feature_dim,
        )
        condition = condition_tensor_factory(
            batch_size=batch_size,
            condition_dim=condition_dim,
        )
        output = norm(features, condition)
        assert isinstance(output, torch.Tensor)
        assert output.shape == features.shape

    def test_output_shape_with_different_sequence_lengths(
        self,
        ada_norm_factory: Callable[..., AdaNorm],
        feature_tensor_factory: Callable[..., torch.Tensor],
        condition_tensor_factory: Callable[..., torch.Tensor],
    ):
        condition_dim = 32
        feature_dim = 64
        batch_size = 2
        sequence_length = 16
        norm = ada_norm_factory(
            condition_dim=condition_dim,
            feature_dim=feature_dim,
            use_gate=True,
        )
        features = feature_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            feature_dim=feature_dim,
        )
        condition = condition_tensor_factory(
            batch_size=batch_size,
            condition_dim=condition_dim,
        )
        output = norm(features, condition)
        assert isinstance(output, tuple)
        modulated, gate = output
        assert modulated.shape == (batch_size, sequence_length, feature_dim)
        # Gate is broadcast: (B, 1, D)
        assert gate.shape == (batch_size, 1, feature_dim)
