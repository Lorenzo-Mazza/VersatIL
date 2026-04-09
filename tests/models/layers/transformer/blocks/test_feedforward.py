"""Tests for versatil.models.layers.transformer.block.feedforward module."""

from collections.abc import Callable

import pytest
import torch
import torch.nn as nn

from tests.models.layers.conftest import reinit_modulation_layers
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.gated_linear_unit import GatedLinearUnit
from versatil.models.layers.normalization.ada_norm import AdaNorm
from versatil.models.layers.normalization.unconditioned_norm import UnconditionedNorm
from versatil.models.layers.transformer.block.feedforward import (
    FeedforwardBlock,
    build_feedforward,
)

from .conftest import EMBEDDING_DIMENSION, FEEDFORWARD_DIMENSION


class TestBuildFeedforward:
    @pytest.mark.parametrize(
        "activation, expected_first_module_type",
        [
            (ActivationFunction.SWIGLU.value, GatedLinearUnit),
            (ActivationFunction.GEGLU.value, GatedLinearUnit),
            (ActivationFunction.GELU.value, nn.Linear),
        ],
        ids=["swiglu", "geglu", "gelu"],
    )
    def test_first_module_type_matches_activation(
        self,
        activation: str,
        expected_first_module_type: type,
    ):
        feedforward = build_feedforward(
            embedding_dimension=EMBEDDING_DIMENSION,
            feedforward_dimension=FEEDFORWARD_DIMENSION,
            activation=activation,
        )
        assert isinstance(feedforward[0], expected_first_module_type)

    def test_square_root_weight_flag_on_final_linear(self):
        feedforward = build_feedforward(
            embedding_dimension=EMBEDDING_DIMENSION,
            feedforward_dimension=FEEDFORWARD_DIMENSION,
        )
        assert feedforward[-1].SQUARE_ROOT_WEIGHT is True


class TestFeedforwardBlockForward:
    def test_unconditioned_norm_ignores_conditioning(
        self,
        unconditioned_norm: UnconditionedNorm,
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        block = FeedforwardBlock(
            feedforward=build_feedforward(
                embedding_dimension=EMBEDDING_DIMENSION,
                feedforward_dimension=FEEDFORWARD_DIMENSION,
                dropout=0.0,
            ),
            normalization=unconditioned_norm,
            dropout=0.0,
        )
        block.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        conditioning = condition_factory(
            batch_size=2, condition_dim=EMBEDDING_DIMENSION
        )
        output_without = block(hidden_states=hidden_states)
        output_with = block(hidden_states=hidden_states, conditioning=conditioning)
        assert torch.allclose(output_without, output_with)

    def test_ada_norm_different_conditioning_produces_different_outputs(
        self,
        ada_norm_no_gate: AdaNorm,
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        block = FeedforwardBlock(
            feedforward=build_feedforward(
                embedding_dimension=EMBEDDING_DIMENSION,
                feedforward_dimension=FEEDFORWARD_DIMENSION,
                dropout=0.0,
            ),
            normalization=ada_norm_no_gate,
            dropout=0.0,
        )
        reinit_modulation_layers(block)
        hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        conditioning_a = condition_factory(
            batch_size=2, condition_dim=EMBEDDING_DIMENSION
        )
        conditioning_b = condition_factory(
            batch_size=2, condition_dim=EMBEDDING_DIMENSION
        )
        output_a = block(hidden_states=hidden_states, conditioning=conditioning_a)
        output_b = block(hidden_states=hidden_states, conditioning=conditioning_b)
        assert not torch.allclose(output_a, output_b)

    def test_residual_connection_adds_input_to_feedforward_output(
        self,
        unconditioned_norm: UnconditionedNorm,
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        feedforward = build_feedforward(
            embedding_dimension=EMBEDDING_DIMENSION,
            feedforward_dimension=FEEDFORWARD_DIMENSION,
            dropout=0.0,
        )
        block = FeedforwardBlock(
            feedforward=feedforward,
            normalization=unconditioned_norm,
            dropout=0.0,
        )
        block.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        block_output = block(hidden_states=hidden_states)
        normed, _ = unconditioned_norm(x=hidden_states)
        feedforward_only = feedforward(normed)
        # block_output = hidden_states + feedforward(norm(hidden_states))
        assert not torch.allclose(block_output, feedforward_only)
        expected = hidden_states + feedforward_only
        assert torch.allclose(block_output, expected, atol=1e-6)
