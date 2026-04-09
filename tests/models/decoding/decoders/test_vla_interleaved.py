"""Tests for versatil.models.decoding.decoders.vla_interleaved module."""

from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.models.decoding.decoders.vla_interleaved import VLACrossAttentionLayer
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.normalization.constants import NormalizationType

VLM_HIDDEN_DIMENSION = 32
NUM_ATTENTION_HEADS = 2
NUM_KEY_VALUE_HEADS = 2
HEAD_DIMENSION = VLM_HIDDEN_DIMENSION // NUM_ATTENTION_HEADS
EXPERT_HIDDEN_SIZE = 16
EXPERT_FEEDFORWARD_DIMENSION = EXPERT_HIDDEN_SIZE * 4
BATCH_SIZE = 2
EXPERT_SEQUENCE_LENGTH = 4
VLM_SEQUENCE_LENGTH = 8


@pytest.fixture
def cross_attention_layer_factory() -> Callable[..., VLACrossAttentionLayer]:

    def factory(
        expert_embedding_dimension: int = EXPERT_HIDDEN_SIZE,
        vlm_key_value_dimension: int = NUM_KEY_VALUE_HEADS * HEAD_DIMENSION,
        expert_number_of_heads: int = NUM_ATTENTION_HEADS,
        expert_number_of_key_value_heads: int = NUM_KEY_VALUE_HEADS,
        expert_head_dimension: int = HEAD_DIMENSION,
        expert_feedforward_dimension: int = EXPERT_FEEDFORWARD_DIMENSION,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        activation: str = ActivationFunction.SILU.value,
        dropout: float = 0.0,
    ) -> VLACrossAttentionLayer:
        return VLACrossAttentionLayer(
            expert_embedding_dimension=expert_embedding_dimension,
            vlm_key_value_dimension=vlm_key_value_dimension,
            expert_number_of_heads=expert_number_of_heads,
            expert_number_of_key_value_heads=expert_number_of_key_value_heads,
            expert_head_dimension=expert_head_dimension,
            expert_feedforward_dimension=expert_feedforward_dimension,
            normalization_type=normalization_type,
            activation=activation,
            dropout=dropout,
        )

    return factory


@pytest.fixture
def expert_hidden_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:

    def factory(
        batch_size: int = BATCH_SIZE,
        sequence_length: int = EXPERT_SEQUENCE_LENGTH,
        embedding_dimension: int = EXPERT_HIDDEN_SIZE,
    ) -> torch.Tensor:
        return torch.from_numpy(
            rng.standard_normal(
                (batch_size, sequence_length, embedding_dimension)
            ).astype(np.float32)
        )

    return factory


@pytest.fixture
def vlm_kv_factory(
    rng: np.random.Generator,
) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:

    def factory(
        batch_size: int = BATCH_SIZE,
        sequence_length: int = VLM_SEQUENCE_LENGTH,
        kv_dimension: int = NUM_KEY_VALUE_HEADS * HEAD_DIMENSION,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (batch_size, sequence_length, kv_dimension)
        keys = torch.from_numpy(rng.standard_normal(shape).astype(np.float32))
        values = torch.from_numpy(rng.standard_normal(shape).astype(np.float32))
        return keys, values

    return factory


class TestVLACrossAttentionLayer:
    @pytest.mark.parametrize("vlm_kv_dim", [NUM_KEY_VALUE_HEADS * HEAD_DIMENSION, 128])
    @pytest.mark.parametrize(
        "activation",
        [ActivationFunction.SILU.value, ActivationFunction.SWIGLU.value],
    )
    def test_forward_produces_valid_output(
        self,
        cross_attention_layer_factory: Callable[..., VLACrossAttentionLayer],
        expert_hidden_factory: Callable[..., torch.Tensor],
        vlm_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
        vlm_kv_dim: int,
        activation: str,
    ):
        layer = cross_attention_layer_factory(
            vlm_key_value_dimension=vlm_kv_dim,
            activation=activation,
        )
        expert_hidden = expert_hidden_factory()
        vlm_keys, vlm_values = vlm_kv_factory(kv_dimension=vlm_kv_dim)
        output = layer(
            expert_hidden_states=expert_hidden,
            vlm_key_states=vlm_keys,
            vlm_value_states=vlm_values,
        )
        assert output.shape == expert_hidden.shape
        assert torch.all(torch.isfinite(output))
        assert not torch.allclose(output, expert_hidden)

    def test_different_vlm_kv_produce_different_outputs(
        self,
        cross_attention_layer_factory: Callable[..., VLACrossAttentionLayer],
        expert_hidden_factory: Callable[..., torch.Tensor],
        vlm_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ):
        layer = cross_attention_layer_factory()
        layer.eval()
        expert_hidden = expert_hidden_factory()
        keys_a, values_a = vlm_kv_factory()
        keys_b, values_b = vlm_kv_factory()
        output_a = layer(
            expert_hidden_states=expert_hidden,
            vlm_key_states=keys_a,
            vlm_value_states=values_a,
        )
        output_b = layer(
            expert_hidden_states=expert_hidden,
            vlm_key_states=keys_b,
            vlm_value_states=values_b,
        )
        assert not torch.allclose(output_a, output_b)
