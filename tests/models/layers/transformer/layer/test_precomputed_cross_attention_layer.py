"""Tests for versatil.models.layers.transformer.layer.precomputed_kv_layer module."""

from collections.abc import Callable

import pytest
import torch

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.transformer.cache.conditioning import (
    ConditioningLayerCache,
)
from versatil.models.layers.transformer.layer.precomputed_kv_layer import (
    PrecomputedKVCrossAttentionLayer,
)

NUM_ATTENTION_HEADS = 2
NUM_KEY_VALUE_HEADS = 2
HEAD_DIMENSION = 8
EMBEDDING_DIMENSION = 16
FEEDFORWARD_DIMENSION = EMBEDDING_DIMENSION * 4
CONDITIONING_KV_DIMENSION = NUM_KEY_VALUE_HEADS * HEAD_DIMENSION
BATCH_SIZE = 2
SEQUENCE_LENGTH = 4
VLM_SEQUENCE_LENGTH = 8


@pytest.fixture
def layer_factory() -> Callable[..., PrecomputedKVCrossAttentionLayer]:

    def factory(
        embedding_dimension: int = EMBEDDING_DIMENSION,
        conditioning_key_value_dimension: int = CONDITIONING_KV_DIMENSION,
        number_of_heads: int = NUM_ATTENTION_HEADS,
        number_of_key_value_heads: int = NUM_KEY_VALUE_HEADS,
        head_dimension: int = HEAD_DIMENSION,
        feedforward_dimension: int = FEEDFORWARD_DIMENSION,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        conditioning_dimension: int | None = None,
        use_gating: bool = False,
        dropout: float = 0.0,
        activation: str = ActivationFunction.SILU.value,
    ) -> PrecomputedKVCrossAttentionLayer:
        return PrecomputedKVCrossAttentionLayer(
            embedding_dimension=embedding_dimension,
            conditioning_key_value_dimension=conditioning_key_value_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_key_value_heads,
            head_dimension=head_dimension,
            feedforward_dimension=feedforward_dimension,
            normalization_type=normalization_type,
            conditioning_dimension=conditioning_dimension,
            use_gating=use_gating,
            dropout=dropout,
            activation=activation,
        )

    return factory


class TestPrecomputedCrossAttentionLayer:
    @pytest.mark.parametrize(
        "conditioning_kv_dimension",
        [CONDITIONING_KV_DIMENSION, 128],
    )
    def test_bridges_different_embedding_dimensions(
        self,
        layer_factory: Callable[..., PrecomputedKVCrossAttentionLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        flat_conditioning_cache_factory: Callable[..., ConditioningLayerCache],
        conditioning_kv_dimension: int,
    ):
        layer = layer_factory(
            conditioning_key_value_dimension=conditioning_kv_dimension,
        )
        hidden = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        cache = flat_conditioning_cache_factory(
            kv_dimension=conditioning_kv_dimension,
        )
        output = layer(hidden_states=hidden, conditioning_cache=cache)
        assert output.shape == (BATCH_SIZE, SEQUENCE_LENGTH, EMBEDDING_DIMENSION)
        assert torch.all(torch.isfinite(output))

    @pytest.mark.parametrize(
        "conditioning_kv_dimension",
        [CONDITIONING_KV_DIMENSION, 128],
    )
    def test_kv_projection_maps_to_local_dimension(
        self,
        layer_factory: Callable[..., PrecomputedKVCrossAttentionLayer],
        conditioning_kv_dimension: int,
    ):
        layer = layer_factory(
            conditioning_key_value_dimension=conditioning_kv_dimension,
        )
        expected_local_dim = NUM_KEY_VALUE_HEADS * HEAD_DIMENSION
        assert layer.key_projection.in_features == conditioning_kv_dimension
        assert layer.key_projection.out_features == expected_local_dim
        assert layer.value_projection.in_features == conditioning_kv_dimension
        assert layer.value_projection.out_features == expected_local_dim

    def test_different_cache_produces_different_outputs(
        self,
        layer_factory: Callable[..., PrecomputedKVCrossAttentionLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        flat_conditioning_cache_factory: Callable[..., ConditioningLayerCache],
    ):
        layer = layer_factory()
        layer.eval()
        hidden = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        cache_a = flat_conditioning_cache_factory(
            kv_dimension=CONDITIONING_KV_DIMENSION
        )
        cache_b = flat_conditioning_cache_factory(
            kv_dimension=CONDITIONING_KV_DIMENSION
        )
        output_a = layer(hidden_states=hidden, conditioning_cache=cache_a)
        output_b = layer(hidden_states=hidden, conditioning_cache=cache_b)
        assert not torch.allclose(output_a, output_b)

    def test_gradients_flow_through_kv_projections(
        self,
        layer_factory: Callable[..., PrecomputedKVCrossAttentionLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        flat_conditioning_cache_factory: Callable[..., ConditioningLayerCache],
    ):
        layer = layer_factory()
        hidden = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        cache = flat_conditioning_cache_factory(kv_dimension=CONDITIONING_KV_DIMENSION)
        output = layer(hidden_states=hidden, conditioning_cache=cache)
        output.sum().backward()
        assert layer.key_projection.weight.grad is not None
        assert layer.value_projection.weight.grad is not None
