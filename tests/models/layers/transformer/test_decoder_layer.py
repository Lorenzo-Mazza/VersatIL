"""Tests for versatil.models.layers.transformer.decoder_layer module."""

import re
from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.transformer.decoder_layer import TransformerDecoderLayer
from versatil.models.layers.transformer.kv_cache import LayerKVCache


@pytest.fixture
def decoder_layer_factory() -> Callable[..., TransformerDecoderLayer]:
    """Factory for TransformerDecoderLayer modules."""

    def factory(
        embedding_dimension: int = 32,
        number_of_heads: int = 4,
        number_of_key_value_heads: int | None = None,
        feedforward_dimension: int | None = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation: str = ActivationFunction.GELU.value,
        normalization_type: str = NormalizationType.LAYER_NORM.value,
        attention_type: str = AttentionType.MULTI_HEAD.value,
        use_cross_attention: bool = True,
        bias: bool = True,
        normalization_epsilon: float = 1e-6,
        autoregressive: bool = True,
    ) -> TransformerDecoderLayer:
        return TransformerDecoderLayer(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_key_value_heads,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation=activation,
            normalization_type=normalization_type,
            attention_type=attention_type,
            use_cross_attention=use_cross_attention,
            bias=bias,
            normalization_epsilon=normalization_epsilon,
            autoregressive=autoregressive,
        )

    return factory


class TestTransformerDecoderLayerInitialization:
    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("number_of_heads", [4, 8])
    @pytest.mark.parametrize("use_cross_attention", [True, False])
    def test_stores_configuration(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        embedding_dimension: int,
        number_of_heads: int,
        use_cross_attention: bool,
    ):
        layer = decoder_layer_factory(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            use_cross_attention=use_cross_attention,
        )
        assert layer.embedding_dimension == embedding_dimension
        assert layer.number_of_heads == number_of_heads
        assert layer.use_cross_attention == use_cross_attention

    def test_cross_attention_modules_created_when_enabled(
        self, decoder_layer_factory: Callable[..., TransformerDecoderLayer]
    ):
        layer = decoder_layer_factory(use_cross_attention=True)
        assert layer.cross_attention is not None
        assert layer.cross_attention_normalization is not None

    def test_cross_attention_modules_none_when_disabled(
        self, decoder_layer_factory: Callable[..., TransformerDecoderLayer]
    ):
        layer = decoder_layer_factory(use_cross_attention=False)
        assert layer.cross_attention is None
        assert layer.cross_attention_normalization is None

    @pytest.mark.parametrize(
        "activation",
        [ActivationFunction.SWIGLU.value, ActivationFunction.GELU.value],
    )
    def test_activation_variants(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        activation: str,
    ):
        layer = decoder_layer_factory(activation=activation)
        assert layer.feedforward_network is not None

    def test_feedforward_last_layer_has_initialization_flag(
        self, decoder_layer_factory: Callable[..., TransformerDecoderLayer]
    ):
        layer = decoder_layer_factory()
        assert layer.feedforward_network[-1].SQUARE_ROOT_WEIGHT is True


class TestTransformerDecoderLayerForward:
    def test_output_shape_with_cross_attention(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        layer = decoder_layer_factory(
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=True,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=5, embedding_dimension=32
        )
        memory = sequence_tensor_factory(
            batch_size=2, sequence_length=8, embedding_dimension=32
        )
        output, cache = layer(hidden_states=hidden_states, encoded_features=memory)
        assert output.shape == (2, 5, 32)
        assert cache is None

    def test_output_shape_without_cross_attention(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        layer = decoder_layer_factory(
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=False,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=5, embedding_dimension=32
        )
        output, cache = layer(hidden_states=hidden_states)
        assert output.shape == (2, 5, 32)

    def test_use_cache_returns_updated_cache(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        layer = decoder_layer_factory(
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=False,
            autoregressive=True,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=1, embedding_dimension=32
        )
        empty_cache = LayerKVCache(
            self_attention_keys=torch.empty(2, 4, 0, 8),
            self_attention_values=torch.empty(2, 4, 0, 8),
        )
        output, new_cache = layer(
            hidden_states=hidden_states,
            layer_cache=empty_cache,
            use_cache=True,
        )
        assert new_cache is not None
        assert new_cache.get_length() == 1

    def test_use_cache_on_non_autoregressive_raises(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        layer = decoder_layer_factory(
            embedding_dimension=32,
            number_of_heads=4,
            autoregressive=False,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=1, embedding_dimension=32
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "use_self_attention_cache=True only valid for autoregressive models"
            ),
        ):
            layer(hidden_states=hidden_states, use_cache=True)

    def test_cross_attention_without_features_or_cache_raises(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        layer = decoder_layer_factory(
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=True,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=5, embedding_dimension=32
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "encoded_features required when use_cross_attention=True and no cached cross KV"
            ),
        ):
            layer(hidden_states=hidden_states, encoded_features=None)

    def test_self_attention_mask_affects_output(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        rng: np.random.Generator,
        attention_mask_factory: Callable[..., torch.Tensor],
    ):
        layer = decoder_layer_factory(
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=False,
            dropout=0.0,
        )
        layer.eval()
        hidden_states = torch.from_numpy(
            rng.standard_normal((2, 4, 32)).astype(np.float32)
        )
        causal_mask = attention_mask_factory(
            batch_size=2, query_length=4, key_length=4, causal=True
        )
        output_causal, _ = layer(
            hidden_states=hidden_states,
            self_attention_mask=causal_mask,
        )
        output_no_mask, _ = layer(hidden_states=hidden_states)
        assert not torch.allclose(output_causal, output_no_mask, atol=1e-6)

    def test_cross_attention_with_cached_kv(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        rng: np.random.Generator,
    ):
        layer = decoder_layer_factory(
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=True,
            autoregressive=True,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=1, embedding_dimension=32
        )
        cross_keys = torch.from_numpy(
            rng.standard_normal((2, 4, 6, 8)).astype(np.float32)
        )
        cross_values = torch.from_numpy(
            rng.standard_normal((2, 4, 6, 8)).astype(np.float32)
        )
        cache = LayerKVCache(
            self_attention_keys=torch.empty(2, 4, 0, 8),
            self_attention_values=torch.empty(2, 4, 0, 8),
            cross_attention_keys=cross_keys,
            cross_attention_values=cross_values,
        )
        output, new_cache = layer(
            hidden_states=hidden_states,
            layer_cache=cache,
            use_cache=True,
        )
        assert output.shape == (2, 1, 32)
        assert new_cache is not None

    def test_different_encoded_features_produce_different_output(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        rng: np.random.Generator,
    ):
        layer = decoder_layer_factory(
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=True,
            dropout=0.0,
        )
        layer.eval()
        hidden_states = torch.from_numpy(
            rng.standard_normal((2, 4, 32)).astype(np.float32)
        )
        memory_a = torch.from_numpy(rng.standard_normal((2, 6, 32)).astype(np.float32))
        memory_b = torch.from_numpy(rng.standard_normal((2, 6, 32)).astype(np.float32))
        output_a, _ = layer(hidden_states=hidden_states, encoded_features=memory_a)
        output_b, _ = layer(hidden_states=hidden_states, encoded_features=memory_b)
        assert not torch.allclose(output_a, output_b, atol=1e-6)
