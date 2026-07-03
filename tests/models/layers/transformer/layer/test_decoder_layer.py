"""Tests for versatil.models.layers.transformer.layer.decoder_layer module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.models.layers.conftest import reinit_modulation_layers
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.transformer.cache.conditioning import (
    CacheableLayer,
    ConditioningLayerCache,
)
from versatil.models.layers.transformer.cache.generation import GenerationLayerCache
from versatil.models.layers.transformer.layer.decoder_layer import (
    TransformerDecoderLayer,
)


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
        conditioning_dimension: int | None = None,
        use_gating: bool = False,
        cross_attention_normalization_type: str | None = None,
        cross_attention_conditioning_dimension: int | None = None,
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
            conditioning_dimension=conditioning_dimension,
            use_gating=use_gating,
            cross_attention_normalization_type=cross_attention_normalization_type,
            cross_attention_conditioning_dimension=cross_attention_conditioning_dimension,
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
        if use_cross_attention:
            assert layer.cross_attention_block is not None
        else:
            assert layer.cross_attention_block is None

    def test_feedforward_last_layer_has_initialization_flag(
        self, decoder_layer_factory: Callable[..., TransformerDecoderLayer]
    ):
        layer = decoder_layer_factory()
        assert layer.feedforward_block.feedforward[-1].SQUARE_ROOT_WEIGHT is True


class TestTransformerDecoderLayerForward:
    @pytest.mark.parametrize("use_cross_attention", [True, False])
    def test_output_shape_and_values(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        use_cross_attention: bool,
    ):
        layer = decoder_layer_factory(
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=use_cross_attention,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=5, embedding_dimension=32
        )
        kwargs = {}
        if use_cross_attention:
            kwargs["encoded_features"] = sequence_tensor_factory(
                batch_size=2, sequence_length=8, embedding_dimension=32
            )
        output, cache = layer(hidden_states=hidden_states, **kwargs)
        assert output.shape == (2, 5, 32)
        assert torch.all(torch.isfinite(output))
        assert not torch.allclose(output, hidden_states)
        assert cache is None

    def test_generation_cache_returns_updated_cache(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        layer = decoder_layer_factory(
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=False,
            dropout=0.0,
        )
        layer.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=1, embedding_dimension=32
        )
        empty_cache = GenerationLayerCache(
            keys=torch.empty(2, 4, 0, 8),
            values=torch.empty(2, 4, 0, 8),
        )
        output, new_cache = layer(
            hidden_states=hidden_states,
            generation_cache=empty_cache,
        )
        assert new_cache is not None
        assert new_cache.get_length() == 1
        # Second step with cache should produce valid output
        step2_input = sequence_tensor_factory(
            batch_size=2, sequence_length=1, embedding_dimension=32
        )
        output2, cache2 = layer(
            hidden_states=step2_input,
            generation_cache=new_cache,
        )
        assert torch.all(torch.isfinite(output2))
        assert cache2.get_length() == 2

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
                "Either encoded_features or conditioning_cache must be provided when using cross-attention"
            ),
        ):
            layer(hidden_states=hidden_states, encoded_features=None)

    def test_self_attention_mask_affects_output(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        attention_mask_factory: Callable[..., torch.Tensor],
    ):
        layer = decoder_layer_factory(
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=False,
            dropout=0.0,
        )
        layer.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
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

    def test_conditioning_cached_kv_matches_fresh_forward(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        layer = decoder_layer_factory(
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=True,
            dropout=0.0,
        )
        layer.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=1, embedding_dimension=32
        )
        memory = sequence_tensor_factory(
            batch_size=2, sequence_length=6, embedding_dimension=32
        )
        # Fresh forward with encoded_features
        output_fresh, _ = layer(hidden_states=hidden_states, encoded_features=memory)
        # Pre-compute conditioning K/V manually via the layer's projections
        cross_attn = layer.cross_attention_block.attention
        projected_keys = cross_attn.compute_key(memory)
        projected_values = cross_attn.compute_value(memory)
        conditioning_cache = ConditioningLayerCache(
            keys=projected_keys,
            values=projected_values,
        )
        output_cached, _ = layer(
            hidden_states=hidden_states,
            conditioning_cache=conditioning_cache,
        )
        assert torch.allclose(output_fresh, output_cached, atol=1e-5)

    def test_positional_encoding_affects_output(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        mock_rope_factory: Callable[..., MagicMock],
    ):
        embedding_dimension = 32
        number_of_heads = 4
        head_dimension = embedding_dimension // number_of_heads
        layer = decoder_layer_factory(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            use_cross_attention=False,
            dropout=0.0,
        )
        layer.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=embedding_dimension
        )
        mock_rope = mock_rope_factory(head_dimension=head_dimension)
        output_with_rope, _ = layer(
            hidden_states=hidden_states, positional_encoding=mock_rope
        )
        output_without_rope, _ = layer(hidden_states=hidden_states)
        assert not torch.allclose(output_with_rope, output_without_rope)
        mock_rope.compute_rotation_components.assert_called_once()

    def test_different_encoded_features_produce_different_output(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        layer = decoder_layer_factory(
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=True,
            dropout=0.0,
        )
        layer.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        memory_a = sequence_tensor_factory(
            batch_size=2, sequence_length=6, embedding_dimension=32
        )
        memory_b = sequence_tensor_factory(
            batch_size=2, sequence_length=6, embedding_dimension=32
        )
        output_a, _ = layer(hidden_states=hidden_states, encoded_features=memory_a)
        output_b, _ = layer(hidden_states=hidden_states, encoded_features=memory_b)
        assert not torch.allclose(output_a, output_b, atol=1e-6)

    @pytest.mark.parametrize(
        "activation",
        [ActivationFunction.GELU.value, ActivationFunction.SWIGLU.value],
    )
    def test_gated_and_nongated_activations_produce_valid_output(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        activation: str,
    ):
        layer = decoder_layer_factory(
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=False,
            activation=activation,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        output, _ = layer(hidden_states=hidden_states)
        assert output.shape == hidden_states.shape
        assert torch.all(torch.isfinite(output))

    def test_grouped_query_attention(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        layer = decoder_layer_factory(
            embedding_dimension=32,
            number_of_heads=4,
            number_of_key_value_heads=2,
            attention_type=AttentionType.GROUPED_QUERY.value,
            use_cross_attention=False,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        output, _ = layer(hidden_states=hidden_states)
        assert output.shape == hidden_states.shape
        assert torch.all(torch.isfinite(output))


class TestTransformerDecoderLayerPrecomputeConditioningKV:
    def test_satisfies_cacheable_layer_protocol(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
    ):
        layer = decoder_layer_factory(
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=True,
        )
        assert isinstance(layer, CacheableLayer)

    def test_returns_layer_cache_when_cross_attention_enabled(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        layer = decoder_layer_factory(
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=True,
        )
        memory = sequence_tensor_factory(
            batch_size=2, sequence_length=6, embedding_dimension=32
        )
        cache = layer.precompute_conditioning_kv(encoded_features=memory)
        assert cache is not None
        # (B=2, heads=4, S=6, head_dim=8)
        assert cache.keys.shape == (2, 4, 6, 8)
        assert cache.values.shape == (2, 4, 6, 8)

    def test_returns_none_when_no_cross_attention(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        layer = decoder_layer_factory(
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=False,
        )
        memory = sequence_tensor_factory(
            batch_size=2, sequence_length=6, embedding_dimension=32
        )
        cache = layer.precompute_conditioning_kv(encoded_features=memory)
        assert cache is None

    def test_precomputed_cache_matches_fresh_forward(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        layer = decoder_layer_factory(
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=True,
            dropout=0.0,
        )
        layer.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        memory = sequence_tensor_factory(
            batch_size=2, sequence_length=6, embedding_dimension=32
        )
        output_fresh, _ = layer(hidden_states=hidden_states, encoded_features=memory)
        conditioning_cache = layer.precompute_conditioning_kv(encoded_features=memory)
        output_cached, _ = layer(
            hidden_states=hidden_states,
            conditioning_cache=conditioning_cache,
        )
        assert torch.allclose(output_fresh, output_cached, atol=1e-5)


class TestTransformerDecoderLayerConditioning:
    def test_adaptive_norm_different_conditioning_produces_different_outputs(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        layer = decoder_layer_factory(
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=False,
            normalization_type=NormalizationType.RMS_NORM.value,
            conditioning_dimension=32,
            dropout=0.0,
        )
        reinit_modulation_layers(layer)
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        conditioning_a = condition_factory(batch_size=2, conditioning_dimension=32)
        conditioning_b = condition_factory(batch_size=2, conditioning_dimension=32)
        output_a, _ = layer(hidden_states=hidden_states, conditioning=conditioning_a)
        output_b, _ = layer(hidden_states=hidden_states, conditioning=conditioning_b)
        assert not torch.allclose(output_a, output_b)

    def test_adaln_zero_gate_makes_output_equal_input_at_init(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        layer = decoder_layer_factory(
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=False,
            normalization_type=NormalizationType.RMS_NORM.value,
            conditioning_dimension=32,
            use_gating=True,
            dropout=0.0,
        )
        layer.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        conditioning = condition_factory(batch_size=2, conditioning_dimension=32)
        output, _ = layer(hidden_states=hidden_states, conditioning=conditioning)
        assert torch.allclose(output, hidden_states, atol=1e-6)

    def test_unconditioned_layer_ignores_conditioning_argument(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        layer = decoder_layer_factory(
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=False,
            normalization_type=NormalizationType.LAYER_NORM.value,
            dropout=0.0,
        )
        layer.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        conditioning = condition_factory(batch_size=2, conditioning_dimension=16)
        output_with_cond, _ = layer(
            hidden_states=hidden_states, conditioning=conditioning
        )
        output_without_cond, _ = layer(hidden_states=hidden_states)
        assert torch.allclose(output_with_cond, output_without_cond)

    def test_conditioning_with_cross_attention(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        layer = decoder_layer_factory(
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=True,
            normalization_type=NormalizationType.RMS_NORM.value,
            conditioning_dimension=32,
            dropout=0.0,
        )
        reinit_modulation_layers(layer)
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        memory = sequence_tensor_factory(
            batch_size=2, sequence_length=6, embedding_dimension=32
        )
        conditioning_a = condition_factory(batch_size=2, conditioning_dimension=32)
        conditioning_b = condition_factory(batch_size=2, conditioning_dimension=32)
        output_a, _ = layer(
            hidden_states=hidden_states,
            encoded_features=memory,
            conditioning=conditioning_a,
        )
        output_b, _ = layer(
            hidden_states=hidden_states,
            encoded_features=memory,
            conditioning=conditioning_b,
        )
        assert not torch.allclose(output_a, output_b)

    @pytest.mark.parametrize(
        "cross_attention_conditioning_dimension, expected_condition_dim",
        [
            (None, None),
            (32, 32),
            (16, 16),
        ],
        ids=["no_conditioning", "same_as_self_attention", "different_dimension"],
    )
    def test_cross_attention_conditioning_dimension_passed_to_factory(
        self,
        cross_attention_conditioning_dimension: int | None,
        expected_condition_dim: int | None,
    ):
        with patch(
            "versatil.models.layers.transformer.layer.decoder_layer.create_block_normalization"
        ) as mock_factory:
            mock_factory.return_value = MagicMock()
            TransformerDecoderLayer(
                embedding_dimension=32,
                number_of_heads=4,
                attention_type=AttentionType.MULTI_HEAD.value,
                use_cross_attention=True,
                conditioning_dimension=32,
                cross_attention_conditioning_dimension=cross_attention_conditioning_dimension,
            )
            # Second call is cross-attention normalization
            cross_attention_call = mock_factory.call_args_list[1]
            assert (
                cross_attention_call.kwargs["conditioning_dimension"]
                == expected_condition_dim
            )

    def test_conditioning_gradient_flows_through_modulation(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        layer = decoder_layer_factory(
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=False,
            normalization_type=NormalizationType.RMS_NORM.value,
            conditioning_dimension=32,
            dropout=0.0,
        )
        reinit_modulation_layers(layer)
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        conditioning = condition_factory(batch_size=2, conditioning_dimension=32)
        conditioning.requires_grad_(True)
        output, _ = layer(hidden_states=hidden_states, conditioning=conditioning)
        output.sum().backward()
        assert conditioning.grad is not None
        assert conditioning.grad.abs().sum().item() > 0.0
