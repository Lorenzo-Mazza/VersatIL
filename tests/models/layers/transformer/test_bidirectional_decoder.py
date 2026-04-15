"""Tests for versatil.models.layers.transformer.bidirectional_decoder module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise

import pytest
import torch

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType, PositionalEncodingType
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.transformer.bidirectional_decoder import (
    BidirectionalDecoder,
)


@pytest.fixture
def bidirectional_decoder_factory() -> Callable[..., BidirectionalDecoder]:
    """Factory for BidirectionalDecoder modules."""

    def factory(
        number_of_layers: int = 2,
        embedding_dimension: int = 32,
        number_of_heads: int = 4,
        number_of_key_value_heads: int | None = None,
        feedforward_dimension: int | None = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation: str = ActivationFunction.GELU.value,
        normalization_type: str = NormalizationType.LAYER_NORM.value,
        attention_type: str = AttentionType.MULTI_HEAD.value,
        positional_encoding_type: str | None = None,
        maximum_sequence_length: int = 128,
        bias: bool = True,
        normalization_epsilon: float = 1e-6,
        initializer_range: float = 0.02,
        use_cross_attention: bool = True,
    ) -> BidirectionalDecoder:
        return BidirectionalDecoder(
            number_of_layers=number_of_layers,
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_key_value_heads,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation=activation,
            normalization_type=normalization_type,
            attention_type=attention_type,
            positional_encoding_type=positional_encoding_type,
            maximum_sequence_length=maximum_sequence_length,
            bias=bias,
            normalization_epsilon=normalization_epsilon,
            initializer_range=initializer_range,
            use_cross_attention=use_cross_attention,
        )

    return factory


class TestBidirectionalDecoderInitialization:
    @pytest.mark.parametrize("number_of_layers", [1, 3])
    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("number_of_heads", [4, 8])
    @pytest.mark.parametrize("use_cross_attention", [True, False])
    def test_stores_configuration(
        self,
        bidirectional_decoder_factory: Callable[..., BidirectionalDecoder],
        number_of_layers: int,
        embedding_dimension: int,
        number_of_heads: int,
        use_cross_attention: bool,
    ):
        decoder = bidirectional_decoder_factory(
            number_of_layers=number_of_layers,
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            use_cross_attention=use_cross_attention,
        )
        assert decoder.number_of_layers == number_of_layers
        assert decoder.embedding_dimension == embedding_dimension
        assert decoder.number_of_heads == number_of_heads
        assert decoder.use_cross_attention == use_cross_attention
        expected_residual_blocks = 3 if use_cross_attention else 2
        assert decoder.number_of_residual_blocks == expected_residual_blocks

    def test_creates_correct_number_of_layers(
        self, bidirectional_decoder_factory: Callable[..., BidirectionalDecoder]
    ):
        decoder = bidirectional_decoder_factory(number_of_layers=3)
        assert len(decoder.layers) == 3

    def test_layers_are_non_autoregressive(
        self, bidirectional_decoder_factory: Callable[..., BidirectionalDecoder]
    ):
        decoder = bidirectional_decoder_factory(number_of_layers=2)
        for layer in decoder.layers:
            assert layer.autoregressive is False

    @pytest.mark.parametrize("use_cross_attention", [True, False])
    def test_layers_cross_attention_matches_config(
        self,
        bidirectional_decoder_factory: Callable[..., BidirectionalDecoder],
        use_cross_attention: bool,
    ):
        decoder = bidirectional_decoder_factory(
            number_of_layers=2, use_cross_attention=use_cross_attention
        )
        for layer in decoder.layers:
            assert layer.use_cross_attention is use_cross_attention

    def test_no_positional_encoding_by_default(
        self, bidirectional_decoder_factory: Callable[..., BidirectionalDecoder]
    ):
        decoder = bidirectional_decoder_factory(positional_encoding_type=None)
        assert decoder.positional_encoding is None

    @pytest.mark.parametrize(
        "positional_encoding_type",
        [PositionalEncodingType.SINUSOIDAL.value, PositionalEncodingType.ROPE.value],
    )
    def test_positional_encoding_created_when_specified(
        self,
        bidirectional_decoder_factory: Callable[..., BidirectionalDecoder],
        positional_encoding_type: str,
    ):
        decoder = bidirectional_decoder_factory(
            positional_encoding_type=positional_encoding_type,
        )
        assert decoder.positional_encoding is not None

    def test_gqa_requires_kv_heads(
        self, bidirectional_decoder_factory: Callable[..., BidirectionalDecoder]
    ):
        with pytest.raises(
            ValueError,
            match=re.escape("number_of_key_value_heads required for GQA"),
        ):
            bidirectional_decoder_factory(
                attention_type=AttentionType.GROUPED_QUERY.value,
                number_of_key_value_heads=None,
            )

    def test_head_dimension_computed(
        self, bidirectional_decoder_factory: Callable[..., BidirectionalDecoder]
    ):
        decoder = bidirectional_decoder_factory(
            embedding_dimension=64, number_of_heads=8
        )
        assert decoder.head_dimension == 8


class TestBidirectionalDecoderForward:
    @pytest.mark.parametrize(
        "use_cross_attention, provide_features, expectation",
        [
            (True, True, does_not_raise()),
            (
                True,
                False,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "Either encoded_features or conditioning_cache must be "
                        "provided when use_cross_attention=True."
                    ),
                ),
            ),
            (False, False, does_not_raise()),
        ],
        ids=[
            "cross_attn_with_features",
            "cross_attn_missing_features",
            "no_cross_attn",
        ],
    )
    def test_encoded_features_validation(
        self,
        bidirectional_decoder_factory: Callable[..., BidirectionalDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        use_cross_attention: bool,
        provide_features: bool,
        expectation: object,
    ):
        decoder = bidirectional_decoder_factory(
            number_of_layers=1,
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=use_cross_attention,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        memory = (
            sequence_tensor_factory(
                batch_size=2, sequence_length=6, embedding_dimension=32
            )
            if provide_features
            else None
        )
        with expectation:
            decoder(hidden_states=hidden_states, encoded_features=memory)

    @pytest.mark.parametrize("use_cross_attention", [True, False])
    def test_output_shape(
        self,
        bidirectional_decoder_factory: Callable[..., BidirectionalDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        use_cross_attention: bool,
    ):
        decoder = bidirectional_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=use_cross_attention,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=5, embedding_dimension=32
        )
        memory = sequence_tensor_factory(
            batch_size=2, sequence_length=8, embedding_dimension=32
        )
        output = decoder(
            hidden_states=hidden_states,
            encoded_features=memory if use_cross_attention else None,
        )
        assert output.shape == (2, 5, 32)

    def test_bidirectional_all_positions_see_all_positions(
        self,
        bidirectional_decoder_factory: Callable[..., BidirectionalDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        decoder = bidirectional_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            number_of_heads=4,
            initializer_range=0.5,
        )
        decoder.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=1, sequence_length=4, embedding_dimension=32
        )
        memory = sequence_tensor_factory(
            batch_size=1, sequence_length=6, embedding_dimension=32
        )
        output_original = decoder(hidden_states=hidden_states, encoded_features=memory)
        # Modify the last position with a large perturbation
        modified_hidden_states = hidden_states.clone()
        modified_hidden_states[0, 3, :] *= 100.0
        output_modified = decoder(
            hidden_states=modified_hidden_states, encoded_features=memory
        )
        # Bidirectional: modifying position 3 should change ALL positions in the output
        for position in range(4):
            assert not torch.allclose(
                output_original[0, position],
                output_modified[0, position],
                atol=1e-5,
            )

    def test_query_padding_mask_affects_output(
        self,
        bidirectional_decoder_factory: Callable[..., BidirectionalDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        padding_mask_factory: Callable[..., torch.Tensor],
    ):
        decoder = bidirectional_decoder_factory(
            number_of_layers=2, embedding_dimension=32, number_of_heads=4
        )
        decoder.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        memory = sequence_tensor_factory(
            batch_size=2, sequence_length=6, embedding_dimension=32
        )
        query_mask = padding_mask_factory(
            batch_size=2, sequence_length=4, padded_positions=[[2, 3], []]
        )
        output_masked = decoder(
            hidden_states=hidden_states,
            encoded_features=memory,
            query_padding_mask=query_mask,
        )
        output_unmasked = decoder(hidden_states=hidden_states, encoded_features=memory)
        assert not torch.allclose(output_masked[0], output_unmasked[0], atol=1e-5)

    def test_memory_padding_mask_affects_output(
        self,
        bidirectional_decoder_factory: Callable[..., BidirectionalDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        padding_mask_factory: Callable[..., torch.Tensor],
    ):
        decoder = bidirectional_decoder_factory(
            number_of_layers=2, embedding_dimension=32, number_of_heads=4
        )
        decoder.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        memory = sequence_tensor_factory(
            batch_size=2, sequence_length=6, embedding_dimension=32
        )
        memory_mask = padding_mask_factory(
            batch_size=2, sequence_length=6, padded_positions=[[4, 5], []]
        )
        output_masked = decoder(
            hidden_states=hidden_states,
            encoded_features=memory,
            memory_padding_mask=memory_mask,
        )
        output_unmasked = decoder(hidden_states=hidden_states, encoded_features=memory)
        assert not torch.allclose(output_masked[0], output_unmasked[0], atol=1e-5)

    def test_different_memory_produces_different_output(
        self,
        bidirectional_decoder_factory: Callable[..., BidirectionalDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        decoder = bidirectional_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            number_of_heads=4,
        )
        decoder.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        memory_a = sequence_tensor_factory(
            batch_size=2, sequence_length=6, embedding_dimension=32
        )
        memory_b = sequence_tensor_factory(
            batch_size=2, sequence_length=6, embedding_dimension=32
        )
        output_a = decoder(hidden_states=hidden_states, encoded_features=memory_a)
        output_b = decoder(hidden_states=hidden_states, encoded_features=memory_b)
        assert not torch.allclose(output_a, output_b, atol=1e-5)

    @pytest.mark.parametrize(
        "positional_encoding_type",
        [PositionalEncodingType.SINUSOIDAL.value, PositionalEncodingType.ROPE.value],
        ids=["sinusoidal", "rope"],
    )
    def test_positional_encoding_produces_valid_output(
        self,
        bidirectional_decoder_factory: Callable[..., BidirectionalDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        positional_encoding_type: str,
    ):
        decoder = bidirectional_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            number_of_heads=4,
            positional_encoding_type=positional_encoding_type,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=5, embedding_dimension=32
        )
        memory = sequence_tensor_factory(
            batch_size=2, sequence_length=8, embedding_dimension=32
        )
        output = decoder(hidden_states=hidden_states, encoded_features=memory)
        assert output.shape == (2, 5, 32)


class TestBidirectionalDecoderSelfAttentionOnly:
    def test_output_shape_without_cross_attention(
        self,
        bidirectional_decoder_factory: Callable[..., BidirectionalDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        decoder = bidirectional_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=False,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=6, embedding_dimension=32
        )
        output = decoder(hidden_states=hidden_states)
        assert output.shape == (2, 6, 32)
        assert torch.all(torch.isfinite(output))

    def test_padding_mask_affects_output_without_cross_attention(
        self,
        bidirectional_decoder_factory: Callable[..., BidirectionalDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        padding_mask_factory: Callable[..., torch.Tensor],
    ):
        decoder = bidirectional_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=False,
        )
        decoder.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        mask = padding_mask_factory(
            batch_size=2, sequence_length=4, padded_positions=[[2, 3], []]
        )
        output_masked = decoder(hidden_states=hidden_states, query_padding_mask=mask)
        output_unmasked = decoder(hidden_states=hidden_states)
        assert not torch.allclose(output_masked[0], output_unmasked[0], atol=1e-5)


class TestBidirectionalDecoderPrecomputeConditioningKV:
    def test_returns_one_layer_cache_per_decoder_layer(
        self,
        bidirectional_decoder_factory: Callable[..., BidirectionalDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        decoder = bidirectional_decoder_factory(
            number_of_layers=3,
            embedding_dimension=32,
            number_of_heads=4,
        )
        memory = sequence_tensor_factory(
            batch_size=2, sequence_length=8, embedding_dimension=32
        )
        conditioning_cache = decoder.precompute_conditioning_kv(encoded_features=memory)
        assert len(conditioning_cache.layers) == 3

    def test_layer_cache_kv_shapes(
        self,
        bidirectional_decoder_factory: Callable[..., BidirectionalDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        decoder = bidirectional_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            number_of_heads=4,
        )
        memory = sequence_tensor_factory(
            batch_size=2, sequence_length=8, embedding_dimension=32
        )
        conditioning_cache = decoder.precompute_conditioning_kv(encoded_features=memory)
        for layer_cache in conditioning_cache.layers:
            # (B=2, heads=4, S=8, head_dim=8)
            assert layer_cache.keys.shape == (2, 4, 8, 8)
            assert layer_cache.values.shape == (2, 4, 8, 8)

    def test_cached_forward_matches_uncached(
        self,
        bidirectional_decoder_factory: Callable[..., BidirectionalDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        decoder = bidirectional_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            number_of_heads=4,
        )
        decoder.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        memory = sequence_tensor_factory(
            batch_size=2, sequence_length=6, embedding_dimension=32
        )
        output_uncached = decoder(hidden_states=hidden_states, encoded_features=memory)
        conditioning_cache = decoder.precompute_conditioning_kv(encoded_features=memory)
        output_cached = decoder(
            hidden_states=hidden_states, conditioning_cache=conditioning_cache
        )
        assert torch.allclose(output_uncached, output_cached, atol=1e-5)


class TestBidirectionalDecoderExpandPaddingMask:
    def test_expands_to_four_dimensions(
        self,
        padding_mask_factory: Callable[..., torch.Tensor],
    ):
        mask = padding_mask_factory(
            batch_size=2, sequence_length=6, padded_positions=[[4, 5], [5]]
        )
        expanded = BidirectionalDecoder._expand_padding_mask(
            padding_mask=mask, query_length=4
        )
        assert expanded.shape == (2, 1, 4, 6)

    def test_padded_positions_broadcast_across_queries(
        self,
        padding_mask_factory: Callable[..., torch.Tensor],
    ):
        mask = padding_mask_factory(
            batch_size=1, sequence_length=6, padded_positions=[[5]]
        )
        expanded = BidirectionalDecoder._expand_padding_mask(
            padding_mask=mask, query_length=4
        )
        # Key position 5 should be masked for all query positions
        for query_index in range(4):
            assert expanded[0, 0, query_index, 5].item() is True
        # Key position 0 should not be masked
        assert expanded[0, 0, 0, 0].item() is False
