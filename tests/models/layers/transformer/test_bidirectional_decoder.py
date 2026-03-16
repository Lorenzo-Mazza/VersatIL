"""Tests for versatil.models.layers.transformer.bidirectional_decoder module."""

import re
from collections.abc import Callable

import numpy as np
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
        )

    return factory


class TestBidirectionalDecoderInitialization:
    @pytest.mark.parametrize("number_of_layers", [1, 3])
    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("number_of_heads", [4, 8])
    def test_stores_configuration(
        self,
        bidirectional_decoder_factory: Callable[..., BidirectionalDecoder],
        number_of_layers: int,
        embedding_dimension: int,
        number_of_heads: int,
    ):
        decoder = bidirectional_decoder_factory(
            number_of_layers=number_of_layers,
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
        )
        assert decoder.number_of_layers == number_of_layers
        assert decoder.embedding_dimension == embedding_dimension
        assert decoder.number_of_heads == number_of_heads

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

    def test_layers_use_cross_attention(
        self, bidirectional_decoder_factory: Callable[..., BidirectionalDecoder]
    ):
        decoder = bidirectional_decoder_factory(number_of_layers=2)
        for layer in decoder.layers:
            assert layer.use_cross_attention is True

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
    def test_output_shape(
        self,
        bidirectional_decoder_factory: Callable[..., BidirectionalDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        decoder = bidirectional_decoder_factory(
            number_of_layers=2, embedding_dimension=32, number_of_heads=4
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=5, embedding_dimension=32
        )
        memory = sequence_tensor_factory(
            batch_size=2, sequence_length=8, embedding_dimension=32
        )
        output = decoder(hidden_states=hidden_states, encoded_features=memory)
        assert output.shape == (2, 5, 32)

    def test_bidirectional_all_positions_see_all_positions(
        self,
        bidirectional_decoder_factory: Callable[..., BidirectionalDecoder],
        rng: np.random.Generator,
    ):
        decoder = bidirectional_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            number_of_heads=4,
            initializer_range=0.5,
        )
        decoder.eval()
        hidden_states = torch.from_numpy(
            rng.standard_normal((1, 4, 32)).astype(np.float32)
        )
        memory = torch.from_numpy(rng.standard_normal((1, 6, 32)).astype(np.float32))
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
        rng: np.random.Generator,
        padding_mask_factory: Callable[..., torch.Tensor],
    ):
        decoder = bidirectional_decoder_factory(
            number_of_layers=2, embedding_dimension=32, number_of_heads=4
        )
        decoder.eval()
        hidden_states = torch.from_numpy(
            rng.standard_normal((2, 4, 32)).astype(np.float32)
        )
        memory = torch.from_numpy(rng.standard_normal((2, 6, 32)).astype(np.float32))
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
        rng: np.random.Generator,
        padding_mask_factory: Callable[..., torch.Tensor],
    ):
        decoder = bidirectional_decoder_factory(
            number_of_layers=2, embedding_dimension=32, number_of_heads=4
        )
        decoder.eval()
        hidden_states = torch.from_numpy(
            rng.standard_normal((2, 4, 32)).astype(np.float32)
        )
        memory = torch.from_numpy(rng.standard_normal((2, 6, 32)).astype(np.float32))
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
        rng: np.random.Generator,
    ):
        decoder = bidirectional_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            number_of_heads=4,
        )
        decoder.eval()
        hidden_states = torch.from_numpy(
            rng.standard_normal((2, 4, 32)).astype(np.float32)
        )
        memory_a = torch.from_numpy(rng.standard_normal((2, 6, 32)).astype(np.float32))
        memory_b = torch.from_numpy(rng.standard_normal((2, 6, 32)).astype(np.float32))
        output_a = decoder(hidden_states=hidden_states, encoded_features=memory_a)
        output_b = decoder(hidden_states=hidden_states, encoded_features=memory_b)
        assert not torch.allclose(output_a, output_b, atol=1e-5)

    def test_with_sinusoidal_positional_encoding(
        self,
        bidirectional_decoder_factory: Callable[..., BidirectionalDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        decoder = bidirectional_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            number_of_heads=4,
            positional_encoding_type=PositionalEncodingType.SINUSOIDAL.value,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=5, embedding_dimension=32
        )
        memory = sequence_tensor_factory(
            batch_size=2, sequence_length=8, embedding_dimension=32
        )
        output = decoder(hidden_states=hidden_states, encoded_features=memory)
        assert output.shape == (2, 5, 32)

    def test_with_rope_positional_encoding(
        self,
        bidirectional_decoder_factory: Callable[..., BidirectionalDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        decoder = bidirectional_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            number_of_heads=4,
            positional_encoding_type=PositionalEncodingType.ROPE.value,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=5, embedding_dimension=32
        )
        memory = sequence_tensor_factory(
            batch_size=2, sequence_length=8, embedding_dimension=32
        )
        output = decoder(hidden_states=hidden_states, encoded_features=memory)
        assert output.shape == (2, 5, 32)


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
