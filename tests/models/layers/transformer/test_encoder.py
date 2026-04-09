"""Tests for versatil.models.layers.transformer.encoder module."""

import re
from collections.abc import Callable

import pytest
import torch

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType, PositionalEncodingType
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.transformer.encoder import TransformerEncoder


@pytest.fixture
def encoder_factory() -> Callable[..., TransformerEncoder]:
    """Factory for TransformerEncoder modules."""

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
    ) -> TransformerEncoder:
        return TransformerEncoder(
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


class TestTransformerEncoderInitialization:
    @pytest.mark.parametrize("number_of_layers", [1, 4])
    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("number_of_heads", [4, 8])
    def test_stores_configuration(
        self,
        encoder_factory: Callable[..., TransformerEncoder],
        number_of_layers: int,
        embedding_dimension: int,
        number_of_heads: int,
    ):
        encoder = encoder_factory(
            number_of_layers=number_of_layers,
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
        )
        assert encoder.number_of_layers == number_of_layers
        assert encoder.embedding_dimension == embedding_dimension
        assert encoder.number_of_heads == number_of_heads
        assert encoder.number_of_residual_blocks == 2  # Self-Attn + FFN

    def test_creates_correct_number_of_layers(
        self, encoder_factory: Callable[..., TransformerEncoder]
    ):
        encoder = encoder_factory(number_of_layers=3)
        assert len(encoder.layers) == 3

    def test_no_positional_encoding_by_default(
        self, encoder_factory: Callable[..., TransformerEncoder]
    ):
        encoder = encoder_factory(positional_encoding_type=None)
        assert encoder.positional_encoding is None

    @pytest.mark.parametrize(
        "positional_encoding_type",
        [PositionalEncodingType.SINUSOIDAL.value, PositionalEncodingType.LEARNED.value],
    )
    def test_positional_encoding_created_when_specified(
        self,
        encoder_factory: Callable[..., TransformerEncoder],
        positional_encoding_type: str,
    ):
        encoder = encoder_factory(positional_encoding_type=positional_encoding_type)
        assert encoder.positional_encoding is not None

    def test_rope_positional_encoding_created(
        self, encoder_factory: Callable[..., TransformerEncoder]
    ):
        encoder = encoder_factory(
            positional_encoding_type=PositionalEncodingType.ROPE.value
        )
        assert encoder.positional_encoding is not None

    def test_gqa_requires_kv_heads(
        self, encoder_factory: Callable[..., TransformerEncoder]
    ):
        with pytest.raises(
            ValueError,
            match=re.escape("number_of_key_value_heads required for GQA"),
        ):
            encoder_factory(
                attention_type=AttentionType.GROUPED_QUERY.value,
                number_of_key_value_heads=None,
            )

    def test_gqa_stores_kv_heads(
        self, encoder_factory: Callable[..., TransformerEncoder]
    ):
        encoder = encoder_factory(
            number_of_heads=8,
            number_of_key_value_heads=2,
            attention_type=AttentionType.GROUPED_QUERY.value,
        )
        assert encoder.number_of_key_value_heads == 2

    def test_mha_sets_kv_heads_to_query_heads(
        self, encoder_factory: Callable[..., TransformerEncoder]
    ):
        encoder = encoder_factory(
            number_of_heads=8,
            attention_type=AttentionType.MULTI_HEAD.value,
        )
        assert encoder.number_of_key_value_heads == 8

    def test_head_dimension_computed(
        self, encoder_factory: Callable[..., TransformerEncoder]
    ):
        encoder = encoder_factory(embedding_dimension=64, number_of_heads=8)
        assert encoder.head_dimension == 8


class TestTransformerEncoderForward:
    def test_output_shape(
        self,
        encoder_factory: Callable[..., TransformerEncoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        encoder = encoder_factory(
            number_of_layers=2, embedding_dimension=32, number_of_heads=4
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=6, embedding_dimension=32
        )
        output = encoder(hidden_states=hidden_states)
        assert output.shape == (2, 6, 32)

    def test_padding_mask_affects_output(
        self,
        encoder_factory: Callable[..., TransformerEncoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        padding_mask_factory: Callable[..., torch.Tensor],
    ):
        encoder = encoder_factory(
            number_of_layers=2, embedding_dimension=32, number_of_heads=4
        )
        encoder.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        mask = padding_mask_factory(
            batch_size=2,
            sequence_length=4,
            padded_positions=[[2, 3], []],
        )
        output_with_mask = encoder(hidden_states=hidden_states, padding_mask=mask)
        output_without_mask = encoder(hidden_states=hidden_states)
        # Batch 0 should differ (has padding), batch 1 output may or may not differ
        assert not torch.allclose(
            output_with_mask[0], output_without_mask[0], atol=1e-5
        )

    def test_bidirectional_all_tokens_influence_all_outputs(
        self,
        encoder_factory: Callable[..., TransformerEncoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        encoder = encoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            number_of_heads=4,
            initializer_range=0.5,
        )
        encoder.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=1, sequence_length=4, embedding_dimension=32
        )
        output_original = encoder(hidden_states=hidden_states)
        # Modify the last position with a large perturbation
        modified_hidden_states = hidden_states.clone()
        modified_hidden_states[0, 3, :] *= 100.0
        output_modified = encoder(hidden_states=modified_hidden_states)
        # Bidirectional: modifying position 3 should change ALL positions in the output
        for position in range(4):
            assert not torch.allclose(
                output_original[0, position],
                output_modified[0, position],
                atol=1e-5,
            )

    def test_with_sinusoidal_positional_encoding(
        self,
        encoder_factory: Callable[..., TransformerEncoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        encoder = encoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            number_of_heads=4,
            positional_encoding_type=PositionalEncodingType.SINUSOIDAL.value,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=6, embedding_dimension=32
        )
        output = encoder(hidden_states=hidden_states)
        assert output.shape == (2, 6, 32)

    def test_with_rope_positional_encoding(
        self,
        encoder_factory: Callable[..., TransformerEncoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        encoder = encoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            number_of_heads=4,
            positional_encoding_type=PositionalEncodingType.ROPE.value,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=6, embedding_dimension=32
        )
        output = encoder(hidden_states=hidden_states)
        assert output.shape == (2, 6, 32)


class TestTransformerEncoderExpandPaddingMask:
    def test_expands_to_four_dimensions(
        self,
        padding_mask_factory: Callable[..., torch.Tensor],
    ):
        mask = padding_mask_factory(
            batch_size=2,
            sequence_length=4,
            padded_positions=[[2, 3], [3]],
        )
        expanded = TransformerEncoder._expand_padding_mask(
            padding_mask=mask, query_length=4
        )
        assert expanded.shape == (2, 1, 4, 4)

    def test_padded_positions_broadcast_across_queries(
        self,
        padding_mask_factory: Callable[..., torch.Tensor],
    ):
        mask = padding_mask_factory(
            batch_size=1,
            sequence_length=4,
            padded_positions=[[3]],
        )
        expanded = TransformerEncoder._expand_padding_mask(
            padding_mask=mask, query_length=4
        )
        # Position 3 should be masked for all query positions
        assert expanded[0, 0, 0, 3].item() is True
        assert expanded[0, 0, 1, 3].item() is True
        assert expanded[0, 0, 2, 3].item() is True
        # Position 0 should not be masked
        assert expanded[0, 0, 0, 0].item() is False
