"""Tests for versatil.models.layers.diffusion_transformer.cross_attention_dit_decoder module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise

import pytest
import torch

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType
from versatil.models.layers.diffusion_transformer.cross_attention_dit_decoder import (
    CrossConditioningDecoder,
)
from versatil.models.layers.normalization.constants import NormalizationType


@pytest.fixture
def cross_decoder_factory() -> Callable[..., CrossConditioningDecoder]:
    def factory(
        number_of_layers: int = 2,
        embedding_dimension: int = 32,
        timestep_dimension: int = 32,
        number_of_heads: int = 4,
        number_of_key_value_heads: int | None = None,
        feedforward_dimension: int | None = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation: str = ActivationFunction.SILU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        attention_type: str = AttentionType.MULTI_HEAD.value,
        positional_encoding_type: str | None = None,
        maximum_sequence_length: int = 256,
        bias: bool = True,
        normalization_epsilon: float = 1e-6,
        use_gating: bool = True,
        initializer_range: float = 0.02,
    ) -> CrossConditioningDecoder:
        return CrossConditioningDecoder(
            number_of_layers=number_of_layers,
            embedding_dimension=embedding_dimension,
            timestep_dimension=timestep_dimension,
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
            use_gating=use_gating,
            initializer_range=initializer_range,
        )

    return factory


class TestCrossConditioningDecoderInitialization:
    @pytest.mark.parametrize("number_of_layers", [1, 3])
    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    def test_stores_configuration(
        self,
        cross_decoder_factory: Callable[..., CrossConditioningDecoder],
        number_of_layers: int,
        embedding_dimension: int,
    ):
        decoder = cross_decoder_factory(
            number_of_layers=number_of_layers,
            embedding_dimension=embedding_dimension,
            number_of_heads=4,
        )
        assert decoder.number_of_layers == number_of_layers
        assert decoder.embedding_dimension == embedding_dimension
        assert len(decoder.layers) == number_of_layers

    @pytest.mark.parametrize(
        "attention_type, number_of_key_value_heads, expectation",
        [
            (AttentionType.MULTI_HEAD.value, None, does_not_raise()),
            (AttentionType.GROUPED_QUERY.value, 2, does_not_raise()),
            (
                AttentionType.GROUPED_QUERY.value,
                None,
                pytest.raises(
                    ValueError,
                    match=re.escape("number_of_key_value_heads required for GQA"),
                ),
            ),
        ],
    )
    def test_grouped_query_attention_validation(
        self,
        cross_decoder_factory: Callable[..., CrossConditioningDecoder],
        attention_type: str,
        number_of_key_value_heads: int | None,
        expectation: object,
    ):
        with expectation:
            cross_decoder_factory(
                attention_type=attention_type,
                number_of_key_value_heads=number_of_key_value_heads,
            )


class TestCrossConditioningDecoderForward:
    @pytest.mark.parametrize(
        "batch_size, decoder_sequence_length, encoder_sequence_length, embedding_dimension",
        [
            (2, 4, 6, 32),
            (1, 8, 10, 64),
        ],
    )
    def test_output_shape(
        self,
        cross_decoder_factory: Callable[..., CrossConditioningDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
        batch_size: int,
        decoder_sequence_length: int,
        encoder_sequence_length: int,
        embedding_dimension: int,
    ):
        decoder = cross_decoder_factory(
            embedding_dimension=embedding_dimension,
            timestep_dimension=embedding_dimension,
            number_of_heads=4,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=decoder_sequence_length,
            embedding_dimension=embedding_dimension,
        )
        encoder_hidden = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=encoder_sequence_length,
            embedding_dimension=embedding_dimension,
        )
        conditioning = condition_factory(
            batch_size=batch_size,
            condition_dim=embedding_dimension,
        )
        output = decoder(
            hidden_states=hidden_states,
            conditioning_embedding=conditioning,
            encoder_hidden_states=encoder_hidden,
        )
        assert output.shape == (
            batch_size,
            decoder_sequence_length,
            embedding_dimension,
        )

    def test_different_encoder_context_produces_different_outputs(
        self,
        cross_decoder_factory: Callable[..., CrossConditioningDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        decoder = cross_decoder_factory(
            embedding_dimension=embedding_dimension,
            timestep_dimension=embedding_dimension,
            use_gating=False,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=embedding_dimension,
        )
        conditioning = condition_factory(
            batch_size=2,
            condition_dim=embedding_dimension,
        )
        encoder_a = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=embedding_dimension,
        )
        encoder_b = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=embedding_dimension,
        )
        output_a = decoder(
            hidden_states=hidden_states,
            conditioning_embedding=conditioning,
            encoder_hidden_states=encoder_a,
        )
        output_b = decoder(
            hidden_states=hidden_states,
            conditioning_embedding=conditioning,
            encoder_hidden_states=encoder_b,
        )
        assert not torch.allclose(output_a, output_b)

    def test_encoder_padding_mask_affects_output(
        self,
        cross_decoder_factory: Callable[..., CrossConditioningDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
        padding_mask_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        encoder_sequence_length = 6
        decoder = cross_decoder_factory(
            embedding_dimension=embedding_dimension,
            timestep_dimension=embedding_dimension,
            use_gating=False,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=embedding_dimension,
        )
        encoder_hidden = sequence_tensor_factory(
            batch_size=2,
            sequence_length=encoder_sequence_length,
            embedding_dimension=embedding_dimension,
        )
        conditioning = condition_factory(
            batch_size=2,
            condition_dim=embedding_dimension,
        )
        output_no_mask = decoder(
            hidden_states=hidden_states,
            conditioning_embedding=conditioning,
            encoder_hidden_states=encoder_hidden,
        )
        encoder_padding_mask = padding_mask_factory(
            batch_size=2,
            sequence_length=encoder_sequence_length,
            mask_last_n=3,
        )
        output_with_mask = decoder(
            hidden_states=hidden_states,
            conditioning_embedding=conditioning,
            encoder_hidden_states=encoder_hidden,
            encoder_padding_mask=encoder_padding_mask,
        )
        assert not torch.allclose(output_no_mask, output_with_mask)


class TestCrossConditioningDecoderExpandPaddingMask:
    @pytest.mark.parametrize(
        "batch_size, query_length, key_length",
        [
            (2, 4, 6),
            (1, 8, 10),
        ],
    )
    def test_expands_to_4d_with_correct_shape(
        self,
        padding_mask_factory: Callable[..., torch.Tensor],
        batch_size: int,
        query_length: int,
        key_length: int,
    ):
        mask = padding_mask_factory(
            batch_size=batch_size,
            sequence_length=key_length,
            mask_last_n=1,
        )
        expanded = CrossConditioningDecoder._expand_padding_mask(
            mask, query_length, key_length
        )
        assert expanded.shape == (batch_size, 1, query_length, key_length)

    def test_masked_positions_propagate_across_queries(
        self,
        padding_mask_factory: Callable[..., torch.Tensor],
    ):
        mask = padding_mask_factory(
            batch_size=2,
            sequence_length=6,
            mask_last_n=2,
        )
        expanded = CrossConditioningDecoder._expand_padding_mask(mask, 4, 6)
        # Last 2 key positions should be masked for all query positions
        assert expanded[0, 0, 0, 4].item() is True
        assert expanded[0, 0, 0, 5].item() is True
        assert expanded[0, 0, 0, 0].item() is False
        # Same masking for different query positions
        assert expanded[0, 0, 2, 4].item() is True
        assert expanded[0, 0, 2, 0].item() is False
