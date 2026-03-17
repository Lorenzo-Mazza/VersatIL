"""Tests for versatil.models.layers.diffusion_transformer.dit_decoder module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise

import pytest
import torch
import torch.nn as nn

from tests.models.layers.diffusion_transformer.conftest import reinit_modulation_layers
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType
from versatil.models.layers.diffusion_transformer.dit_decoder import (
    DiffusionTransformerDecoder,
)
from versatil.models.layers.normalization.constants import NormalizationType


@pytest.fixture
def dit_decoder_factory() -> Callable[..., DiffusionTransformerDecoder]:
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
        use_final_normalization: bool = True,
        initializer_range: float = 0.02,
    ) -> DiffusionTransformerDecoder:
        return DiffusionTransformerDecoder(
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
            use_final_normalization=use_final_normalization,
            initializer_range=initializer_range,
        )

    return factory


class TestDiffusionTransformerDecoderInitialization:
    @pytest.mark.parametrize("number_of_layers", [1, 3])
    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    def test_stores_configuration(
        self,
        dit_decoder_factory: Callable[..., DiffusionTransformerDecoder],
        number_of_layers: int,
        embedding_dimension: int,
    ):
        decoder = dit_decoder_factory(
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
        dit_decoder_factory: Callable[..., DiffusionTransformerDecoder],
        attention_type: str,
        number_of_key_value_heads: int | None,
        expectation: object,
    ):
        with expectation:
            dit_decoder_factory(
                attention_type=attention_type,
                number_of_key_value_heads=number_of_key_value_heads,
            )

    def test_init_weights_preserves_modulation_layer_zero_init(
        self,
        dit_decoder_factory: Callable[..., DiffusionTransformerDecoder],
    ):
        # _init_weights must skip modulation layers (marked with _is_modulation_layer)
        # so they retain their zero initialization from ConditionalModulation.init_parameters
        decoder = dit_decoder_factory(
            number_of_layers=1,
            embedding_dimension=32,
            timestep_dimension=32,
            use_gating=True,
        )
        first_layer = decoder.layers[0]
        for (
            linear
        ) in first_layer.self_attention_normalization.modulation.projection.modules():
            if isinstance(linear, nn.Linear):
                assert torch.all(linear.weight == 0.0)


class TestDiffusionTransformerDecoderForward:
    @pytest.mark.parametrize(
        "batch_size, sequence_length, embedding_dimension",
        [
            (2, 4, 32),
            (1, 8, 64),
        ],
    )
    def test_output_shape(
        self,
        dit_decoder_factory: Callable[..., DiffusionTransformerDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
        batch_size: int,
        sequence_length: int,
        embedding_dimension: int,
    ):
        decoder = dit_decoder_factory(
            embedding_dimension=embedding_dimension,
            timestep_dimension=embedding_dimension,
            number_of_heads=4,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=embedding_dimension,
        )
        conditioning = condition_factory(
            batch_size=batch_size,
            condition_dim=embedding_dimension,
        )
        output = decoder(
            hidden_states=hidden_states,
            conditioning_embedding=conditioning,
        )
        assert output.shape == (batch_size, sequence_length, embedding_dimension)

    def test_different_conditioning_produces_different_outputs(
        self,
        dit_decoder_factory: Callable[..., DiffusionTransformerDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        decoder = dit_decoder_factory(
            embedding_dimension=embedding_dimension,
            timestep_dimension=embedding_dimension,
            use_gating=False,
        )
        reinit_modulation_layers(decoder)
        hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=embedding_dimension,
        )
        conditioning_a = condition_factory(
            batch_size=2,
            condition_dim=embedding_dimension,
        )
        conditioning_b = condition_factory(
            batch_size=2,
            condition_dim=embedding_dimension,
        )
        output_a = decoder(
            hidden_states=hidden_states,
            conditioning_embedding=conditioning_a,
        )
        output_b = decoder(
            hidden_states=hidden_states,
            conditioning_embedding=conditioning_b,
        )
        assert not torch.allclose(output_a, output_b)

    def test_padding_mask_affects_output(
        self,
        dit_decoder_factory: Callable[..., DiffusionTransformerDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
        padding_mask_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        sequence_length = 6
        decoder = dit_decoder_factory(
            embedding_dimension=embedding_dimension,
            timestep_dimension=embedding_dimension,
            use_gating=False,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=sequence_length,
            embedding_dimension=embedding_dimension,
        )
        conditioning = condition_factory(
            batch_size=2,
            condition_dim=embedding_dimension,
        )
        output_no_mask = decoder(
            hidden_states=hidden_states,
            conditioning_embedding=conditioning,
        )
        padding_mask = padding_mask_factory(
            batch_size=2,
            sequence_length=sequence_length,
            mask_last_n=2,
        )
        output_with_mask = decoder(
            hidden_states=hidden_states,
            conditioning_embedding=conditioning,
            padding_mask=padding_mask,
        )
        assert not torch.allclose(output_no_mask, output_with_mask)

    def test_final_normalization_changes_output(
        self,
        dit_decoder_factory: Callable[..., DiffusionTransformerDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        decoder_with_norm = dit_decoder_factory(
            embedding_dimension=embedding_dimension,
            timestep_dimension=embedding_dimension,
            use_final_normalization=True,
            use_gating=False,
        )
        decoder_without_norm = dit_decoder_factory(
            embedding_dimension=embedding_dimension,
            timestep_dimension=embedding_dimension,
            use_final_normalization=False,
            use_gating=False,
        )
        # Copy layer weights so the only difference is final normalization
        decoder_without_norm.layers.load_state_dict(
            decoder_with_norm.layers.state_dict()
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
        output_with_norm = decoder_with_norm(
            hidden_states=hidden_states,
            conditioning_embedding=conditioning,
        )
        output_without_norm = decoder_without_norm(
            hidden_states=hidden_states,
            conditioning_embedding=conditioning,
        )
        assert output_with_norm.shape == output_without_norm.shape
        assert not torch.allclose(output_with_norm, output_without_norm)


class TestExpandPaddingMask:
    @pytest.mark.parametrize(
        "batch_size, sequence_length",
        [
            (2, 4),
            (1, 8),
        ],
    )
    def test_expands_to_4d(
        self,
        padding_mask_factory: Callable[..., torch.Tensor],
        batch_size: int,
        sequence_length: int,
    ):
        mask = padding_mask_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            mask_last_n=1,
        )
        expanded = DiffusionTransformerDecoder._expand_padding_mask(
            mask, sequence_length
        )
        assert expanded.shape == (batch_size, 1, sequence_length, sequence_length)

    def test_masked_positions_propagate(
        self,
        padding_mask_factory: Callable[..., torch.Tensor],
    ):
        mask = padding_mask_factory(
            batch_size=2,
            sequence_length=4,
            mask_last_n=1,
        )
        expanded = DiffusionTransformerDecoder._expand_padding_mask(mask, 4)
        # Last position is masked in the key dimension for all query positions
        assert expanded[0, 0, 0, 3].item() is True
        assert expanded[0, 0, 0, 0].item() is False
