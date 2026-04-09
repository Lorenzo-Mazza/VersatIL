"""Tests for versatil.models.layers.transformer.conditional_bidirectional_decoder module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from tests.models.layers.conftest import reinit_modulation_layers
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType, PositionalEncodingType
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.transformer.conditional_bidirectional_decoder import (
    ConditionalBidirectionalDecoder,
)


@pytest.fixture
def conditional_bidirectional_decoder_factory() -> Callable[
    ..., ConditionalBidirectionalDecoder
]:
    """Factory for ConditionalBidirectionalDecoder modules."""

    def factory(
        number_of_layers: int = 2,
        embedding_dimension: int = 32,
        conditioning_dimension: int = 16,
        number_of_heads: int = 4,
        number_of_key_value_heads: int | None = None,
        feedforward_dimension: int | None = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation: str = ActivationFunction.GELU.value,
        normalization_type: str = NormalizationType.LAYER_NORM.value,
        use_gating: bool = False,
        attention_type: str = AttentionType.MULTI_HEAD.value,
        positional_encoding_type: str | None = None,
        maximum_sequence_length: int = 128,
        bias: bool = True,
        normalization_epsilon: float = 1e-6,
        initializer_range: float = 0.02,
        use_cross_attention: bool = True,
        cross_attention_conditioning_dimension: int | None = None,
        cross_attention_normalization_type: str | None = None,
        use_final_normalization: bool = True,
        condition_final_normalization: bool = True,
    ) -> ConditionalBidirectionalDecoder:
        return ConditionalBidirectionalDecoder(
            number_of_layers=number_of_layers,
            embedding_dimension=embedding_dimension,
            conditioning_dimension=conditioning_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_key_value_heads,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation=activation,
            normalization_type=normalization_type,
            use_gating=use_gating,
            attention_type=attention_type,
            positional_encoding_type=positional_encoding_type,
            maximum_sequence_length=maximum_sequence_length,
            bias=bias,
            normalization_epsilon=normalization_epsilon,
            initializer_range=initializer_range,
            use_cross_attention=use_cross_attention,
            cross_attention_conditioning_dimension=cross_attention_conditioning_dimension,
            cross_attention_normalization_type=cross_attention_normalization_type,
            use_final_normalization=use_final_normalization,
            condition_final_normalization=condition_final_normalization,
        )

    return factory


class TestConditionalBidirectionalDecoderInitialization:
    @pytest.mark.parametrize("number_of_layers", [1, 3])
    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("conditioning_dimension", [16, 32])
    @pytest.mark.parametrize("use_cross_attention", [True, False])
    @pytest.mark.parametrize("use_gating", [True, False])
    def test_stores_configuration(
        self,
        conditional_bidirectional_decoder_factory: Callable[
            ..., ConditionalBidirectionalDecoder
        ],
        number_of_layers: int,
        embedding_dimension: int,
        conditioning_dimension: int,
        use_cross_attention: bool,
        use_gating: bool,
    ):
        decoder = conditional_bidirectional_decoder_factory(
            number_of_layers=number_of_layers,
            embedding_dimension=embedding_dimension,
            conditioning_dimension=conditioning_dimension,
            use_cross_attention=use_cross_attention,
            use_gating=use_gating,
        )
        assert decoder.number_of_layers == number_of_layers
        assert decoder.embedding_dimension == embedding_dimension
        assert decoder.condition_dimension == conditioning_dimension
        assert decoder.use_cross_attention == use_cross_attention
        expected_residual_blocks = 3 if use_cross_attention else 2
        assert decoder.number_of_residual_blocks == expected_residual_blocks

    def test_creates_correct_number_of_conditional_layers(
        self,
        conditional_bidirectional_decoder_factory: Callable[
            ..., ConditionalBidirectionalDecoder
        ],
    ):
        decoder = conditional_bidirectional_decoder_factory(number_of_layers=3)
        assert len(decoder.layers) == 3

    @pytest.mark.parametrize(
        "cross_attention_conditioning_dimension, expected",
        [(16, 16), (None, None)],
        ids=["explicit", "default_none"],
    )
    def test_cross_attention_conditioning_forwarded_to_layer(
        self,
        cross_attention_conditioning_dimension: int | None,
        expected: int | None,
    ):
        with patch(
            "versatil.models.layers.transformer.conditional_bidirectional_decoder.TransformerDecoderLayer",
            return_value=MagicMock(spec=torch.nn.Module),
        ) as mock_layer:
            ConditionalBidirectionalDecoder(
                number_of_layers=1,
                embedding_dimension=32,
                number_of_heads=4,
                conditioning_dimension=16,
                attention_type=AttentionType.MULTI_HEAD.value,
                cross_attention_conditioning_dimension=cross_attention_conditioning_dimension,
            )
            call_kwargs = mock_layer.call_args.kwargs
            assert call_kwargs["cross_attention_conditioning_dimension"] == expected
            assert call_kwargs["conditioning_dimension"] == 16

    def test_layers_have_cross_attention_enabled(
        self,
        conditional_bidirectional_decoder_factory: Callable[
            ..., ConditionalBidirectionalDecoder
        ],
    ):
        decoder = conditional_bidirectional_decoder_factory(number_of_layers=2)
        for layer in decoder.layers:
            assert layer.use_cross_attention is True

    def test_no_positional_encoding_by_default(
        self,
        conditional_bidirectional_decoder_factory: Callable[
            ..., ConditionalBidirectionalDecoder
        ],
    ):
        decoder = conditional_bidirectional_decoder_factory(
            positional_encoding_type=None
        )
        assert decoder.positional_encoding is None

    def test_positional_encoding_created_when_specified(
        self,
        conditional_bidirectional_decoder_factory: Callable[
            ..., ConditionalBidirectionalDecoder
        ],
    ):
        decoder = conditional_bidirectional_decoder_factory(
            positional_encoding_type=PositionalEncodingType.SINUSOIDAL.value,
        )
        assert decoder.positional_encoding is not None

    def test_gqa_requires_kv_heads(
        self,
        conditional_bidirectional_decoder_factory: Callable[
            ..., ConditionalBidirectionalDecoder
        ],
    ):
        with pytest.raises(
            ValueError,
            match=re.escape("number_of_key_value_heads required for GQA"),
        ):
            conditional_bidirectional_decoder_factory(
                attention_type=AttentionType.GROUPED_QUERY.value,
                number_of_key_value_heads=None,
            )

    def test_mha_sets_kv_heads_to_query_heads(
        self,
        conditional_bidirectional_decoder_factory: Callable[
            ..., ConditionalBidirectionalDecoder
        ],
    ):
        decoder = conditional_bidirectional_decoder_factory(
            number_of_heads=8,
            attention_type=AttentionType.MULTI_HEAD.value,
        )
        assert decoder.number_of_key_value_heads == 8

    @pytest.mark.parametrize(
        "normalization_type",
        [
            NormalizationType.RMS_NORM.value,
            NormalizationType.LAYER_NORM.value,
        ],
    )
    def test_normalization_type_accepted(
        self,
        conditional_bidirectional_decoder_factory: Callable[
            ..., ConditionalBidirectionalDecoder
        ],
        normalization_type: str,
    ):
        conditional_bidirectional_decoder_factory(
            number_of_layers=1,
            embedding_dimension=32,
            conditioning_dimension=16,
            number_of_heads=4,
            normalization_type=normalization_type,
        )


class TestConditionalBidirectionalDecoderForward:
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
                        "encoded_features required when use_cross_attention=True."
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
        conditional_bidirectional_decoder_factory: Callable[
            ..., ConditionalBidirectionalDecoder
        ],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
        use_cross_attention: bool,
        provide_features: bool,
        expectation: object,
    ):
        decoder = conditional_bidirectional_decoder_factory(
            number_of_layers=1,
            embedding_dimension=32,
            conditioning_dimension=16,
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
        condition = condition_factory(batch_size=2, condition_dim=16)
        with expectation:
            decoder(
                hidden_states=hidden_states,
                condition=condition,
                encoded_features=memory,
            )

    @pytest.mark.parametrize("use_cross_attention", [True, False])
    def test_output_shape(
        self,
        conditional_bidirectional_decoder_factory: Callable[
            ..., ConditionalBidirectionalDecoder
        ],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
        use_cross_attention: bool,
    ):
        decoder = conditional_bidirectional_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            conditioning_dimension=16,
            number_of_heads=4,
            use_cross_attention=use_cross_attention,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=5, embedding_dimension=32
        )
        memory = sequence_tensor_factory(
            batch_size=2, sequence_length=8, embedding_dimension=32
        )
        condition = condition_factory(batch_size=2, condition_dim=16)
        output = decoder(
            hidden_states=hidden_states,
            condition=condition,
            encoded_features=memory if use_cross_attention else None,
        )
        assert output.shape == (2, 5, 32)

    def test_identity_init_conditioning_has_no_effect(
        self,
        conditional_bidirectional_decoder_factory: Callable[
            ..., ConditionalBidirectionalDecoder
        ],
        rng: np.random.Generator,
    ):
        decoder = conditional_bidirectional_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            conditioning_dimension=16,
            number_of_heads=4,
        )
        decoder.eval()
        hidden_states = torch.from_numpy(
            rng.standard_normal((2, 4, 32)).astype(np.float32)
        )
        memory = torch.from_numpy(rng.standard_normal((2, 6, 32)).astype(np.float32))
        condition_a = torch.from_numpy(rng.standard_normal((2, 16)).astype(np.float32))
        condition_b = torch.from_numpy(rng.standard_normal((2, 16)).astype(np.float32))
        output_a = decoder(
            hidden_states=hidden_states,
            condition=condition_a,
            encoded_features=memory,
        )
        output_b = decoder(
            hidden_states=hidden_states,
            condition=condition_b,
            encoded_features=memory,
        )
        assert torch.allclose(output_a, output_b, atol=1e-6)

    @pytest.mark.parametrize("use_cross_attention", [True, False])
    def test_xavier_init_different_conditions_produce_different_outputs(
        self,
        conditional_bidirectional_decoder_factory: Callable[
            ..., ConditionalBidirectionalDecoder
        ],
        rng: np.random.Generator,
        use_cross_attention: bool,
    ):
        decoder = conditional_bidirectional_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            conditioning_dimension=16,
            number_of_heads=4,
            use_cross_attention=use_cross_attention,
        )
        reinit_modulation_layers(decoder)
        decoder.eval()
        hidden_states = torch.from_numpy(
            rng.standard_normal((2, 4, 32)).astype(np.float32)
        )
        memory = torch.from_numpy(rng.standard_normal((2, 6, 32)).astype(np.float32))
        condition_a = torch.from_numpy(rng.standard_normal((2, 16)).astype(np.float32))
        condition_b = torch.from_numpy(rng.standard_normal((2, 16)).astype(np.float32))
        output_a = decoder(
            hidden_states=hidden_states,
            condition=condition_a,
            encoded_features=memory if use_cross_attention else None,
        )
        output_b = decoder(
            hidden_states=hidden_states,
            condition=condition_b,
            encoded_features=memory if use_cross_attention else None,
        )
        assert not torch.allclose(output_a, output_b, atol=1e-5)

    def test_bidirectional_all_positions_see_all_positions(
        self,
        conditional_bidirectional_decoder_factory: Callable[
            ..., ConditionalBidirectionalDecoder
        ],
        rng: np.random.Generator,
    ):
        decoder = conditional_bidirectional_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            conditioning_dimension=16,
            number_of_heads=4,
            initializer_range=0.5,
        )
        reinit_modulation_layers(decoder)
        decoder.eval()
        hidden_states = torch.from_numpy(
            rng.standard_normal((1, 4, 32)).astype(np.float32)
        )
        memory = torch.from_numpy(rng.standard_normal((1, 6, 32)).astype(np.float32))
        condition = torch.from_numpy(rng.standard_normal((1, 16)).astype(np.float32))
        output_original = decoder(
            hidden_states=hidden_states,
            condition=condition,
            encoded_features=memory,
        )
        # Modify the last position with a large perturbation
        modified_hidden_states = hidden_states.clone()
        modified_hidden_states[0, 3, :] *= 100.0
        output_modified = decoder(
            hidden_states=modified_hidden_states,
            condition=condition,
            encoded_features=memory,
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
        conditional_bidirectional_decoder_factory: Callable[
            ..., ConditionalBidirectionalDecoder
        ],
        rng: np.random.Generator,
        padding_mask_factory: Callable[..., torch.Tensor],
    ):
        decoder = conditional_bidirectional_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            conditioning_dimension=16,
            number_of_heads=4,
        )
        decoder.eval()
        hidden_states = torch.from_numpy(
            rng.standard_normal((2, 4, 32)).astype(np.float32)
        )
        memory = torch.from_numpy(rng.standard_normal((2, 6, 32)).astype(np.float32))
        condition = torch.from_numpy(rng.standard_normal((2, 16)).astype(np.float32))
        query_mask = padding_mask_factory(
            batch_size=2, sequence_length=4, padded_positions=[[2, 3], []]
        )
        output_masked = decoder(
            hidden_states=hidden_states,
            condition=condition,
            encoded_features=memory,
            query_padding_mask=query_mask,
        )
        output_unmasked = decoder(
            hidden_states=hidden_states,
            condition=condition,
            encoded_features=memory,
        )
        assert not torch.allclose(output_masked[0], output_unmasked[0], atol=1e-5)

    def test_memory_padding_mask_affects_output(
        self,
        conditional_bidirectional_decoder_factory: Callable[
            ..., ConditionalBidirectionalDecoder
        ],
        rng: np.random.Generator,
        padding_mask_factory: Callable[..., torch.Tensor],
    ):
        decoder = conditional_bidirectional_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            conditioning_dimension=16,
            number_of_heads=4,
        )
        decoder.eval()
        hidden_states = torch.from_numpy(
            rng.standard_normal((2, 4, 32)).astype(np.float32)
        )
        memory = torch.from_numpy(rng.standard_normal((2, 6, 32)).astype(np.float32))
        condition = torch.from_numpy(rng.standard_normal((2, 16)).astype(np.float32))
        memory_mask = padding_mask_factory(
            batch_size=2, sequence_length=6, padded_positions=[[4, 5], []]
        )
        output_masked = decoder(
            hidden_states=hidden_states,
            condition=condition,
            encoded_features=memory,
            memory_padding_mask=memory_mask,
        )
        output_unmasked = decoder(
            hidden_states=hidden_states,
            condition=condition,
            encoded_features=memory,
        )
        assert not torch.allclose(output_masked[0], output_unmasked[0], atol=1e-5)

    def test_different_memory_produces_different_output(
        self,
        conditional_bidirectional_decoder_factory: Callable[
            ..., ConditionalBidirectionalDecoder
        ],
        rng: np.random.Generator,
    ):
        decoder = conditional_bidirectional_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            conditioning_dimension=16,
            number_of_heads=4,
        )
        decoder.eval()
        hidden_states = torch.from_numpy(
            rng.standard_normal((2, 4, 32)).astype(np.float32)
        )
        condition = torch.from_numpy(rng.standard_normal((2, 16)).astype(np.float32))
        memory_a = torch.from_numpy(rng.standard_normal((2, 6, 32)).astype(np.float32))
        memory_b = torch.from_numpy(rng.standard_normal((2, 6, 32)).astype(np.float32))
        output_a = decoder(
            hidden_states=hidden_states,
            condition=condition,
            encoded_features=memory_a,
        )
        output_b = decoder(
            hidden_states=hidden_states,
            condition=condition,
            encoded_features=memory_b,
        )
        assert not torch.allclose(output_a, output_b, atol=1e-5)

    def test_with_sinusoidal_positional_encoding(
        self,
        conditional_bidirectional_decoder_factory: Callable[
            ..., ConditionalBidirectionalDecoder
        ],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        decoder = conditional_bidirectional_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            conditioning_dimension=16,
            number_of_heads=4,
            positional_encoding_type=PositionalEncodingType.SINUSOIDAL.value,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=5, embedding_dimension=32
        )
        memory = sequence_tensor_factory(
            batch_size=2, sequence_length=8, embedding_dimension=32
        )
        condition = condition_factory(batch_size=2, condition_dim=16)
        output = decoder(
            hidden_states=hidden_states,
            condition=condition,
            encoded_features=memory,
        )
        assert output.shape == (2, 5, 32)

    def test_with_rope_positional_encoding(
        self,
        conditional_bidirectional_decoder_factory: Callable[
            ..., ConditionalBidirectionalDecoder
        ],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        decoder = conditional_bidirectional_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            conditioning_dimension=16,
            number_of_heads=4,
            positional_encoding_type=PositionalEncodingType.ROPE.value,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=5, embedding_dimension=32
        )
        memory = sequence_tensor_factory(
            batch_size=2, sequence_length=8, embedding_dimension=32
        )
        condition = condition_factory(batch_size=2, condition_dim=16)
        output = decoder(
            hidden_states=hidden_states,
            condition=condition,
            encoded_features=memory,
        )
        assert output.shape == (2, 5, 32)


class TestConditionalBidirectionalDecoderSelfAttentionOnly:
    def test_output_shape_without_cross_attention(
        self,
        conditional_bidirectional_decoder_factory: Callable[
            ..., ConditionalBidirectionalDecoder
        ],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        decoder = conditional_bidirectional_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            conditioning_dimension=32,
            number_of_heads=4,
            use_cross_attention=False,
            use_gating=True,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=6, embedding_dimension=32
        )
        condition = condition_factory(batch_size=2, condition_dim=32)
        output = decoder(
            hidden_states=hidden_states,
            condition=condition,
        )
        assert output.shape == (2, 6, 32)
        assert torch.all(torch.isfinite(output))

    def test_conditioning_affects_output_without_cross_attention(
        self,
        conditional_bidirectional_decoder_factory: Callable[
            ..., ConditionalBidirectionalDecoder
        ],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        decoder = conditional_bidirectional_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            conditioning_dimension=32,
            number_of_heads=4,
            use_cross_attention=False,
            use_gating=False,
        )
        reinit_modulation_layers(decoder)
        decoder.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        condition_a = condition_factory(batch_size=2, condition_dim=32)
        condition_b = condition_factory(batch_size=2, condition_dim=32)
        output_a = decoder(hidden_states=hidden_states, condition=condition_a)
        output_b = decoder(hidden_states=hidden_states, condition=condition_b)
        assert not torch.allclose(output_a, output_b, atol=1e-5)

    def test_padding_mask_affects_output_without_cross_attention(
        self,
        conditional_bidirectional_decoder_factory: Callable[
            ..., ConditionalBidirectionalDecoder
        ],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
        padding_mask_factory: Callable[..., torch.Tensor],
    ):
        decoder = conditional_bidirectional_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            conditioning_dimension=32,
            number_of_heads=4,
            use_cross_attention=False,
        )
        decoder.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        condition = condition_factory(batch_size=2, condition_dim=32)
        mask = padding_mask_factory(
            batch_size=2, sequence_length=4, padded_positions=[[2, 3], []]
        )
        output_masked = decoder(
            hidden_states=hidden_states,
            condition=condition,
            query_padding_mask=mask,
        )
        output_unmasked = decoder(
            hidden_states=hidden_states,
            condition=condition,
        )
        assert not torch.allclose(output_masked[0], output_unmasked[0], atol=1e-5)


class TestConditionalBidirectionalDecoderFinalNormalization:
    def test_no_final_normalization(
        self,
        conditional_bidirectional_decoder_factory: Callable[
            ..., ConditionalBidirectionalDecoder
        ],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        decoder_with = conditional_bidirectional_decoder_factory(
            number_of_layers=1,
            embedding_dimension=32,
            conditioning_dimension=32,
            number_of_heads=4,
            use_cross_attention=False,
            use_final_normalization=True,
        )
        decoder_without = conditional_bidirectional_decoder_factory(
            number_of_layers=1,
            embedding_dimension=32,
            conditioning_dimension=32,
            number_of_heads=4,
            use_cross_attention=False,
            use_final_normalization=False,
        )
        assert decoder_with.final_normalization is not None
        assert decoder_without.final_normalization is None
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        condition = condition_factory(batch_size=2, condition_dim=32)
        output_with = decoder_with(hidden_states=hidden_states, condition=condition)
        output_without = decoder_without(
            hidden_states=hidden_states, condition=condition
        )
        assert output_with.shape == output_without.shape
        assert not torch.allclose(output_with, output_without)

    def test_unconditioned_final_normalization_ignores_condition(
        self,
        conditional_bidirectional_decoder_factory: Callable[
            ..., ConditionalBidirectionalDecoder
        ],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        decoder = conditional_bidirectional_decoder_factory(
            number_of_layers=1,
            embedding_dimension=32,
            conditioning_dimension=32,
            number_of_heads=4,
            use_cross_attention=False,
            condition_final_normalization=False,
        )
        decoder.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        condition_a = condition_factory(batch_size=2, condition_dim=32)
        condition_b = condition_a * 10.0
        output_a = decoder(hidden_states=hidden_states, condition=condition_a)
        output_b = decoder(hidden_states=hidden_states, condition=condition_b)
        assert torch.allclose(output_a, output_b, atol=1e-6)

    def test_cross_attention_changes_output(
        self,
        conditional_bidirectional_decoder_factory: Callable[
            ..., ConditionalBidirectionalDecoder
        ],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        shared_kwargs = {
            "number_of_layers": 2,
            "embedding_dimension": 32,
            "conditioning_dimension": 16,
            "number_of_heads": 4,
        }
        decoder_with_ca = conditional_bidirectional_decoder_factory(
            use_cross_attention=True, **shared_kwargs
        )
        decoder_without_ca = conditional_bidirectional_decoder_factory(
            use_cross_attention=False, **shared_kwargs
        )
        decoder_with_ca.load_state_dict(decoder_without_ca.state_dict(), strict=False)
        decoder_with_ca.eval()
        decoder_without_ca.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        memory = sequence_tensor_factory(
            batch_size=2, sequence_length=6, embedding_dimension=32
        )
        condition = condition_factory(batch_size=2, condition_dim=16)
        output_with = decoder_with_ca(
            hidden_states=hidden_states,
            condition=condition,
            encoded_features=memory,
        )
        output_without = decoder_without_ca(
            hidden_states=hidden_states,
            condition=condition,
        )
        assert not torch.allclose(output_with, output_without)

    @pytest.mark.parametrize(
        "use_cross_attention, use_gating, cross_attention_normalization_type",
        [
            (False, True, None),
            (True, False, NormalizationType.LAYER_NORM.value),
        ],
        ids=["self_attention_gated", "cross_attention_custom_norm"],
    )
    def test_new_params_forwarded_to_layer(
        self,
        use_cross_attention: bool,
        use_gating: bool,
        cross_attention_normalization_type: str | None,
    ):
        with patch(
            "versatil.models.layers.transformer.conditional_bidirectional_decoder.TransformerDecoderLayer",
            return_value=MagicMock(spec=torch.nn.Module),
        ) as mock_layer:
            ConditionalBidirectionalDecoder(
                number_of_layers=1,
                embedding_dimension=32,
                number_of_heads=4,
                conditioning_dimension=16,
                attention_type=AttentionType.MULTI_HEAD.value,
                use_cross_attention=use_cross_attention,
                use_gating=use_gating,
                cross_attention_normalization_type=cross_attention_normalization_type,
            )
            call_kwargs = mock_layer.call_args.kwargs
            assert call_kwargs["use_cross_attention"] == use_cross_attention
            assert call_kwargs["use_gating"] == use_gating
            assert (
                call_kwargs["cross_attention_normalization_type"]
                == cross_attention_normalization_type
            )

    def test_unconditioned_final_norm_ignores_condition(
        self,
        conditional_bidirectional_decoder_factory: Callable[
            ..., ConditionalBidirectionalDecoder
        ],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        decoder = conditional_bidirectional_decoder_factory(
            number_of_layers=1,
            embedding_dimension=32,
            conditioning_dimension=32,
            number_of_heads=4,
            use_cross_attention=False,
            condition_final_normalization=False,
        )
        decoder.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        condition_a = condition_factory(batch_size=2, condition_dim=32)
        condition_b = condition_a * 10.0
        output_a = decoder(hidden_states=hidden_states, condition=condition_a)
        output_b = decoder(hidden_states=hidden_states, condition=condition_b)
        assert torch.allclose(output_a, output_b, atol=1e-6)

    @pytest.mark.parametrize("use_final_normalization", [True, False])
    def test_final_normalization_presence(
        self,
        conditional_bidirectional_decoder_factory: Callable[
            ..., ConditionalBidirectionalDecoder
        ],
        use_final_normalization: bool,
    ):
        decoder = conditional_bidirectional_decoder_factory(
            number_of_layers=1,
            embedding_dimension=32,
            conditioning_dimension=32,
            number_of_heads=4,
            use_final_normalization=use_final_normalization,
        )
        if use_final_normalization:
            assert decoder.final_normalization is not None
        else:
            assert decoder.final_normalization is None


class TestConditionalBidirectionalDecoderExpandPaddingMask:
    def test_expands_to_four_dimensions(
        self,
        padding_mask_factory: Callable[..., torch.Tensor],
    ):
        mask = padding_mask_factory(
            batch_size=2, sequence_length=6, padded_positions=[[4, 5], [5]]
        )
        expanded = ConditionalBidirectionalDecoder._expand_padding_mask(
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
        expanded = ConditionalBidirectionalDecoder._expand_padding_mask(
            padding_mask=mask, query_length=3
        )
        for query_index in range(3):
            assert expanded[0, 0, query_index, 5].item() is True
        assert expanded[0, 0, 0, 0].item() is False
