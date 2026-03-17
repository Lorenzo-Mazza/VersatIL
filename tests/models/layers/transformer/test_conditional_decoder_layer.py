"""Tests for versatil.models.layers.transformer.conditional_decoder_layer module."""

import re
from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.transformer.conditional_decoder_layer import (
    ConditionalTransformerDecoderLayer,
)


@pytest.fixture
def conditional_decoder_layer_factory() -> Callable[
    ..., ConditionalTransformerDecoderLayer
]:
    """Factory for ConditionalTransformerDecoderLayer modules."""

    def factory(
        embedding_dimension: int = 32,
        condition_dimension: int = 16,
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
        modulation_init_strategy: str = "identity",
    ) -> ConditionalTransformerDecoderLayer:
        return ConditionalTransformerDecoderLayer(
            embedding_dimension=embedding_dimension,
            condition_dimension=condition_dimension,
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
            modulation_init_strategy=modulation_init_strategy,
        )

    return factory


class TestConditionalDecoderLayerInitialization:
    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("condition_dimension", [16, 32])
    @pytest.mark.parametrize("use_cross_attention", [True, False])
    def test_stores_configuration(
        self,
        conditional_decoder_layer_factory: Callable[
            ..., ConditionalTransformerDecoderLayer
        ],
        embedding_dimension: int,
        condition_dimension: int,
        use_cross_attention: bool,
    ):
        layer = conditional_decoder_layer_factory(
            embedding_dimension=embedding_dimension,
            condition_dimension=condition_dimension,
            use_cross_attention=use_cross_attention,
        )
        assert layer.embedding_dimension == embedding_dimension
        assert layer.condition_dimension == condition_dimension
        assert layer.use_cross_attention == use_cross_attention

    def test_modulation_layers_created_for_self_attention_and_ffn(
        self,
        conditional_decoder_layer_factory: Callable[
            ..., ConditionalTransformerDecoderLayer
        ],
    ):
        layer = conditional_decoder_layer_factory(use_cross_attention=False)
        assert layer.self_attention_modulation is not None
        assert layer.feedforward_modulation is not None

    def test_cross_attention_modulation_created_when_enabled(
        self,
        conditional_decoder_layer_factory: Callable[
            ..., ConditionalTransformerDecoderLayer
        ],
    ):
        layer = conditional_decoder_layer_factory(use_cross_attention=True)
        assert layer.cross_attention_modulation is not None

    def test_cross_attention_modulation_none_when_disabled(
        self,
        conditional_decoder_layer_factory: Callable[
            ..., ConditionalTransformerDecoderLayer
        ],
    ):
        layer = conditional_decoder_layer_factory(use_cross_attention=False)
        assert layer.cross_attention_modulation is None

    @pytest.mark.parametrize(
        "activation",
        [ActivationFunction.SWIGLU.value, ActivationFunction.GELU.value],
    )
    def test_activation_variants(
        self,
        conditional_decoder_layer_factory: Callable[
            ..., ConditionalTransformerDecoderLayer
        ],
        activation: str,
    ):
        layer = conditional_decoder_layer_factory(activation=activation)
        assert layer.feedforward_network is not None

    def test_feedforward_last_layer_has_initialization_flag(
        self,
        conditional_decoder_layer_factory: Callable[
            ..., ConditionalTransformerDecoderLayer
        ],
    ):
        layer = conditional_decoder_layer_factory()
        assert layer.feedforward_network[-1].SQUARE_ROOT_WEIGHT is True


class TestConditionalDecoderLayerForward:
    def test_output_shape_with_cross_attention(
        self,
        conditional_decoder_layer_factory: Callable[
            ..., ConditionalTransformerDecoderLayer
        ],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        layer = conditional_decoder_layer_factory(
            embedding_dimension=32,
            condition_dimension=16,
            number_of_heads=4,
            use_cross_attention=True,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=5, embedding_dimension=32
        )
        memory = sequence_tensor_factory(
            batch_size=2, sequence_length=8, embedding_dimension=32
        )
        condition = condition_factory(batch_size=2, condition_dim=16)
        output = layer(
            hidden_states=hidden_states,
            condition=condition,
            encoded_features=memory,
        )
        assert output.shape == (2, 5, 32)

    def test_output_shape_without_cross_attention(
        self,
        conditional_decoder_layer_factory: Callable[
            ..., ConditionalTransformerDecoderLayer
        ],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        layer = conditional_decoder_layer_factory(
            embedding_dimension=32,
            condition_dimension=16,
            number_of_heads=4,
            use_cross_attention=False,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=5, embedding_dimension=32
        )
        condition = condition_factory(batch_size=2, condition_dim=16)
        output = layer(
            hidden_states=hidden_states,
            condition=condition,
        )
        assert output.shape == (2, 5, 32)

    def test_identity_init_conditioning_has_no_effect_at_initialization(
        self,
        conditional_decoder_layer_factory: Callable[
            ..., ConditionalTransformerDecoderLayer
        ],
        rng: np.random.Generator,
    ):
        layer = conditional_decoder_layer_factory(
            embedding_dimension=32,
            condition_dimension=16,
            number_of_heads=4,
            use_cross_attention=False,
            modulation_init_strategy="identity",
            dropout=0.0,
        )
        layer.eval()
        hidden_states = torch.from_numpy(
            rng.standard_normal((2, 4, 32)).astype(np.float32)
        )
        condition_a = torch.from_numpy(rng.standard_normal((2, 16)).astype(np.float32))
        condition_b = torch.from_numpy(rng.standard_normal((2, 16)).astype(np.float32))
        output_a = layer(hidden_states=hidden_states, condition=condition_a)
        output_b = layer(hidden_states=hidden_states, condition=condition_b)
        # With identity init (zero weights), modulation has no effect so
        # different conditions should produce the same output
        assert torch.allclose(output_a, output_b, atol=1e-6)

    def test_different_conditions_produce_different_outputs_after_training(
        self,
        conditional_decoder_layer_factory: Callable[
            ..., ConditionalTransformerDecoderLayer
        ],
        rng: np.random.Generator,
    ):
        layer = conditional_decoder_layer_factory(
            embedding_dimension=32,
            condition_dimension=16,
            number_of_heads=4,
            use_cross_attention=False,
            modulation_init_strategy="xavier",
            dropout=0.0,
        )
        layer.eval()
        hidden_states = torch.from_numpy(
            rng.standard_normal((2, 4, 32)).astype(np.float32)
        )
        condition_a = torch.from_numpy(rng.standard_normal((2, 16)).astype(np.float32))
        condition_b = torch.from_numpy(rng.standard_normal((2, 16)).astype(np.float32))
        output_a = layer(hidden_states=hidden_states, condition=condition_a)
        output_b = layer(hidden_states=hidden_states, condition=condition_b)
        assert not torch.allclose(output_a, output_b, atol=1e-5)

    def test_cross_attention_without_features_raises(
        self,
        conditional_decoder_layer_factory: Callable[
            ..., ConditionalTransformerDecoderLayer
        ],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        layer = conditional_decoder_layer_factory(
            embedding_dimension=32,
            condition_dimension=16,
            number_of_heads=4,
            use_cross_attention=True,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=5, embedding_dimension=32
        )
        condition = condition_factory(batch_size=2, condition_dim=16)
        with pytest.raises(
            ValueError,
            match=re.escape("encoded_features required when use_cross_attention=True"),
        ):
            layer(
                hidden_states=hidden_states,
                condition=condition,
                encoded_features=None,
            )

    def test_different_encoded_features_produce_different_output(
        self,
        conditional_decoder_layer_factory: Callable[
            ..., ConditionalTransformerDecoderLayer
        ],
        rng: np.random.Generator,
    ):
        layer = conditional_decoder_layer_factory(
            embedding_dimension=32,
            condition_dimension=16,
            number_of_heads=4,
            use_cross_attention=True,
            dropout=0.0,
        )
        layer.eval()
        hidden_states = torch.from_numpy(
            rng.standard_normal((2, 4, 32)).astype(np.float32)
        )
        condition = torch.from_numpy(rng.standard_normal((2, 16)).astype(np.float32))
        memory_a = torch.from_numpy(rng.standard_normal((2, 6, 32)).astype(np.float32))
        memory_b = torch.from_numpy(rng.standard_normal((2, 6, 32)).astype(np.float32))
        output_a = layer(
            hidden_states=hidden_states,
            condition=condition,
            encoded_features=memory_a,
        )
        output_b = layer(
            hidden_states=hidden_states,
            condition=condition,
            encoded_features=memory_b,
        )
        assert not torch.allclose(output_a, output_b, atol=1e-6)

    def test_self_attention_mask_affects_output(
        self,
        conditional_decoder_layer_factory: Callable[
            ..., ConditionalTransformerDecoderLayer
        ],
        rng: np.random.Generator,
        attention_mask_factory: Callable[..., torch.Tensor],
    ):
        layer = conditional_decoder_layer_factory(
            embedding_dimension=32,
            condition_dimension=16,
            number_of_heads=4,
            use_cross_attention=False,
            modulation_init_strategy="xavier",
            dropout=0.0,
        )
        layer.eval()
        hidden_states = torch.from_numpy(
            rng.standard_normal((2, 4, 32)).astype(np.float32)
        )
        condition = torch.from_numpy(rng.standard_normal((2, 16)).astype(np.float32))
        causal_mask = attention_mask_factory(
            batch_size=2, query_length=4, key_length=4, causal=True
        )
        output_causal = layer(
            hidden_states=hidden_states,
            condition=condition,
            self_attention_mask=causal_mask,
        )
        output_bidir = layer(
            hidden_states=hidden_states,
            condition=condition,
        )
        assert not torch.allclose(output_causal, output_bidir, atol=1e-6)
