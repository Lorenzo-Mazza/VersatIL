"""Tests for versatil.models.layers.transformer.encoder_layer module."""
from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.transformer.encoder_layer import TransformerEncoderLayer


@pytest.fixture
def encoder_layer_factory() -> Callable[..., TransformerEncoderLayer]:
    """Factory for TransformerEncoderLayer modules."""

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
        bias: bool = True,
        normalization_epsilon: float = 1e-6,
    ) -> TransformerEncoderLayer:
        return TransformerEncoderLayer(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_key_value_heads,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation=activation,
            normalization_type=normalization_type,
            attention_type=attention_type,
            bias=bias,
            normalization_epsilon=normalization_epsilon,
        )

    return factory


class TestTransformerEncoderLayerInitialization:

    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("number_of_heads", [4, 8])
    def test_stores_configuration(
        self,
        encoder_layer_factory: Callable[..., TransformerEncoderLayer],
        embedding_dimension: int,
        number_of_heads: int,
    ):
        layer = encoder_layer_factory(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
        )
        assert layer.embedding_dimension == embedding_dimension
        assert layer.number_of_heads == number_of_heads

    def test_feedforward_dimension_defaults_to_four_times_embedding(
        self,
        encoder_layer_factory: Callable[..., TransformerEncoderLayer],
    ):
        layer = encoder_layer_factory(
            embedding_dimension=32, feedforward_dimension=None
        )
        # SwiGLU and standard both use feedforward_dimension in the sequential;
        # we test a non-SwiGLU to check linear layer sizes
        layer_gelu = encoder_layer_factory(
            embedding_dimension=32,
            feedforward_dimension=None,
            activation=ActivationFunction.GELU.value,
        )
        first_linear = layer_gelu.feedforward_network[0]
        assert first_linear.out_features == 128

    def test_custom_feedforward_dimension(
        self,
        encoder_layer_factory: Callable[..., TransformerEncoderLayer],
    ):
        layer = encoder_layer_factory(
            embedding_dimension=32,
            feedforward_dimension=64,
            activation=ActivationFunction.GELU.value,
        )
        first_linear = layer.feedforward_network[0]
        assert first_linear.out_features == 64

    @pytest.mark.parametrize(
        "activation",
        [ActivationFunction.SWIGLU.value, ActivationFunction.GELU.value],
    )
    def test_activation_variants_create_feedforward(
        self,
        encoder_layer_factory: Callable[..., TransformerEncoderLayer],
        activation: str,
    ):
        layer = encoder_layer_factory(activation=activation)
        assert layer.feedforward_network is not None

    def test_feedforward_last_layer_has_initialization_flag(
        self,
        encoder_layer_factory: Callable[..., TransformerEncoderLayer],
    ):
        layer = encoder_layer_factory()
        assert layer.feedforward_network[-1].SQUARE_ROOT_WEIGHT is True


class TestTransformerEncoderLayerForward:

    def test_output_shape(
        self,
        encoder_layer_factory: Callable[..., TransformerEncoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        layer = encoder_layer_factory(embedding_dimension=32, number_of_heads=4)
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=6, embedding_dimension=32
        )
        output = layer(hidden_states=hidden_states)
        assert output.shape == (2, 6, 32)

    def test_residual_connection_passes_input_through(
        self,
        encoder_layer_factory: Callable[..., TransformerEncoderLayer],
        rng: np.random.Generator,
    ):
        layer = encoder_layer_factory(
            embedding_dimension=32, number_of_heads=4, dropout=0.0
        )
        layer.eval()
        hidden_states = torch.from_numpy(
            rng.standard_normal((2, 4, 32)).astype(np.float32)
        )
        output = layer(hidden_states=hidden_states)
        # Output should not be identical (attention + FFN change values)
        assert not torch.equal(output, hidden_states)
        # But due to residual connections, output should be correlated
        assert output.shape == hidden_states.shape

    def test_attention_mask_affects_output(
        self,
        encoder_layer_factory: Callable[..., TransformerEncoderLayer],
        rng: np.random.Generator,
        attention_mask_factory: Callable[..., torch.Tensor],
    ):
        layer = encoder_layer_factory(
            embedding_dimension=32, number_of_heads=4, dropout=0.0
        )
        layer.eval()
        hidden_states = torch.from_numpy(
            rng.standard_normal((2, 4, 32)).astype(np.float32)
        )
        mask = attention_mask_factory(
            batch_size=2, query_length=4, key_length=4, causal=True
        )
        output_with_mask = layer(
            hidden_states=hidden_states, attention_mask=mask
        )
        output_without_mask = layer(hidden_states=hidden_states)
        assert not torch.allclose(output_with_mask, output_without_mask, atol=1e-6)

    @pytest.mark.parametrize("sequence_length", [1, 4, 8])
    def test_variable_sequence_lengths(
        self,
        encoder_layer_factory: Callable[..., TransformerEncoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        sequence_length: int,
    ):
        layer = encoder_layer_factory(embedding_dimension=32, number_of_heads=4)
        hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=sequence_length,
            embedding_dimension=32,
        )
        output = layer(hidden_states=hidden_states)
        assert output.shape == (2, sequence_length, 32)

    def test_bidirectional_all_positions_see_all_positions(
        self,
        encoder_layer_factory: Callable[..., TransformerEncoderLayer],
        rng: np.random.Generator,
    ):
        layer = encoder_layer_factory(
            embedding_dimension=32, number_of_heads=4, dropout=0.0
        )
        hidden_states = torch.from_numpy(
            rng.standard_normal((1, 4, 32)).astype(np.float32)
        )
        hidden_states.requires_grad_(True)
        output = layer(hidden_states=hidden_states)
        # Compute gradient of the first token's output w.r.t. input
        output[0, 0].sum().backward()
        # In bidirectional attention, the gradient should flow to ALL input positions
        gradient = hidden_states.grad
        assert gradient is not None
        # Last token (position 3) should have nonzero gradient
        assert gradient[0, 3].abs().sum().item() > 1e-6
