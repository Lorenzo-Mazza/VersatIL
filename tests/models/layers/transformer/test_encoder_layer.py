"""Tests for versatil.models.layers.transformer.encoder_layer module."""

from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from tests.models.layers.conftest import reinit_modulation_layers
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
        condition_dim: int | None = None,
        use_gating: bool = False,
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
            conditioning_dimension=condition_dim,
            use_gating=use_gating,
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
            embedding_dimension=32,
            feedforward_dimension=None,
            activation=ActivationFunction.GELU.value,
        )
        first_linear = layer.feedforward_block.feedforward[0]
        assert first_linear.out_features == 128  # 4 * 32

    def test_custom_feedforward_dimension(
        self,
        encoder_layer_factory: Callable[..., TransformerEncoderLayer],
    ):
        layer = encoder_layer_factory(
            embedding_dimension=32,
            feedforward_dimension=64,
            activation=ActivationFunction.GELU.value,
        )
        first_linear = layer.feedforward_block.feedforward[0]
        assert first_linear.out_features == 64

    def test_feedforward_last_layer_has_initialization_flag(
        self,
        encoder_layer_factory: Callable[..., TransformerEncoderLayer],
    ):
        layer = encoder_layer_factory()
        assert layer.feedforward_block.feedforward[-1].SQUARE_ROOT_WEIGHT is True


class TestTransformerEncoderLayerForward:
    def test_output_shape_and_values_are_valid(
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
        assert torch.all(torch.isfinite(output))
        assert not torch.allclose(output, hidden_states)

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
        output_with_mask = layer(hidden_states=hidden_states, attention_mask=mask)
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
        output[0, 0].sum().backward()
        gradient = hidden_states.grad
        assert gradient is not None
        # In bidirectional attention, last token should have nonzero gradient
        assert gradient[0, 3].abs().sum().item() > 1e-6

    def test_positional_encoding_affects_output(
        self,
        encoder_layer_factory: Callable[..., TransformerEncoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        mock_rope_factory: Callable[..., MagicMock],
    ):
        embedding_dimension = 32
        number_of_heads = 4
        head_dimension = embedding_dimension // number_of_heads
        layer = encoder_layer_factory(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            dropout=0.0,
        )
        layer.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=6, embedding_dimension=embedding_dimension
        )
        mock_rope = mock_rope_factory(head_dimension=head_dimension)
        output_with_rope = layer(
            hidden_states=hidden_states, positional_encoding=mock_rope
        )
        output_without_rope = layer(hidden_states=hidden_states)
        assert not torch.allclose(output_with_rope, output_without_rope)
        mock_rope.compute_rotation_components.assert_called_once()

    @pytest.mark.parametrize(
        "activation",
        [ActivationFunction.GELU.value, ActivationFunction.SWIGLU.value],
    )
    def test_gated_and_nongated_activations_produce_valid_output(
        self,
        encoder_layer_factory: Callable[..., TransformerEncoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        activation: str,
    ):
        layer = encoder_layer_factory(
            embedding_dimension=32,
            number_of_heads=4,
            activation=activation,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        output = layer(hidden_states=hidden_states)
        assert output.shape == hidden_states.shape
        assert torch.all(torch.isfinite(output))

    def test_grouped_query_attention(
        self,
        encoder_layer_factory: Callable[..., TransformerEncoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        layer = encoder_layer_factory(
            embedding_dimension=32,
            number_of_heads=4,
            number_of_key_value_heads=2,
            attention_type=AttentionType.GROUPED_QUERY.value,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        output = layer(hidden_states=hidden_states)
        assert output.shape == hidden_states.shape
        assert torch.all(torch.isfinite(output))


class TestTransformerEncoderLayerConditioning:
    def test_adaptive_norm_different_conditioning_produces_different_outputs(
        self,
        encoder_layer_factory: Callable[..., TransformerEncoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        layer = encoder_layer_factory(
            embedding_dimension=32,
            number_of_heads=4,
            normalization_type=NormalizationType.RMS_NORM.value,
            condition_dim=32,
            dropout=0.0,
        )
        reinit_modulation_layers(layer)
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        conditioning_a = condition_factory(batch_size=2, condition_dim=32)
        conditioning_b = condition_factory(batch_size=2, condition_dim=32)
        output_a = layer(hidden_states=hidden_states, conditioning=conditioning_a)
        output_b = layer(hidden_states=hidden_states, conditioning=conditioning_b)
        assert not torch.allclose(output_a, output_b)

    def test_adaln_zero_gate_makes_output_equal_input_at_init(
        self,
        encoder_layer_factory: Callable[..., TransformerEncoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        layer = encoder_layer_factory(
            embedding_dimension=32,
            number_of_heads=4,
            normalization_type=NormalizationType.RMS_NORM.value,
            condition_dim=32,
            use_gating=True,
            dropout=0.0,
        )
        layer.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        conditioning = condition_factory(batch_size=2, condition_dim=32)
        output = layer(hidden_states=hidden_states, conditioning=conditioning)
        # With gating at zero init, gate=0 → output = input
        assert torch.allclose(output, hidden_states, atol=1e-6)

    def test_unconditioned_layer_ignores_conditioning_argument(
        self,
        encoder_layer_factory: Callable[..., TransformerEncoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        layer = encoder_layer_factory(
            embedding_dimension=32,
            number_of_heads=4,
            normalization_type=NormalizationType.LAYER_NORM.value,
            dropout=0.0,
        )
        layer.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        conditioning = condition_factory(batch_size=2, condition_dim=16)
        output_with_cond = layer(hidden_states=hidden_states, conditioning=conditioning)
        output_without_cond = layer(hidden_states=hidden_states)
        assert torch.allclose(output_with_cond, output_without_cond)

    def test_conditioning_gradient_flows_through_modulation(
        self,
        encoder_layer_factory: Callable[..., TransformerEncoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        layer = encoder_layer_factory(
            embedding_dimension=32,
            number_of_heads=4,
            normalization_type=NormalizationType.RMS_NORM.value,
            condition_dim=32,
            dropout=0.0,
        )
        reinit_modulation_layers(layer)
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        conditioning = condition_factory(batch_size=2, condition_dim=32)
        conditioning.requires_grad_(True)
        output = layer(hidden_states=hidden_states, conditioning=conditioning)
        output.sum().backward()
        assert conditioning.grad is not None
        assert conditioning.grad.abs().sum().item() > 0.0
