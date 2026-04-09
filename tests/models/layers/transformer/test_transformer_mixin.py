"""Tests for versatil.models.layers.transformer.transformer_mixin module."""

import math
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from versatil.models.layers.constants import AttentionType, PositionalEncodingType
from versatil.models.layers.normalization.rms_norm import RMSNorm
from versatil.models.layers.transformer.encoder import TransformerEncoder
from versatil.models.layers.transformer.transformer_mixin import (
    RESIDUAL_STREAM_FLAG,
    TransformerMixin,
)


class ConcreteMixin(TransformerMixin, nn.Module):
    """Minimal concrete class for testing the mixin."""

    def __init__(
        self,
        number_of_layers: int = 4,
        initializer_range: float = 0.02,
        number_of_residual_blocks: int = 3,
    ):
        super().__init__()
        self.number_of_layers = number_of_layers
        self.initializer_range = initializer_range
        self.number_of_residual_blocks = number_of_residual_blocks
        self.positional_encoding = None


@pytest.fixture
def mixin_factory() -> Callable[..., ConcreteMixin]:
    def factory(
        number_of_layers: int = 4,
        initializer_range: float = 0.02,
        number_of_residual_blocks: int = 3,
    ) -> ConcreteMixin:
        return ConcreteMixin(
            number_of_layers=number_of_layers,
            initializer_range=initializer_range,
            number_of_residual_blocks=number_of_residual_blocks,
        )

    return factory


class TestTotalResidualStreams:
    @pytest.mark.parametrize(
        "number_of_layers, number_of_residual_blocks, expected",
        [(4, 3, 12), (2, 2, 4), (6, 1, 6)],
    )
    def test_product_of_layers_and_blocks(
        self,
        mixin_factory: Callable[..., ConcreteMixin],
        number_of_layers: int,
        number_of_residual_blocks: int,
        expected: int,
    ):
        mixin = mixin_factory(
            number_of_layers=number_of_layers,
            number_of_residual_blocks=number_of_residual_blocks,
        )
        assert mixin._total_residual_streams == expected


class TestInitWeights:
    @pytest.mark.parametrize(
        "number_of_layers, number_of_residual_blocks",
        [(4, 3), (6, 2), (1, 3)],
    )
    def test_residual_stream_std_is_scaled(
        self,
        mixin_factory: Callable[..., ConcreteMixin],
        number_of_layers: int,
        number_of_residual_blocks: int,
    ):
        mixin = mixin_factory(
            number_of_layers=number_of_layers,
            number_of_residual_blocks=number_of_residual_blocks,
            initializer_range=0.02,
        )
        linear = nn.Linear(32, 32)
        setattr(linear, RESIDUAL_STREAM_FLAG, True)
        torch.manual_seed(0)
        mixin._init_weights(linear)
        expected_std = 0.02 / math.sqrt(number_of_residual_blocks * number_of_layers)
        actual_std = linear.weight.data.std().item()
        assert abs(actual_std - expected_std) < 0.01

    @pytest.mark.parametrize("number_of_layers", [4, 8, 16])
    def test_output_variance_preserved_through_forward_pass(
        self,
        sequence_tensor_factory: Callable[..., torch.Tensor],
        number_of_layers: int,
    ):
        encoder = TransformerEncoder(
            number_of_layers=number_of_layers,
            embedding_dimension=64,
            number_of_heads=4,
            attention_type=AttentionType.MULTI_HEAD.value,
            dropout=0.0,
        )
        encoder.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=4, sequence_length=8, embedding_dimension=64
        )
        input_std = hidden_states.std().item()
        output = encoder(hidden_states=hidden_states)
        output_std = output.std().item()
        assert np.isclose(output_std / input_std, 1.0, atol=0.1)

    def test_non_residual_linear_uses_base_std(
        self,
        mixin_factory: Callable[..., ConcreteMixin],
    ):
        mixin = mixin_factory(initializer_range=0.05)
        linear = nn.Linear(64, 64)
        torch.manual_seed(0)
        mixin._init_weights(linear)
        actual_std = linear.weight.data.std().item()
        assert abs(actual_std - 0.05) < 0.01

    def test_linear_bias_zeroed(
        self,
        mixin_factory: Callable[..., ConcreteMixin],
    ):
        mixin = mixin_factory()
        linear = nn.Linear(32, 32, bias=True)
        nn.init.ones_(linear.bias)
        mixin._init_weights(linear)
        assert torch.all(linear.bias.data == 0.0)

    def test_modulation_layer_skipped(
        self,
        mixin_factory: Callable[..., ConcreteMixin],
    ):
        mixin = mixin_factory()
        linear = nn.Linear(32, 32)
        linear._is_modulation_layer = True
        original_weight = linear.weight.data.clone()
        mixin._init_weights(linear)
        assert torch.equal(linear.weight.data, original_weight)

    def test_embedding_uses_base_std(
        self,
        mixin_factory: Callable[..., ConcreteMixin],
    ):
        mixin = mixin_factory(initializer_range=0.05)
        embedding = nn.Embedding(100, 32)
        torch.manual_seed(0)
        mixin._init_weights(embedding)
        actual_std = embedding.weight.data.std().item()
        assert abs(actual_std - 0.05) < 0.01

    def test_embedding_padding_idx_zeroed(
        self,
        mixin_factory: Callable[..., ConcreteMixin],
    ):
        mixin = mixin_factory()
        embedding = nn.Embedding(100, 32, padding_idx=0)
        mixin._init_weights(embedding)
        assert torch.all(embedding.weight.data[0] == 0.0)

    def test_layer_norm_weight_ones_bias_zero(
        self,
        mixin_factory: Callable[..., ConcreteMixin],
    ):
        mixin = mixin_factory()
        layer_norm = nn.LayerNorm(32)
        nn.init.uniform_(layer_norm.weight)
        nn.init.uniform_(layer_norm.bias)
        mixin._init_weights(layer_norm)
        assert torch.all(layer_norm.weight.data == 1.0)
        assert torch.all(layer_norm.bias.data == 0.0)

    def test_rms_norm_weight_ones(
        self,
        mixin_factory: Callable[..., ConcreteMixin],
    ):
        mixin = mixin_factory()
        rms_norm = RMSNorm(32)
        nn.init.uniform_(rms_norm.weight)
        mixin._init_weights(rms_norm)
        assert torch.all(rms_norm.weight.data == 1.0)


class TestExpandPaddingMask:
    def test_output_shape(
        self,
        padding_mask_factory: Callable[..., torch.Tensor],
    ):
        mask = padding_mask_factory(
            batch_size=2, sequence_length=6, padded_positions=[[4, 5], []]
        )
        expanded = TransformerMixin._expand_padding_mask(
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
        expanded = TransformerMixin._expand_padding_mask(
            padding_mask=mask, query_length=3
        )
        for query_index in range(3):
            assert expanded[0, 0, query_index, 5].item() is True
        assert expanded[0, 0, 0, 0].item() is False

    def test_unpadded_positions_are_false(
        self,
        padding_mask_factory: Callable[..., torch.Tensor],
    ):
        mask = padding_mask_factory(
            batch_size=1, sequence_length=4, padded_positions=[[3]]
        )
        expanded = TransformerMixin._expand_padding_mask(
            padding_mask=mask, query_length=2
        )
        assert expanded[0, 0, 0, 0].item() is False
        assert expanded[0, 0, 0, 1].item() is False
        assert expanded[0, 0, 0, 2].item() is False
        assert expanded[0, 0, 0, 3].item() is True


class TestSetupPositionalEncoding:
    def test_none_type_produces_no_encoding(
        self,
        mixin_factory: Callable[..., ConcreteMixin],
    ):
        mixin = mixin_factory()
        mixin._setup_positional_encoding(
            positional_encoding_type=None,
            embedding_dimension=32,
            maximum_sequence_length=128,
            number_of_heads=4,
        )
        assert mixin.positional_encoding is None

    @pytest.mark.parametrize(
        "encoding_type",
        [
            PositionalEncodingType.SINUSOIDAL.value,
            PositionalEncodingType.ROPE.value,
        ],
    )
    def test_delegates_to_create_positional_encoding(
        self,
        mixin_factory: Callable[..., ConcreteMixin],
        encoding_type: str,
    ):
        mixin = mixin_factory()
        with patch(
            "versatil.models.layers.transformer.transformer_mixin.create_positional_encoding",
            return_value=MagicMock(spec=nn.Module),
        ) as mock_create:
            mixin._setup_positional_encoding(
                positional_encoding_type=encoding_type,
                embedding_dimension=32,
                maximum_sequence_length=128,
                number_of_heads=4,
            )
            mock_create.assert_called_once_with(
                encoding_type=encoding_type,
                embedding_dimension=32,
                maximum_length=128,
                num_heads=4,
            )
            assert mixin.positional_encoding is mock_create.return_value


class TestApplyPositionalEncoding:
    def test_no_encoding_returns_unchanged_input(
        self,
        mixin_factory: Callable[..., ConcreteMixin],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        mixin = mixin_factory()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        output, rope = mixin._apply_positional_encoding(hidden_states)
        assert torch.equal(output, hidden_states)
        assert rope is None

    @pytest.mark.parametrize("offset", [0, 5])
    def test_additive_encoding_modifies_hidden_states(
        self,
        mixin_factory: Callable[..., ConcreteMixin],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        mock_sinusoidal_factory: Callable[..., MagicMock],
        offset: int,
    ):
        mixin = mixin_factory()
        mock_pe = mock_sinusoidal_factory()
        mixin.positional_encoding = mock_pe
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        output, rope = mixin._apply_positional_encoding(hidden_states, offset=offset)
        expected = hidden_states + torch.ones_like(hidden_states)
        assert torch.equal(output, expected)
        assert rope is None
        mock_pe.assert_called_once_with(hidden_states, offset=offset)

    def test_rope_returns_encoding_without_modifying_input(
        self,
        mixin_factory: Callable[..., ConcreteMixin],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        mock_rope_factory: Callable[..., MagicMock],
    ):
        mixin = mixin_factory()
        mock_rope = mock_rope_factory()
        mixin.positional_encoding = mock_rope
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        output, rope = mixin._apply_positional_encoding(hidden_states)
        assert torch.equal(output, hidden_states)
        assert rope is mock_rope
