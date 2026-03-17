"""Tests for versatil.models.layers.detr_transformer.attention module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise

import pytest
import torch

from versatil.models.layers.detr_transformer.attention import FlashAttention

EMBEDDING_DIMENSION = 64
NUMBER_OF_HEADS = 4
SOURCE_LENGTH = 8
TARGET_LENGTH = 6


class TestFlashAttentionInitialization:
    @pytest.mark.parametrize("embedding_dimension", [EMBEDDING_DIMENSION, 128])
    @pytest.mark.parametrize("number_of_heads", [NUMBER_OF_HEADS, 8])
    @pytest.mark.parametrize("dropout", [0.0, 0.1])
    def test_stores_configuration(
        self,
        flash_attention_factory: Callable[..., FlashAttention],
        embedding_dimension: int,
        number_of_heads: int,
        dropout: float,
    ):
        attention = flash_attention_factory(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            dropout=dropout,
        )
        assert attention.embedding_dimension == embedding_dimension
        assert attention.number_of_heads == number_of_heads
        assert attention.head_size == embedding_dimension // number_of_heads
        assert attention.dropout == dropout

    @pytest.mark.parametrize(
        "embedding_dimension, number_of_heads, expectation",
        [
            (64, 4, does_not_raise()),
            (128, 8, does_not_raise()),
            (
                64,
                3,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "Attention layer embedding_dimension must be divisible by number_of_heads."
                    ),
                ),
            ),
            (
                64,
                5,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "Attention layer embedding_dimension must be divisible by number_of_heads."
                    ),
                ),
            ),
        ],
    )
    def test_embedding_dimension_divisibility_validation(
        self,
        embedding_dimension: int,
        number_of_heads: int,
        expectation,
    ):
        with expectation:
            FlashAttention(
                embedding_dimension=embedding_dimension,
                number_of_heads=number_of_heads,
            )

    def test_projection_layer_dimensions(
        self,
        flash_attention_factory: Callable[..., FlashAttention],
    ):
        attention = flash_attention_factory(
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        assert attention.q_proj.in_features == EMBEDDING_DIMENSION
        assert attention.q_proj.out_features == EMBEDDING_DIMENSION
        assert attention.k_proj.in_features == EMBEDDING_DIMENSION
        assert attention.k_proj.out_features == EMBEDDING_DIMENSION
        assert attention.v_proj.in_features == EMBEDDING_DIMENSION
        assert attention.v_proj.out_features == EMBEDDING_DIMENSION
        assert attention.out_proj.in_features == EMBEDDING_DIMENSION
        assert attention.out_proj.out_features == EMBEDDING_DIMENSION

    def test_output_projection_has_square_root_weight_flag(
        self,
        flash_attention_factory: Callable[..., FlashAttention],
    ):
        attention = flash_attention_factory()
        assert attention.out_proj.SQUARE_ROOT_WEIGHT is True


class TestFlashAttentionForward:
    def test_self_attention_output_shape(
        self,
        flash_attention_factory: Callable[..., FlashAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        attention = flash_attention_factory()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output = attention(
            query=source,
            key=source,
            value=source,
        )
        assert output.shape == (batch_size, SOURCE_LENGTH, EMBEDDING_DIMENSION)

    def test_cross_attention_output_shape(
        self,
        flash_attention_factory: Callable[..., FlashAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        attention = flash_attention_factory()
        query = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        key_value = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output = attention(
            query=query,
            key=key_value,
            value=key_value,
        )
        assert output.shape == (batch_size, TARGET_LENGTH, EMBEDDING_DIMENSION)

    def test_different_queries_produce_different_outputs(
        self,
        flash_attention_factory: Callable[..., FlashAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        attention = flash_attention_factory()
        attention.eval()
        query_a = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        query_b = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        key_value = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output_a = attention(query=query_a, key=key_value, value=key_value)
        output_b = attention(query=query_b, key=key_value, value=key_value)
        assert not torch.allclose(output_a, output_b)

    def test_different_keys_produce_different_outputs(
        self,
        flash_attention_factory: Callable[..., FlashAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        attention = flash_attention_factory()
        attention.eval()
        query = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        key_value_a = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        key_value_b = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output_a = attention(query=query, key=key_value_a, value=key_value_a)
        output_b = attention(query=query, key=key_value_b, value=key_value_b)
        assert not torch.allclose(output_a, output_b)

    def test_positional_encoding_changes_output(
        self,
        flash_attention_factory: Callable[..., FlashAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        attention = flash_attention_factory()
        attention.eval()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        positional_encoding = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output_without_pe = attention(
            query=source,
            key=source,
            value=source,
        )
        output_with_pe = attention(
            query=source,
            key=source,
            value=source,
            query_positional_encoding=positional_encoding,
            key_positional_encoding=positional_encoding,
        )
        assert not torch.allclose(output_without_pe, output_with_pe)

    def test_positional_encoding_does_not_affect_value_projection(
        self,
        flash_attention_factory: Callable[..., FlashAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        # Positional encoding is added to query and key but NOT to value.
        # We verify this by checking that with a single token (no attention routing effect),
        # the value path is not directly altered by PE.
        attention = flash_attention_factory(
            embedding_dimension=EMBEDDING_DIMENSION,
            number_of_heads=NUMBER_OF_HEADS,
        )
        attention.eval()
        # Use a single-token sequence so attention weights are always [1.0] regardless of PE
        single_token = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=1,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        positional_encoding = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=1,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output_without_pe = attention(
            query=single_token,
            key=single_token,
            value=single_token,
        )
        output_with_pe = attention(
            query=single_token,
            key=single_token,
            value=single_token,
            query_positional_encoding=positional_encoding,
            key_positional_encoding=positional_encoding,
        )
        # With single token the attention weights are always 1.0,
        # so the output only depends on value projection + output projection.
        # Since PE is NOT added to value, both outputs should be identical.
        assert torch.allclose(output_without_pe, output_with_pe, atol=1e-6)


class TestFlashAttentionMasking:
    def test_key_padding_mask_suppresses_padded_positions(
        self,
        flash_attention_factory: Callable[..., FlashAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        padding_mask_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        attention = flash_attention_factory()
        attention.eval()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        padding_mask = padding_mask_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            mask_last_n=1,
        )
        output_without_mask = attention(
            query=source,
            key=source,
            value=source,
        )
        output_with_mask = attention(
            query=source,
            key=source,
            value=source,
            key_padding_mask=padding_mask,
        )
        # Masking out positions changes the output
        assert not torch.allclose(output_without_mask, output_with_mask)

    def test_fully_padded_key_except_one_produces_that_value(
        self,
        flash_attention_factory: Callable[..., FlashAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        attention = flash_attention_factory()
        attention.eval()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        # Mask everything except the first position
        padding_mask = torch.ones(batch_size, SOURCE_LENGTH, dtype=torch.bool)
        padding_mask[:, 0] = False
        output = attention(
            query=source,
            key=source,
            value=source,
            key_padding_mask=padding_mask,
        )
        # All query positions attend only to position 0,
        # so all positions should get the same output value
        for position_index in range(1, SOURCE_LENGTH):
            assert torch.allclose(
                output[:, 0, :], output[:, position_index, :], atol=1e-5
            )

    def test_attention_mask_2d_changes_output(
        self,
        flash_attention_factory: Callable[..., FlashAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        attention = flash_attention_factory()
        attention.eval()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        # 2D causal-style mask (True = masked)
        attention_mask = torch.triu(
            torch.ones(SOURCE_LENGTH, SOURCE_LENGTH, dtype=torch.bool),
            diagonal=1,
        )
        output_without_mask = attention(
            query=source,
            key=source,
            value=source,
        )
        output_with_mask = attention(
            query=source,
            key=source,
            value=source,
            attention_mask=attention_mask,
        )
        assert not torch.allclose(output_without_mask, output_with_mask)

    def test_combined_attention_and_padding_mask(
        self,
        flash_attention_factory: Callable[..., FlashAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        padding_mask_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        attention = flash_attention_factory()
        attention.eval()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        padding_mask = padding_mask_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            mask_last_n=1,
        )
        attention_mask = torch.triu(
            torch.ones(SOURCE_LENGTH, SOURCE_LENGTH, dtype=torch.bool),
            diagonal=1,
        )
        output_padding_only = attention(
            query=source,
            key=source,
            value=source,
            key_padding_mask=padding_mask,
        )
        output_both = attention(
            query=source,
            key=source,
            value=source,
            attention_mask=attention_mask,
            key_padding_mask=padding_mask,
        )
        # Adding an attention mask on top of padding mask should further change output
        assert not torch.allclose(output_padding_only, output_both)

    def test_neginf_attention_mask_treated_as_boolean(
        self,
        flash_attention_factory: Callable[..., FlashAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        attention = flash_attention_factory()
        attention.eval()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        # Bool mask (True = masked)
        bool_mask = torch.triu(
            torch.ones(SOURCE_LENGTH, SOURCE_LENGTH, dtype=torch.bool),
            diagonal=1,
        )
        # Float mask with -inf for masked positions
        float_mask = torch.zeros(SOURCE_LENGTH, SOURCE_LENGTH)
        float_mask[bool_mask] = float("-inf")
        output_bool = attention(
            query=source,
            key=source,
            value=source,
            attention_mask=bool_mask,
        )
        output_float = attention(
            query=source,
            key=source,
            value=source,
            attention_mask=float_mask,
        )
        assert torch.allclose(output_bool, output_float, atol=1e-5)


class TestFlashAttentionTraining:
    def test_dropout_disabled_during_eval(
        self,
        flash_attention_factory: Callable[..., FlashAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        attention = flash_attention_factory(dropout=0.5)
        attention.eval()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output_a = attention(query=source, key=source, value=source)
        output_b = attention(query=source, key=source, value=source)
        # In eval mode, dropout=0.0 so outputs are deterministic
        assert torch.allclose(output_a, output_b)

    def test_gradients_flow_through_all_projections(
        self,
        flash_attention_factory: Callable[..., FlashAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        attention = flash_attention_factory()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        source.requires_grad_(True)
        output = attention(query=source, key=source, value=source)
        loss = output.sum()
        loss.backward()
        assert source.grad is not None
        for name, parameter in attention.named_parameters():
            assert parameter.grad is not None, f"No gradient for {name}"
