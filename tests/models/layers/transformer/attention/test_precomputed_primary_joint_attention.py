"""Tests for versatil.models.layers.transformer.attention.precomputed_primary_joint_attention module."""

import unittest.mock
from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.models.layers.transformer.attention.precomputed_primary_joint_attention import (
    PrecomputedPrimaryJointAttention,
)

PRIMARY_EMBEDDING_DIMENSION = 32
SECONDARY_EMBEDDING_DIMENSION = 24
NUMBER_OF_HEADS = 4
HEAD_DIMENSION = PRIMARY_EMBEDDING_DIMENSION // NUMBER_OF_HEADS


@pytest.fixture
def precomputed_joint_attention_factory() -> Callable[
    ..., PrecomputedPrimaryJointAttention
]:
    def factory(
        primary_embedding_dimension: int = PRIMARY_EMBEDDING_DIMENSION,
        number_of_heads: int = NUMBER_OF_HEADS,
        secondary_embedding_dimension: int = SECONDARY_EMBEDDING_DIMENSION,
        number_of_key_value_heads: int | None = None,
        head_dimension: int | None = None,
        dropout: float = 0.0,
        use_query_key_norm: bool = True,
        bias: bool = True,
    ) -> PrecomputedPrimaryJointAttention:
        return PrecomputedPrimaryJointAttention(
            primary_embedding_dimension=primary_embedding_dimension,
            number_of_heads=number_of_heads,
            secondary_embedding_dimension=secondary_embedding_dimension,
            number_of_key_value_heads=number_of_key_value_heads,
            head_dimension=head_dimension,
            dropout=dropout,
            use_query_key_norm=use_query_key_norm,
            bias=bias,
        )

    return factory


@pytest.fixture
def precomputed_primary_qkv_factory(
    rng: np.random.Generator,
    precomputed_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Factory for precomputed primary (Q, K, V) tuples, reusing precomputed_kv_factory."""

    def factory(
        batch_size: int = 2,
        sequence_length: int = 6,
        number_of_heads: int = NUMBER_OF_HEADS,
        head_dimension: int = HEAD_DIMENSION,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key, value = precomputed_kv_factory(
            batch_size=batch_size,
            key_value_length=sequence_length,
            number_of_heads=number_of_heads,
            head_dimension=head_dimension,
        )
        query_shape = (batch_size, number_of_heads, sequence_length, head_dimension)
        query = torch.from_numpy(rng.standard_normal(query_shape).astype(np.float32))
        return query, key, value

    return factory


class TestPrecomputedPrimaryJointAttentionInitialization:
    def test_output_projection_has_sqrt_weight_flag(
        self,
        precomputed_joint_attention_factory: Callable[
            ..., PrecomputedPrimaryJointAttention
        ],
    ):
        attention = precomputed_joint_attention_factory()
        assert attention.output_projection_secondary.SQUARE_ROOT_WEIGHT is True


class TestPrecomputedPrimaryJointAttentionForward:
    def test_primary_output_is_raw_secondary_is_projected(
        self,
        precomputed_joint_attention_factory: Callable[
            ..., PrecomputedPrimaryJointAttention
        ],
        precomputed_primary_qkv_factory: Callable,
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        attention = precomputed_joint_attention_factory()
        primary_qkv = precomputed_primary_qkv_factory(batch_size=2, sequence_length=6)
        secondary = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=SECONDARY_EMBEDDING_DIMENSION,
        )
        output_primary, output_secondary = attention(
            precomputed_primary=primary_qkv,
            hidden_states_secondary=secondary,
        )
        # Primary: raw attention output (B, S, H*D_head) — no O-projection
        query_dimension = NUMBER_OF_HEADS * HEAD_DIMENSION
        assert output_primary.shape == (2, 6, query_dimension)
        # Secondary: projected back to secondary embedding dim
        assert output_secondary.shape == (2, 4, SECONDARY_EMBEDDING_DIMENSION)

    def test_different_secondary_inputs_produce_different_secondary_outputs(
        self,
        precomputed_joint_attention_factory: Callable[
            ..., PrecomputedPrimaryJointAttention
        ],
        precomputed_primary_qkv_factory: Callable,
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        attention = precomputed_joint_attention_factory()
        attention.eval()
        primary_qkv = precomputed_primary_qkv_factory(batch_size=2, sequence_length=6)
        secondary_a = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=SECONDARY_EMBEDDING_DIMENSION,
        )
        secondary_b = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=SECONDARY_EMBEDDING_DIMENSION,
        )
        _, output_a = attention(
            precomputed_primary=primary_qkv,
            hidden_states_secondary=secondary_a,
        )
        _, output_b = attention(
            precomputed_primary=primary_qkv,
            hidden_states_secondary=secondary_b,
        )
        assert not torch.allclose(output_a, output_b)

    def test_different_primary_qkv_affects_both_outputs(
        self,
        precomputed_joint_attention_factory: Callable[
            ..., PrecomputedPrimaryJointAttention
        ],
        precomputed_primary_qkv_factory: Callable,
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        attention = precomputed_joint_attention_factory()
        attention.eval()
        primary_qkv_a = precomputed_primary_qkv_factory(batch_size=2, sequence_length=6)
        primary_qkv_b = precomputed_primary_qkv_factory(batch_size=2, sequence_length=6)
        secondary = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=SECONDARY_EMBEDDING_DIMENSION,
        )
        output_primary_a, output_secondary_a = attention(
            precomputed_primary=primary_qkv_a,
            hidden_states_secondary=secondary,
        )
        output_primary_b, output_secondary_b = attention(
            precomputed_primary=primary_qkv_b,
            hidden_states_secondary=secondary,
        )
        assert not torch.allclose(output_primary_a, output_primary_b)
        assert not torch.allclose(output_secondary_a, output_secondary_b)

    def test_query_key_norm_changes_output(
        self,
        precomputed_primary_qkv_factory: Callable,
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        attention_with_norm = PrecomputedPrimaryJointAttention(
            primary_embedding_dimension=PRIMARY_EMBEDDING_DIMENSION,
            number_of_heads=NUMBER_OF_HEADS,
            secondary_embedding_dimension=SECONDARY_EMBEDDING_DIMENSION,
            use_query_key_norm=True,
        )
        attention_without_norm = PrecomputedPrimaryJointAttention(
            primary_embedding_dimension=PRIMARY_EMBEDDING_DIMENSION,
            number_of_heads=NUMBER_OF_HEADS,
            secondary_embedding_dimension=SECONDARY_EMBEDDING_DIMENSION,
            use_query_key_norm=False,
        )
        # Copy secondary projections so only QK-norm differs
        attention_without_norm.load_state_dict(
            attention_with_norm.state_dict(), strict=False
        )
        attention_with_norm.eval()
        attention_without_norm.eval()
        primary_qkv = precomputed_primary_qkv_factory(batch_size=2, sequence_length=6)
        secondary = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=SECONDARY_EMBEDDING_DIMENSION,
        )
        _, output_with = attention_with_norm(
            precomputed_primary=primary_qkv,
            hidden_states_secondary=secondary,
        )
        _, output_without = attention_without_norm(
            precomputed_primary=primary_qkv,
            hidden_states_secondary=secondary,
        )
        assert not torch.allclose(output_with, output_without)

    def test_precomputed_secondary_rope_changes_output(
        self,
        precomputed_joint_attention_factory: Callable[
            ..., PrecomputedPrimaryJointAttention
        ],
        precomputed_primary_qkv_factory: Callable,
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        attention = precomputed_joint_attention_factory(use_query_key_norm=False)
        attention.eval()
        primary_qkv = precomputed_primary_qkv_factory(batch_size=2, sequence_length=6)
        secondary = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=SECONDARY_EMBEDDING_DIMENSION,
        )
        _, output_no_rope = attention(
            precomputed_primary=primary_qkv,
            hidden_states_secondary=secondary,
        )
        # Create fake (cos, sin) RoPE tensors matching secondary query shape
        # Shape: (1, 1, T, D_head) to broadcast over batch and heads
        cos = torch.ones(1, 1, 4, HEAD_DIMENSION)
        sin = torch.full((1, 1, 4, HEAD_DIMENSION), 0.5)
        _, output_with_rope = attention(
            precomputed_primary=primary_qkv,
            hidden_states_secondary=secondary,
            precomputed_secondary_rope=(cos, sin),
        )
        assert not torch.allclose(output_no_rope, output_with_rope)

    def test_positional_encoding_module_applied_to_secondary(
        self,
        precomputed_joint_attention_factory: Callable[
            ..., PrecomputedPrimaryJointAttention
        ],
        precomputed_primary_qkv_factory: Callable,
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        attention = precomputed_joint_attention_factory(use_query_key_norm=False)
        attention.eval()
        primary_qkv = precomputed_primary_qkv_factory(batch_size=2, sequence_length=6)
        secondary = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=SECONDARY_EMBEDDING_DIMENSION,
        )
        _, output_no_rope = attention(
            precomputed_primary=primary_qkv,
            hidden_states_secondary=secondary,
        )
        mock_rope = unittest.mock.MagicMock()
        with unittest.mock.patch(
            "versatil.models.layers.transformer.attention.precomputed_primary_joint_attention.apply_rope_positional_encoding",
            side_effect=lambda queries, keys, **kwargs: (queries * 1.5, keys * 1.5),
        ):
            _, output_with_rope = attention(
                precomputed_primary=primary_qkv,
                hidden_states_secondary=secondary,
                positional_encoding_secondary=mock_rope,
            )
        assert not torch.allclose(output_no_rope, output_with_rope)
