"""Tests for versatil.models.layers.transformer.attention.precomputed_joint_attention module."""

import re
import unittest.mock
from collections.abc import Callable

import pytest
import torch

from versatil.models.layers.transformer.attention.precomputed_joint_attention import (
    PrecomputedPrimaryJointAttention,
)
from versatil.models.layers.transformer.cache.conditioning import (
    ConditioningLayerCache,
)

PRIMARY_EMBEDDING_DIMENSION = 32
SECONDARY_EMBEDDING_DIMENSION = 24
NUMBER_OF_HEADS = 4
HEAD_DIMENSION = SECONDARY_EMBEDDING_DIMENSION // NUMBER_OF_HEADS


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
def precomputed_secondary_qkv_factory(
    conditioning_cache_with_queries_factory: Callable[..., ConditioningLayerCache],
) -> Callable[..., ConditioningLayerCache]:
    """Factory for precomputed secondary Q/K/V as ConditioningLayerCache."""

    def factory(
        batch_size: int = 2,
        sequence_length: int = 6,
        number_of_heads: int = NUMBER_OF_HEADS,
        head_dimension: int = HEAD_DIMENSION,
    ) -> ConditioningLayerCache:
        return conditioning_cache_with_queries_factory(
            batch_size=batch_size,
            number_of_heads=number_of_heads,
            sequence_length=sequence_length,
            head_dimension=head_dimension,
        )

    return factory


class TestPrecomputedPrimaryJointAttentionInitialization:
    def test_output_projection_has_sqrt_weight_flag(
        self,
        precomputed_joint_attention_factory: Callable[
            ..., PrecomputedPrimaryJointAttention
        ],
    ):
        attention = precomputed_joint_attention_factory()
        assert attention.output_projection_primary.SQUARE_ROOT_WEIGHT is True

    def test_invalid_default_head_dimension_raises(
        self,
        precomputed_joint_attention_factory: Callable[
            ..., PrecomputedPrimaryJointAttention
        ],
    ):
        error_message = (
            "secondary_embedding_dimension (26) must be divisible by "
            "number_of_heads (4) when head_dimension is not provided."
        )
        with pytest.raises(ValueError, match=re.escape(error_message)):
            precomputed_joint_attention_factory(
                secondary_embedding_dimension=26,
                number_of_heads=4,
            )


class TestPrecomputedPrimaryJointAttentionForward:
    def test_primary_output_is_projected_secondary_is_raw(
        self,
        precomputed_joint_attention_factory: Callable[
            ..., PrecomputedPrimaryJointAttention
        ],
        precomputed_secondary_qkv_factory: Callable,
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        attention = precomputed_joint_attention_factory()
        secondary_qkv = precomputed_secondary_qkv_factory(
            batch_size=2, sequence_length=6
        )
        primary = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=PRIMARY_EMBEDDING_DIMENSION,
        )
        output_primary, output_secondary = attention(
            conditioning_cache=secondary_qkv,
            hidden_states_primary=primary,
        )
        # Primary: projected back to primary embedding dim
        assert output_primary.shape == (2, 4, PRIMARY_EMBEDDING_DIMENSION)
        # Secondary: raw attention output (B, S, H*D_head) — no O-projection
        query_dimension = NUMBER_OF_HEADS * HEAD_DIMENSION
        assert output_secondary.shape == (2, 6, query_dimension)

    def test_different_primary_inputs_produce_different_primary_outputs(
        self,
        precomputed_joint_attention_factory: Callable[
            ..., PrecomputedPrimaryJointAttention
        ],
        precomputed_secondary_qkv_factory: Callable,
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        attention = precomputed_joint_attention_factory()
        attention.eval()
        secondary_qkv = precomputed_secondary_qkv_factory(
            batch_size=2, sequence_length=6
        )
        primary_a = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=PRIMARY_EMBEDDING_DIMENSION,
        )
        primary_b = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=PRIMARY_EMBEDDING_DIMENSION,
        )
        output_a, _ = attention(
            conditioning_cache=secondary_qkv,
            hidden_states_primary=primary_a,
        )
        output_b, _ = attention(
            conditioning_cache=secondary_qkv,
            hidden_states_primary=primary_b,
        )
        assert not torch.allclose(output_a, output_b)

    def test_different_precomputed_secondary_qkv_affects_both_outputs(
        self,
        precomputed_joint_attention_factory: Callable[
            ..., PrecomputedPrimaryJointAttention
        ],
        precomputed_secondary_qkv_factory: Callable,
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        attention = precomputed_joint_attention_factory()
        attention.eval()
        secondary_qkv_a = precomputed_secondary_qkv_factory(
            batch_size=2, sequence_length=6
        )
        secondary_qkv_b = precomputed_secondary_qkv_factory(
            batch_size=2, sequence_length=6
        )
        primary = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=PRIMARY_EMBEDDING_DIMENSION,
        )
        output_primary_a, output_secondary_a = attention(
            conditioning_cache=secondary_qkv_a,
            hidden_states_primary=primary,
        )
        output_primary_b, output_secondary_b = attention(
            conditioning_cache=secondary_qkv_b,
            hidden_states_primary=primary,
        )
        assert not torch.allclose(output_primary_a, output_primary_b)
        assert not torch.allclose(output_secondary_a, output_secondary_b)

    def test_query_key_norm_changes_output(
        self,
        precomputed_joint_attention_factory: Callable[
            ..., PrecomputedPrimaryJointAttention
        ],
        precomputed_secondary_qkv_factory: Callable,
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        attention_with_norm = precomputed_joint_attention_factory(
            use_query_key_norm=True,
        )
        attention_without_norm = precomputed_joint_attention_factory(
            use_query_key_norm=False,
        )
        # Copy primary projections so only QK-norm differs
        attention_without_norm.load_state_dict(
            attention_with_norm.state_dict(), strict=False
        )
        attention_with_norm.eval()
        attention_without_norm.eval()
        secondary_qkv = precomputed_secondary_qkv_factory(
            batch_size=2, sequence_length=6
        )
        primary = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=PRIMARY_EMBEDDING_DIMENSION,
        )
        _, output_with = attention_with_norm(
            conditioning_cache=secondary_qkv,
            hidden_states_primary=primary,
        )
        _, output_without = attention_without_norm(
            conditioning_cache=secondary_qkv,
            hidden_states_primary=primary,
        )
        assert not torch.allclose(output_with, output_without)

    def test_precomputed_primary_rope_changes_output(
        self,
        precomputed_joint_attention_factory: Callable[
            ..., PrecomputedPrimaryJointAttention
        ],
        precomputed_secondary_qkv_factory: Callable,
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        attention = precomputed_joint_attention_factory(use_query_key_norm=False)
        attention.eval()
        secondary_qkv = precomputed_secondary_qkv_factory(
            batch_size=2, sequence_length=6
        )
        primary = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=PRIMARY_EMBEDDING_DIMENSION,
        )
        output_no_rope, _ = attention(
            conditioning_cache=secondary_qkv,
            hidden_states_primary=primary,
        )
        # Create fake (cos, sin) RoPE tensors matching primary query shape
        cos = torch.ones(1, 1, 4, HEAD_DIMENSION)
        sin = torch.full((1, 1, 4, HEAD_DIMENSION), 0.5)
        output_with_rope, _ = attention(
            conditioning_cache=secondary_qkv,
            hidden_states_primary=primary,
            precomputed_primary_rope=(cos, sin),
        )
        assert not torch.allclose(output_no_rope, output_with_rope)

    def test_positional_encoding_module_applied_to_primary(
        self,
        precomputed_joint_attention_factory: Callable[
            ..., PrecomputedPrimaryJointAttention
        ],
        precomputed_secondary_qkv_factory: Callable,
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        attention = precomputed_joint_attention_factory(use_query_key_norm=False)
        attention.eval()
        secondary_qkv = precomputed_secondary_qkv_factory(
            batch_size=2, sequence_length=6
        )
        primary = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=PRIMARY_EMBEDDING_DIMENSION,
        )
        output_no_rope, _ = attention(
            conditioning_cache=secondary_qkv,
            hidden_states_primary=primary,
        )
        with unittest.mock.patch(
            "versatil.models.layers.transformer.attention.precomputed_joint_attention.apply_rope_positional_encoding",
            side_effect=lambda queries, keys, **kwargs: (queries * 1.5, keys * 1.5),
        ):
            output_with_rope, _ = attention(
                conditioning_cache=secondary_qkv,
                hidden_states_primary=primary,
                positional_encoding_primary=unittest.mock.MagicMock(),
            )
        assert not torch.allclose(output_no_rope, output_with_rope)

    def test_missing_precomputed_secondary_queries_raises(
        self,
        precomputed_joint_attention_factory: Callable[
            ..., PrecomputedPrimaryJointAttention
        ],
        conditioning_cache_factory: Callable[..., ConditioningLayerCache],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        attention = precomputed_joint_attention_factory()
        conditioning_cache = conditioning_cache_factory(
            batch_size=2,
            number_of_key_value_heads=NUMBER_OF_HEADS,
            memory_length=6,
            head_dimension=HEAD_DIMENSION,
        )
        primary = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=PRIMARY_EMBEDDING_DIMENSION,
        )
        error_message = (
            "conditioning_cache.queries must be provided for precomputed "
            "joint attention."
        )
        with pytest.raises(ValueError, match=re.escape(error_message)):
            attention(
                conditioning_cache=conditioning_cache,
                hidden_states_primary=primary,
            )
