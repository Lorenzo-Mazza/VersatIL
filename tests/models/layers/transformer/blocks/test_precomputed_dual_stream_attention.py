"""Tests for versatil.models.layers.transformer.block.precomputed_dual_stream_attention module."""

from collections.abc import Callable

import pytest
import torch

from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.normalization.factory import create_block_normalization
from versatil.models.layers.transformer.attention.precomputed_joint_attention import (
    PrecomputedPrimaryJointAttention,
)
from versatil.models.layers.transformer.block.precomputed_dual_stream_attention import (
    PrecomputedDualStreamAttentionBlock,
)
from versatil.models.layers.transformer.cache.conditioning import (
    ConditioningLayerCache,
)

SECONDARY_EMBEDDING_DIMENSION = 32
NUMBER_OF_HEADS = 4
PRIMARY_EMBEDDING_DIMENSION = 16
BATCH_SIZE = 2
SECONDARY_SEQUENCE_LENGTH = 8
PRIMARY_SEQUENCE_LENGTH = 4
HEAD_DIMENSION = SECONDARY_EMBEDDING_DIMENSION // NUMBER_OF_HEADS


@pytest.fixture
def precomputed_joint_attention() -> PrecomputedPrimaryJointAttention:
    return PrecomputedPrimaryJointAttention(
        primary_embedding_dimension=PRIMARY_EMBEDDING_DIMENSION,
        number_of_heads=NUMBER_OF_HEADS,
        secondary_embedding_dimension=SECONDARY_EMBEDDING_DIMENSION,
        number_of_key_value_heads=NUMBER_OF_HEADS,
        head_dimension=HEAD_DIMENSION,
        dropout=0.0,
        use_query_key_norm=False,
        bias=False,
    )


@pytest.fixture
def block_factory(
    precomputed_joint_attention: PrecomputedPrimaryJointAttention,
) -> Callable[..., PrecomputedDualStreamAttentionBlock]:

    def factory(
        conditioning_dimension: int | None = None,
        use_gating: bool = False,
    ) -> PrecomputedDualStreamAttentionBlock:
        return PrecomputedDualStreamAttentionBlock(
            joint_attention=precomputed_joint_attention,
            attention_normalization_primary=create_block_normalization(
                normalization_type=NormalizationType.RMS_NORM.value,
                dimension=PRIMARY_EMBEDDING_DIMENSION,
                condition_dim=conditioning_dimension,
                use_gating=use_gating,
            ),
            dropout=0.0,
        )

    return factory


@pytest.fixture
def conditioning_factory(
    conditioning_cache_with_queries_factory: Callable[..., ConditioningLayerCache],
) -> Callable[..., ConditioningLayerCache]:

    def factory(
        batch_size: int = BATCH_SIZE,
        sequence_length: int = SECONDARY_SEQUENCE_LENGTH,
    ) -> ConditioningLayerCache:
        return conditioning_cache_with_queries_factory(
            batch_size=batch_size,
            number_of_heads=NUMBER_OF_HEADS,
            sequence_length=sequence_length,
            head_dimension=HEAD_DIMENSION,
        )

    return factory


class TestPrecomputedDualStreamAttentionBlockForward:
    def test_output_shapes(
        self,
        block_factory: Callable[..., PrecomputedDualStreamAttentionBlock],
        conditioning_factory: Callable[..., ConditioningLayerCache],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        block = block_factory()
        conditioning_cache = conditioning_factory()
        primary = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=PRIMARY_SEQUENCE_LENGTH,
            embedding_dimension=PRIMARY_EMBEDDING_DIMENSION,
        )
        primary_out, conditioning_out = block(
            conditioning_cache=conditioning_cache,
            hidden_states_primary=primary,
        )
        assert primary_out.shape == primary.shape
        assert conditioning_out.shape == (
            BATCH_SIZE,
            SECONDARY_SEQUENCE_LENGTH,
            NUMBER_OF_HEADS * HEAD_DIMENSION,
        )
        assert torch.all(torch.isfinite(primary_out))
        assert torch.all(torch.isfinite(conditioning_out))

    def test_different_conditioning_changes_primary_output(
        self,
        block_factory: Callable[..., PrecomputedDualStreamAttentionBlock],
        conditioning_factory: Callable[..., ConditioningLayerCache],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        block = block_factory()
        block.eval()
        primary = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=PRIMARY_SEQUENCE_LENGTH,
            embedding_dimension=PRIMARY_EMBEDDING_DIMENSION,
        )
        conditioning_a = conditioning_factory()
        conditioning_b = conditioning_factory()
        primary_out_a, _ = block(
            conditioning_cache=conditioning_a,
            hidden_states_primary=primary,
        )
        primary_out_b, _ = block(
            conditioning_cache=conditioning_b,
            hidden_states_primary=primary,
        )
        assert not torch.allclose(primary_out_a, primary_out_b)

    def test_residual_connection_preserves_primary_at_gated_init(
        self,
        block_factory: Callable[..., PrecomputedDualStreamAttentionBlock],
        conditioning_factory: Callable[..., ConditioningLayerCache],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        block = block_factory(
            conditioning_dimension=PRIMARY_EMBEDDING_DIMENSION,
            use_gating=True,
        )
        block.eval()
        conditioning_cache = conditioning_factory()
        primary = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=PRIMARY_SEQUENCE_LENGTH,
            embedding_dimension=PRIMARY_EMBEDDING_DIMENSION,
        )
        conditioning = condition_factory(
            batch_size=BATCH_SIZE, condition_dim=PRIMARY_EMBEDDING_DIMENSION
        )
        primary_out, _ = block(
            conditioning_cache=conditioning_cache,
            hidden_states_primary=primary,
            conditioning=conditioning,
        )
        assert torch.allclose(primary_out, primary, atol=1e-6)
