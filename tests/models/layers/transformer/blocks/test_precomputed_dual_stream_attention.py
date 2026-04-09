"""Tests for versatil.models.layers.transformer.block.precomputed_dual_stream_attention module."""

from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.normalization.factory import create_block_normalization
from versatil.models.layers.transformer.attention.precomputed_primary_joint_attention import (
    PrecomputedPrimaryJointAttention,
)
from versatil.models.layers.transformer.block.precomputed_dual_stream_attention import (
    PrecomputedDualStreamAttentionBlock,
)

from .conftest import EMBEDDING_DIMENSION, NUMBER_OF_HEADS

SECONDARY_EMBEDDING_DIMENSION = 16
BATCH_SIZE = 2
PRIMARY_SEQUENCE_LENGTH = 8
SECONDARY_SEQUENCE_LENGTH = 4
HEAD_DIMENSION = EMBEDDING_DIMENSION // NUMBER_OF_HEADS


@pytest.fixture
def precomputed_joint_attention() -> PrecomputedPrimaryJointAttention:
    return PrecomputedPrimaryJointAttention(
        primary_embedding_dimension=EMBEDDING_DIMENSION,
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
            attention_normalization_secondary=create_block_normalization(
                normalization_type=NormalizationType.RMS_NORM.value,
                dimension=SECONDARY_EMBEDDING_DIMENSION,
                condition_dim=conditioning_dimension,
                use_gating=use_gating,
            ),
            dropout=0.0,
        )

    return factory


@pytest.fixture
def precomputed_primary_factory(
    rng: np.random.Generator,
) -> Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

    def factory(
        batch_size: int = BATCH_SIZE,
        sequence_length: int = PRIMARY_SEQUENCE_LENGTH,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_shape = (batch_size, NUMBER_OF_HEADS, sequence_length, HEAD_DIMENSION)
        kv_shape = (batch_size, NUMBER_OF_HEADS, sequence_length, HEAD_DIMENSION)
        queries = torch.from_numpy(rng.standard_normal(query_shape).astype(np.float32))
        keys = torch.from_numpy(rng.standard_normal(kv_shape).astype(np.float32))
        values = torch.from_numpy(rng.standard_normal(kv_shape).astype(np.float32))
        return queries, keys, values

    return factory


class TestPrecomputedDualStreamAttentionBlockForward:
    def test_output_shapes(
        self,
        block_factory: Callable[..., PrecomputedDualStreamAttentionBlock],
        precomputed_primary_factory: Callable[
            ..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        block = block_factory()
        precomputed = precomputed_primary_factory()
        secondary = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=SECONDARY_SEQUENCE_LENGTH,
            embedding_dimension=SECONDARY_EMBEDDING_DIMENSION,
        )
        primary_out, secondary_out = block(
            precomputed_primary=precomputed,
            hidden_states_secondary=secondary,
        )
        assert primary_out.shape == (
            BATCH_SIZE,
            PRIMARY_SEQUENCE_LENGTH,
            NUMBER_OF_HEADS * HEAD_DIMENSION,
        )
        assert secondary_out.shape == secondary.shape
        assert torch.all(torch.isfinite(primary_out))
        assert torch.all(torch.isfinite(secondary_out))

    def test_different_primary_changes_secondary_output(
        self,
        block_factory: Callable[..., PrecomputedDualStreamAttentionBlock],
        precomputed_primary_factory: Callable[
            ..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        block = block_factory()
        block.eval()
        secondary = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=SECONDARY_SEQUENCE_LENGTH,
            embedding_dimension=SECONDARY_EMBEDDING_DIMENSION,
        )
        primary_a = precomputed_primary_factory()
        primary_b = precomputed_primary_factory()
        _, secondary_out_a = block(
            precomputed_primary=primary_a,
            hidden_states_secondary=secondary,
        )
        _, secondary_out_b = block(
            precomputed_primary=primary_b,
            hidden_states_secondary=secondary,
        )
        assert not torch.allclose(secondary_out_a, secondary_out_b)

    def test_residual_connection_preserves_secondary_at_gated_init(
        self,
        block_factory: Callable[..., PrecomputedDualStreamAttentionBlock],
        precomputed_primary_factory: Callable[
            ..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        block = block_factory(
            conditioning_dimension=SECONDARY_EMBEDDING_DIMENSION,
            use_gating=True,
        )
        block.eval()
        precomputed = precomputed_primary_factory()
        secondary = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=SECONDARY_SEQUENCE_LENGTH,
            embedding_dimension=SECONDARY_EMBEDDING_DIMENSION,
        )
        conditioning = condition_factory(
            batch_size=BATCH_SIZE, condition_dim=SECONDARY_EMBEDDING_DIMENSION
        )
        _, secondary_out = block(
            precomputed_primary=precomputed,
            hidden_states_secondary=secondary,
            conditioning=conditioning,
        )
        assert torch.allclose(secondary_out, secondary, atol=1e-6)
