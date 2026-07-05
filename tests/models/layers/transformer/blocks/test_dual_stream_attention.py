"""Tests for versatil.models.layers.transformer.block.dual_stream_attention module."""

from collections.abc import Callable

import pytest
import torch

from tests.models.layers.conftest import reinit_modulation_layers
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.normalization.factory import create_block_normalization
from versatil.models.layers.transformer.attention.joint_attention import JointAttention
from versatil.models.layers.transformer.block.dual_stream_attention import (
    DualStreamAttentionBlock,
)

EMBEDDING_DIMENSION = 32
NUMBER_OF_HEADS = 4

BATCH_SIZE = 2
PRIMARY_SEQUENCE_LENGTH = 6
SECONDARY_SEQUENCE_LENGTH = 4


@pytest.fixture
def joint_attention_module() -> JointAttention:
    return JointAttention(
        primary_embedding_dimension=EMBEDDING_DIMENSION,
        number_of_heads=NUMBER_OF_HEADS,
        dropout=0.0,
        use_query_key_norm=False,
        bias=True,
    )


@pytest.fixture
def block_factory(
    joint_attention_module: JointAttention,
) -> Callable[..., DualStreamAttentionBlock]:

    def factory(
        conditioning_dimension: int | None = None,
        use_gating: bool = False,
    ) -> DualStreamAttentionBlock:
        return DualStreamAttentionBlock(
            joint_attention=joint_attention_module,
            attention_normalization_primary=create_block_normalization(
                normalization_type=NormalizationType.RMS_NORM.value,
                dimension=EMBEDDING_DIMENSION,
                conditioning_dimension=conditioning_dimension,
                use_gating=use_gating,
            ),
            attention_normalization_secondary=create_block_normalization(
                normalization_type=NormalizationType.RMS_NORM.value,
                dimension=EMBEDDING_DIMENSION,
                conditioning_dimension=conditioning_dimension,
                use_gating=use_gating,
            ),
            dropout=0.0,
        )

    return factory


class TestDualStreamAttentionBlockForward:
    def test_output_shapes_match_inputs(
        self,
        block_factory: Callable[..., DualStreamAttentionBlock],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        block = block_factory()
        primary = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=PRIMARY_SEQUENCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        secondary = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=SECONDARY_SEQUENCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        primary_out, secondary_out = block(
            hidden_states_primary=primary,
            hidden_states_secondary=secondary,
        )
        assert primary_out.shape == primary.shape
        assert secondary_out.shape == secondary.shape
        assert torch.all(torch.isfinite(primary_out))
        assert torch.all(torch.isfinite(secondary_out))

    def test_streams_influence_each_other_through_joint_attention(
        self,
        block_factory: Callable[..., DualStreamAttentionBlock],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        block = block_factory()
        block.eval()
        primary = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=PRIMARY_SEQUENCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        secondary_a = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=SECONDARY_SEQUENCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        secondary_b = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=SECONDARY_SEQUENCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        primary_out_a, _ = block(
            hidden_states_primary=primary,
            hidden_states_secondary=secondary_a,
        )
        primary_out_b, _ = block(
            hidden_states_primary=primary,
            hidden_states_secondary=secondary_b,
        )
        assert not torch.allclose(primary_out_a, primary_out_b)

    def test_residual_connection_preserves_input_at_gated_init(
        self,
        block_factory: Callable[..., DualStreamAttentionBlock],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        block = block_factory(
            conditioning_dimension=EMBEDDING_DIMENSION,
            use_gating=True,
        )
        block.eval()
        primary = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=PRIMARY_SEQUENCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        secondary = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=SECONDARY_SEQUENCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        conditioning = condition_factory(
            batch_size=BATCH_SIZE, conditioning_dimension=EMBEDDING_DIMENSION
        )
        primary_out, secondary_out = block(
            hidden_states_primary=primary,
            hidden_states_secondary=secondary,
            conditioning=conditioning,
        )
        assert torch.allclose(primary_out, primary, atol=1e-6)
        assert torch.allclose(secondary_out, secondary, atol=1e-6)

    def test_conditioning_produces_different_outputs(
        self,
        block_factory: Callable[..., DualStreamAttentionBlock],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        block = block_factory(conditioning_dimension=EMBEDDING_DIMENSION)
        reinit_modulation_layers(block)
        primary = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=PRIMARY_SEQUENCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        secondary = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=SECONDARY_SEQUENCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        cond_a = condition_factory(
            batch_size=BATCH_SIZE, conditioning_dimension=EMBEDDING_DIMENSION
        )
        cond_b = condition_factory(
            batch_size=BATCH_SIZE, conditioning_dimension=EMBEDDING_DIMENSION
        )
        primary_a, _ = block(
            hidden_states_primary=primary,
            hidden_states_secondary=secondary,
            conditioning=cond_a,
        )
        primary_b, _ = block(
            hidden_states_primary=primary,
            hidden_states_secondary=secondary,
            conditioning=cond_b,
        )
        assert not torch.allclose(primary_a, primary_b)
