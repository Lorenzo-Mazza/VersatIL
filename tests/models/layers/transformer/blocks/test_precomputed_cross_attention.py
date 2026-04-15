"""Tests for versatil.models.layers.transformer.block.precomputed_cross_attention module."""

from collections.abc import Callable

import torch

from tests.models.layers.conftest import reinit_modulation_layers
from versatil.models.layers.normalization.ada_norm import AdaNorm
from versatil.models.layers.transformer.attention.cached_attention import (
    CachedAttention,
)
from versatil.models.layers.transformer.block.precomputed_cross_attention import (
    PrecomputedCrossAttentionBlock,
)

EMBEDDING_DIMENSION = 32


class TestPrecomputedCrossAttentionBlockForward:
    def test_different_keys_produce_different_outputs(
        self,
        cached_attention: CachedAttention,
        unconditioned_norm,
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        block = PrecomputedCrossAttentionBlock(
            attention=cached_attention,
            normalization=unconditioned_norm,
        )
        block.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        keys_a = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        values_a = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        keys_b = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        values_b = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output_a = block(hidden_states=hidden_states, keys=keys_a, values=values_a)
        output_b = block(hidden_states=hidden_states, keys=keys_b, values=values_b)
        assert not torch.allclose(output_a, output_b)

    def test_precomputed_query_rope_changes_output(
        self,
        cached_attention: CachedAttention,
        unconditioned_norm,
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        block = PrecomputedCrossAttentionBlock(
            attention=cached_attention,
            normalization=unconditioned_norm,
        )
        block.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        keys = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        values = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output_no_rope = block(hidden_states=hidden_states, keys=keys, values=values)
        head_dimension = EMBEDDING_DIMENSION // cached_attention.number_of_heads
        cos = torch.ones(1, 1, 4, head_dimension)
        sin = torch.full((1, 1, 4, head_dimension), 0.5)
        output_with_rope = block(
            hidden_states=hidden_states,
            keys=keys,
            values=values,
            precomputed_query_rope=(cos, sin),
        )
        assert not torch.allclose(output_no_rope, output_with_rope)

    def test_ada_norm_conditioning_affects_output(
        self,
        cached_attention: CachedAttention,
        ada_norm_no_gate: AdaNorm,
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        block = PrecomputedCrossAttentionBlock(
            attention=cached_attention,
            normalization=ada_norm_no_gate,
        )
        reinit_modulation_layers(block)
        hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        keys = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        values = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        conditioning_a = condition_factory(
            batch_size=2, condition_dim=EMBEDDING_DIMENSION
        )
        conditioning_b = condition_factory(
            batch_size=2, condition_dim=EMBEDDING_DIMENSION
        )
        output_a = block(
            hidden_states=hidden_states,
            keys=keys,
            values=values,
            conditioning=conditioning_a,
        )
        output_b = block(
            hidden_states=hidden_states,
            keys=keys,
            values=values,
            conditioning=conditioning_b,
        )
        assert not torch.allclose(output_a, output_b)

    def test_attention_mask_affects_output(
        self,
        cached_attention: CachedAttention,
        unconditioned_norm,
        sequence_tensor_factory: Callable[..., torch.Tensor],
        attention_mask_factory: Callable[..., torch.Tensor],
    ):
        block = PrecomputedCrossAttentionBlock(
            attention=cached_attention,
            normalization=unconditioned_norm,
        )
        block.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        keys = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        values = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        mask = attention_mask_factory(batch_size=2, query_length=4, key_length=6)
        mask[:, :, :, 3:] = True
        output_masked = block(
            hidden_states=hidden_states,
            keys=keys,
            values=values,
            attention_mask=mask,
        )
        output_unmasked = block(
            hidden_states=hidden_states,
            keys=keys,
            values=values,
        )
        assert not torch.allclose(output_masked, output_unmasked)
