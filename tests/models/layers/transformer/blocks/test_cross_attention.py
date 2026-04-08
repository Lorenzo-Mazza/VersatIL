"""Tests for versatil.models.layers.transformer.blocks.cross_attention module."""

from collections.abc import Callable

import torch

from tests.models.layers.conftest import reinit_modulation_layers
from versatil.models.layers.normalization.ada_norm import AdaNorm
from versatil.models.layers.transformer.attention.cached_attention import (
    CachedAttention,
)
from versatil.models.layers.transformer.blocks.cross_attention import (
    CrossAttentionBlock,
)
from versatil.models.layers.transformer.kv_cache import LayerKVCache

from .conftest import EMBEDDING_DIMENSION


class TestCrossAttentionBlockForward:
    def test_different_encoder_hidden_states_produce_different_outputs(
        self,
        cached_attention: CachedAttention,
        unconditioned_norm,
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        block = CrossAttentionBlock(
            attention=cached_attention,
            normalization=unconditioned_norm,
        )
        block.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        encoder_a = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        encoder_b = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output_a, _ = block(
            hidden_states=hidden_states, encoder_hidden_states=encoder_a
        )
        output_b, _ = block(
            hidden_states=hidden_states, encoder_hidden_states=encoder_b
        )
        assert not torch.allclose(output_a, output_b)

    def test_ada_norm_conditioning_affects_output(
        self,
        cached_attention: CachedAttention,
        ada_norm_no_gate: AdaNorm,
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        block = CrossAttentionBlock(
            attention=cached_attention,
            normalization=ada_norm_no_gate,
        )
        reinit_modulation_layers(block)
        hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        encoder_hidden_states = sequence_tensor_factory(
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
        output_a, _ = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            conditioning=conditioning_a,
        )
        output_b, _ = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            conditioning=conditioning_b,
        )
        assert not torch.allclose(output_a, output_b)

    def test_cached_cross_kv_skips_encoder_projection(
        self,
        cached_attention: CachedAttention,
        unconditioned_norm,
        sequence_tensor_factory: Callable[..., torch.Tensor],
        precomputed_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ):
        block = CrossAttentionBlock(
            attention=cached_attention,
            normalization=unconditioned_norm,
        )
        block.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        cross_keys, cross_values = precomputed_kv_factory(
            batch_size=2, key_value_length=6
        )
        cache = LayerKVCache(
            self_attention_keys=torch.empty(2, 4, 0, 8),
            self_attention_values=torch.empty(2, 4, 0, 8),
            cross_attention_keys=cross_keys,
            cross_attention_values=cross_values,
        )
        # With cache: encoder_hidden_states=None, uses cached K/V
        output_cached, _ = block(
            hidden_states=hidden_states,
            encoder_hidden_states=None,
            layer_cache=cache,
        )
        assert output_cached.shape == (2, 4, EMBEDDING_DIMENSION)

    def test_attention_mask_affects_output(
        self,
        cached_attention: CachedAttention,
        unconditioned_norm,
        sequence_tensor_factory: Callable[..., torch.Tensor],
        attention_mask_factory: Callable[..., torch.Tensor],
    ):
        block = CrossAttentionBlock(
            attention=cached_attention,
            normalization=unconditioned_norm,
        )
        block.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        encoder_hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        mask = attention_mask_factory(batch_size=2, query_length=4, key_length=6)
        mask[:, :, :, 3:] = True
        output_masked, _ = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=mask,
        )
        output_unmasked, _ = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )
        assert not torch.allclose(output_masked, output_unmasked)
