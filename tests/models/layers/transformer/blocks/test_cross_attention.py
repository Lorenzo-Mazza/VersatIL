"""Tests for versatil.models.layers.transformer.block.cross_attention module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise

import pytest
import torch

from tests.models.layers.conftest import reinit_modulation_layers
from versatil.models.layers.normalization.ada_norm import AdaNorm
from versatil.models.layers.transformer.attention.cached_attention import (
    CachedAttention,
)
from versatil.models.layers.transformer.block.cross_attention import (
    CrossAttentionBlock,
)
from versatil.models.layers.transformer.cache.conditioning import ConditioningLayerCache

EMBEDDING_DIMENSION = 32


class TestCrossAttentionBlockForward:
    @pytest.mark.parametrize(
        "use_encoder_states, use_conditioning_cache, expectation",
        [
            (True, False, does_not_raise()),
            (False, True, does_not_raise()),
            (True, True, does_not_raise()),
            (
                False,
                False,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "Either encoder_hidden_states or conditioning_cache must be provided"
                    ),
                ),
            ),
        ],
        ids=[
            "encoder_states_only",
            "cache_only",
            "both_provided",
            "neither_raises",
        ],
    )
    def test_requires_encoder_states_or_conditioning_cache(
        self,
        cached_attention: CachedAttention,
        unconditioned_norm,
        sequence_tensor_factory: Callable[..., torch.Tensor],
        conditioning_cache_factory: Callable[..., ConditioningLayerCache],
        use_encoder_states: bool,
        use_conditioning_cache: bool,
        expectation,
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
        encoder_hidden_states = (
            sequence_tensor_factory(
                batch_size=2,
                sequence_length=6,
                embedding_dimension=EMBEDDING_DIMENSION,
            )
            if use_encoder_states
            else None
        )
        cache = (
            conditioning_cache_factory(
                batch_size=2,
                number_of_key_value_heads=4,
                memory_length=6,
                head_dimension=8,
            )
            if use_conditioning_cache
            else None
        )
        with expectation:
            output = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                conditioning_cache=cache,
            )
            assert output.shape == (2, 4, EMBEDDING_DIMENSION)

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
        output_a = block(hidden_states=hidden_states, encoder_hidden_states=encoder_a)
        output_b = block(hidden_states=hidden_states, encoder_hidden_states=encoder_b)
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
        output_a = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            conditioning=conditioning_a,
        )
        output_b = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            conditioning=conditioning_b,
        )
        assert not torch.allclose(output_a, output_b)

    def test_cached_conditioning_kv_skips_encoder_projection(
        self,
        cached_attention: CachedAttention,
        unconditioned_norm,
        sequence_tensor_factory: Callable[..., torch.Tensor],
        conditioning_cache_factory: Callable[..., ConditioningLayerCache],
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
        cache = conditioning_cache_factory(
            batch_size=2,
            number_of_key_value_heads=4,
            memory_length=6,
            head_dimension=8,
        )
        output = block(
            hidden_states=hidden_states,
            encoder_hidden_states=None,
            conditioning_cache=cache,
        )
        assert output.shape == (2, 4, EMBEDDING_DIMENSION)

    def test_precompute_kv_shapes(
        self,
        cached_attention: CachedAttention,
        unconditioned_norm,
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        block = CrossAttentionBlock(
            attention=cached_attention,
            normalization=unconditioned_norm,
        )
        encoder_hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        cache = block.precompute_kv(encoded_features=encoder_hidden_states)
        # (B=2, heads=4, S=6, head_dim=8)
        assert cache.keys.shape == (2, 4, 6, 8)
        assert cache.values.shape == (2, 4, 6, 8)

    def test_precompute_kv_matches_fresh_forward(
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
        encoder_hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output_fresh = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )
        cache = block.precompute_kv(encoded_features=encoder_hidden_states)
        output_cached = block(
            hidden_states=hidden_states,
            conditioning_cache=cache,
        )
        assert torch.allclose(output_fresh, output_cached, atol=1e-5)

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
        output_masked = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=mask,
        )
        output_unmasked = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )
        assert not torch.allclose(output_masked, output_unmasked)
