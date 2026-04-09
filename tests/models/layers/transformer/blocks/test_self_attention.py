"""Tests for versatil.models.layers.transformer.block.self_attention module."""

import unittest.mock
from collections.abc import Callable

import torch

from tests.models.layers.conftest import reinit_modulation_layers
from versatil.models.layers.normalization.ada_norm import AdaNorm
from versatil.models.layers.transformer.attention.cached_attention import (
    CachedAttention,
)
from versatil.models.layers.transformer.block.self_attention import (
    SelfAttentionBlock,
)
from versatil.models.layers.transformer.cache.generation import (
    initialize_generation_cache,
)

EMBEDDING_DIMENSION = 32
NUMBER_OF_HEADS = 4


class TestSelfAttentionBlockForward:
    def test_cache_none_when_no_generation_cache_provided(
        self,
        cached_attention: CachedAttention,
        unconditioned_norm,
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        block = SelfAttentionBlock(
            attention=cached_attention,
            normalization=unconditioned_norm,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        _, cache = block(hidden_states=hidden_states)
        assert cache is None

    def test_cache_accumulates_sequence_length(
        self,
        cached_attention: CachedAttention,
        unconditioned_norm,
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        block = SelfAttentionBlock(
            attention=cached_attention,
            normalization=unconditioned_norm,
        )
        block.eval()
        head_dimension = EMBEDDING_DIMENSION // NUMBER_OF_HEADS
        layer_caches = initialize_generation_cache(
            batch_size=2,
            num_layers=1,
            num_heads=NUMBER_OF_HEADS,
            head_dimension=head_dimension,
            device=torch.device("cpu"),
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        _, cache = block(
            hidden_states=hidden_states,
            generation_cache=layer_caches[0],
        )
        assert cache.keys.shape[2] == 4

    def test_ada_norm_different_conditioning_produces_different_outputs(
        self,
        cached_attention: CachedAttention,
        ada_norm_no_gate: AdaNorm,
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        block = SelfAttentionBlock(
            attention=cached_attention,
            normalization=ada_norm_no_gate,
        )
        reinit_modulation_layers(block)
        hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        conditioning_a = condition_factory(
            batch_size=2, condition_dim=EMBEDDING_DIMENSION
        )
        conditioning_b = condition_factory(
            batch_size=2, condition_dim=EMBEDDING_DIMENSION
        )
        output_a, _ = block(hidden_states=hidden_states, conditioning=conditioning_a)
        output_b, _ = block(hidden_states=hidden_states, conditioning=conditioning_b)
        assert not torch.allclose(output_a, output_b)

    def test_adaln_zero_gate_makes_output_equal_input_at_init(
        self,
        cached_attention: CachedAttention,
        ada_norm_with_gate: AdaNorm,
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        # With use_gate=True and zero init, gate=0 →
        # output = input + 0 * attention(norm(input)) = input
        block = SelfAttentionBlock(
            attention=cached_attention,
            normalization=ada_norm_with_gate,
        )
        block.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        conditioning = condition_factory(
            batch_size=2, condition_dim=EMBEDDING_DIMENSION
        )
        output, _ = block(hidden_states=hidden_states, conditioning=conditioning)
        assert torch.allclose(output, hidden_states, atol=1e-6)

    def test_attention_mask_affects_output(
        self,
        cached_attention: CachedAttention,
        unconditioned_norm,
        sequence_tensor_factory: Callable[..., torch.Tensor],
        attention_mask_factory: Callable[..., torch.Tensor],
    ):
        block = SelfAttentionBlock(
            attention=cached_attention,
            normalization=unconditioned_norm,
        )
        block.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        causal_mask = attention_mask_factory(
            batch_size=2, query_length=4, key_length=4, causal=True
        )
        output_no_mask, _ = block(hidden_states=hidden_states)
        output_causal, _ = block(
            hidden_states=hidden_states, attention_mask=causal_mask
        )
        assert not torch.allclose(output_no_mask, output_causal)

    def test_positional_encoding_passed_to_attention(
        self,
        cached_attention: CachedAttention,
        unconditioned_norm,
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        block = SelfAttentionBlock(
            attention=cached_attention,
            normalization=unconditioned_norm,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        mock_rope = unittest.mock.MagicMock()
        with unittest.mock.patch.object(
            block.attention, "forward", wraps=block.attention.forward
        ) as mock_forward:
            block(
                hidden_states=hidden_states,
                positional_encoding=mock_rope,
            )
            call_kwargs = mock_forward.call_args.kwargs
            assert call_kwargs["positional_encoding"] is mock_rope
