"""Tests for CachedAttention module."""

import pytest
import torch

from versatil.models.layers.constants import AttentionType
from versatil.models.layers.transformer.attention import CachedAttention
from versatil.models.layers.transformer.kv_cache import LayerKVCache


@pytest.mark.unit
class TestCachedAttention:
    """Tests for CachedAttention."""

    def test_initialization_mha(self):
        """Test MHA initialization."""
        attention = CachedAttention(
            embedding_dimension=512,
            number_of_heads=8,
            attention_type=AttentionType.MULTI_HEAD.value,
        )

        assert attention.embedding_dimension == 512
        assert attention.number_of_heads == 8
        assert attention.head_dimension == 64
        assert attention.number_of_key_value_heads == 8
        assert attention.group_size == 1

    def test_initialization_gqa(self):
        """Test GQA initialization."""
        attention = CachedAttention(
            embedding_dimension=512,
            number_of_heads=8,
            number_of_key_value_heads=2,
            attention_type=AttentionType.GROUPED_QUERY.value,
        )

        assert attention.number_of_heads == 8
        assert attention.number_of_key_value_heads == 2
        assert attention.group_size == 4

    def test_forward_without_cache(self):
        """Test forward pass without caching."""
        batch_size, seq_len, embedding_dim = 2, 10, 512
        num_heads = 8

        attention = CachedAttention(
            embedding_dimension=embedding_dim,
            number_of_heads=num_heads,
        )

        query_input = torch.randn(batch_size, seq_len, embedding_dim)
        key_input = torch.randn(batch_size, seq_len, embedding_dim)
        value_input = torch.randn(batch_size, seq_len, embedding_dim)

        output, cache = attention(
            query_input=query_input,
            key_input=key_input,
            value_input=value_input,
            use_cache=False,
        )

        assert output.shape == (batch_size, seq_len, embedding_dim)
        assert cache is None

    def test_forward_with_cache_initialization(self):
        """Test forward pass with cache initialization."""
        batch_size, seq_len, embedding_dim = 2, 10, 512
        num_heads = 8
        head_dim = embedding_dim // num_heads

        attention = CachedAttention(
            embedding_dimension=embedding_dim,
            number_of_heads=num_heads,
        )

        query_input = torch.randn(batch_size, seq_len, embedding_dim)

        output, cache = attention(
            query_input=query_input,
            key_input=query_input,
            value_input=query_input,
            use_cache=True,
        )

        assert output.shape == (batch_size, seq_len, embedding_dim)
        assert cache is not None
        assert cache.self_attention_keys.shape == (batch_size, num_heads, seq_len, head_dim)
        assert cache.self_attention_values.shape == (batch_size, num_heads, seq_len, head_dim)

    def test_forward_with_existing_cache(self):
        """Test forward pass with existing cache (autoregressive generation)."""
        batch_size, initial_len, new_len, embedding_dim = 2, 10, 1, 512
        num_heads = 8
        head_dim = embedding_dim // num_heads

        attention = CachedAttention(
            embedding_dimension=embedding_dim,
            number_of_heads=num_heads,
        )

        # Initial forward to create cache
        initial_input = torch.randn(batch_size, initial_len, embedding_dim)
        _, cache = attention(
            query_input=initial_input,
            key_input=initial_input,
            value_input=initial_input,
            use_cache=True,
        )

        # Autoregressive step with cache
        new_input = torch.randn(batch_size, new_len, embedding_dim)
        output, new_cache = attention(
            query_input=new_input,
            key_input=new_input,
            value_input=new_input,
            layer_cache=cache,
            use_cache=True,
        )

        assert output.shape == (batch_size, new_len, embedding_dim)
        assert new_cache.self_attention_keys.shape == (batch_size, num_heads, initial_len + new_len, head_dim)
        assert new_cache.self_attention_values.shape == (batch_size, num_heads, initial_len + new_len, head_dim)

    def test_gqa_cache_compact_storage(self):
        """Test that GQA stores compact K/V in cache."""
        batch_size, seq_len, embedding_dim = 2, 10, 512
        num_heads, num_kv_heads = 8, 2
        head_dim = embedding_dim // num_heads

        attention = CachedAttention(
            embedding_dimension=embedding_dim,
            number_of_heads=num_heads,
            number_of_key_value_heads=num_kv_heads,
            attention_type=AttentionType.GROUPED_QUERY.value,
        )

        query_input = torch.randn(batch_size, seq_len, embedding_dim)

        output, cache = attention(
            query_input=query_input,
            key_input=query_input,
            value_input=query_input,
            use_cache=True,
        )

        # Cache should store compact KV (num_kv_heads, not num_heads)
        assert cache.self_attention_keys.shape == (batch_size, num_kv_heads, seq_len, head_dim)
        assert cache.self_attention_values.shape == (batch_size, num_kv_heads, seq_len, head_dim)

    def test_cross_attention_with_precomputed_kv(self):
        """Test cross-attention with precomputed K/V in cache."""
        batch_size, query_len, kv_len, embedding_dim = 2, 5, 20, 512
        num_heads = 8
        head_dim = embedding_dim // num_heads

        attention = CachedAttention(
            embedding_dimension=embedding_dim,
            number_of_heads=num_heads,
        )

        # Precomputed cross K/V
        cross_keys = torch.randn(batch_size, num_heads, kv_len, head_dim)
        cross_values = torch.randn(batch_size, num_heads, kv_len, head_dim)

        cache = LayerKVCache(
            self_attention_keys=torch.empty(batch_size, num_heads, 0, head_dim),
            self_attention_values=torch.empty(batch_size, num_heads, 0, head_dim),
            cross_attention_keys=cross_keys,
            cross_attention_values=cross_values,
        )

        query_input = torch.randn(batch_size, query_len, embedding_dim)

        output, _ = attention(
            query_input=query_input,
            layer_cache=cache,
            use_cross_attention_cache=True,
        )

        assert output.shape == (batch_size, query_len, embedding_dim)

    def test_attention_mask(self):
        """Test attention with causal mask."""
        batch_size, seq_len, embedding_dim = 2, 10, 512
        num_heads = 8

        attention = CachedAttention(
            embedding_dimension=embedding_dim,
            number_of_heads=num_heads,
        )

        query_input = torch.randn(batch_size, seq_len, embedding_dim)

        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        output, _ = attention(
            query_input=query_input,
            key_input=query_input,
            value_input=query_input,
            attention_mask=causal_mask,
        )

        assert output.shape == (batch_size, seq_len, embedding_dim)