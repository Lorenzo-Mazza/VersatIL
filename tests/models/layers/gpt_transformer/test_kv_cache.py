"""Tests for KV cache utilities."""

import pytest
import torch

from refactoring.models.layers.gpt_transformer.kv_cache import (
    DecoderKVCache,
    LayerKVCache,
    initialize_decoder_cache,
    update_layer_cache,
)


@pytest.mark.unit
class TestLayerKVCache:
    """Tests for LayerKVCache."""

    def test_creation(self):
        """Test LayerKVCache creation."""
        batch_size, num_heads, seq_len, head_dim = 2, 4, 10, 64
        keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
        values = torch.randn(batch_size, num_heads, seq_len, head_dim)

        cache = LayerKVCache(
            self_attention_keys=keys,
            self_attention_values=values,
        )

        assert cache.self_attention_keys.shape == (batch_size, num_heads, seq_len, head_dim)
        assert cache.self_attention_values.shape == (batch_size, num_heads, seq_len, head_dim)
        assert cache.cross_attention_keys is None
        assert cache.cross_attention_values is None

    def test_creation_with_cross_attention(self):
        """Test LayerKVCache creation with cross-attention K/V."""
        batch_size, num_heads, seq_len, head_dim = 2, 4, 10, 64
        cross_seq_len = 20

        self_keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
        self_values = torch.randn(batch_size, num_heads, seq_len, head_dim)
        cross_keys = torch.randn(batch_size, num_heads, cross_seq_len, head_dim)
        cross_values = torch.randn(batch_size, num_heads, cross_seq_len, head_dim)

        cache = LayerKVCache(
            self_attention_keys=self_keys,
            self_attention_values=self_values,
            cross_attention_keys=cross_keys,
            cross_attention_values=cross_values,
        )

        assert cache.cross_attention_keys.shape == (batch_size, num_heads, cross_seq_len, head_dim)
        assert cache.cross_attention_values.shape == (batch_size, num_heads, cross_seq_len, head_dim)

    def test_get_length(self):
        """Test get_length method."""
        batch_size, num_heads, seq_len, head_dim = 2, 4, 10, 64
        keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
        values = torch.randn(batch_size, num_heads, seq_len, head_dim)

        cache = LayerKVCache(
            self_attention_keys=keys,
            self_attention_values=values,
        )

        assert cache.get_length() == seq_len


@pytest.mark.unit
class TestDecoderKVCache:
    """Tests for DecoderKVCache."""

    def test_creation(self):
        """Test DecoderKVCache creation."""
        num_layers = 3
        batch_size, num_heads, seq_len, head_dim = 2, 4, 10, 64

        layers = []
        for _ in range(num_layers):
            keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
            values = torch.randn(batch_size, num_heads, seq_len, head_dim)
            layers.append(LayerKVCache(
                self_attention_keys=keys,
                self_attention_values=values,
            ))

        cache = DecoderKVCache(layers=layers)

        assert len(cache.layers) == num_layers
        assert cache.get_length() == seq_len

    def test_get_length_empty(self):
        """Test get_length with empty cache."""
        cache = DecoderKVCache(layers=[])
        assert cache.get_length() == 0


@pytest.mark.unit
class TestInitializeDecoderCache:
    """Tests for initialize_decoder_cache."""

    def test_initialization(self):
        """Test cache initialization."""
        batch_size, num_layers, num_heads, head_dim = 2, 3, 4, 64

        caches = initialize_decoder_cache(
            batch_size=batch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dimension=head_dim,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        assert len(caches) == num_layers

        for cache in caches:
            assert isinstance(cache, LayerKVCache)
            assert cache.self_attention_keys.shape == (batch_size, num_heads, 0, head_dim)
            assert cache.self_attention_values.shape == (batch_size, num_heads, 0, head_dim)
            assert cache.get_length() == 0


@pytest.mark.unit
class TestUpdateLayerCache:
    """Tests for update_layer_cache."""

    def test_update(self):
        """Test cache update."""
        batch_size, num_heads, seq_len, head_dim = 2, 4, 10, 64

        # Create initial cache
        old_keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
        old_values = torch.randn(batch_size, num_heads, seq_len, head_dim)
        cache = LayerKVCache(
            self_attention_keys=old_keys,
            self_attention_values=old_values,
        )

        # Create new K/V to append
        new_keys = torch.randn(batch_size, num_heads, 1, head_dim)
        new_values = torch.randn(batch_size, num_heads, 1, head_dim)

        # Update cache
        updated_cache = update_layer_cache(cache, new_keys, new_values)

        assert updated_cache.get_length() == seq_len + 1
        assert updated_cache.self_attention_keys.shape == (batch_size, num_heads, seq_len + 1, head_dim)
        assert updated_cache.self_attention_values.shape == (batch_size, num_heads, seq_len + 1, head_dim)

        # Verify old values are preserved
        assert torch.allclose(updated_cache.self_attention_keys[:, :, :seq_len], old_keys)
        assert torch.allclose(updated_cache.self_attention_values[:, :, :seq_len], old_values)

        # Verify new values are appended
        assert torch.allclose(updated_cache.self_attention_keys[:, :, seq_len:], new_keys)
        assert torch.allclose(updated_cache.self_attention_values[:, :, seq_len:], new_values)