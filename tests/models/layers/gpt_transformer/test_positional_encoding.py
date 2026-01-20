"""Tests for positional encoding factory and utilities."""

import pytest
import torch

from versatil.models.layers.constants import PositionalEncodingType
from versatil.models.layers.transformer.positional_encoding import (
    apply_rope_positional_encoding,
    create_positional_encoding,
)
from versatil.models.layers.positional_encoding.rotary import RotaryPositionalEncoding1D
from versatil.models.layers.positional_encoding.sinusoidal import SinusoidalPositionalEncoding1D


@pytest.mark.unit
class TestCreatePositionalEncoding:
    """Tests for create_positional_encoding factory function."""

    def test_create_sinusoidal(self):
        """Test creation of SinusoidalPositionalEncoding1D."""
        embedding_dim = 512
        max_len = 2048

        encoding = create_positional_encoding(
            encoding_type=PositionalEncodingType.SINUSOIDAL.value,
            embedding_dimension=embedding_dim,
            maximum_length=max_len,
        )

        assert isinstance(encoding, SinusoidalPositionalEncoding1D)
        assert encoding.embedding_dimension == embedding_dim
        assert encoding.maximum_length == max_len

    def test_create_rope(self):
        """Test creation of RotaryPositionalEncoding1D."""
        embedding_dim = 512
        max_len = 2048
        num_heads = 8

        encoding = create_positional_encoding(
            encoding_type=PositionalEncodingType.ROPE.value,
            embedding_dimension=embedding_dim,
            maximum_length=max_len,
            num_heads=num_heads,
        )

        assert isinstance(encoding, RotaryPositionalEncoding1D)
        assert encoding.embedding_dimension == embedding_dim
        assert encoding.num_heads == num_heads

    def test_create_rope_with_custom_base_frequency(self):
        """Test creation of RoPE with custom base frequency."""
        base_freq = 50000.0

        encoding = create_positional_encoding(
            encoding_type=PositionalEncodingType.ROPE.value,
            embedding_dimension=512,
            maximum_length=2048,
            num_heads=8,
            base_frequency=base_freq,
        )

        # Base frequency affects the computed frequencies
        # Just verify the encoding was created successfully
        assert isinstance(encoding, RotaryPositionalEncoding1D)
        assert encoding.frequencies is not None

    def test_raises_error_rope_without_num_heads(self):
        """Test that error is raised when RoPE is created without num_heads."""
        with pytest.raises(ValueError, match="num_heads is required for RoPE"):
            create_positional_encoding(
                encoding_type=PositionalEncodingType.ROPE.value,
                embedding_dimension=512,
                maximum_length=2048,
            )

    def test_raises_error_unsupported_type(self):
        """Test that error is raised for unsupported encoding type."""
        with pytest.raises(ValueError, match="Unsupported positional encoding type"):
            create_positional_encoding(
                encoding_type="invalid_type",
                embedding_dimension=512,
                maximum_length=2048,
            )


@pytest.mark.unit
class TestApplyPositionalEncoding:
    """Tests for apply_positional_encoding function."""

    def test_apply_rope(self):
        """Test applying RoPE to queries and keys."""
        batch_size, num_heads, seq_len, head_dim = 2, 8, 10, 64
        embedding_dim = num_heads * head_dim

        encoding = create_positional_encoding(
            encoding_type=PositionalEncodingType.ROPE.value,
            embedding_dimension=embedding_dim,
            maximum_length=2048,
            num_heads=num_heads,
        )

        queries = torch.randn(batch_size, num_heads, seq_len, head_dim)
        keys = torch.randn(batch_size, num_heads, seq_len, head_dim)

        queries_with_pos, keys_with_pos = apply_rope_positional_encoding(
            queries=queries,
            keys=keys,
            positional_encoding=encoding,
        )

        assert queries_with_pos.shape == (batch_size, num_heads, seq_len, head_dim)
        assert keys_with_pos.shape == (batch_size, num_heads, seq_len, head_dim)
        # RoPE should modify the tensors
        assert not torch.allclose(queries_with_pos, queries)
        assert not torch.allclose(keys_with_pos, keys)

    def test_apply_rope_with_cache_position(self):
        """Test applying RoPE with non-zero cache position."""
        batch_size, num_heads, seq_len, head_dim = 2, 8, 1, 64
        embedding_dim = num_heads * head_dim
        cache_position = 10

        encoding = create_positional_encoding(
            encoding_type=PositionalEncodingType.ROPE.value,
            embedding_dimension=embedding_dim,
            maximum_length=2048,
            num_heads=num_heads,
        )

        queries = torch.randn(batch_size, num_heads, seq_len, head_dim)
        keys = torch.randn(batch_size, num_heads, seq_len, head_dim)

        queries_with_pos, keys_with_pos = apply_rope_positional_encoding(
            queries=queries,
            keys=keys,
            positional_encoding=encoding,
            cache_position=cache_position,
        )

        assert queries_with_pos.shape == (batch_size, num_heads, seq_len, head_dim)
        assert keys_with_pos.shape == (batch_size, num_heads, seq_len, head_dim)

    def test_apply_sinusoidal(self):
        """Test applying Sinusoidal encoding (no-op in attention)."""
        batch_size, num_heads, seq_len, head_dim = 2, 8, 10, 64

        encoding = create_positional_encoding(
            encoding_type=PositionalEncodingType.SINUSOIDAL.value,
            embedding_dimension=num_heads * head_dim,
            maximum_length=2048,
        )

        queries = torch.randn(batch_size, num_heads, seq_len, head_dim)
        keys = torch.randn(batch_size, num_heads, seq_len, head_dim)

        queries_with_pos, keys_with_pos = apply_rope_positional_encoding(
            queries=queries,
            keys=keys,
            positional_encoding=encoding,
        )

        # Sinusoidal is applied before attention, so should be unchanged here
        assert torch.allclose(queries_with_pos, queries)
        assert torch.allclose(keys_with_pos, keys)

    def test_rope_different_positions_different_rotations(self):
        """Test that RoPE applies different rotations at different positions."""
        batch_size, num_heads, seq_len, head_dim = 1, 4, 5, 32
        embedding_dim = num_heads * head_dim

        encoding = create_positional_encoding(
            encoding_type=PositionalEncodingType.ROPE.value,
            embedding_dimension=embedding_dim,
            maximum_length=2048,
            num_heads=num_heads,
        )

        # Same input at different positions
        queries = torch.ones(batch_size, num_heads, seq_len, head_dim)

        queries_with_pos, _ = apply_rope_positional_encoding(
            queries=queries,
            keys=queries.clone(),
            positional_encoding=encoding,
            cache_position=0,
        )

        # Check that different positions have different values after RoPE
        # Compare first and last position
        first_position = queries_with_pos[0, 0, 0, :]
        last_position = queries_with_pos[0, 0, -1, :]
        assert not torch.allclose(first_position, last_position)

    def test_apply_rope_multi_token_sequence(self):
        """Test applying RoPE to multi-token sequences."""
        batch_size, num_heads, seq_len, head_dim = 2, 8, 20, 64
        embedding_dim = num_heads * head_dim

        encoding = create_positional_encoding(
            encoding_type=PositionalEncodingType.ROPE.value,
            embedding_dimension=embedding_dim,
            maximum_length=2048,
            num_heads=num_heads,
        )

        queries = torch.randn(batch_size, num_heads, seq_len, head_dim)
        keys = torch.randn(batch_size, num_heads, seq_len, head_dim)

        queries_with_pos, keys_with_pos = apply_rope_positional_encoding(
            queries=queries,
            keys=keys,
            positional_encoding=encoding,
            cache_position=0,
        )

        assert queries_with_pos.shape == (batch_size, num_heads, seq_len, head_dim)
        assert keys_with_pos.shape == (batch_size, num_heads, seq_len, head_dim)
        assert not torch.isnan(queries_with_pos).any()
        assert not torch.isnan(keys_with_pos).any()

    def test_apply_unknown_encoding_returns_unchanged(self):
        """Test that unknown encoding type returns tensors unchanged."""
        batch_size, num_heads, seq_len, head_dim = 2, 8, 10, 64

        queries = torch.randn(batch_size, num_heads, seq_len, head_dim)
        keys = torch.randn(batch_size, num_heads, seq_len, head_dim)

        # Pass a mock encoding that's not Sinusoidal or RoPE
        class UnknownEncoding(torch.nn.Module):
            pass

        queries_with_pos, keys_with_pos = apply_rope_positional_encoding(
            queries=queries,
            keys=keys,
            positional_encoding=UnknownEncoding(),
        )

        # Should return unchanged
        assert torch.allclose(queries_with_pos, queries)
        assert torch.allclose(keys_with_pos, keys)