"""Positional encoding for GPT transformer."""
import logging

import torch
import torch.nn as nn

from versatil.models.layers.constants import PositionalEncodingType
from versatil.models.layers.positional_encoding.base import PositionalEncoding1D
from versatil.models.layers.positional_encoding.learned import (
    LearnedPositionalEncoding1D,
)
from versatil.models.layers.positional_encoding.rotary import (
    RotaryPositionalEncoding1D,
)
from versatil.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding1D,
)


def create_positional_encoding(
    encoding_type: str,
    embedding_dimension: int,
    maximum_length: int,
    num_heads: int | None = None,
    base_frequency: float = 10000.0,
    learnable_frequencies: bool = False,
) -> PositionalEncoding1D | RotaryPositionalEncoding1D:
    """Factory function to create positional encoding.

    Args:
        encoding_type: Type of encoding (use PositionalEncodingType enum values)
        embedding_dimension: Model embedding dimension
        maximum_length: Maximum sequence length
        num_heads: Number of attention heads (required for RoPE)
        base_frequency: Base frequency for RoPE
        learnable_frequencies: Whether to make RoPE frequencies learnable

    Returns:
        Positional encoding module

    Raises:
        ValueError: If encoding_type is not supported or required args missing
    """
    if encoding_type == PositionalEncodingType.SINUSOIDAL.value:
        return SinusoidalPositionalEncoding1D(
            embedding_dimension=embedding_dimension,
            maximum_length=maximum_length,
        )
    elif encoding_type == PositionalEncodingType.LEARNED.value:
        return LearnedPositionalEncoding1D(
            embedding_dimension=embedding_dimension,
            maximum_length=maximum_length,
        )
    elif encoding_type == PositionalEncodingType.ROPE.value:
        if num_heads is None:
            raise ValueError("num_heads is required for RoPE positional encoding")
        return RotaryPositionalEncoding1D(
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
            base_frequency=base_frequency,
            learnable_frequencies=learnable_frequencies,
        )
    else:
        raise ValueError(
            f"Unsupported positional encoding type: {encoding_type}. "
            f"Must be one of {[e.value for e in PositionalEncodingType if e.value in ('sinusoidal', 'rope')]}."
        )


def apply_rope_positional_encoding(
    queries: torch.Tensor,
    keys: torch.Tensor,
    positional_encoding: nn.Module,
    cache_position: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply positional encoding to queries and keys.

    Handles both Sinusoidal (added to embeddings) and RoPE (applied via rotation).

    Args:
        queries: Query tensor (B, num_heads, query_len, head_dim)
        keys: Key tensor (B, num_heads, key_len, head_dim) including cached keys
        positional_encoding: Positional encoding module
        cache_position: Starting position for queries (0 for initial forward, cache_len for generation)

    Returns:
        Tuple of (queries_with_pos, keys_with_pos)
    """
    if isinstance(positional_encoding, RotaryPositionalEncoding1D):
        # RoPE: apply rotation to Q and K
        # Both queries and keys are new segments at the same positions
        sequence_length = queries.shape[2]

        # Compute rotation components for positions [cache_position, ..., cache_position + seq_len - 1]
        sine, cosine = positional_encoding.compute_rotation_components(
            cache_position + sequence_length
        )
        sine = sine[cache_position : cache_position + sequence_length]
        cosine = cosine[cache_position : cache_position + sequence_length]

        # Expand for batch and heads: (seq_len, head_dim) -> (1, 1, seq_len, head_dim)
        sine = sine.unsqueeze(0).unsqueeze(0)
        cosine = cosine.unsqueeze(0).unsqueeze(0)

        # Apply rotation to both queries and keys
        queries = positional_encoding.apply_rotation(queries, sine, cosine)
        keys = positional_encoding.apply_rotation(keys, sine, cosine)

        return queries, keys

    else:
        # Unknown type - return unchanged
        logging.warning(
            "Positional encoding module is not an instance of RotaryPositionalEncoding. Skipping."
        )
        return queries, keys
