"""Key-Value cache utilities for autoregressive GPT decoder."""

from dataclasses import dataclass

import torch


@dataclass
class LayerKVCache:
    """Key-Value cache for a single transformer layer.

    Stores cached keys and values from self-attention and optional precomputed cross-attention K/V.

    Attributes:
        self_attention_keys: Cached keys from self-attention (B, kv_heads, T, head_dim)
        self_attention_values: Cached values from self-attention (B, kv_heads, T, head_dim)
        cross_attention_keys: Optional precomputed K from encoded features (B, kv_heads, S, head_dim)
        cross_attention_values: Optional precomputed V from encoded features (B, kv_heads, S, head_dim)
    """

    self_attention_keys: torch.Tensor
    self_attention_values: torch.Tensor
    cross_attention_keys: torch.Tensor | None = None
    cross_attention_values: torch.Tensor | None = None

    def get_length(self) -> int:
        """Get current sequence length in cache.

        Returns:
            Current sequence length (number of cached positions)
        """
        return self.self_attention_keys.shape[2]


@dataclass
class DecoderKVCache:
    """Key-Value cache for the full GPT decoder.

    Manages caches for all layers. Each layer has its own self-attention cache
    and precomputed cross-attention K/V.

    Attributes:
        layers: List of LayerKVCache, one per decoder layer
    """

    layers: list[LayerKVCache]
    key_padding_mask: torch.Tensor | None = (
        None  # (B, cache_len) bool, True = masked (do not attend)
    )

    def get_length(self) -> int:
        """Get current sequence length (from first layer).

        Returns:
            Current sequence length
        """
        if len(self.layers) == 0 or self.layers[0] is None:
            return 0
        return self.layers[0].get_length()


def initialize_decoder_cache(
    batch_size: int,
    num_layers: int,
    num_heads: int,
    head_dimension: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> list[LayerKVCache]:
    """Initialize empty cache for all decoder layers.

    Pre-allocates empty tensors for efficiency during generation.

    Args:
        batch_size: Batch size
        num_layers: Number of decoder layers
        num_heads: Number of attention heads per layer
        head_dimension: Dimension per attention head
        device: Device to allocate tensors on
        dtype: Data type for cache tensors

    Returns:
        List of empty LayerKVCache, one per layer
    """
    caches = []
    for _ in range(num_layers):
        cache = LayerKVCache(
            self_attention_keys=torch.empty(
                batch_size, num_heads, 0, head_dimension, device=device, dtype=dtype
            ),
            self_attention_values=torch.empty(
                batch_size, num_heads, 0, head_dimension, device=device, dtype=dtype
            ),
        )
        caches.append(cache)
    return caches


def update_layer_cache(
    cache: LayerKVCache,
    new_keys: torch.Tensor,
    new_values: torch.Tensor,
) -> LayerKVCache:
    """Update cache with new keys and values.

    Concatenates new key-values along sequence dimension.

    Args:
        cache: Existing cache
        new_keys: New keys to append (B, num_heads, 1, head_dim)
        new_values: New values to append (B, num_heads, 1, head_dim)

    Returns:
        Updated LayerKVCache
    """
    return LayerKVCache(
        self_attention_keys=torch.cat([cache.self_attention_keys, new_keys], dim=2),
        self_attention_values=torch.cat(
            [cache.self_attention_values, new_values], dim=2
        ),
        cross_attention_keys=cache.cross_attention_keys,
        cross_attention_values=cache.cross_attention_values,
    )
