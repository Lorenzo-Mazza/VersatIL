from dataclasses import dataclass

import torch


@dataclass
class GenerationLayerCache:
    """Per-layer K/V cache for the main sequence (grows during generation).

    Keys and values accumulate along the sequence dimension as new tokens are
    decoded. Each generation step appends one new position.

    Attributes:
        keys: Cached keys (B, kv_heads, T, head_dim) where T grows each step.
        values: Cached values (B, kv_heads, T, head_dim).
    """

    keys: torch.Tensor
    values: torch.Tensor

    def get_length(self) -> int:
        """Get number of cached positions in the sequence dimension."""
        return self.keys.shape[2]


@dataclass
class GenerationCache:
    """Multi-layer generation cache for an autoregressive decoder.

    Attributes:
        layers: One GenerationLayerCache per decoder layer.
        key_padding_mask: Bool mask (B, cache_len), True = masked (do not attend).
    """

    layers: list[GenerationLayerCache]
    key_padding_mask: torch.Tensor | None = None

    def get_length(self) -> int:
        """Get cached sequence length from the first layer."""
        if len(self.layers) == 0 or self.layers[0] is None:
            return 0
        return self.layers[0].get_length()


def initialize_generation_cache(
    batch_size: int,
    num_layers: int,
    number_of_heads: int,
    head_dimension: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> list[GenerationLayerCache]:
    """Initialize empty generation cache for all decoder layers.

    Pre-allocates empty tensors with zero sequence length for efficiency
    during autoregressive generation.

    Args:
        batch_size: Batch size.
        num_layers: Number of decoder layers.
        number_of_heads: Number of K/V attention heads per layer.
        head_dimension: Dimension per attention head.
        device: Device to allocate tensors on.
        dtype: Data type for cache tensors.

    Returns:
        List of empty GenerationLayerCache, one per layer.
    """
    caches = []
    for _ in range(num_layers):
        cache = GenerationLayerCache(
            keys=torch.empty(
                batch_size,
                number_of_heads,
                0,
                head_dimension,
                device=device,
                dtype=dtype,
            ),
            values=torch.empty(
                batch_size,
                number_of_heads,
                0,
                head_dimension,
                device=device,
                dtype=dtype,
            ),
        )
        caches.append(cache)
    return caches


def update_generation_layer_cache(
    cache: GenerationLayerCache,
    new_keys: torch.Tensor,
    new_values: torch.Tensor,
) -> GenerationLayerCache:
    """Append new keys and values to a generation layer cache.

    Concatenates along the sequence dimension (dim=2).

    Args:
        cache: Existing layer cache.
        new_keys: New keys to append (B, number_of_heads, new_len, head_dim).
        new_values: New values to append (B, number_of_heads, new_len, head_dim).

    Returns:
        New GenerationLayerCache with accumulated K/V.
    """
    return GenerationLayerCache(
        keys=torch.cat([cache.keys, new_keys], dim=2),
        values=torch.cat([cache.values, new_values], dim=2),
    )
