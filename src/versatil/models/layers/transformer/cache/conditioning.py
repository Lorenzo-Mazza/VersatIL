"""Conditioning cache for static context (observations, encoder features).

Conditioning information does not change between forward calls — whether
consumed via cross-attention, joint attention, or prefix concatenation.
The cache is computed once before generation starts and reused unchanged
across all forward calls.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@dataclass
class ConditioningLayerCache:
    """Per-layer K/V cache for static conditioning (computed once, reused).

    Stores precomputed projections for conditioning information that does not
    change between forward calls. Populated once before generation starts.

    Attributes:
        keys: Precomputed keys (B, kv_heads, S, head_dim).
        values: Precomputed values (B, kv_heads, S, head_dim).
        queries: Precomputed queries (B, heads, S, head_dim). Populated when
            conditioning participates bidirectionally (both streams attend to
            each other). None when conditioning is only attended to.
    """

    keys: torch.Tensor
    values: torch.Tensor
    queries: torch.Tensor | None = None


@dataclass
class ConditioningCache:
    """Multi-layer conditioning cache for static context.

    Attributes:
        layers: One ConditioningLayerCache (or None) per decoder layer.
    """

    layers: list[ConditioningLayerCache | None]

    def __getitem__(self, index: int) -> ConditioningLayerCache | None:
        """Get cache layer by slicing."""
        return self.layers[index]


@runtime_checkable
class CacheableLayer(Protocol):
    """Layer that can precompute conditioning K/V projections."""

    def precompute_conditioning_kv(
        self, encoded_features: torch.Tensor
    ) -> ConditioningLayerCache | None:
        """Project encoded features to cacheable conditioning keys and values."""
        ...


def precompute_conditioning(
    layers: Iterable[CacheableLayer],
    encoded_features: torch.Tensor,
) -> ConditioningCache:
    """Precompute conditioning K/V for all layers.

    Calls each layer's precompute_conditioning_kv to project encoder features
    through cross-attention K/V weights once, for reuse across forward calls.

    Args:
        layers: Decoder layers satisfying CacheableLayer protocol.
        encoded_features: Encoder features (B, memory_length, D).

    Returns:
        ConditioningCache with one ConditioningLayerCache per layer.
    """
    return ConditioningCache(
        layers=[layer.precompute_conditioning_kv(encoded_features) for layer in layers]
    )
