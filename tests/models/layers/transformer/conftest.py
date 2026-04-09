"""Shared fixtures for transformer layer tests."""

from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from versatil.models.layers.constants import AttentionType
from versatil.models.layers.positional_encoding.rotary import (
    RotaryPositionalEncoding1D,
)
from versatil.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding1D,
)
from versatil.models.layers.transformer.attention.cached_attention import (
    CachedAttention,
)
from versatil.models.layers.transformer.cache.conditioning import ConditioningLayerCache
from versatil.models.layers.transformer.cache.generation import GenerationLayerCache

EMBEDDING_DIMENSION = 32
NUMBER_OF_HEADS = 4
HEAD_DIMENSION = EMBEDDING_DIMENSION // NUMBER_OF_HEADS


@pytest.fixture
def cached_attention_factory() -> Callable[..., CachedAttention]:
    """Factory for CachedAttention modules."""

    def factory(
        embedding_dimension: int = 32,
        number_of_heads: int = 4,
        number_of_key_value_heads: int | None = None,
        head_dimension: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        attention_type: str = AttentionType.MULTI_HEAD.value,
    ) -> CachedAttention:
        return CachedAttention(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_key_value_heads,
            head_dimension=head_dimension,
            dropout=dropout,
            bias=bias,
            attention_type=attention_type,
        )

    return factory


@pytest.fixture
def precomputed_kv_factory(
    rng: np.random.Generator,
) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    """Factory for precomputed K/V tensors in head-split shape (B, heads, S, head_dim)."""

    def factory(
        batch_size: int = 2,
        key_value_length: int = 6,
        number_of_heads: int = NUMBER_OF_HEADS,
        head_dimension: int = HEAD_DIMENSION,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (batch_size, number_of_heads, key_value_length, head_dimension)
        keys = torch.from_numpy(rng.standard_normal(shape).astype(np.float32))
        values = torch.from_numpy(rng.standard_normal(shape).astype(np.float32))
        return keys, values

    return factory


@pytest.fixture
def generation_cache_factory(
    precomputed_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> Callable[..., GenerationLayerCache]:
    """Factory for GenerationLayerCache with populated keys/values."""

    def factory(
        batch_size: int = 2,
        number_of_heads: int = NUMBER_OF_HEADS,
        cached_length: int = 3,
        head_dimension: int = HEAD_DIMENSION,
    ) -> GenerationLayerCache:
        keys, values = precomputed_kv_factory(
            batch_size=batch_size,
            key_value_length=cached_length,
            number_of_heads=number_of_heads,
            head_dimension=head_dimension,
        )
        return GenerationLayerCache(keys=keys, values=values)

    return factory


@pytest.fixture
def conditioning_cache_factory(
    precomputed_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> Callable[..., ConditioningLayerCache]:
    """Factory for ConditioningLayerCache with precomputed keys/values."""

    def factory(
        batch_size: int = 2,
        number_of_key_value_heads: int = NUMBER_OF_HEADS,
        memory_length: int = 6,
        head_dimension: int = HEAD_DIMENSION,
    ) -> ConditioningLayerCache:
        keys, values = precomputed_kv_factory(
            batch_size=batch_size,
            key_value_length=memory_length,
            number_of_heads=number_of_key_value_heads,
            head_dimension=head_dimension,
        )
        return ConditioningLayerCache(keys=keys, values=values)

    return factory


@pytest.fixture
def conditioning_cache_with_queries_factory(
    rng: np.random.Generator,
    precomputed_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> Callable[..., ConditioningLayerCache]:
    """Factory for ConditioningLayerCache with precomputed queries, keys, and values."""

    def factory(
        batch_size: int = 2,
        number_of_heads: int = NUMBER_OF_HEADS,
        number_of_key_value_heads: int = NUMBER_OF_HEADS,
        sequence_length: int = 6,
        head_dimension: int = HEAD_DIMENSION,
    ) -> ConditioningLayerCache:
        keys, values = precomputed_kv_factory(
            batch_size=batch_size,
            key_value_length=sequence_length,
            number_of_heads=number_of_key_value_heads,
            head_dimension=head_dimension,
        )
        query_shape = (batch_size, number_of_heads, sequence_length, head_dimension)
        queries = torch.from_numpy(rng.standard_normal(query_shape).astype(np.float32))
        return ConditioningLayerCache(keys=keys, values=values, queries=queries)

    return factory


@pytest.fixture
def head_split_tensor_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for head-split tensors (B, H, S, D_head)."""

    def factory(
        batch_size: int = 2,
        number_of_heads: int = NUMBER_OF_HEADS,
        sequence_length: int = 6,
        head_dimension: int = HEAD_DIMENSION,
    ) -> torch.Tensor:
        shape = (batch_size, number_of_heads, sequence_length, head_dimension)
        return torch.from_numpy(rng.standard_normal(shape).astype(np.float32))

    return factory


@pytest.fixture
def new_kv_factory(
    precomputed_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    """Factory for new key/value tensors to append to generation cache."""

    def factory(
        batch_size: int = 2,
        number_of_heads: int = NUMBER_OF_HEADS,
        new_length: int = 1,
        head_dimension: int = HEAD_DIMENSION,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return precomputed_kv_factory(
            batch_size=batch_size,
            key_value_length=new_length,
            number_of_heads=number_of_heads,
            head_dimension=head_dimension,
        )

    return factory


@pytest.fixture
def flat_conditioning_cache_factory(
    sequence_tensor_factory: Callable[..., torch.Tensor],
) -> Callable[..., ConditioningLayerCache]:
    """Factory for ConditioningLayerCache with flat (B, S, D) keys/values.

    Used by layers that project conditioning K/V internally (e.g.
    PrecomputedKVCrossAttentionLayer).
    """

    def factory(
        batch_size: int = 2,
        sequence_length: int = 8,
        kv_dimension: int = NUMBER_OF_HEADS * HEAD_DIMENSION,
    ) -> ConditioningLayerCache:
        keys = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=kv_dimension,
        )
        values = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=kv_dimension,
        )
        return ConditioningLayerCache(keys=keys, values=values)

    return factory


@pytest.fixture
def mock_rope_factory() -> Callable[..., MagicMock]:
    """Factory for mock RotaryPositionalEncoding1D that scales Q/K by 0.5.

    The mock passes isinstance checks and wires compute_rotation_components
    and apply_rotation to produce deterministic transformations.
    """

    def factory(head_dimension: int = HEAD_DIMENSION) -> MagicMock:
        mock = MagicMock(spec=RotaryPositionalEncoding1D)
        mock.compute_rotation_components.side_effect = lambda seq_len: (
            torch.zeros(seq_len, head_dimension),
            torch.zeros(seq_len, head_dimension),
        )
        mock.apply_rotation.side_effect = lambda tensor, sine, cosine: tensor * 0.5
        return mock

    return factory


@pytest.fixture
def mock_sinusoidal_factory() -> Callable[..., MagicMock]:
    """Factory for mock SinusoidalPositionalEncoding1D that adds ones to input."""

    def factory(
        embedding_dimension: int = EMBEDDING_DIMENSION,
    ) -> MagicMock:
        mock = MagicMock()
        mock.__class__ = SinusoidalPositionalEncoding1D
        mock.side_effect = lambda x, offset=0: torch.ones_like(x)
        return mock

    return factory
