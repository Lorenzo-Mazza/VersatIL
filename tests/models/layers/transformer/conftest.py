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
from versatil.models.layers.transformer.attention.cached_attention import (
    CachedAttention,
)

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
