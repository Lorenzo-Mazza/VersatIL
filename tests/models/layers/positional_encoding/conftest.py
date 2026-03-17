"""Shared fixtures for positional encoding tests."""

from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.models.layers.positional_encoding.base import PositionSource
from versatil.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding1D,
)


@pytest.fixture
def scalar_tensor_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for scalar input tensors with shape (B,)."""

    def factory(
        batch_size: int = 2,
    ) -> torch.Tensor:
        data = rng.standard_normal((batch_size,)).astype(np.float32)
        return torch.from_numpy(data)

    return factory


@pytest.fixture
def sinusoidal_1d_factory() -> Callable[..., SinusoidalPositionalEncoding1D]:
    """Factory for SinusoidalPositionalEncoding1D instances."""

    def factory(
        embedding_dimension: int = 64,
        position_source: str = PositionSource.TENSOR_INDICES.value,
        precompute_encodings: bool = True,
        maximum_length: int | None = 5000,
        learnable_frequencies: bool = False,
        temperature: float = 10000.0,
        mlp_hidden_dimensions: list[int] | None = None,
        mlp_activation: type | None = None,
    ) -> SinusoidalPositionalEncoding1D:
        return SinusoidalPositionalEncoding1D(
            embedding_dimension=embedding_dimension,
            position_source=position_source,
            precompute_encodings=precompute_encodings,
            maximum_length=maximum_length,
            learnable_frequencies=learnable_frequencies,
            temperature=temperature,
            mlp_hidden_dimensions=mlp_hidden_dimensions,
            mlp_activation=mlp_activation,
        )

    return factory
