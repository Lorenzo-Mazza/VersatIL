"""Shared fixtures for transformer block tests."""

from collections.abc import Callable

import pytest

from versatil.models.layers.normalization.ada_norm import AdaNorm
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.normalization.factory import create_normalization_layer
from versatil.models.layers.normalization.unconditioned_norm import UnconditionedNorm
from versatil.models.layers.transformer.attention.cached_attention import (
    CachedAttention,
)

EMBEDDING_DIMENSION = 32
FEEDFORWARD_DIMENSION = 64
NUMBER_OF_HEADS = 4
HEAD_DIMENSION = EMBEDDING_DIMENSION // NUMBER_OF_HEADS


@pytest.fixture
def cached_attention(
    cached_attention_factory: Callable[..., CachedAttention],
) -> CachedAttention:
    return cached_attention_factory(
        embedding_dimension=EMBEDDING_DIMENSION,
        number_of_heads=NUMBER_OF_HEADS,
    )


@pytest.fixture
def unconditioned_norm() -> UnconditionedNorm:
    return UnconditionedNorm(
        create_normalization_layer(
            normalization_type=NormalizationType.RMS_NORM.value,
            dimension=EMBEDDING_DIMENSION,
        )
    )


@pytest.fixture
def ada_norm_no_gate(
    ada_norm_factory: Callable[..., AdaNorm],
) -> AdaNorm:
    return ada_norm_factory(
        conditioning_dimension=EMBEDDING_DIMENSION,
        feature_dim=EMBEDDING_DIMENSION,
        use_gate=False,
    )


@pytest.fixture
def ada_norm_with_gate(
    ada_norm_factory: Callable[..., AdaNorm],
) -> AdaNorm:
    return ada_norm_factory(
        conditioning_dimension=EMBEDDING_DIMENSION,
        feature_dim=EMBEDDING_DIMENSION,
        use_gate=True,
    )
