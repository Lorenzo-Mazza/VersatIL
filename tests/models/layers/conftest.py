"""Shared fixtures for layer tests."""

from collections.abc import Callable

import numpy as np
import pytest
import torch
import torch.nn as nn

from versatil.models.layers.modulation.conditional_modulation import (
    ConditionalModulation,
)
from versatil.models.layers.normalization.ada_norm import AdaNorm


@pytest.fixture
def condition_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for conditioning vectors (B, condition_dim)."""

    def factory(
        batch_size: int = 2,
        condition_dim: int = 32,
    ) -> torch.Tensor:
        data = rng.standard_normal((batch_size, condition_dim)).astype(np.float32)
        return torch.from_numpy(data)

    return factory


@pytest.fixture
def flat_tensor_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for flat tensors (B, D)."""

    def factory(
        batch_size: int = 2,
        feature_dimension: int = 32,
    ) -> torch.Tensor:
        data = rng.standard_normal((batch_size, feature_dimension)).astype(np.float32)
        return torch.from_numpy(data)

    return factory


@pytest.fixture
def sequence_tensor_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for sequence tensors (B, S, D)."""

    def factory(
        batch_size: int = 2,
        sequence_length: int = 4,
        embedding_dimension: int = 32,
    ) -> torch.Tensor:
        data = rng.standard_normal(
            (batch_size, sequence_length, embedding_dimension)
        ).astype(np.float32)
        return torch.from_numpy(data)

    return factory


@pytest.fixture
def nchw_tensor_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for channels-first spatial tensors (B, C, H, W)."""

    def factory(
        batch_size: int = 2,
        channels: int = 16,
        height: int = 8,
        width: int = 8,
    ) -> torch.Tensor:
        data = rng.standard_normal((batch_size, channels, height, width)).astype(
            np.float32
        )
        return torch.from_numpy(data)

    return factory


@pytest.fixture
def nhwc_tensor_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for channels-last spatial tensors (B, H, W, C)."""

    def factory(
        batch_size: int = 2,
        height: int = 8,
        width: int = 8,
        channels: int = 32,
    ) -> torch.Tensor:
        data = rng.standard_normal((batch_size, height, width, channels)).astype(
            np.float32
        )
        return torch.from_numpy(data)

    return factory


@pytest.fixture
def conv1d_tensor_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for 1D convolution tensors (B, C, T)."""

    def factory(
        batch_size: int = 2,
        channels: int = 16,
        sequence_length: int = 32,
    ) -> torch.Tensor:
        data = rng.standard_normal((batch_size, channels, sequence_length)).astype(
            np.float32
        )
        return torch.from_numpy(data)

    return factory


@pytest.fixture
def timestep_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for integer timestep tensors (B,)."""

    def factory(
        batch_size: int = 2,
        num_train_timesteps: int = 100,
    ) -> torch.Tensor:
        values = rng.integers(low=0, high=num_train_timesteps, size=(batch_size,))
        return torch.from_numpy(values).long()

    return factory


@pytest.fixture
def attention_mask_factory() -> Callable[..., torch.Tensor]:
    """Factory for 4D attention masks (B, 1, Q, K) with True=masked."""

    def factory(
        batch_size: int = 2,
        query_length: int = 4,
        key_length: int = 4,
        causal: bool = False,
    ) -> torch.Tensor:
        if causal:
            mask = torch.triu(
                torch.ones(query_length, key_length, dtype=torch.bool),
                diagonal=1,
            )
            return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        return torch.zeros(batch_size, 1, query_length, key_length, dtype=torch.bool)

    return factory


def reinit_modulation_layers(module: nn.Module) -> None:
    """Re-initialize ConditionalModulation projections with xavier to break zero init.

    At initialization, AdaNorm modulation layers have zero weights, so conditioning
    has no effect. This helper enables conditioning sensitivity for behavioral tests.
    """
    for submodule in module.modules():
        if isinstance(submodule, ConditionalModulation):
            for linear in submodule.projection.modules():
                if isinstance(linear, nn.Linear):
                    nn.init.xavier_uniform_(linear.weight)
                    if linear.bias is not None:
                        nn.init.zeros_(linear.bias)


@pytest.fixture
def ada_norm_factory() -> Callable[..., AdaNorm]:
    """Factory for AdaNorm instances with configurable parameters."""

    def factory(
        condition_dim: int = 32,
        feature_dim: int = 64,
        use_gate: bool = False,
        base_norm: nn.Module | None = None,
        init_strategy: str = "zero",
    ) -> AdaNorm:
        if base_norm is None:
            base_norm = nn.LayerNorm(feature_dim, elementwise_affine=False)
        return AdaNorm(
            base_norm=base_norm,
            condition_dim=condition_dim,
            feature_dim=feature_dim,
            use_gate=use_gate,
            init_strategy=init_strategy,
        )

    return factory
