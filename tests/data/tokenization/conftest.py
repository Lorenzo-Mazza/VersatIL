"""Shared fixtures for tokenization tests."""

from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.data.tokenization.binned_value_discretizer import BinnedValueDiscretizer


@pytest.fixture
def binned_value_discretizer_factory() -> Callable[..., BinnedValueDiscretizer]:
    """Factory for BinnedValueDiscretizer with configurable parameters."""

    def factory(
        num_bins: int = 16,
        device: torch.device = torch.device("cpu"),
    ) -> BinnedValueDiscretizer:
        return BinnedValueDiscretizer(num_bins=num_bins, device=device)

    return factory


@pytest.fixture
def fitted_binned_value_discretizer_factory(
    rng, binned_value_discretizer_factory
) -> Callable[..., BinnedValueDiscretizer]:
    """Factory for a fitted BinnedValueDiscretizer with configurable data."""

    def factory(
        num_bins: int = 16,
        num_samples: int = 100,
        num_dimensions: int = 7,
        device: torch.device = torch.device("cpu"),
    ) -> BinnedValueDiscretizer:
        tokenizer = binned_value_discretizer_factory(num_bins=num_bins, device=device)
        data = rng.standard_normal((num_samples, num_dimensions)).astype(np.float32)
        tokenizer.fit(data)
        return tokenizer

    return factory


@pytest.fixture
def action_chunk_factory(rng) -> Callable[..., np.ndarray | torch.Tensor]:
    """Factory for action chunk arrays with configurable shape and format."""

    def factory(
        time_horizon: int = 5,
        action_dimension: int = 7,
        batch_size: int | None = None,
        scale: float = 1.0,
        as_torch: bool = False,
    ) -> np.ndarray | torch.Tensor:
        if batch_size is not None:
            shape = (batch_size, time_horizon, action_dimension)
        else:
            shape = (time_horizon, action_dimension)
        data = rng.standard_normal(shape).astype(np.float32) * scale
        if as_torch:
            return torch.from_numpy(data)
        return data

    return factory


@pytest.fixture
def pad_mask_factory() -> Callable[..., np.ndarray | torch.Tensor]:
    """Factory for padding masks with configurable valid/pad split."""

    def factory(
        total: int = 5,
        num_valid: int = 2,
        as_torch: bool = False,
    ) -> np.ndarray | torch.Tensor:
        mask = np.array([False] * num_valid + [True] * (total - num_valid))
        if as_torch:
            return torch.tensor(mask)
        return mask

    return factory
