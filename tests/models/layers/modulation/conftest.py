"""Shared fixtures for modulation layer tests."""
from collections.abc import Callable

import numpy as np
import pytest
import torch


@pytest.fixture
def condition_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for conditioning vectors (B, condition_dim)."""
    def factory(
        batch_size: int = 2,
        condition_dim: int = 32,
    ) -> torch.Tensor:
        data = rng.standard_normal(
            (batch_size, condition_dim)
        ).astype(np.float32)
        return torch.from_numpy(data)
    return factory
