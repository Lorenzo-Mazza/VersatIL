"""Shared fixtures for diffusion transformer layer tests."""

from collections.abc import Callable

import numpy as np
import pytest
import torch


@pytest.fixture
def continuous_timestep_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for diffusion timestep tensors (B,)."""

    def factory(
        batch_size: int = 2,
    ) -> torch.Tensor:
        data = rng.uniform(low=0.0, high=1.0, size=(batch_size,)).astype(np.float32)
        return torch.from_numpy(data)

    return factory
