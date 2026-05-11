"""Shared fixtures for versatil.metrics tests."""

from collections.abc import Callable

import numpy as np
import pytest
import torch


@pytest.fixture
def latent_sample_factory(rng: np.random.Generator) -> Callable[..., torch.Tensor]:
    def factory(
        batch_size: int = 16,
        latent_dimension: int = 8,
    ) -> torch.Tensor:
        data = rng.standard_normal((batch_size, latent_dimension)).astype(np.float32)
        return torch.from_numpy(data)

    return factory
