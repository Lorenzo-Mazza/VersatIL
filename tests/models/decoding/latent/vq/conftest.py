"""Shared fixtures for vector quantization tests."""

from collections.abc import Callable

import numpy as np
import pytest
import torch


@pytest.fixture
def z_e_factory(rng: np.random.Generator) -> Callable[..., torch.Tensor]:
    """Factory returning random encoder output tensors in code space."""

    def factory(
        batch_size: int = 8,
        dim: int = 8,
    ) -> torch.Tensor:
        return torch.from_numpy(
            rng.standard_normal((batch_size, dim)).astype(np.float32)
        )

    return factory
