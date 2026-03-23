"""Shared fixtures for post-training compression tests."""

from collections.abc import Callable

import numpy as np
import pytest
import torch


@pytest.fixture
def spatial_input_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for spatial input tensors (B, C, H, W)."""

    def factory(
        batch_size: int = 2,
        channels: int = 3,
        height: int = 8,
        width: int = 8,
    ) -> torch.Tensor:
        return torch.from_numpy(
            rng.standard_normal((batch_size, channels, height, width)).astype(
                np.float32
            )
        )

    return factory
