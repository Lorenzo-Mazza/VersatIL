"""Shared fixtures for pooling layer tests."""
from collections.abc import Callable

import numpy as np
import pytest
import torch


@pytest.fixture
def feature_map_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for 4D feature maps (B, C, H, W)."""
    def factory(
        batch_size: int = 2,
        channels: int = 16,
        height: int = 8,
        width: int = 8,
    ) -> torch.Tensor:
        data = rng.standard_normal(
            (batch_size, channels, height, width)
        ).astype(np.float32)
        return torch.from_numpy(data)
    return factory
