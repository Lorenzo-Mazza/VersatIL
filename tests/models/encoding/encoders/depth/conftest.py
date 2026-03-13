"""Shared fixtures for depth encoder tests."""
from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.data.constants import Cameras


@pytest.fixture
def rgbd_input_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    """Factory for paired RGB + depth input tensors."""
    def factory(
        rgb_key: str = Cameras.LEFT.value,
        depth_key: str = Cameras.DEPTH.value,
        batch_size: int = 2,
        rgb_channels: int = 3,
        depth_channels: int = 1,
        height: int = 224,
        width: int = 224,
        time_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if time_steps is not None:
            rgb_shape = (batch_size, time_steps, rgb_channels, height, width)
            depth_shape = (batch_size, time_steps, depth_channels, height, width)
        else:
            rgb_shape = (batch_size, rgb_channels, height, width)
            depth_shape = (batch_size, depth_channels, height, width)
        rgb_tensor = torch.from_numpy(
            rng.standard_normal(rgb_shape).astype(np.float32)
        )
        depth_tensor = torch.from_numpy(
            rng.standard_normal(depth_shape).astype(np.float32)
        )
        return {rgb_key: rgb_tensor, depth_key: depth_tensor}
    return factory
