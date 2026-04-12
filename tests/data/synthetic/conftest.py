"""Shared fixtures for synthetic data tests."""

from collections.abc import Callable

import numpy as np
import pytest

from versatil.data.synthetic.constants import SyntheticTaskName
from versatil.data.synthetic.generators import generate_task_episodes


@pytest.fixture
def episode_factory(
    rng: np.random.Generator,
) -> Callable[..., list[dict[str, np.ndarray]]]:
    """Factory that generates small synthetic episode batches for testing."""

    def factory(
        task_name: str = SyntheticTaskName.CIRCLE.value,
        num_episodes: int = 6,
        image_size: int = 16,
        trajectory_length: int = 10,
        num_modes: int = 3,
        noise_std: float = 0.01,
        num_styles: int = 4,
    ) -> list[dict[str, np.ndarray]]:
        seed = int(rng.integers(0, 2**31))
        return generate_task_episodes(
            task_name=task_name,
            num_episodes=num_episodes,
            seed=seed,
            image_size=image_size,
            num_modes=num_modes,
            trajectory_length=trajectory_length,
            noise_std=noise_std,
            num_styles=num_styles,
        )

    return factory


@pytest.fixture
def trajectory_factory(
    rng: np.random.Generator,
) -> Callable[..., np.ndarray]:
    """Factory returning random (num_points, 2) float32 trajectories in [0, 1]."""

    def factory(
        num_points: int = 10,
    ) -> np.ndarray:
        return rng.uniform(0.0, 1.0, size=(num_points, 2)).astype(np.float32)

    return factory
