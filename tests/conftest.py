"""Root test fixtures shared across the entire test suite."""
import versatil  # noqa: F401 — triggers dotenv loading and cache directory setup

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch

from versatil.metrics.base import LossOutput


@pytest.fixture
def rng() -> np.random.Generator:
    """Fixed-seed RNG for data generators. Fresh instance per test for isolation."""
    return np.random.default_rng(42)


@pytest.fixture
def device() -> torch.device:
    """Get available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size() -> int:
    """Default batch size for tests."""
    return 2


@pytest.fixture
def temporal_length() -> int:
    """Default temporal sequence length."""
    return 2


@pytest.fixture
def image_size() -> tuple[int, int]:
    """Default image size (height, width)."""
    return 224, 224


@pytest.fixture
def loss_output_factory() -> Callable[..., LossOutput]:
    """Factory for LossOutput instances with configurable loss values."""

    def factory(
        total_loss_value: float = 1.0,
        component_losses: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
        device: str = "cpu",
        requires_grad: bool = False,
    ) -> LossOutput:
        total = torch.tensor(
            total_loss_value, device=device, requires_grad=requires_grad
        )
        components = {}
        if component_losses is not None:
            for key, value in component_losses.items():
                components[key] = torch.tensor(value, device=device)
        return LossOutput(
            total_loss=total,
            component_losses=components,
            metadata=metadata if metadata is not None else {},
        )

    return factory


@pytest.fixture
def padding_mask_factory() -> Callable[..., torch.Tensor]:
    """Factory for padding masks (B, S) with True=padded."""

    def factory(
        batch_size: int = 2,
        sequence_length: int = 4,
        padded_from: int | None = None,
        padded_positions: list[list[int]] | None = None,
        mask_last_n: int | None = None,
    ) -> torch.Tensor:
        mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool)
        if padded_positions is not None:
            for batch_index, positions in enumerate(padded_positions):
                for position in positions:
                    mask[batch_index, position] = True
        elif mask_last_n is not None:
            mask[:, -mask_last_n:] = True
        elif padded_from is not None:
            mask[:, padded_from:] = True
        return mask

    return factory


@pytest.fixture
def action_tensor_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for action tensors (B, T, D)."""

    def factory(
        batch_size: int = 2,
        sequence_length: int = 4,
        action_dimension: int = 3,
    ) -> torch.Tensor:
        data = rng.standard_normal(
            (batch_size, sequence_length, action_dimension)
        ).astype(np.float32)
        return torch.from_numpy(data)

    return factory