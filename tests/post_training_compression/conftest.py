"""Shared fixtures for post-training compression tests."""

from collections.abc import Callable

import numpy as np
import pytest
import torch
import torch.nn as nn


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


def verify_reload_fidelity(
    original_model: nn.Module,
    reloaded_model: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
) -> bool:
    """Verify exact numerical match between original and reloaded model.

    Args:
        original_model: The model before save.
        reloaded_model: The model after loading from disk.
        example_inputs: Inputs to run through both models.

    Returns:
        True if all output tensors match exactly.
    """
    with torch.no_grad():
        original_outputs = original_model(*example_inputs)
        reloaded_outputs = reloaded_model(*example_inputs)
    if len(original_outputs) != len(reloaded_outputs):
        return False
    return all(
        torch.equal(original, reloaded)
        for original, reloaded in zip(original_outputs, reloaded_outputs)
    )
