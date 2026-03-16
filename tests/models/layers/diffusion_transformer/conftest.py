"""Shared fixtures for diffusion transformer layer tests."""

from collections.abc import Callable

import numpy as np
import pytest
import torch
import torch.nn as nn

from versatil.models.layers.modulation.conditional_modulation import (
    ConditionalModulation,
)


def reinit_modulation_layers(module: nn.Module) -> None:
    """Re-initialize ConditionalModulation projections with xavier to break zero init.

    At initialization, AdaNorm modulation layers have zero weights, so conditioning
    has no effect. This helper enables conditioning sensitivity for behavioral tests.
    """
    for submodule in module.modules():
        if isinstance(submodule, ConditionalModulation):
            for linear in submodule.projection.modules():
                if isinstance(linear, nn.Linear):
                    nn.init.xavier_uniform_(linear.weight)
                    if linear.bias is not None:
                        nn.init.zeros_(linear.bias)


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
