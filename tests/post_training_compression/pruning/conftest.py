"""Shared fixtures for pruning tests."""

from collections.abc import Callable

import numpy as np
import pytest
import torch
from torch import nn


@pytest.fixture
def pruning_model_factory(
    rng: np.random.Generator,
) -> Callable[..., nn.Module]:
    """Factory that creates a small model with Conv2d and Linear layers."""

    def factory(
        input_channels: int = 3,
        hidden_channels: int = 8,
        linear_features: int = 16,
        output_features: int = 4,
    ) -> nn.Module:
        model = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(
                in_features=hidden_channels,
                out_features=linear_features,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=linear_features,
                out_features=output_features,
            ),
        )
        with torch.no_grad():
            for parameter in model.parameters():
                data = rng.standard_normal(parameter.shape).astype(np.float32)
                parameter.copy_(torch.from_numpy(data))
        return model

    return factory
