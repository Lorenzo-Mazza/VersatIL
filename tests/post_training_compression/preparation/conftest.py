"""Shared fixtures for quantization preparation tests."""

from collections.abc import Callable

import numpy as np
import pytest
import torch
from torch import nn

from versatil.models.layers.frozen_batchnorm import FrozenBatchNorm2d


@pytest.fixture
def frozen_batchnorm_factory(
    rng: np.random.Generator,
) -> Callable[..., FrozenBatchNorm2d]:
    """Factory for VersatIL FrozenBatchNorm2d with randomized parameters."""

    def factory(
        num_features: int = 16,
    ) -> FrozenBatchNorm2d:
        module = FrozenBatchNorm2d(dimension=num_features)
        module.weight.data = torch.from_numpy(
            rng.standard_normal(num_features).astype(np.float32)
        )
        module.bias.data = torch.from_numpy(
            rng.standard_normal(num_features).astype(np.float32)
        )
        module.running_mean.data = torch.from_numpy(
            rng.standard_normal(num_features).astype(np.float32)
        )
        module.running_var.data = torch.from_numpy(
            rng.uniform(low=0.1, high=2.0, size=num_features).astype(np.float32)
        )
        return module

    return factory


@pytest.fixture
def frozen_batchnorm_model_factory(
    frozen_batchnorm_factory: Callable[..., FrozenBatchNorm2d],
) -> Callable[..., nn.Module]:
    """Factory for models containing VersatIL FrozenBatchNorm2d layers."""

    def factory(
        num_features: int = 16,
        num_frozen_layers: int = 2,
    ) -> nn.Module:
        layers = {}
        for index in range(num_frozen_layers):
            layers[f"conv_{index}"] = nn.Conv2d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=3,
                padding=1,
            )
            layers[f"bn_{index}"] = frozen_batchnorm_factory(
                num_features=num_features,
            )
        model = nn.Module()
        for name, layer in layers.items():
            model.add_module(name, layer)
        return model

    return factory


@pytest.fixture
def conv_bn_model_factory(
    rng: np.random.Generator,
) -> Callable[..., nn.Module]:
    """Factory for models with Conv2d followed by BatchNorm2d pairs."""

    def factory(
        in_channels: int = 3,
        out_channels: int = 16,
        num_pairs: int = 2,
        conv_bias: bool = False,
    ) -> nn.Module:
        layers = []
        current_channels = in_channels
        for _index in range(num_pairs):
            layers.append(
                nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=conv_bias,
                )
            )
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            current_channels = out_channels
        return nn.Sequential(*layers)

    return factory
