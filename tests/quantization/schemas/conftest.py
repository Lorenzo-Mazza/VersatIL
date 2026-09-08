"""Shared fixtures for quantization schema tests."""

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
from torch import nn
from torchao.core.config import AOBaseConfig
from torchao.prototype.smoothquant.core import (
    RunningAbsMaxSmoothQuantObserver,
    SmoothQuantObservedLinear,
)
from torchao.quantization import Int8DynamicActivationInt8WeightConfig


@pytest.fixture
def schema_base_config_factory() -> Callable[..., MagicMock]:
    def factory(config_type: type[AOBaseConfig]) -> MagicMock:
        return MagicMock(spec=config_type)

    return factory


@pytest.fixture
def smoothquant_base_config_factory() -> Callable[..., MagicMock]:
    def factory(version: int = 2, weight_only_decode: bool = False) -> MagicMock:
        config = MagicMock(spec=Int8DynamicActivationInt8WeightConfig)
        config.version = version
        config.weight_only_decode = weight_only_decode
        return config

    return factory


@pytest.fixture
def observed_model_factory() -> Callable[..., MagicMock]:
    def factory(prepared: bool, observed: bool) -> MagicMock:
        layer = MagicMock(spec=SmoothQuantObservedLinear if prepared else nn.Linear)
        if prepared:
            layer.in_features = 32
            layer.obs = MagicMock(spec=RunningAbsMaxSmoothQuantObserver)
            layer.obs.calibration_count = 2 if observed else 0
        model = MagicMock(spec=nn.Module)
        model.get_submodule.return_value = layer
        return model

    return factory
