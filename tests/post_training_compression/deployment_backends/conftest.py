"""Fixtures for deployment-specific quantization validation."""

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from torchao.core.config import AOBaseConfig

from versatil.quantization.constants import QuantizationModuleType
from versatil.quantization.module_target import EagerQuantizationModuleTarget
from versatil.quantization.schemas.base import QuantizationSchema
from versatil.quantization.schemas.smoothquant import SmoothQuantSchema


@pytest.fixture
def deployment_target_factory() -> Callable[..., MagicMock]:
    def factory(
        config: AOBaseConfig,
        group_size: int | None = None,
        smoothquant: bool = False,
        module_type: QuantizationModuleType = QuantizationModuleType.LINEAR,
    ) -> MagicMock:
        schema = MagicMock(
            spec=SmoothQuantSchema if smoothquant else QuantizationSchema
        )
        schema.base_config = config
        schema.weight_group_size = group_size
        target = MagicMock(spec=EagerQuantizationModuleTarget)
        target.schema = schema
        target.quantize_config = config
        target.label = "(root)"
        target.module_type = module_type
        return target

    return factory


@pytest.fixture
def deployment_model_factory() -> Callable[..., MagicMock]:
    def factory(device: str) -> MagicMock:
        layer = MagicMock(spec=nn.Linear)
        layer.weight = MagicMock(spec=torch.Tensor)
        layer.weight.device = torch.device(device)
        model = MagicMock(spec=nn.Module)
        model.get_submodule.return_value = layer
        return model

    return factory
