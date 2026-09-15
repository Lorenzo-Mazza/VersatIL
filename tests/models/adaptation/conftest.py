"""Shared fixtures for adaptation tests."""

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch.nn as nn
from peft import PeftModel


@pytest.fixture
def adaptation_model_factory() -> Callable[..., MagicMock]:
    def factory(adapted: bool = False) -> MagicMock:
        model = MagicMock(spec=PeftModel if adapted else nn.Module)
        model.vision_tower = MagicMock(spec=nn.Sequential)
        model.vision_tower.projection = MagicMock(spec=nn.Linear)
        model.vision_tower.normalization = MagicMock(spec=nn.LayerNorm)
        model.vision_tower.nested = MagicMock(spec=nn.Sequential)
        model.vision_tower.nested.projection = MagicMock(spec=nn.Linear)
        model.projector = MagicMock(spec=nn.Linear)
        model.language_model = MagicMock(spec=nn.Linear)
        model.named_modules.return_value = [
            ("", model),
            ("vision_tower", model.vision_tower),
            ("vision_tower.projection", model.vision_tower.projection),
            ("vision_tower.normalization", model.vision_tower.normalization),
            ("vision_tower.nested", model.vision_tower.nested),
            ("vision_tower.nested.projection", model.vision_tower.nested.projection),
            ("projector", model.projector),
            ("language_model", model.language_model),
        ]
        return model

    return factory
