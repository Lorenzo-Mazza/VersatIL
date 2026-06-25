"""Tests for versatil.quantization.workflows.base module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock

import pytest
import torch.nn as nn

from versatil.quantization.module_target import EagerQuantizationModuleTarget
from versatil.quantization.workflows.base import validate_quantization_targets


class QuantizationValidationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
        self.decoder = nn.Linear(8, 2)


@pytest.fixture
def eager_target_factory() -> Callable[[str], EagerQuantizationModuleTarget]:
    """Factory for eager targets used by shared validator tests."""

    def factory(module_path: str) -> EagerQuantizationModuleTarget:
        return EagerQuantizationModuleTarget(
            module_path=module_path,
            quantize_config=MagicMock(spec=[]),
        )

    return factory


@pytest.mark.unit
@pytest.mark.parametrize(
    "module_path, expectation",
    [
        ("", does_not_raise()),
        ("encoder", does_not_raise()),
        ("encoder.0", does_not_raise()),
        (
            "missing",
            pytest.raises(
                ValueError,
                match=re.escape(
                    "Quantization target 'missing' not found in model. "
                    "Available top-level modules: ['encoder', 'decoder']."
                ),
            ),
        ),
    ],
)
def test_validate_quantization_targets_checks_module_paths(
    eager_target_factory: Callable[[str], EagerQuantizationModuleTarget],
    module_path: str,
    expectation,
) -> None:
    model = QuantizationValidationModel()
    targets = [eager_target_factory(module_path)]

    with expectation:
        validate_quantization_targets(model=model, targets=targets)


@pytest.mark.unit
@pytest.mark.parametrize(
    "module_paths, expectation",
    [
        (["encoder", "decoder"], does_not_raise()),
        (
            ["encoder", "encoder.0"],
            pytest.raises(
                ValueError,
                match=re.escape(
                    "Quantization targets overlap: 'encoder' and 'encoder.0'."
                ),
            ),
        ),
        (
            ["", "decoder"],
            pytest.raises(
                ValueError,
                match=re.escape("Quantization targets overlap: '' and 'decoder'."),
            ),
        ),
    ],
)
def test_validate_quantization_targets_rejects_overlapping_paths(
    eager_target_factory: Callable[[str], EagerQuantizationModuleTarget],
    module_paths: list[str],
    expectation,
) -> None:
    model = QuantizationValidationModel()
    targets = [eager_target_factory(module_path) for module_path in module_paths]

    with expectation:
        validate_quantization_targets(model=model, targets=targets)
