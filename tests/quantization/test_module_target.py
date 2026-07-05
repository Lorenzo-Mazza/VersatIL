"""Tests for versatil.quantization.module_target module."""

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest

from versatil.quantization.module_target import (
    EagerQuantizationModuleTarget,
    PT2EQuantizationModuleTarget,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "module_path, module_name, expected",
    [
        ("", "encoder.0", True),
        ("decoder", "decoder", True),
        ("decoder", "decoder.head", True),
        ("decoder", "encoder.head", False),
        ("decoder", "decoder_head", False),
    ],
)
def test_contains_module_matches_exact_path_and_children(
    module_path: str,
    module_name: str,
    expected: bool,
) -> None:
    target = EagerQuantizationModuleTarget(
        module_path=module_path,
        quantize_config=MagicMock(spec=[]),
    )

    assert target.contains_module(module_name=module_name) is expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "module_path, expected",
    [
        ("", "(root)"),
        ("decoder", "decoder"),
    ],
)
def test_label_returns_root_name_for_empty_path(
    module_path: str,
    expected: str,
) -> None:
    target = EagerQuantizationModuleTarget(
        module_path=module_path,
        quantize_config=MagicMock(spec=[]),
    )

    assert target.label == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "left_path, right_path, expected",
    [
        ("", "decoder", True),
        ("encoder", "encoder", True),
        ("encoder", "encoder.backbone", True),
        ("encoder.backbone", "encoder", True),
        ("encoder", "decoder", False),
        ("encoder", "encoder_head", False),
    ],
)
def test_overlaps_detects_root_same_and_nested_targets(
    left_path: str,
    right_path: str,
    expected: bool,
) -> None:
    left = EagerQuantizationModuleTarget(
        module_path=left_path,
        quantize_config=MagicMock(spec=[]),
    )
    right = EagerQuantizationModuleTarget(
        module_path=right_path,
        quantize_config=MagicMock(spec=[]),
    )

    assert left.overlaps(other=right) is expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "is_dynamic, expected",
    [
        (True, False),
        (False, True),
    ],
)
def test_pt2e_target_needs_calibration_reflects_backend_dynamic_flag(
    mock_pt2e_backend_factory: Callable[..., MagicMock],
    is_dynamic: bool,
    expected: bool,
) -> None:
    target = PT2EQuantizationModuleTarget(
        module_path="",
        pt2e_backend=mock_pt2e_backend_factory(is_dynamic=is_dynamic),
    )

    assert target.needs_calibration is expected
