"""Tests for versatil.quantization.workflows.base module."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from versatil.quantization.workflows.base import (
    BaseQuantizationWorkflow,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "requested, expected",
    [(None, "cuda:1"), ("cpu", "cpu"), ("cuda", "cuda:0"), ("cuda:1", "cuda:1")],
)
def test_deployment_device_resolves_current_cuda_index(
    requested: str | None, expected: str
) -> None:
    with patch(
        "versatil.quantization.workflows.base.torch.cuda.current_device", return_value=0
    ) as current_device:
        device = BaseQuantizationWorkflow._resolve_deployment_device(
            device=torch.device(requested) if requested is not None else None,
            source_device=torch.device("cuda:1"),
        )
    assert device == torch.device(expected)
    if requested == "cuda":
        current_device.assert_called_once_with()
    else:
        current_device.assert_not_called()


@pytest.mark.unit
def test_validate_targets_delegates_to_resolution() -> None:
    workflow = MagicMock(spec=BaseQuantizationWorkflow)
    model = MagicMock(spec=nn.Module)
    with patch(
        "versatil.quantization.workflows.base.validate_quantization_targets"
    ) as validate:
        BaseQuantizationWorkflow.validate_targets(workflow, model=model)
    validate.assert_called_once_with(model=model, targets=workflow.targets)
