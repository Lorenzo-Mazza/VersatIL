"""Tests for versatil.post_training_compression.policy_context module."""

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest

from versatil.post_training_compression.policy_context import PolicyContext
from versatil.training.constants import PrecisionType


@pytest.fixture
def precision_context_factory() -> Callable[..., MagicMock]:
    def factory(
        checkpoint_precision: PrecisionType, compression_precision: PrecisionType | None
    ) -> MagicMock:
        context = MagicMock(spec=PolicyContext)
        context.precision = compression_precision
        context.config = MagicMock()
        context.config.experiment.precision = checkpoint_precision.value
        return context

    return factory


@pytest.mark.unit
@pytest.mark.parametrize(
    "checkpoint_precision",
    [PrecisionType.FP32, PrecisionType.FP16_MIXED, PrecisionType.BF16_MIXED],
)
@pytest.mark.parametrize(
    "compression_precision", [None, PrecisionType.FP32, PrecisionType.BF16_TRUE]
)
def test_checkpoint_autocast_applies_only_when_preserving_loaded_dtypes(
    precision_context_factory: Callable[..., MagicMock],
    checkpoint_precision: PrecisionType,
    compression_precision: PrecisionType | None,
) -> None:
    context = precision_context_factory(
        checkpoint_precision=checkpoint_precision,
        compression_precision=compression_precision,
    )
    result = PolicyContext.autocast_precision.fget(context)
    expected = (
        checkpoint_precision
        if compression_precision is None
        and checkpoint_precision in (PrecisionType.FP16_MIXED, PrecisionType.BF16_MIXED)
        else None
    )
    assert result == expected
    assert context.config.experiment.precision == checkpoint_precision.value
