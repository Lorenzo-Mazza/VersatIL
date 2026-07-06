"""Tests for versatil.training.constants module."""

import pytest
import torch

from versatil.training.constants import PrecisionType


class TestPrecisionTypeIsMixed:
    @pytest.mark.parametrize(
        "precision, expected",
        [
            (PrecisionType.FP32, False),
            (PrecisionType.FP16_MIXED, True),
            (PrecisionType.BF16_MIXED, True),
            (PrecisionType.FP16_TRUE, False),
            (PrecisionType.BF16_TRUE, False),
            (PrecisionType.FP64, False),
            (PrecisionType.INT8, False),
        ],
    )
    def test_only_mixed_half_precisions_are_mixed(
        self,
        precision: PrecisionType,
        expected: bool,
    ):
        assert precision.is_mixed() is expected


class TestPrecisionTypeAutocast:
    @pytest.mark.parametrize(
        "precision, expected_enabled",
        [
            (PrecisionType.FP32, False),
            (PrecisionType.BF16_MIXED, True),
            (PrecisionType.FP16_MIXED, True),
            (PrecisionType.BF16_TRUE, False),
        ],
    )
    def test_enabled_only_for_mixed_precisions(
        self,
        precision: PrecisionType,
        expected_enabled: bool,
    ):
        with precision.autocast(device_type="cpu"):
            assert torch.is_autocast_enabled("cpu") is expected_enabled

    def test_mixed_precision_context_casts_compute_to_model_dtype(self):
        linear = torch.nn.Linear(4, 4)
        with PrecisionType.BF16_MIXED.autocast(device_type="cpu"):
            output = linear(torch.zeros(1, 4))
        assert output.dtype == torch.bfloat16
