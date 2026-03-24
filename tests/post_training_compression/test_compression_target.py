"""Tests for versatil.post_training_compression.compression_target module."""

import re
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock

import pytest

from versatil.configs.post_training_compression import PreparationConfig
from versatil.post_training_compression.compression_target import CompressionTarget
from versatil.post_training_compression.pruning.base import BasePruner
from versatil.quantization.backends.x86_inductor import X86InductorBackend
from versatil.quantization.strategies import PT2EStrategy, QuantizeApiStrategy


@pytest.mark.unit
class TestCompressionTargetStorage:
    @pytest.mark.parametrize("module_path", ["", "encoder.backbone"])
    @pytest.mark.parametrize(
        "quantization",
        [None, PT2EStrategy(pt2e_backend=X86InductorBackend())],
    )
    def test_stores_configuration(self, module_path, quantization):
        preparation = PreparationConfig()
        pruner_a = MagicMock(spec=BasePruner)
        pruner_b = MagicMock(spec=BasePruner)
        pruning = [pruner_a, pruner_b]

        target = CompressionTarget(
            module_path=module_path,
            preparation=preparation,
            pruning=pruning,
            quantization=quantization,
        )

        assert target.module_path == module_path
        assert target.preparation is preparation
        assert target.pruning is pruning
        assert target.quantization is quantization

    def test_none_pruning_normalizes_to_empty_list(self):
        target = CompressionTarget(module_path="", pruning=None)

        assert target.pruning == []

    def test_none_preparation_stored_as_none(self):
        target = CompressionTarget(module_path="", preparation=None)

        assert target.preparation is None


@pytest.mark.unit
class TestCompressionTargetValidation:
    @pytest.mark.parametrize(
        "quantization, module_path, expectation",
        [
            (None, "encoder", does_not_raise()),
            (MagicMock(), "encoder", does_not_raise()),
            (
                QuantizeApiStrategy(quantize_config=MagicMock(spec=[])),
                "encoder",
                does_not_raise(),
            ),
            (
                QuantizeApiStrategy(
                    quantize_config=MagicMock(act_quant_scale=None),
                ),
                "backbone",
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "Module 'backbone' uses a static activation "
                        "quantize_() config. Static quantization is only "
                        "supported via PT2E. Use PT2EStrategy or a "
                        "dynamic/weight-only config."
                    ),
                ),
            ),
            (
                QuantizeApiStrategy(
                    quantize_config=MagicMock(act_quant_scale=None),
                ),
                "",
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "Module '(root)' uses a static activation quantize_() config."
                    ),
                ),
            ),
        ],
        ids=[
            "none",
            "non_quantize_api",
            "dynamic_quantize_api",
            "static_quantize_api",
            "static_quantize_api_root",
        ],
    )
    def test_quantization_validation(self, quantization, module_path, expectation):
        with expectation:
            CompressionTarget(
                module_path=module_path,
                quantization=quantization,
            )
