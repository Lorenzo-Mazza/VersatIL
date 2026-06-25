"""Tests for versatil.post_training_compression.compression_target module."""

from unittest.mock import MagicMock

import pytest

from versatil.configs.post_training_compression import PreparationConfig
from versatil.post_training_compression.compression_target import CompressionTarget
from versatil.post_training_compression.pruning.base import BasePruner
from versatil.quantization.pt2e.backends.x86_inductor import X86InductorBackend
from versatil.quantization.workflows.eager import EagerQuantizationWorkflow
from versatil.quantization.workflows.pt2e import PT2EQuantizationWorkflow


@pytest.mark.unit
class TestCompressionTargetStorage:
    @pytest.mark.parametrize("module_path", ["", "encoder.backbone"])
    @pytest.mark.parametrize(
        "quantization",
        [
            None,
            PT2EQuantizationWorkflow(pt2e_backend=X86InductorBackend()),
            EagerQuantizationWorkflow(quantize_config=MagicMock(spec=[])),
        ],
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
