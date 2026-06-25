"""Tests for versatil.post_training_compression.compression_target module."""

from unittest.mock import MagicMock

import pytest

from versatil.configs.post_training_compression import PreparationConfig
from versatil.post_training_compression.compression_target import CompressionTarget
from versatil.post_training_compression.pruning.base import BasePruner


@pytest.mark.unit
class TestCompressionTargetStorage:
    @pytest.mark.parametrize("module_path", ["", "encoder.backbone"])
    def test_stores_configuration(self, module_path):
        preparation = PreparationConfig()
        pruner_a = MagicMock(spec=BasePruner)
        pruner_b = MagicMock(spec=BasePruner)
        pruning = [pruner_a, pruner_b]

        target = CompressionTarget(
            module_path=module_path,
            preparation=preparation,
            pruning=pruning,
        )

        assert target.module_path == module_path
        assert target.preparation is preparation
        assert target.pruning is pruning

    def test_none_pruning_normalizes_to_empty_list(self):
        target = CompressionTarget(module_path="", pruning=None)

        assert target.pruning == []

    def test_none_preparation_stored_as_none(self):
        target = CompressionTarget(module_path="", preparation=None)

        assert target.preparation is None
