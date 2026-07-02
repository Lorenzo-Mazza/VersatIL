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


@pytest.mark.unit
class TestCompressionTargetOverlaps:
    @pytest.mark.parametrize(
        "first_path, second_path, expected",
        [
            ("", "decoder", True),
            ("decoder", "", True),
            ("decoder", "decoder", True),
            ("decoder", "decoder.0", True),
            ("decoder.0", "decoder", True),
            ("encoder", "decoder", False),
            ("decoder.0", "decoder.1", False),
            ("decoder", "decoder_head", False),
        ],
        ids=[
            "root_first",
            "root_second",
            "same_path",
            "nested_child",
            "nested_parent",
            "disjoint",
            "disjoint_siblings",
            "shared_prefix_not_nested",
        ],
    )
    def test_overlap_detection(self, first_path, second_path, expected):
        first = CompressionTarget(module_path=first_path)
        second = CompressionTarget(module_path=second_path)

        assert first.overlaps(other=second) is expected
