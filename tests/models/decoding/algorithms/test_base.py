"""Tests for versatil.models.decoding.algorithm.base module."""
import pytest
import torch.nn as nn

from versatil.models.decoding.algorithm.base import DecodingAlgorithm


class TestDecodingAlgorithmInterface:

    def test_is_abstract(self):
        with pytest.raises(
            TypeError,
            match="Can't instantiate abstract class DecodingAlgorithm",
        ):
            DecodingAlgorithm()

    def test_inherits_from_nn_module(self):
        assert issubclass(DecodingAlgorithm, nn.Module)