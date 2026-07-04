"""Tests for versatil.explainability.attribution.maps helpers."""

import pytest
import torch

from versatil.explainability.attribution.maps import (
    activation_to_nchw,
    get_batch_temporal_shape,
    get_image_size,
)
from versatil.models.encoding.explainability import ActivationLayout


@pytest.mark.unit
class TestCameraTensorShapes:
    def test_5d_tensor_returns_batch_and_time(self):
        assert get_batch_temporal_shape(torch.zeros(2, 3, 3, 8, 8)) == (2, 3)

    def test_4d_tensor_returns_time_one(self):
        assert get_batch_temporal_shape(torch.zeros(2, 3, 8, 8)) == (2, 1)

    def test_wrong_rank_raises(self):
        with pytest.raises(ValueError, match="Camera tensor"):
            get_batch_temporal_shape(torch.zeros(2, 8))

    def test_image_size_from_5d_and_4d(self):
        assert get_image_size(torch.zeros(2, 3, 3, 16, 24)) == (16, 24)
        assert get_image_size(torch.zeros(2, 3, 16, 24)) == (16, 24)

    def test_image_size_wrong_rank_raises(self):
        with pytest.raises(ValueError):
            get_image_size(torch.zeros(2, 8))


@pytest.mark.unit
class TestEnsureChannelFirst:
    def test_nchw_passthrough(self):
        tensor = torch.randn(2, 4, 8, 8)
        result = activation_to_nchw(tensor, ActivationLayout.NCHW.value)
        assert result is tensor

    def test_nhwc_is_permuted(self):
        tensor = torch.randn(2, 8, 8, 4)
        result = activation_to_nchw(tensor, ActivationLayout.NHWC.value)
        assert result.shape == (2, 4, 8, 8)

    def test_non_spatial_layout_raises(self):
        with pytest.raises(ValueError, match="not a spatial"):
            activation_to_nchw(torch.randn(2, 4), "tokens")
