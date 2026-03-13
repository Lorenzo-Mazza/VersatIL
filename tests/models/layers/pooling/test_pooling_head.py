"""Tests for versatil.models.layers.pooling.pooling_head module."""
import re
from collections.abc import Callable

import pytest
import torch

from versatil.models.encoding.encoders.constants import PoolingMethod
from versatil.models.layers.pooling.pooling_head import (
    GlobalAveragePooling,
    IdentityPooling,
    LearnedAggregationPooling,
    MaxPooling,
    PoolingHead,
    SpatialSoftmaxPooling,
    create_pooling_head,
)


class TestCreatePoolingHeadFactory:

    @pytest.mark.parametrize("pooling_method, expected_type", [
        (PoolingMethod.SPATIAL_SOFTMAX.value, SpatialSoftmaxPooling),
        (PoolingMethod.AVERAGE.value, GlobalAveragePooling),
        (PoolingMethod.MAX.value, MaxPooling),
        (PoolingMethod.DEFAULT.value, MaxPooling),
        (PoolingMethod.NONE.value, IdentityPooling),
        (PoolingMethod.LEARNED_AGGREGATION.value, LearnedAggregationPooling),
    ])
    def test_returns_correct_type_for_pooling_method(
        self,
        pooling_method: str,
        expected_type: type,
    ):
        head = create_pooling_head(
            pooling_method=pooling_method,
            feature_channels=16,
            spatial_height=8,
            spatial_width=8,
        )
        assert isinstance(head, expected_type)

    def test_raises_for_unsupported_pooling_method(self):
        invalid_method = "nonexistent_pooling"
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Unsupported pooling method: {invalid_method}."
                f"Supported: {[e.value for e in PoolingMethod]}"
            ),
        ):
            create_pooling_head(
                pooling_method=invalid_method,
                feature_channels=16,
                spatial_height=8,
                spatial_width=8,
            )

    def test_all_factory_outputs_are_pooling_heads(self):
        for method in PoolingMethod:
            head = create_pooling_head(
                pooling_method=method.value,
                feature_channels=16,
                spatial_height=8,
                spatial_width=8,
            )
            assert isinstance(head, PoolingHead)


class TestSpatialSoftmaxPooling:

    @pytest.mark.parametrize("channels", [8, 16])
    def test_output_dim(self, channels: int):
        head = SpatialSoftmaxPooling(
            spatial_height=8,
            spatial_width=8,
            channels=channels,
        )
        assert head.get_output_dim(input_channels=channels) == channels * 2

    def test_forward_shape(
        self,
        feature_map_factory: Callable[..., torch.Tensor],
    ):
        channels = 16
        height, width = 8, 8
        head = SpatialSoftmaxPooling(
            spatial_height=height,
            spatial_width=width,
            channels=channels,
        )
        tensor = feature_map_factory(
            batch_size=2,
            channels=channels,
            height=height,
            width=width,
        )
        output = head(tensor)
        assert output.shape == (2, channels * 2)


class TestGlobalAveragePooling:

    @pytest.mark.parametrize("channels", [8, 32])
    def test_output_dim(self, channels: int):
        head = GlobalAveragePooling()
        assert head.get_output_dim(input_channels=channels) == channels

    def test_forward_shape(
        self,
        feature_map_factory: Callable[..., torch.Tensor],
    ):
        channels = 16
        head = GlobalAveragePooling()
        tensor = feature_map_factory(
            batch_size=2,
            channels=channels,
            height=8,
            width=8,
        )
        output = head(tensor)
        assert output.shape == (2, channels)


class TestMaxPooling:

    @pytest.mark.parametrize("channels", [8, 32])
    def test_output_dim(self, channels: int):
        head = MaxPooling()
        assert head.get_output_dim(input_channels=channels) == channels

    def test_forward_shape(
        self,
        feature_map_factory: Callable[..., torch.Tensor],
    ):
        channels = 16
        head = MaxPooling()
        tensor = feature_map_factory(
            batch_size=2,
            channels=channels,
            height=8,
            width=8,
        )
        output = head(tensor)
        assert output.shape == (2, channels)


class TestIdentityPooling:

    def test_output_dim_returns_tuple(self):
        head = IdentityPooling(channels=16)
        result = head.get_output_dim(input_channels=16)
        assert isinstance(result, tuple)
        assert result[0] == 16

    def test_forward_returns_input_unchanged(
        self,
        feature_map_factory: Callable[..., torch.Tensor],
    ):
        channels = 16
        head = IdentityPooling(channels=channels)
        tensor = feature_map_factory(
            batch_size=2,
            channels=channels,
            height=8,
            width=8,
        )
        output = head(tensor)
        assert torch.equal(output, tensor)


class TestLearnedAggregationPooling:

    @pytest.mark.parametrize("channels", [16, 32])
    def test_output_dim(self, channels: int):
        head = LearnedAggregationPooling(channels=channels)
        assert head.get_output_dim(input_channels=channels) == channels

    def test_forward_shape(
        self,
        feature_map_factory: Callable[..., torch.Tensor],
    ):
        channels = 16
        batch_size = 2
        head = LearnedAggregationPooling(channels=channels)
        tensor = feature_map_factory(
            batch_size=batch_size,
            channels=channels,
            height=4,
            width=4,
        )
        output = head(tensor)
        # cls_q (C,) broadcasts with attn output (B, C) to produce (B, C)
        assert output.shape == (batch_size, channels)
