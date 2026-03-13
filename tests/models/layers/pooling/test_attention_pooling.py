"""Tests for versatil.models.layers.pooling.attention_pooling module."""
import re
from collections.abc import Callable

import numpy as np
import pytest
import torch
import torch.nn as nn

from versatil.models.layers.pooling.attention_pooling import (
    AttentionPool2d,
    LearnedAggregation,
)


@pytest.fixture
def sequence_input_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for 3D sequence inputs (B, N, C)."""
    def factory(
        batch_size: int = 2,
        sequence_length: int = 16,
        channels: int = 16,
    ) -> torch.Tensor:
        data = rng.standard_normal(
            (batch_size, sequence_length, channels)
        ).astype(np.float32)
        return torch.from_numpy(data)
    return factory


@pytest.fixture
def cls_query_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for CLS query tokens (C,)."""
    def factory(
        channels: int = 16,
    ) -> torch.Tensor:
        data = rng.standard_normal((channels,)).astype(np.float32)
        return torch.from_numpy(data)
    return factory


@pytest.fixture
def attention_pool_factory() -> Callable[..., AttentionPool2d]:
    """Factory for AttentionPool2d instances."""
    def factory(
        feature_dimension: int = 16,
        bias: bool = True,
    ) -> AttentionPool2d:
        return AttentionPool2d(
            feature_dimension=feature_dimension,
            bias=bias,
        )
    return factory


@pytest.fixture
def learned_aggregation_factory() -> Callable[..., LearnedAggregation]:
    """Factory for LearnedAggregation instances."""
    def factory(
        feature_dimension: int = 16,
        attention_bias: bool = True,
        feedforward_expand: int | float = 3,
    ) -> LearnedAggregation:
        return LearnedAggregation(
            feature_dimension=feature_dimension,
            attention_bias=attention_bias,
            feedforward_expand=feedforward_expand,
        )
    return factory


class TestAttentionPool2dForward:

    @pytest.mark.parametrize("feature_dimension", [16, 32])
    def test_output_shape_for_4d_input(
        self,
        attention_pool_factory: Callable[..., AttentionPool2d],
        feature_map_factory: Callable[..., torch.Tensor],
        cls_query_factory: Callable[..., torch.Tensor],
        feature_dimension: int,
    ):
        module = attention_pool_factory(feature_dimension=feature_dimension)
        tensor = feature_map_factory(
            batch_size=2, channels=feature_dimension, height=4, width=4,
        )
        cls_q = cls_query_factory(channels=feature_dimension)
        output = module(tensor, cls_q)
        assert output.shape == (2, feature_dimension)

    @pytest.mark.parametrize("feature_dimension", [16, 32])
    def test_output_shape_for_3d_input(
        self,
        attention_pool_factory: Callable[..., AttentionPool2d],
        sequence_input_factory: Callable[..., torch.Tensor],
        cls_query_factory: Callable[..., torch.Tensor],
        feature_dimension: int,
    ):
        module = attention_pool_factory(feature_dimension=feature_dimension)
        tensor = sequence_input_factory(
            batch_size=2,
            sequence_length=10,
            channels=feature_dimension,
        )
        cls_q = cls_query_factory(channels=feature_dimension)
        output = module(tensor, cls_q)
        assert output.shape == (2, feature_dimension)

    def test_raises_for_incompatible_4d_shape(
        self,
        rng: np.random.Generator,
        attention_pool_factory: Callable[..., AttentionPool2d],
        cls_query_factory: Callable[..., torch.Tensor],
    ):
        feature_dimension = 16
        module = attention_pool_factory(feature_dimension=feature_dimension)
        wrong_channels = 24
        data = rng.standard_normal((2, wrong_channels, 4, 4)).astype(np.float32)
        tensor = torch.from_numpy(data)
        cls_q = cls_query_factory(channels=feature_dimension)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Input shape {tensor.shape} not compatible with AttentionPool2d "
                f"of size {feature_dimension}"
            ),
        ):
            module(tensor, cls_q)

    def test_raises_for_incompatible_3d_shape(
        self,
        rng: np.random.Generator,
        attention_pool_factory: Callable[..., AttentionPool2d],
        cls_query_factory: Callable[..., torch.Tensor],
    ):
        feature_dimension = 16
        module = attention_pool_factory(feature_dimension=feature_dimension)
        wrong_dim = 24
        data = rng.standard_normal((2, wrong_dim, wrong_dim)).astype(np.float32)
        tensor = torch.from_numpy(data)
        cls_q = cls_query_factory(channels=feature_dimension)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Input shape {tensor.shape} not compatible with AttentionPool2d "
                f"of size {feature_dimension}"
            ),
        ):
            module(tensor, cls_q)

    def test_inherits_nn_module(
        self,
        attention_pool_factory: Callable[..., AttentionPool2d],
    ):
        module = attention_pool_factory(feature_dimension=16, bias=True)
        assert isinstance(module, nn.Module)


class TestLearnedAggregation:

    def test_inherits_nn_module(
        self,
        learned_aggregation_factory: Callable[..., LearnedAggregation],
    ):
        module = learned_aggregation_factory(feature_dimension=16)
        assert isinstance(module, nn.Module)

    @pytest.mark.parametrize("feature_dimension", [16, 32])
    def test_output_shape_from_4d_input(
        self,
        learned_aggregation_factory: Callable[..., LearnedAggregation],
        feature_map_factory: Callable[..., torch.Tensor],
        feature_dimension: int,
    ):
        batch_size = 2
        module = learned_aggregation_factory(feature_dimension=feature_dimension)
        tensor = feature_map_factory(
            batch_size=batch_size, channels=feature_dimension, height=4, width=4,
        )
        output = module(tensor)
        assert output.shape == (batch_size, feature_dimension)

    @pytest.mark.parametrize("feature_dimension", [16, 32])
    def test_output_shape_from_3d_input(
        self,
        learned_aggregation_factory: Callable[..., LearnedAggregation],
        sequence_input_factory: Callable[..., torch.Tensor],
        feature_dimension: int,
    ):
        batch_size = 2
        module = learned_aggregation_factory(feature_dimension=feature_dimension)
        tensor = sequence_input_factory(
            batch_size=batch_size,
            sequence_length=10,
            channels=feature_dimension,
        )
        output = module(tensor)
        assert output.shape == (batch_size, feature_dimension)

    def test_cls_query_is_learnable_parameter(
        self,
        learned_aggregation_factory: Callable[..., LearnedAggregation],
    ):
        module = learned_aggregation_factory(feature_dimension=16)
        assert isinstance(module.cls_q, nn.Parameter)
        assert module.cls_q.shape == (16,)
