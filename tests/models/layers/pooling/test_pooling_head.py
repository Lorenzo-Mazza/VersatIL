"""Tests for versatil.models.layers.pooling.pooling_head module."""

import re
from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.models.encoding.encoders.constants import PoolingMethod
from versatil.models.layers.pooling.pooling_head import (
    GlobalAveragePooling,
    MaxPooling,
    PoolingHead,
    SpatialIdentityPooling,
    SpatialLearnedAggregationPooling,
    SpatialSoftmaxPooling,
    TokenPoolingHead,
    create_spatial_pooling_head,
    create_token_pooling_head,
)


class TestCreateSpatialPoolingHeadFactory:
    @pytest.mark.parametrize(
        "pooling_method, expected_type",
        [
            (PoolingMethod.SPATIAL_SOFTMAX.value, SpatialSoftmaxPooling),
            (PoolingMethod.AVERAGE.value, GlobalAveragePooling),
            (PoolingMethod.MAX.value, MaxPooling),
            (PoolingMethod.DEFAULT.value, MaxPooling),
            (PoolingMethod.NONE.value, SpatialIdentityPooling),
            (PoolingMethod.LEARNED_AGGREGATION.value, SpatialLearnedAggregationPooling),
        ],
    )
    def test_returns_correct_type_for_pooling_method(
        self,
        pooling_method: str,
        expected_type: type,
    ):
        head = create_spatial_pooling_head(
            pooling_method=pooling_method,
            input_dimension=16,
            spatial_height=8,
            spatial_width=8,
        )
        assert isinstance(head, expected_type)

    def test_raises_for_unsupported_pooling_method(self):
        invalid_method = "nonexistent_pooling"
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Unsupported spatial pooling method: {invalid_method}. "
                f"Supported: {[e.value for e in PoolingMethod]}"
            ),
        ):
            create_spatial_pooling_head(
                pooling_method=invalid_method,
                input_dimension=16,
                spatial_height=8,
                spatial_width=8,
            )

    def test_all_factory_outputs_are_pooling_heads(self):
        for method in PoolingMethod:
            head = create_spatial_pooling_head(
                pooling_method=method.value,
                input_dimension=16,
                spatial_height=8,
                spatial_width=8,
            )
            assert isinstance(head, PoolingHead)


class TestSpatialSoftmaxPooling:
    @pytest.mark.parametrize("input_dimension", [8, 16])
    def test_output_dim(self, input_dimension: int):
        head = SpatialSoftmaxPooling(
            input_dimension=input_dimension,
            spatial_height=8,
            spatial_width=8,
        )
        assert head.output_dim == input_dimension * 2

    def test_forward_shape(
        self,
        nchw_tensor_factory: Callable[..., torch.Tensor],
    ):
        input_dimension = 16
        height, width = 8, 8
        head = SpatialSoftmaxPooling(
            input_dimension=input_dimension,
            spatial_height=height,
            spatial_width=width,
        )
        tensor = nchw_tensor_factory(
            batch_size=2,
            channels=input_dimension,
            height=height,
            width=width,
        )
        output = head(tensor)
        assert output.shape == (2, input_dimension * 2)


class TestGlobalAveragePooling:
    @pytest.mark.parametrize("input_dimension", [8, 32])
    def test_output_dim(self, input_dimension: int):
        head = GlobalAveragePooling(input_dimension=input_dimension)
        assert head.output_dim == input_dimension

    def test_forward_shape(
        self,
        nchw_tensor_factory: Callable[..., torch.Tensor],
    ):
        input_dimension = 16
        head = GlobalAveragePooling(input_dimension=input_dimension)
        tensor = nchw_tensor_factory(
            batch_size=2,
            channels=input_dimension,
            height=8,
            width=8,
        )
        output = head(tensor)
        assert output.shape == (2, input_dimension)

    def test_forward_computes_spatial_mean(self):
        head = GlobalAveragePooling(input_dimension=1)
        tensor = torch.tensor([[[[1.0, 3.0], [5.0, 7.0]]]])  # (1, 1, 2, 2)
        output = head(tensor)
        expected_mean = (1.0 + 3.0 + 5.0 + 7.0) / 4.0
        assert torch.allclose(output, torch.tensor([[expected_mean]]))


class TestMaxPooling:
    @pytest.mark.parametrize("input_dimension", [8, 32])
    def test_output_dim(self, input_dimension: int):
        head = MaxPooling(input_dimension=input_dimension)
        assert head.output_dim == input_dimension

    def test_forward_shape(
        self,
        nchw_tensor_factory: Callable[..., torch.Tensor],
    ):
        input_dimension = 16
        head = MaxPooling(input_dimension=input_dimension)
        tensor = nchw_tensor_factory(
            batch_size=2,
            channels=input_dimension,
            height=8,
            width=8,
        )
        output = head(tensor)
        assert output.shape == (2, input_dimension)

    def test_forward_returns_spatial_maximum(self):
        head = MaxPooling(input_dimension=1)
        tensor = torch.tensor([[[[1.0, 3.0], [5.0, 7.0]]]])  # (1, 1, 2, 2)
        output = head(tensor)
        assert torch.allclose(output, torch.tensor([[7.0]]))


class TestSpatialIdentityPooling:
    def test_output_dim_returns_dimension_and_unknown_spatial(self):
        input_dimension = 16
        head = SpatialIdentityPooling(input_dimension=input_dimension)
        assert head.output_dim == (input_dimension, -1, -1)

    def test_forward_returns_input_unchanged(
        self,
        nchw_tensor_factory: Callable[..., torch.Tensor],
    ):
        input_dimension = 16
        head = SpatialIdentityPooling(input_dimension=input_dimension)
        tensor = nchw_tensor_factory(
            batch_size=2,
            channels=input_dimension,
            height=8,
            width=8,
        )
        output = head(tensor)
        assert torch.equal(output, tensor)


class TestSpatialLearnedAggregationPooling:
    @pytest.mark.parametrize("input_dimension", [16, 32])
    def test_output_dim(self, input_dimension: int):
        head = SpatialLearnedAggregationPooling(input_dimension=input_dimension)
        assert head.output_dim == input_dimension

    def test_forward_produces_input_sensitive_output(
        self,
        nchw_tensor_factory: Callable[..., torch.Tensor],
    ):
        input_dimension = 16
        batch_size = 2
        head = SpatialLearnedAggregationPooling(input_dimension=input_dimension)
        tensor_a = nchw_tensor_factory(
            batch_size=batch_size,
            channels=input_dimension,
            height=4,
            width=4,
        )
        tensor_b = nchw_tensor_factory(
            batch_size=batch_size,
            channels=input_dimension,
            height=4,
            width=4,
        )
        output_a = head(tensor_a)
        output_b = head(tensor_b)
        assert output_a.shape == (batch_size, input_dimension)
        assert not torch.allclose(output_a, output_b)


class TestCreateTokenPoolingHeadFactory:
    @pytest.mark.parametrize(
        "pooling_method",
        [
            PoolingMethod.DEFAULT.value,
            PoolingMethod.AVERAGE.value,
            PoolingMethod.LEARNED_AGGREGATION.value,
            PoolingMethod.NONE.value,
        ],
    )
    def test_returns_token_pooling_head(self, pooling_method: str):
        head = create_token_pooling_head(
            pooling_method=pooling_method,
            input_dimension=64,
        )
        assert isinstance(head, TokenPoolingHead)

    def test_spatial_softmax_method_raises_on_forward(self):
        head = create_token_pooling_head(
            pooling_method=PoolingMethod.SPATIAL_SOFTMAX.value,
            input_dimension=64,
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Unsupported token pooling method: {PoolingMethod.SPATIAL_SOFTMAX.value}. "
                f"Supported: {[e.value for e in PoolingMethod]}"
            ),
        ):
            head(torch.zeros(2, 10, 64))


class TestTokenPoolingHead:
    @pytest.mark.parametrize(
        "pooling_method, expected_dim",
        [
            (PoolingMethod.DEFAULT.value, 64),
            (PoolingMethod.AVERAGE.value, 64),
            (PoolingMethod.LEARNED_AGGREGATION.value, 64),
            (PoolingMethod.NONE.value, (20, 64)),
        ],
    )
    def test_output_dim(self, pooling_method: str, expected_dim: int | tuple[int, int]):
        head = TokenPoolingHead(
            input_dimension=64,
            pooling_method=pooling_method,
            sequence_length=20,
        )
        assert head.output_dim == expected_dim

    @pytest.mark.parametrize("num_prefix_tokens", [1, 3])
    def test_none_output_dim_subtracts_prefix_tokens(self, num_prefix_tokens: int):
        head = TokenPoolingHead(
            input_dimension=64,
            pooling_method=PoolingMethod.NONE.value,
            sequence_length=20,
            num_prefix_tokens=num_prefix_tokens,
        )
        assert head.output_dim == (20 - num_prefix_tokens, 64)

    def test_default_returns_cls_token(
        self,
        rng: np.random.Generator,
    ):
        batch_size = 2
        sequence_length = 10
        feature_dimension = 32
        head = TokenPoolingHead(
            input_dimension=feature_dimension,
            pooling_method=PoolingMethod.DEFAULT.value,
        )
        hidden_states = torch.from_numpy(
            rng.standard_normal(
                (batch_size, sequence_length, feature_dimension)
            ).astype(np.float32)
        )
        output = head(hidden_states)
        assert output.shape == (batch_size, feature_dimension)
        expected = hidden_states[:, 0]
        assert torch.allclose(output, expected)

    @pytest.mark.parametrize("num_prefix_tokens", [0, 1, 5])
    def test_average_respects_num_prefix_tokens(
        self,
        rng: np.random.Generator,
        num_prefix_tokens: int,
    ):
        batch_size = 2
        sequence_length = 10
        feature_dimension = 32
        head = TokenPoolingHead(
            input_dimension=feature_dimension,
            pooling_method=PoolingMethod.AVERAGE.value,
            num_prefix_tokens=num_prefix_tokens,
        )
        hidden_states = torch.from_numpy(
            rng.standard_normal(
                (batch_size, sequence_length, feature_dimension)
            ).astype(np.float32)
        )
        output = head(hidden_states)
        start = num_prefix_tokens
        expected = hidden_states[:, start:].mean(dim=1)
        assert torch.allclose(output, expected)

    @pytest.mark.parametrize("num_prefix_tokens", [0, 1, 5])
    def test_learned_aggregation_produces_input_sensitive_output(
        self,
        rng: np.random.Generator,
        num_prefix_tokens: int,
    ):
        batch_size = 2
        sequence_length = 10
        feature_dimension = 32
        head = TokenPoolingHead(
            input_dimension=feature_dimension,
            pooling_method=PoolingMethod.LEARNED_AGGREGATION.value,
            num_prefix_tokens=num_prefix_tokens,
        )
        hidden_states_a = torch.from_numpy(
            rng.standard_normal(
                (batch_size, sequence_length, feature_dimension)
            ).astype(np.float32)
        )
        hidden_states_b = torch.from_numpy(
            rng.standard_normal(
                (batch_size, sequence_length, feature_dimension)
            ).astype(np.float32)
        )
        output_a = head(hidden_states_a)
        output_b = head(hidden_states_b)
        assert output_a.shape == (batch_size, feature_dimension)
        assert not torch.allclose(output_a, output_b)

    @pytest.mark.parametrize("num_prefix_tokens", [0, 1, 5])
    def test_none_returns_sequence_with_prefix_tokens_stripped(
        self,
        rng: np.random.Generator,
        num_prefix_tokens: int,
    ):
        batch_size = 2
        sequence_length = 10
        feature_dimension = 32
        head = TokenPoolingHead(
            input_dimension=feature_dimension,
            pooling_method=PoolingMethod.NONE.value,
            num_prefix_tokens=num_prefix_tokens,
        )
        hidden_states = torch.from_numpy(
            rng.standard_normal(
                (batch_size, sequence_length, feature_dimension)
            ).astype(np.float32)
        )
        output = head(hidden_states)
        start = num_prefix_tokens
        expected = hidden_states[:, start:]
        assert torch.equal(output, expected)

    def test_unsupported_method_raises(self):
        invalid_method = "nonexistent_pooling"
        head = TokenPoolingHead(
            input_dimension=64,
            pooling_method=invalid_method,
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Unsupported token pooling method: {invalid_method}. "
                f"Supported: {[e.value for e in PoolingMethod]}"
            ),
        ):
            head(torch.zeros(2, 10, 64))
