"""Tests for versatil.models.layers.diffusion_transformer.query_key_norm module."""

from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.models.layers.diffusion_transformer.query_key_norm import QueryKeyNorm


@pytest.fixture
def query_key_norm_factory() -> Callable[..., QueryKeyNorm]:
    def factory(
        head_dimension: int = 16,
        epsilon: float = 1e-6,
    ) -> QueryKeyNorm:
        return QueryKeyNorm(
            head_dimension=head_dimension,
            epsilon=epsilon,
        )

    return factory


@pytest.fixture
def attention_head_tensor_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    # Semantic factory for 4D attention head tensors (B, num_heads, S, head_dim)

    def factory(
        batch_size: int = 2,
        number_of_heads: int = 4,
        sequence_length: int = 8,
        head_dimension: int = 16,
    ) -> torch.Tensor:
        data = rng.standard_normal(
            (batch_size, number_of_heads, sequence_length, head_dimension)
        ).astype(np.float32)
        return torch.from_numpy(data)

    return factory


class TestQueryKeyNormInitialization:
    @pytest.mark.parametrize("head_dimension", [16, 64])
    @pytest.mark.parametrize("epsilon", [1e-6, 1e-8])
    def test_stores_configuration(
        self,
        query_key_norm_factory: Callable[..., QueryKeyNorm],
        head_dimension: int,
        epsilon: float,
    ):
        norm = query_key_norm_factory(
            head_dimension=head_dimension,
            epsilon=epsilon,
        )
        assert norm.query_norm.eps == epsilon
        assert norm.key_norm.eps == epsilon

    def test_query_and_key_norms_have_independent_weights(
        self,
        query_key_norm_factory: Callable[..., QueryKeyNorm],
    ):
        norm = query_key_norm_factory(head_dimension=16)
        norm.query_norm.weight.data.fill_(3.0)
        assert not torch.allclose(norm.query_norm.weight, norm.key_norm.weight)


class TestQueryKeyNormForward:
    def test_output_shapes_match_inputs(
        self,
        query_key_norm_factory: Callable[..., QueryKeyNorm],
        attention_head_tensor_factory: Callable[..., torch.Tensor],
    ):
        head_dimension = 16
        norm = query_key_norm_factory(head_dimension=head_dimension)
        query = attention_head_tensor_factory(head_dimension=head_dimension)
        key = attention_head_tensor_factory(head_dimension=head_dimension)
        normalized_query, normalized_key = norm(query, key)
        assert normalized_query.shape == query.shape
        assert normalized_key.shape == key.shape

    def test_normalization_produces_unit_rms_outputs(
        self,
        query_key_norm_factory: Callable[..., QueryKeyNorm],
        attention_head_tensor_factory: Callable[..., torch.Tensor],
    ):
        head_dimension = 64
        norm = query_key_norm_factory(head_dimension=head_dimension)
        query = attention_head_tensor_factory(head_dimension=head_dimension)
        key = attention_head_tensor_factory(head_dimension=head_dimension)
        normalized_query, normalized_key = norm(query, key)
        query_rms = torch.sqrt(torch.mean(normalized_query**2, dim=-1))
        key_rms = torch.sqrt(torch.mean(normalized_key**2, dim=-1))
        assert torch.allclose(query_rms, torch.ones_like(query_rms), atol=1e-4)
        assert torch.allclose(key_rms, torch.ones_like(key_rms), atol=1e-4)

    def test_query_and_key_normalized_independently(
        self,
        query_key_norm_factory: Callable[..., QueryKeyNorm],
        attention_head_tensor_factory: Callable[..., torch.Tensor],
    ):
        head_dimension = 16
        norm = query_key_norm_factory(head_dimension=head_dimension)
        query = attention_head_tensor_factory(head_dimension=head_dimension)
        key = attention_head_tensor_factory(head_dimension=head_dimension)
        norm.key_norm.weight.data.fill_(5.0)
        normalized_query, normalized_key = norm(query, key)
        query_rms = torch.sqrt(torch.mean(normalized_query**2, dim=-1))
        key_rms = torch.sqrt(torch.mean(normalized_key**2, dim=-1))
        assert torch.allclose(query_rms, torch.ones_like(query_rms), atol=1e-4)
        assert torch.allclose(key_rms, torch.full_like(key_rms, 5.0), atol=1e-2)

    def test_gradient_flows_through_both_outputs(
        self,
        query_key_norm_factory: Callable[..., QueryKeyNorm],
        attention_head_tensor_factory: Callable[..., torch.Tensor],
    ):
        head_dimension = 16
        norm = query_key_norm_factory(head_dimension=head_dimension)
        query = attention_head_tensor_factory(head_dimension=head_dimension)
        key = attention_head_tensor_factory(head_dimension=head_dimension)
        query.requires_grad_(True)
        key.requires_grad_(True)
        normalized_query, normalized_key = norm(query, key)
        loss = normalized_query.sum() + normalized_key.sum()
        loss.backward()
        assert query.grad is not None
        assert key.grad is not None
        assert torch.all(torch.isfinite(query.grad))
        assert torch.all(torch.isfinite(key.grad))
