"""Tests for versatil.models.layers.transformer.attention.query_key_norm module."""

from collections.abc import Callable

import pytest
import torch

from versatil.models.layers.transformer.attention.query_key_norm import QueryKeyNorm


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
        assert norm.query_norm.epsilon == epsilon
        assert norm.key_norm.epsilon == epsilon

    def test_query_and_key_norms_have_independent_weights(
        self,
        query_key_norm_factory: Callable[..., QueryKeyNorm],
    ):
        norm = query_key_norm_factory(head_dimension=16)
        norm.query_norm.weight.data.fill_(3.0)
        assert not torch.allclose(norm.query_norm.weight, norm.key_norm.weight)


class TestQueryKeyNormForward:
    def test_normalization_produces_unit_rms_outputs(
        self,
        query_key_norm_factory: Callable[..., QueryKeyNorm],
        precomputed_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ):
        head_dimension = 16
        norm = query_key_norm_factory(head_dimension=head_dimension)
        query, key = precomputed_kv_factory(
            key_value_length=8, head_dimension=head_dimension
        )
        normalized_query, normalized_key = norm(query, key)
        query_rms = torch.sqrt(torch.mean(normalized_query**2, dim=-1))
        key_rms = torch.sqrt(torch.mean(normalized_key**2, dim=-1))
        assert torch.allclose(query_rms, torch.ones_like(query_rms), atol=1e-4)
        assert torch.allclose(key_rms, torch.ones_like(key_rms), atol=1e-4)

    def test_query_and_key_normalized_independently(
        self,
        query_key_norm_factory: Callable[..., QueryKeyNorm],
        precomputed_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ):
        head_dimension = 16
        norm = query_key_norm_factory(head_dimension=head_dimension)
        query, key = precomputed_kv_factory(
            key_value_length=8, head_dimension=head_dimension
        )
        norm.key_norm.weight.data.fill_(5.0)
        normalized_query, normalized_key = norm(query, key)
        query_rms = torch.sqrt(torch.mean(normalized_query**2, dim=-1))
        key_rms = torch.sqrt(torch.mean(normalized_key**2, dim=-1))
        assert torch.allclose(query_rms, torch.ones_like(query_rms), atol=1e-4)
        assert torch.allclose(key_rms, torch.full_like(key_rms, 5.0), atol=1e-2)
