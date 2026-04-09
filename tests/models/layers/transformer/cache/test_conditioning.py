"""Tests for versatil.models.layers.transformer.cache.conditioning module."""

from collections.abc import Callable

import pytest
import torch

from versatil.models.layers.transformer.cache.conditioning import (
    ConditioningCache,
    ConditioningLayerCache,
)


@pytest.fixture
def conditioning_layer_cache_factory(
    precomputed_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> Callable[..., ConditioningLayerCache]:
    """Factory for ConditioningLayerCache with optional queries field."""

    def factory(
        batch_size: int = 2,
        number_of_heads: int = 4,
        conditioning_length: int = 5,
        head_dimension: int = 8,
        include_queries: bool = False,
    ) -> ConditioningLayerCache:
        keys, values = precomputed_kv_factory(
            batch_size=batch_size,
            key_value_length=conditioning_length,
            number_of_heads=number_of_heads,
            head_dimension=head_dimension,
        )
        queries = None
        if include_queries:
            queries, _ = precomputed_kv_factory(
                batch_size=batch_size,
                key_value_length=conditioning_length,
                number_of_heads=number_of_heads,
                head_dimension=head_dimension,
            )
        return ConditioningLayerCache(keys=keys, values=values, queries=queries)

    return factory


class TestConditioningLayerCache:
    def test_queries_default_to_none(
        self,
        conditioning_layer_cache_factory: Callable[..., ConditioningLayerCache],
    ):
        cache = conditioning_layer_cache_factory(include_queries=False)
        assert cache.queries is None

    def test_queries_populated_when_requested(
        self,
        conditioning_layer_cache_factory: Callable[..., ConditioningLayerCache],
    ):
        cache = conditioning_layer_cache_factory(
            include_queries=True,
            conditioning_length=6,
            number_of_heads=4,
            head_dimension=8,
        )
        assert cache.queries.shape == (2, 4, 6, 8)

    def test_keys_and_values_have_matching_shapes(
        self,
        conditioning_layer_cache_factory: Callable[..., ConditioningLayerCache],
    ):
        cache = conditioning_layer_cache_factory(
            batch_size=3, number_of_heads=8, conditioning_length=5, head_dimension=16
        )
        assert cache.keys.shape == (3, 8, 5, 16)
        assert cache.values.shape == cache.keys.shape


class TestConditioningCache:
    def test_stores_layer_caches(
        self,
        conditioning_layer_cache_factory: Callable[..., ConditioningLayerCache],
    ):
        layers = [
            conditioning_layer_cache_factory(conditioning_length=6),
            conditioning_layer_cache_factory(conditioning_length=6),
            conditioning_layer_cache_factory(conditioning_length=6),
        ]
        cache = ConditioningCache(layers=layers)
        assert len(cache.layers) == 3

    def test_getitem_returns_layer_cache(
        self,
        conditioning_layer_cache_factory: Callable[..., ConditioningLayerCache],
    ):
        layer_0 = conditioning_layer_cache_factory(conditioning_length=5)
        layer_1 = conditioning_layer_cache_factory(conditioning_length=7)
        cache = ConditioningCache(layers=[layer_0, layer_1])
        assert cache[0].keys.shape[2] == 5
        assert cache[1].keys.shape[2] == 7

    def test_getitem_returns_none_for_none_layer(
        self,
        conditioning_layer_cache_factory: Callable[..., ConditioningLayerCache],
    ):
        layer_0 = conditioning_layer_cache_factory(conditioning_length=5)
        cache = ConditioningCache(layers=[layer_0, None])
        assert cache[0] is not None
        assert cache[1] is None
