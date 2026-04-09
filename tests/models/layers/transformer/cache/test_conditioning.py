"""Tests for versatil.models.layers.transformer.cache.conditioning module."""

from collections.abc import Callable

from versatil.models.layers.transformer.cache.conditioning import (
    ConditioningCache,
    ConditioningLayerCache,
)


class TestConditioningLayerCache:
    def test_queries_default_to_none(
        self,
        conditioning_cache_factory: Callable[..., ConditioningLayerCache],
    ):
        cache = conditioning_cache_factory()
        assert cache.queries is None

    def test_queries_populated_when_requested(
        self,
        conditioning_cache_with_queries_factory: Callable[..., ConditioningLayerCache],
    ):
        cache = conditioning_cache_with_queries_factory(
            sequence_length=6,
            number_of_heads=4,
            head_dimension=8,
        )
        assert cache.queries.shape == (2, 4, 6, 8)

    def test_keys_and_values_have_matching_shapes(
        self,
        conditioning_cache_factory: Callable[..., ConditioningLayerCache],
    ):
        cache = conditioning_cache_factory(
            batch_size=3,
            number_of_key_value_heads=8,
            memory_length=5,
            head_dimension=16,
        )
        assert cache.keys.shape == (3, 8, 5, 16)
        assert cache.values.shape == cache.keys.shape


class TestConditioningCache:
    def test_stores_layer_caches(
        self,
        conditioning_cache_factory: Callable[..., ConditioningLayerCache],
    ):
        layers = [
            conditioning_cache_factory(memory_length=6),
            conditioning_cache_factory(memory_length=6),
            conditioning_cache_factory(memory_length=6),
        ]
        cache = ConditioningCache(layers=layers)
        assert len(cache.layers) == 3

    def test_getitem_returns_layer_cache(
        self,
        conditioning_cache_factory: Callable[..., ConditioningLayerCache],
    ):
        layer_0 = conditioning_cache_factory(memory_length=5)
        layer_1 = conditioning_cache_factory(memory_length=7)
        cache = ConditioningCache(layers=[layer_0, layer_1])
        assert cache[0].keys.shape[2] == 5
        assert cache[1].keys.shape[2] == 7

    def test_getitem_returns_none_for_none_layer(
        self,
        conditioning_cache_factory: Callable[..., ConditioningLayerCache],
    ):
        layer_0 = conditioning_cache_factory(memory_length=5)
        cache = ConditioningCache(layers=[layer_0, None])
        assert cache[0] is not None
        assert cache[1] is None
