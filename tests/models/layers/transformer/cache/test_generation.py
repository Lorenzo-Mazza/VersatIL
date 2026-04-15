"""Tests for versatil.models.layers.transformer.cache.generation module."""

from collections.abc import Callable

import pytest
import torch

from versatil.models.layers.transformer.cache.generation import (
    GenerationCache,
    GenerationLayerCache,
    initialize_generation_cache,
    update_generation_layer_cache,
)


class TestGenerationLayerCache:
    @pytest.mark.parametrize("cached_length", [0, 5, 10])
    def test_get_length_returns_cached_sequence_length(
        self,
        generation_cache_factory: Callable[..., GenerationLayerCache],
        cached_length: int,
    ):
        cache = generation_cache_factory(cached_length=cached_length)
        assert cache.get_length() == cached_length

    def test_keys_and_values_have_matching_shapes(
        self,
        generation_cache_factory: Callable[..., GenerationLayerCache],
    ):
        cache = generation_cache_factory(
            batch_size=3, number_of_heads=8, cached_length=7, head_dimension=16
        )
        assert cache.keys.shape == (3, 8, 7, 16)
        assert cache.values.shape == cache.keys.shape


class TestGenerationCache:
    def test_get_length_delegates_to_first_layer(
        self,
        generation_cache_factory: Callable[..., GenerationLayerCache],
    ):
        layers = [
            generation_cache_factory(cached_length=10),
            generation_cache_factory(cached_length=10),
        ]
        cache = GenerationCache(layers=layers)
        assert cache.get_length() == 10

    def test_get_length_returns_zero_for_empty_layers(self):
        cache = GenerationCache(layers=[])
        assert cache.get_length() == 0

    def test_key_padding_mask_defaults_to_none(
        self,
        generation_cache_factory: Callable[..., GenerationLayerCache],
    ):
        cache = GenerationCache(layers=[generation_cache_factory(cached_length=3)])
        assert cache.key_padding_mask is None

    def test_key_padding_mask_stored(
        self,
        generation_cache_factory: Callable[..., GenerationLayerCache],
    ):
        mask = torch.tensor([[False, True, False], [True, False, False]])
        cache = GenerationCache(
            layers=[generation_cache_factory(cached_length=3)],
            key_padding_mask=mask,
        )
        assert torch.equal(cache.key_padding_mask, mask)


class TestInitializeGenerationCache:
    @pytest.mark.parametrize("number_of_layers", [1, 4])
    def test_creates_correct_number_of_layer_caches(
        self, device: torch.device, number_of_layers: int
    ):
        caches = initialize_generation_cache(
            batch_size=2,
            num_layers=number_of_layers,
            num_heads=4,
            head_dimension=8,
            device=device,
        )
        assert len(caches) == number_of_layers

    def test_initial_caches_have_zero_length_and_correct_shape(
        self, device: torch.device
    ):
        caches = initialize_generation_cache(
            batch_size=3,
            num_layers=2,
            num_heads=4,
            head_dimension=16,
            device=device,
        )
        for cache in caches:
            assert cache.get_length() == 0
            assert cache.keys.shape == (3, 4, 0, 16)
            assert cache.values.shape == (3, 4, 0, 16)

    def test_cache_device_placement(self, device: torch.device):
        caches = initialize_generation_cache(
            batch_size=2,
            num_layers=1,
            num_heads=4,
            head_dimension=8,
            device=device,
        )
        assert caches[0].keys.device.type == device.type

    def test_cache_dtype(self, device: torch.device):
        caches = initialize_generation_cache(
            batch_size=2,
            num_layers=1,
            num_heads=4,
            head_dimension=8,
            device=device,
            dtype=torch.float16,
        )
        assert caches[0].keys.dtype == torch.float16


class TestUpdateGenerationLayerCache:
    def test_concatenates_new_keys_along_sequence_dimension(
        self,
        generation_cache_factory: Callable[..., GenerationLayerCache],
        new_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ):
        cache = generation_cache_factory(cached_length=3)
        new_keys, new_values = new_kv_factory(new_length=1)
        updated = update_generation_layer_cache(
            cache=cache, new_keys=new_keys, new_values=new_values
        )
        assert updated.get_length() == 4

    def test_preserves_existing_cached_values(
        self,
        generation_cache_factory: Callable[..., GenerationLayerCache],
        new_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ):
        cache = generation_cache_factory(cached_length=3)
        original_keys = cache.keys.clone()
        new_keys, new_values = new_kv_factory(new_length=1)
        updated = update_generation_layer_cache(
            cache=cache, new_keys=new_keys, new_values=new_values
        )
        assert torch.equal(updated.keys[:, :, :3, :], original_keys)

    def test_appends_new_values_at_end(
        self,
        generation_cache_factory: Callable[..., GenerationLayerCache],
        new_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ):
        cache = generation_cache_factory(cached_length=2)
        new_keys, new_values = new_kv_factory(new_length=1)
        updated = update_generation_layer_cache(
            cache=cache, new_keys=new_keys, new_values=new_values
        )
        assert torch.equal(updated.keys[:, :, 2:, :], new_keys)
        assert torch.equal(updated.values[:, :, 2:, :], new_values)

    def test_multiple_updates_accumulate(
        self,
        generation_cache_factory: Callable[..., GenerationLayerCache],
        new_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ):
        cache = generation_cache_factory(cached_length=0)
        for _step in range(5):
            new_keys, new_values = new_kv_factory(new_length=1)
            cache = update_generation_layer_cache(
                cache=cache, new_keys=new_keys, new_values=new_values
            )
        assert cache.get_length() == 5
