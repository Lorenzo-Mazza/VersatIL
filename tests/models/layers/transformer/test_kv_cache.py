"""Tests for versatil.models.layers.transformer.kv_cache module."""

from collections.abc import Callable

import pytest
import torch

from versatil.models.layers.transformer.kv_cache import (
    DecoderKVCache,
    LayerKVCache,
    initialize_decoder_cache,
    update_layer_cache,
)


@pytest.fixture
def layer_cache_factory(
    precomputed_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> Callable[..., LayerKVCache]:
    """Factory for LayerKVCache with populated self-attention keys/values."""

    def factory(
        batch_size: int = 2,
        number_of_heads: int = 4,
        cached_length: int = 3,
        head_dimension: int = 8,
        include_cross_attention: bool = False,
        cross_attention_length: int = 5,
    ) -> LayerKVCache:
        self_keys, self_values = precomputed_kv_factory(
            batch_size=batch_size,
            key_value_length=cached_length,
            number_of_heads=number_of_heads,
            head_dimension=head_dimension,
        )
        cross_keys = None
        cross_values = None
        if include_cross_attention:
            cross_keys, cross_values = precomputed_kv_factory(
                batch_size=batch_size,
                key_value_length=cross_attention_length,
                number_of_heads=number_of_heads,
                head_dimension=head_dimension,
            )
        return LayerKVCache(
            self_attention_keys=self_keys,
            self_attention_values=self_values,
            cross_attention_keys=cross_keys,
            cross_attention_values=cross_values,
        )

    return factory


@pytest.fixture
def new_kv_factory(
    precomputed_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    """Factory for new key/value tensors to append to cache."""

    def factory(
        batch_size: int = 2,
        number_of_heads: int = 4,
        new_length: int = 1,
        head_dimension: int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return precomputed_kv_factory(
            batch_size=batch_size,
            key_value_length=new_length,
            number_of_heads=number_of_heads,
            head_dimension=head_dimension,
        )

    return factory


class TestLayerKVCache:
    @pytest.mark.parametrize("cached_length", [0, 5, 10])
    def test_get_length_returns_cached_sequence_length(
        self,
        layer_cache_factory: Callable[..., LayerKVCache],
        cached_length: int,
    ):
        cache = layer_cache_factory(cached_length=cached_length)
        assert cache.get_length() == cached_length

    def test_cross_attention_fields_default_to_none(
        self,
        layer_cache_factory: Callable[..., LayerKVCache],
    ):
        cache = layer_cache_factory(include_cross_attention=False)
        assert cache.cross_attention_keys is None
        assert cache.cross_attention_values is None

    def test_cross_attention_fields_populated_when_requested(
        self,
        layer_cache_factory: Callable[..., LayerKVCache],
    ):
        cache = layer_cache_factory(
            include_cross_attention=True,
            cross_attention_length=6,
            number_of_heads=4,
            head_dimension=8,
        )
        assert cache.cross_attention_keys.shape == (2, 4, 6, 8)
        assert cache.cross_attention_values.shape == (2, 4, 6, 8)


class TestDecoderKVCache:
    def test_get_length_delegates_to_first_layer(
        self,
        layer_cache_factory: Callable[..., LayerKVCache],
    ):
        layers = [
            layer_cache_factory(cached_length=10),
            layer_cache_factory(cached_length=10),
        ]
        decoder_cache = DecoderKVCache(layers=layers)
        assert decoder_cache.get_length() == 10

    def test_get_length_returns_zero_for_empty_layers(self):
        decoder_cache = DecoderKVCache(layers=[])
        assert decoder_cache.get_length() == 0

    def test_key_padding_mask_defaults_to_none(
        self,
        layer_cache_factory: Callable[..., LayerKVCache],
    ):
        decoder_cache = DecoderKVCache(layers=[layer_cache_factory(cached_length=3)])
        assert decoder_cache.key_padding_mask is None

    def test_key_padding_mask_stored(
        self,
        layer_cache_factory: Callable[..., LayerKVCache],
    ):
        mask = torch.tensor([[False, True, False], [True, False, False]])
        decoder_cache = DecoderKVCache(
            layers=[layer_cache_factory(cached_length=3)],
            key_padding_mask=mask,
        )
        assert torch.equal(decoder_cache.key_padding_mask, mask)


class TestInitializeDecoderCache:
    @pytest.mark.parametrize("number_of_layers", [1, 4])
    def test_creates_correct_number_of_layer_caches(
        self, device: torch.device, number_of_layers: int
    ):
        caches = initialize_decoder_cache(
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
        caches = initialize_decoder_cache(
            batch_size=3,
            num_layers=2,
            num_heads=4,
            head_dimension=16,
            device=device,
        )
        for cache in caches:
            assert cache.get_length() == 0
            assert cache.self_attention_keys.shape == (3, 4, 0, 16)
            assert cache.self_attention_values.shape == (3, 4, 0, 16)

    def test_cache_device_placement(self, device: torch.device):
        caches = initialize_decoder_cache(
            batch_size=2,
            num_layers=1,
            num_heads=4,
            head_dimension=8,
            device=device,
        )
        assert caches[0].self_attention_keys.device.type == device.type

    def test_cache_dtype(self, device: torch.device):
        caches = initialize_decoder_cache(
            batch_size=2,
            num_layers=1,
            num_heads=4,
            head_dimension=8,
            device=device,
            dtype=torch.float16,
        )
        assert caches[0].self_attention_keys.dtype == torch.float16

    def test_cross_attention_fields_are_none(self, device: torch.device):
        caches = initialize_decoder_cache(
            batch_size=2,
            num_layers=2,
            num_heads=4,
            head_dimension=8,
            device=device,
        )
        for cache in caches:
            assert cache.cross_attention_keys is None
            assert cache.cross_attention_values is None


class TestUpdateLayerCache:
    def test_concatenates_new_keys_along_sequence_dimension(
        self,
        layer_cache_factory: Callable[..., LayerKVCache],
        new_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ):
        cache = layer_cache_factory(cached_length=3)
        new_keys, new_values = new_kv_factory(new_length=1)
        updated = update_layer_cache(
            cache=cache, new_keys=new_keys, new_values=new_values
        )
        assert updated.get_length() == 4

    def test_preserves_existing_cached_values(
        self,
        layer_cache_factory: Callable[..., LayerKVCache],
        new_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ):
        cache = layer_cache_factory(cached_length=3)
        original_keys = cache.self_attention_keys.clone()
        new_keys, new_values = new_kv_factory(new_length=1)
        updated = update_layer_cache(
            cache=cache, new_keys=new_keys, new_values=new_values
        )
        assert torch.equal(updated.self_attention_keys[:, :, :3, :], original_keys)

    def test_appends_new_values_at_end(
        self,
        layer_cache_factory: Callable[..., LayerKVCache],
        new_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ):
        cache = layer_cache_factory(cached_length=2)
        new_keys, new_values = new_kv_factory(new_length=1)
        updated = update_layer_cache(
            cache=cache, new_keys=new_keys, new_values=new_values
        )
        assert torch.equal(updated.self_attention_keys[:, :, 2:, :], new_keys)
        assert torch.equal(updated.self_attention_values[:, :, 2:, :], new_values)

    def test_multiple_updates_accumulate(
        self,
        layer_cache_factory: Callable[..., LayerKVCache],
        new_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ):
        cache = layer_cache_factory(cached_length=0)
        for _step in range(5):
            new_keys, new_values = new_kv_factory(new_length=1)
            cache = update_layer_cache(
                cache=cache, new_keys=new_keys, new_values=new_values
            )
        assert cache.get_length() == 5

    def test_update_preserves_cross_attention_data(
        self,
        layer_cache_factory: Callable[..., LayerKVCache],
        new_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ):
        cache = layer_cache_factory(
            cached_length=3,
            include_cross_attention=True,
            cross_attention_length=6,
        )
        original_cross_keys = cache.cross_attention_keys.clone()
        new_keys, new_values = new_kv_factory(new_length=1)
        updated = update_layer_cache(
            cache=cache, new_keys=new_keys, new_values=new_values
        )
        assert torch.equal(updated.cross_attention_keys, original_cross_keys)
        assert updated.cross_attention_values is not None
