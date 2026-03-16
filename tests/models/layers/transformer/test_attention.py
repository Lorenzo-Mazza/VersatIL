"""Tests for versatil.models.layers.transformer.attention module."""
import re
from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.models.layers.constants import AttentionType
from versatil.models.layers.transformer.attention import CachedAttention
from versatil.models.layers.transformer.kv_cache import LayerKVCache


@pytest.fixture
def cached_attention_factory() -> Callable[..., CachedAttention]:
    """Factory for CachedAttention modules."""

    def factory(
        embedding_dimension: int = 32,
        number_of_heads: int = 4,
        number_of_key_value_heads: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        attention_type: str = AttentionType.MULTI_HEAD.value,
    ) -> CachedAttention:
        return CachedAttention(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_key_value_heads,
            dropout=dropout,
            bias=bias,
            attention_type=attention_type,
        )

    return factory


@pytest.fixture
def self_attention_cache_factory(
    rng: np.random.Generator,
) -> Callable[..., LayerKVCache]:
    """Factory for LayerKVCache with populated self-attention keys/values."""

    def factory(
        batch_size: int = 2,
        number_of_heads: int = 4,
        cached_length: int = 3,
        head_dimension: int = 8,
    ) -> LayerKVCache:
        self_keys = torch.from_numpy(
            rng.standard_normal(
                (batch_size, number_of_heads, cached_length, head_dimension)
            ).astype(np.float32)
        )
        self_values = torch.from_numpy(
            rng.standard_normal(
                (batch_size, number_of_heads, cached_length, head_dimension)
            ).astype(np.float32)
        )
        return LayerKVCache(
            self_attention_keys=self_keys,
            self_attention_values=self_values,
        )

    return factory


@pytest.fixture
def cross_attention_cache_factory(
    rng: np.random.Generator,
) -> Callable[..., LayerKVCache]:
    """Factory for LayerKVCache with precomputed cross-attention keys/values."""

    def factory(
        batch_size: int = 2,
        number_of_query_heads: int = 4,
        number_of_key_value_heads: int = 4,
        memory_length: int = 6,
        head_dimension: int = 8,
    ) -> LayerKVCache:
        self_keys = torch.from_numpy(
            rng.standard_normal(
                (batch_size, number_of_key_value_heads, 0, head_dimension)
            ).astype(np.float32)
        )
        self_values = torch.from_numpy(
            rng.standard_normal(
                (batch_size, number_of_key_value_heads, 0, head_dimension)
            ).astype(np.float32)
        )
        cross_keys = torch.from_numpy(
            rng.standard_normal(
                (batch_size, number_of_key_value_heads, memory_length, head_dimension)
            ).astype(np.float32)
        )
        cross_values = torch.from_numpy(
            rng.standard_normal(
                (batch_size, number_of_key_value_heads, memory_length, head_dimension)
            ).astype(np.float32)
        )
        return LayerKVCache(
            self_attention_keys=self_keys,
            self_attention_values=self_values,
            cross_attention_keys=cross_keys,
            cross_attention_values=cross_values,
        )

    return factory


class TestCachedAttentionInitialization:

    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("number_of_heads", [4, 8])
    @pytest.mark.parametrize("attention_type", [AttentionType.MULTI_HEAD.value, AttentionType.GROUPED_QUERY.value])
    def test_stores_configuration(
        self,
        cached_attention_factory: Callable[..., CachedAttention],
        embedding_dimension: int,
        number_of_heads: int,
        attention_type: str,
    ):
        number_of_key_value_heads = (
            number_of_heads // 2 if attention_type == AttentionType.GROUPED_QUERY.value else None
        )
        module = cached_attention_factory(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_key_value_heads,
            attention_type=attention_type,
        )
        assert module.embedding_dimension == embedding_dimension
        assert module.number_of_heads == number_of_heads
        assert module.head_dimension == embedding_dimension // number_of_heads
        assert module.attention_type == attention_type

    def test_multi_head_sets_kv_heads_equal_to_query_heads(
        self, cached_attention_factory: Callable[..., CachedAttention]
    ):
        module = cached_attention_factory(
            number_of_heads=8,
            attention_type=AttentionType.MULTI_HEAD.value,
        )
        assert module.number_of_key_value_heads == 8
        assert module.group_size == 1

    def test_grouped_query_sets_group_size(
        self, cached_attention_factory: Callable[..., CachedAttention]
    ):
        module = cached_attention_factory(
            number_of_heads=8,
            number_of_key_value_heads=2,
            attention_type=AttentionType.GROUPED_QUERY.value,
        )
        assert module.number_of_key_value_heads == 2
        assert module.group_size == 4

    def test_output_projection_has_initialization_flag(
        self, cached_attention_factory: Callable[..., CachedAttention]
    ):
        module = cached_attention_factory()
        assert module.output_projection.SQUARE_ROOT_WEIGHT is True


class TestCachedAttentionValidation:

    def test_embedding_not_divisible_by_heads_raises(
        self, cached_attention_factory: Callable[..., CachedAttention]
    ):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "embedding_dimension (33) must be divisible by number_of_heads (4)"
            ),
        ):
            cached_attention_factory(
                embedding_dimension=33, number_of_heads=4
            )

    def test_gqa_without_kv_heads_raises(
        self, cached_attention_factory: Callable[..., CachedAttention]
    ):
        with pytest.raises(
            ValueError,
            match=re.escape("number_of_key_value_heads required for GQA"),
        ):
            cached_attention_factory(
                attention_type=AttentionType.GROUPED_QUERY.value,
                number_of_key_value_heads=None,
            )

    def test_gqa_heads_not_divisible_raises(
        self, cached_attention_factory: Callable[..., CachedAttention]
    ):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "number_of_heads (8) must be divisible by number_of_key_value_heads (3)"
            ),
        ):
            cached_attention_factory(
                embedding_dimension=64,
                number_of_heads=8,
                number_of_key_value_heads=3,
                attention_type=AttentionType.GROUPED_QUERY.value,
            )

    def test_unsupported_attention_type_raises(
        self, cached_attention_factory: Callable[..., CachedAttention]
    ):
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Unsupported attention type: invalid_type. "
                f"Must be one of {[e.value for e in AttentionType]}."
            ),
        ):
            cached_attention_factory(attention_type="invalid_type")


class TestCachedAttentionForward:

    def test_self_attention_output_shape(
        self,
        cached_attention_factory: Callable[..., CachedAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        module = cached_attention_factory(
            embedding_dimension=32, number_of_heads=4
        )
        sequence = sequence_tensor_factory(
            batch_size=2, sequence_length=5, embedding_dimension=32
        )
        output, cache = module(
            query_input=sequence,
            key_input=sequence,
            value_input=sequence,
        )
        assert output.shape == (2, 5, 32)
        assert cache is None

    def test_cross_attention_output_shape(
        self,
        cached_attention_factory: Callable[..., CachedAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        module = cached_attention_factory(
            embedding_dimension=32, number_of_heads=4
        )
        query = sequence_tensor_factory(
            batch_size=2, sequence_length=5, embedding_dimension=32
        )
        memory = sequence_tensor_factory(
            batch_size=2, sequence_length=8, embedding_dimension=32
        )
        output, cache = module(
            query_input=query,
            key_input=memory,
            value_input=memory,
        )
        assert output.shape == (2, 5, 32)

    def test_grouped_query_attention_output_shape(
        self,
        cached_attention_factory: Callable[..., CachedAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        module = cached_attention_factory(
            embedding_dimension=32,
            number_of_heads=4,
            number_of_key_value_heads=2,
            attention_type=AttentionType.GROUPED_QUERY.value,
        )
        sequence = sequence_tensor_factory(
            batch_size=2, sequence_length=5, embedding_dimension=32
        )
        output, cache = module(
            query_input=sequence,
            key_input=sequence,
            value_input=sequence,
        )
        assert output.shape == (2, 5, 32)

    def test_attention_mask_zeroes_padded_positions(
        self,
        cached_attention_factory: Callable[..., CachedAttention],
        rng: np.random.Generator,
    ):
        embedding_dimension = 32
        number_of_heads = 4
        batch_size = 2
        sequence_length = 4
        module = cached_attention_factory(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
        )
        module.eval()
        sequence = torch.from_numpy(
            rng.standard_normal(
                (batch_size, sequence_length, embedding_dimension)
            ).astype(np.float32)
        )
        # Mask last 2 positions in the key for batch 0
        mask = torch.zeros(
            batch_size, 1, sequence_length, sequence_length, dtype=torch.bool
        )
        mask[0, :, :, 2:] = True
        output_masked, _ = module(
            query_input=sequence,
            key_input=sequence,
            value_input=sequence,
            attention_mask=mask,
        )
        # Without mask
        output_unmasked, _ = module(
            query_input=sequence,
            key_input=sequence,
            value_input=sequence,
        )
        # Outputs should differ for batch 0 (masked) but not necessarily for batch 1
        assert not torch.allclose(output_masked[0], output_unmasked[0], atol=1e-6)

    def test_self_attention_cache_returns_updated_cache(
        self,
        cached_attention_factory: Callable[..., CachedAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        module = cached_attention_factory(
            embedding_dimension=32, number_of_heads=4
        )
        sequence = sequence_tensor_factory(
            batch_size=2, sequence_length=1, embedding_dimension=32
        )
        empty_cache = LayerKVCache(
            self_attention_keys=torch.empty(2, 4, 0, 8),
            self_attention_values=torch.empty(2, 4, 0, 8),
        )
        output, new_cache = module(
            query_input=sequence,
            key_input=sequence,
            value_input=sequence,
            layer_cache=empty_cache,
            use_self_attention_cache=True,
        )
        assert new_cache is not None
        assert new_cache.get_length() == 1

    def test_cached_forward_matches_causal_uncached_forward(
        self,
        cached_attention_factory: Callable[..., CachedAttention],
        rng: np.random.Generator,
    ):
        embedding_dimension = 32
        number_of_heads = 4
        head_dimension = embedding_dimension // number_of_heads
        batch_size = 2
        sequence_length = 4
        module = cached_attention_factory(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
        )
        module.eval()
        full_sequence = torch.from_numpy(
            rng.standard_normal(
                (batch_size, sequence_length, embedding_dimension)
            ).astype(np.float32)
        )
        # Full forward with causal mask (to match what step-by-step caching sees)
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        full_output, _ = module(
            query_input=full_sequence,
            key_input=full_sequence,
            value_input=full_sequence,
            attention_mask=causal_mask,
        )
        # Incremental forward with cache
        cache = LayerKVCache(
            self_attention_keys=torch.empty(batch_size, number_of_heads, 0, head_dimension),
            self_attention_values=torch.empty(batch_size, number_of_heads, 0, head_dimension),
        )
        cached_outputs = []
        for step in range(sequence_length):
            token = full_sequence[:, step : step + 1, :]
            step_output, cache = module(
                query_input=token,
                key_input=token,
                value_input=token,
                layer_cache=cache,
                use_self_attention_cache=True,
            )
            cached_outputs.append(step_output)
        cached_full_output = torch.cat(cached_outputs, dim=1)
        assert torch.allclose(full_output, cached_full_output, atol=1e-5)

    def test_cross_attention_cache_uses_precomputed_kv(
        self,
        cached_attention_factory: Callable[..., CachedAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        cross_attention_cache_factory: Callable[..., LayerKVCache],
    ):
        module = cached_attention_factory(
            embedding_dimension=32, number_of_heads=4
        )
        query = sequence_tensor_factory(
            batch_size=2, sequence_length=3, embedding_dimension=32
        )
        cache = cross_attention_cache_factory(
            batch_size=2,
            number_of_query_heads=4,
            number_of_key_value_heads=4,
            memory_length=6,
            head_dimension=8,
        )
        output, _ = module(
            query_input=query,
            layer_cache=cache,
            use_cross_attention_cache=True,
        )
        assert output.shape == (2, 3, 32)

    def test_missing_key_value_without_cross_cache_raises(
        self,
        cached_attention_factory: Callable[..., CachedAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        module = cached_attention_factory()
        query = sequence_tensor_factory(
            batch_size=2, sequence_length=3, embedding_dimension=32
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "key_input and value_input required when not using cross_attention_cache"
            ),
        ):
            module(query_input=query)

    def test_cross_cache_without_precomputed_kv_raises(
        self,
        cached_attention_factory: Callable[..., CachedAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        module = cached_attention_factory()
        query = sequence_tensor_factory(
            batch_size=2, sequence_length=3, embedding_dimension=32
        )
        cache = LayerKVCache(
            self_attention_keys=torch.empty(2, 4, 0, 8),
            self_attention_values=torch.empty(2, 4, 0, 8),
            cross_attention_keys=None,
            cross_attention_values=None,
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "layer_cache must contain precomputed cross_attention K/V when use_cross_attention_cache=True"
            ),
        ):
            module(
                query_input=query,
                layer_cache=cache,
                use_cross_attention_cache=True,
            )


class TestCachedAttentionComputeQueryKeyValue:

    def test_mha_projects_all_heads(
        self,
        cached_attention_factory: Callable[..., CachedAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        module = cached_attention_factory(
            embedding_dimension=32,
            number_of_heads=4,
            attention_type=AttentionType.MULTI_HEAD.value,
        )
        sequence = sequence_tensor_factory(
            batch_size=2, sequence_length=5, embedding_dimension=32
        )
        queries, keys, values = module.compute_query_key_value(
            query_input=sequence,
            key_input=sequence,
            value_input=sequence,
        )
        assert queries.shape == (2, 4, 5, 8)
        assert keys.shape == (2, 4, 5, 8)
        assert values.shape == (2, 4, 5, 8)

    def test_gqa_produces_fewer_kv_heads(
        self,
        cached_attention_factory: Callable[..., CachedAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        module = cached_attention_factory(
            embedding_dimension=32,
            number_of_heads=4,
            number_of_key_value_heads=2,
            attention_type=AttentionType.GROUPED_QUERY.value,
        )
        sequence = sequence_tensor_factory(
            batch_size=2, sequence_length=5, embedding_dimension=32
        )
        queries, keys, values = module.compute_query_key_value(
            query_input=sequence,
            key_input=sequence,
            value_input=sequence,
        )
        assert queries.shape == (2, 4, 5, 8)
        assert keys.shape == (2, 2, 5, 8)
        assert values.shape == (2, 2, 5, 8)


class TestCachedAttentionCrossAttentionBehavior:

    def test_different_memory_produces_different_output(
        self,
        cached_attention_factory: Callable[..., CachedAttention],
        rng: np.random.Generator,
    ):
        embedding_dimension = 32
        number_of_heads = 4
        batch_size = 2
        module = cached_attention_factory(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
        )
        module.eval()
        query = torch.from_numpy(
            rng.standard_normal(
                (batch_size, 3, embedding_dimension)
            ).astype(np.float32)
        )
        memory_a = torch.from_numpy(
            rng.standard_normal(
                (batch_size, 6, embedding_dimension)
            ).astype(np.float32)
        )
        memory_b = torch.from_numpy(
            rng.standard_normal(
                (batch_size, 6, embedding_dimension)
            ).astype(np.float32)
        )
        output_a, _ = module(
            query_input=query, key_input=memory_a, value_input=memory_a
        )
        output_b, _ = module(
            query_input=query, key_input=memory_b, value_input=memory_b
        )
        assert not torch.allclose(output_a, output_b, atol=1e-6)

    def test_gqa_and_mha_produce_same_output_with_shared_weights(
        self,
        rng: np.random.Generator,
    ):
        # GQA with group_size=1 (kv_heads == query_heads) should behave like MHA
        embedding_dimension = 32
        number_of_heads = 4
        batch_size = 2
        sequence_length = 5
        mha = CachedAttention(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            attention_type=AttentionType.MULTI_HEAD.value,
        )
        gqa_same_heads = CachedAttention(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_heads,
            attention_type=AttentionType.GROUPED_QUERY.value,
        )
        # Copy weights from MHA to GQA
        gqa_same_heads.load_state_dict(mha.state_dict())
        mha.eval()
        gqa_same_heads.eval()
        sequence = torch.from_numpy(
            rng.standard_normal(
                (batch_size, sequence_length, embedding_dimension)
            ).astype(np.float32)
        )
        output_mha, _ = mha(
            query_input=sequence, key_input=sequence, value_input=sequence
        )
        output_gqa, _ = gqa_same_heads(
            query_input=sequence, key_input=sequence, value_input=sequence
        )
        assert torch.allclose(output_mha, output_gqa, atol=1e-6)
