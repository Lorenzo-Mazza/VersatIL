"""Tests for versatil.models.layers.transformer.attention.cached_attention module."""

import re
import unittest.mock
from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.models.layers.constants import AttentionType
from versatil.models.layers.transformer.attention.cached_attention import (
    CachedAttention,
)
from versatil.models.layers.transformer.cache.conditioning import ConditioningLayerCache
from versatil.models.layers.transformer.cache.generation import GenerationLayerCache


class TestCachedAttentionInitialization:
    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("number_of_heads", [4, 8])
    @pytest.mark.parametrize(
        "attention_type",
        [AttentionType.MULTI_HEAD.value, AttentionType.GROUPED_QUERY.value],
    )
    def test_stores_configuration(
        self,
        cached_attention_factory: Callable[..., CachedAttention],
        embedding_dimension: int,
        number_of_heads: int,
        attention_type: str,
    ):
        number_of_key_value_heads = (
            number_of_heads // 2
            if attention_type == AttentionType.GROUPED_QUERY.value
            else None
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
    @pytest.mark.parametrize(
        (
            "number_of_heads, number_of_key_value_heads, head_dimension, "
            "attention_type, error_message"
        ),
        [
            (
                0,
                None,
                None,
                AttentionType.MULTI_HEAD.value,
                "number_of_heads must be positive, got 0.",
            ),
            (
                4,
                None,
                0,
                AttentionType.MULTI_HEAD.value,
                "head_dimension must be positive, got 0.",
            ),
            (
                4,
                0,
                None,
                AttentionType.GROUPED_QUERY.value,
                "number_of_key_value_heads must be positive, got 0.",
            ),
            (
                4,
                2,
                None,
                AttentionType.MULTI_HEAD.value,
                "number_of_key_value_heads must be None or equal to "
                "number_of_heads for multi-head attention, got 2.",
            ),
        ],
        ids=[
            "zero_heads",
            "zero_head_dimension",
            "zero_key_value_heads",
            "mha_key_value_mismatch",
        ],
    )
    def test_invalid_attention_configuration_raises(
        self,
        cached_attention_factory: Callable[..., CachedAttention],
        number_of_heads: int,
        number_of_key_value_heads: int | None,
        head_dimension: int | None,
        attention_type: str,
        error_message: str,
    ):
        with pytest.raises(ValueError, match=re.escape(error_message)):
            cached_attention_factory(
                number_of_heads=number_of_heads,
                number_of_key_value_heads=number_of_key_value_heads,
                head_dimension=head_dimension,
                attention_type=attention_type,
            )

    def test_embedding_not_divisible_by_heads_raises(
        self, cached_attention_factory: Callable[..., CachedAttention]
    ):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "embedding_dimension (33) must be divisible by number_of_heads (4)."
            ),
        ):
            cached_attention_factory(embedding_dimension=33, number_of_heads=4)

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
                "number_of_heads (8) must be divisible by number_of_key_value_heads (3)."
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
        module = cached_attention_factory(embedding_dimension=32, number_of_heads=4)
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

    def test_generation_cache_returns_updated_cache(
        self,
        cached_attention_factory: Callable[..., CachedAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        module = cached_attention_factory(embedding_dimension=32, number_of_heads=4)
        sequence = sequence_tensor_factory(
            batch_size=2, sequence_length=1, embedding_dimension=32
        )
        empty_cache = GenerationLayerCache(
            keys=torch.empty(2, 4, 0, 8),
            values=torch.empty(2, 4, 0, 8),
        )
        output, new_cache = module(
            query_input=sequence,
            key_input=sequence,
            value_input=sequence,
            generation_cache=empty_cache,
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
        causal_mask = (
            torch.triu(
                torch.ones(sequence_length, sequence_length, dtype=torch.bool),
                diagonal=1,
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, -1, -1, -1)
        )
        full_output, _ = module(
            query_input=full_sequence,
            key_input=full_sequence,
            value_input=full_sequence,
            attention_mask=causal_mask,
        )
        # Incremental forward with generation cache
        cache = GenerationLayerCache(
            keys=torch.empty(batch_size, number_of_heads, 0, head_dimension),
            values=torch.empty(batch_size, number_of_heads, 0, head_dimension),
        )
        cached_outputs = []
        for step in range(sequence_length):
            token = full_sequence[:, step : step + 1, :]
            step_output, cache = module(
                query_input=token,
                key_input=token,
                value_input=token,
                generation_cache=cache,
            )
            cached_outputs.append(step_output)
        cached_full_output = torch.cat(cached_outputs, dim=1)
        assert torch.allclose(full_output, cached_full_output, atol=1e-5)

    def test_conditioning_cache_produces_output_sensitive_to_cached_kv(
        self,
        cached_attention_factory: Callable[..., CachedAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        conditioning_cache_factory: Callable[..., ConditioningLayerCache],
    ):
        module = cached_attention_factory(embedding_dimension=32, number_of_heads=4)
        module.eval()
        query = sequence_tensor_factory(
            batch_size=2, sequence_length=3, embedding_dimension=32
        )
        cache_a = conditioning_cache_factory(
            batch_size=2,
            number_of_key_value_heads=4,
            memory_length=6,
            head_dimension=8,
        )
        cache_b = conditioning_cache_factory(
            batch_size=2,
            number_of_key_value_heads=4,
            memory_length=6,
            head_dimension=8,
        )
        output_a, _ = module(query_input=query, conditioning_cache=cache_a)
        output_b, _ = module(query_input=query, conditioning_cache=cache_b)
        assert output_a.shape == (2, 3, 32)
        assert not torch.allclose(output_a, output_b, atol=1e-6)

    def test_conditioning_cache_without_generation_cache_returns_none(
        self,
        cached_attention_factory: Callable[..., CachedAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        conditioning_cache_factory: Callable[..., ConditioningLayerCache],
    ):
        module = cached_attention_factory(embedding_dimension=32, number_of_heads=4)
        query = sequence_tensor_factory(
            batch_size=2, sequence_length=3, embedding_dimension=32
        )
        cache = conditioning_cache_factory(
            batch_size=2,
            number_of_key_value_heads=4,
            memory_length=6,
            head_dimension=8,
        )
        _, generation_cache = module(
            query_input=query,
            conditioning_cache=cache,
        )
        assert generation_cache is None

    def test_missing_key_value_without_conditioning_cache_raises(
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
                "key_input and value_input required when conditioning_cache is not provided"
            ),
        ):
            module(query_input=query)


class TestCachedAttentionComputeQueryKeyValue:
    @pytest.mark.parametrize(
        "attention_type, number_of_key_value_heads, expected_kv_heads",
        [
            (AttentionType.MULTI_HEAD.value, None, 4),
            (AttentionType.GROUPED_QUERY.value, 2, 2),
        ],
        ids=["mha", "gqa"],
    )
    def test_projection_shapes_by_attention_type(
        self,
        cached_attention_factory: Callable[..., CachedAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        attention_type: str,
        number_of_key_value_heads: int | None,
        expected_kv_heads: int,
    ):
        module = cached_attention_factory(
            embedding_dimension=32,
            number_of_heads=4,
            number_of_key_value_heads=number_of_key_value_heads,
            attention_type=attention_type,
        )
        sequence = sequence_tensor_factory(
            batch_size=2, sequence_length=5, embedding_dimension=32
        )
        queries, keys, values = module.compute_query_key_value(
            query_input=sequence,
            key_input=sequence,
            value_input=sequence,
        )
        assert queries.shape == (2, 4, 5, 8)  # (B, H, S, D_head)
        assert keys.shape == (2, expected_kv_heads, 5, 8)  # (B, KV_H, S, D_head)
        assert values.shape == (2, expected_kv_heads, 5, 8)

    def test_head_dimension_override(
        self,
        cached_attention_factory: Callable[..., CachedAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        module = cached_attention_factory(
            embedding_dimension=32,
            number_of_heads=4,
            head_dimension=16,
            attention_type=AttentionType.MULTI_HEAD.value,
        )
        assert module.head_dimension == 16
        sequence = sequence_tensor_factory(
            batch_size=2, sequence_length=5, embedding_dimension=32
        )
        queries = module.compute_query(query_input=sequence)
        assert queries.shape == (2, 4, 5, 16)  # (B, H, S, overridden_D_head)

    def test_individual_projections_match_combined(
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
        queries_combined, keys_combined, values_combined = (
            module.compute_query_key_value(
                query_input=sequence,
                key_input=sequence,
                value_input=sequence,
            )
        )
        queries_individual = module.compute_query(query_input=sequence)
        keys_individual = module.compute_key(key_input=sequence)
        values_individual = module.compute_value(value_input=sequence)
        assert torch.equal(queries_combined, queries_individual)
        assert torch.equal(keys_combined, keys_individual)
        assert torch.equal(values_combined, values_individual)


class TestComputeAttention:
    @pytest.mark.parametrize(
        "query_length, key_value_length",
        [(5, 8), (3, 6)],
    )
    def test_output_shape_and_mask_sensitivity(
        self,
        cached_attention_factory: Callable[..., CachedAttention],
        precomputed_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
        rng: np.random.Generator,
        query_length: int,
        key_value_length: int,
    ):
        module = cached_attention_factory(embedding_dimension=32, number_of_heads=4)
        module.eval()
        query_input = torch.from_numpy(
            rng.standard_normal((2, query_length, 32)).astype(np.float32)
        )
        queries = module.compute_query(query_input=query_input)
        keys, values = precomputed_kv_factory(
            batch_size=2, key_value_length=key_value_length
        )
        output_no_mask = module.compute_attention(
            queries=queries, keys=keys, values=values
        )
        assert output_no_mask.shape == (2, query_length, 32)
        mask = torch.zeros(2, 1, query_length, key_value_length, dtype=torch.bool)
        mask[:, :, :, key_value_length // 2 :] = True
        output_masked = module.compute_attention(
            queries=queries, keys=keys, values=values, attention_mask=mask
        )
        assert not torch.allclose(output_no_mask, output_masked)

    def test_conditioning_cache_path_only_projects_queries(
        self,
        cached_attention_factory: Callable[..., CachedAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        conditioning_cache_factory: Callable[..., ConditioningLayerCache],
    ):
        module = cached_attention_factory(embedding_dimension=32, number_of_heads=4)
        query = sequence_tensor_factory(
            batch_size=2, sequence_length=3, embedding_dimension=32
        )
        cache = conditioning_cache_factory(
            batch_size=2,
            number_of_key_value_heads=4,
            memory_length=6,
            head_dimension=8,
        )
        with (
            unittest.mock.patch.object(
                module, "compute_query", wraps=module.compute_query
            ) as mock_query,
            unittest.mock.patch.object(
                module, "compute_key", wraps=module.compute_key
            ) as mock_key,
            unittest.mock.patch.object(
                module, "compute_value", wraps=module.compute_value
            ) as mock_value,
        ):
            module(
                query_input=query,
                conditioning_cache=cache,
            )
            mock_query.assert_called_once()
            mock_key.assert_not_called()
            mock_value.assert_not_called()


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
            rng.standard_normal((batch_size, 3, embedding_dimension)).astype(np.float32)
        )
        memory_a = torch.from_numpy(
            rng.standard_normal((batch_size, 6, embedding_dimension)).astype(np.float32)
        )
        memory_b = torch.from_numpy(
            rng.standard_normal((batch_size, 6, embedding_dimension)).astype(np.float32)
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
