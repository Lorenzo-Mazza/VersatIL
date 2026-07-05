"""Tests for versatil.models.layers.transformer.layer.precomputed_dual_stream_layer module."""

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
import torch

from versatil.models.layers.transformer.cache.conditioning import (
    ConditioningLayerCache,
)
from versatil.models.layers.transformer.layer.precomputed_dual_stream_layer import (
    PrecomputedDualStreamLayer,
)

PRIMARY_EMBEDDING_DIMENSION = 32
SECONDARY_EMBEDDING_DIMENSION = 16
NUMBER_OF_HEADS = 2
HEAD_DIMENSION = PRIMARY_EMBEDDING_DIMENSION // NUMBER_OF_HEADS
BATCH_SIZE = 2
SECONDARY_SEQUENCE_LENGTH = 8
PRIMARY_SEQUENCE_LENGTH = 4


class TestPrecomputedDualStreamLayerInitialization:
    @pytest.mark.parametrize(
        "conditioning_dimension, use_gating",
        [(None, False), (16, True)],
        ids=["unconditioned", "conditioned_gated"],
    )
    def test_creates_attention_block_and_feedforward(
        self,
        conditioning_dimension: int | None,
        use_gating: bool,
    ):
        with (
            patch(
                "versatil.models.layers.transformer.layer.precomputed_dual_stream_layer.PrecomputedDualStreamAttentionBlock"
            ) as mock_block,
            patch(
                "versatil.models.layers.transformer.layer.precomputed_dual_stream_layer.FeedforwardBlock"
            ) as mock_ff,
            patch(
                "versatil.models.layers.transformer.layer.precomputed_dual_stream_layer.create_block_normalization"
            ) as mock_norm,
        ):
            PrecomputedDualStreamLayer(
                primary_embedding_dimension=PRIMARY_EMBEDDING_DIMENSION,
                secondary_embedding_dimension=SECONDARY_EMBEDDING_DIMENSION,
                number_of_heads=NUMBER_OF_HEADS,
                number_of_key_value_heads=NUMBER_OF_HEADS,
                head_dimension=HEAD_DIMENSION,
                primary_feedforward_dimension=PRIMARY_EMBEDDING_DIMENSION * 4,
                conditioning_dimension=conditioning_dimension,
                use_gating=use_gating,
            )
            mock_block.assert_called_once()
            mock_ff.assert_called_once()
            for call in mock_norm.call_args_list:
                assert call.kwargs["conditioning_dimension"] == conditioning_dimension
                assert call.kwargs["use_gating"] == use_gating


class TestPrecomputedDualStreamLayerForward:
    def test_attention_block_called_then_feedforward(
        self,
        sequence_tensor_factory: Callable[..., torch.Tensor],
        conditioning_cache_with_queries_factory: Callable[..., ConditioningLayerCache],
    ):
        layer = PrecomputedDualStreamLayer(
            primary_embedding_dimension=PRIMARY_EMBEDDING_DIMENSION,
            secondary_embedding_dimension=SECONDARY_EMBEDDING_DIMENSION,
            number_of_heads=NUMBER_OF_HEADS,
            number_of_key_value_heads=NUMBER_OF_HEADS,
            head_dimension=HEAD_DIMENSION,
            primary_feedforward_dimension=PRIMARY_EMBEDDING_DIMENSION * 4,
        )
        conditioning_cache = conditioning_cache_with_queries_factory(
            batch_size=BATCH_SIZE,
            number_of_heads=NUMBER_OF_HEADS,
            sequence_length=SECONDARY_SEQUENCE_LENGTH,
            head_dimension=HEAD_DIMENSION,
        )
        primary = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=PRIMARY_SEQUENCE_LENGTH,
            embedding_dimension=PRIMARY_EMBEDDING_DIMENSION,
        )
        attention_primary_output = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=PRIMARY_SEQUENCE_LENGTH,
            embedding_dimension=PRIMARY_EMBEDDING_DIMENSION,
        )
        secondary_attention_output = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=SECONDARY_SEQUENCE_LENGTH,
            embedding_dimension=NUMBER_OF_HEADS * HEAD_DIMENSION,
        )
        feedforward_primary_output = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=PRIMARY_SEQUENCE_LENGTH,
            embedding_dimension=PRIMARY_EMBEDDING_DIMENSION,
        )
        layer.attention_block = MagicMock(
            spec=torch.nn.Module,
            return_value=(attention_primary_output, secondary_attention_output),
        )
        layer.feedforward_block_primary = MagicMock(
            spec=torch.nn.Module,
            return_value=feedforward_primary_output,
        )
        primary_out = layer(
            hidden_states=primary,
            conditioning_cache=conditioning_cache,
        )
        layer.attention_block.assert_called_once()
        layer.feedforward_block_primary.assert_called_once_with(
            hidden_states=attention_primary_output, conditioning=None
        )
        assert torch.equal(primary_out, feedforward_primary_output)

    def test_conditioning_forwarded_to_all_submodules(
        self,
        sequence_tensor_factory: Callable[..., torch.Tensor],
        conditioning_cache_with_queries_factory: Callable[..., ConditioningLayerCache],
        condition_factory: Callable[..., torch.Tensor],
    ):
        layer = PrecomputedDualStreamLayer(
            primary_embedding_dimension=PRIMARY_EMBEDDING_DIMENSION,
            secondary_embedding_dimension=SECONDARY_EMBEDDING_DIMENSION,
            number_of_heads=NUMBER_OF_HEADS,
            number_of_key_value_heads=NUMBER_OF_HEADS,
            head_dimension=HEAD_DIMENSION,
            primary_feedforward_dimension=PRIMARY_EMBEDDING_DIMENSION * 4,
            conditioning_dimension=PRIMARY_EMBEDDING_DIMENSION,
        )
        conditioning_cache = conditioning_cache_with_queries_factory(
            batch_size=BATCH_SIZE,
            number_of_heads=NUMBER_OF_HEADS,
            sequence_length=SECONDARY_SEQUENCE_LENGTH,
            head_dimension=HEAD_DIMENSION,
        )
        primary = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=PRIMARY_SEQUENCE_LENGTH,
            embedding_dimension=PRIMARY_EMBEDDING_DIMENSION,
        )
        conditioning = condition_factory(
            batch_size=BATCH_SIZE, conditioning_dimension=PRIMARY_EMBEDDING_DIMENSION
        )
        attention_primary_output = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=PRIMARY_SEQUENCE_LENGTH,
            embedding_dimension=PRIMARY_EMBEDDING_DIMENSION,
        )
        secondary_attention_output = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=SECONDARY_SEQUENCE_LENGTH,
            embedding_dimension=NUMBER_OF_HEADS * HEAD_DIMENSION,
        )
        layer.attention_block = MagicMock(
            spec=torch.nn.Module,
            return_value=(attention_primary_output, secondary_attention_output),
        )
        layer.feedforward_block_primary = MagicMock(
            spec=torch.nn.Module,
            return_value=sequence_tensor_factory(
                batch_size=BATCH_SIZE,
                sequence_length=PRIMARY_SEQUENCE_LENGTH,
                embedding_dimension=PRIMARY_EMBEDDING_DIMENSION,
            ),
        )
        layer(
            hidden_states=primary,
            conditioning_cache=conditioning_cache,
            conditioning=conditioning,
        )
        attention_call_kwargs = layer.attention_block.call_args.kwargs
        assert torch.equal(attention_call_kwargs["conditioning"], conditioning)
        feedforward_call_kwargs = layer.feedforward_block_primary.call_args.kwargs
        assert torch.equal(feedforward_call_kwargs["conditioning"], conditioning)


class TestPrecomputedDualStreamLayerForwardWithSecondary:
    def test_returns_primary_and_secondary_with_correct_shapes(
        self,
        sequence_tensor_factory: Callable[..., torch.Tensor],
        conditioning_cache_with_queries_factory: Callable[..., ConditioningLayerCache],
    ):
        layer = PrecomputedDualStreamLayer(
            primary_embedding_dimension=PRIMARY_EMBEDDING_DIMENSION,
            secondary_embedding_dimension=SECONDARY_EMBEDDING_DIMENSION,
            number_of_heads=NUMBER_OF_HEADS,
            number_of_key_value_heads=NUMBER_OF_HEADS,
            head_dimension=HEAD_DIMENSION,
            primary_feedforward_dimension=PRIMARY_EMBEDDING_DIMENSION * 4,
        )
        conditioning_cache = conditioning_cache_with_queries_factory(
            batch_size=BATCH_SIZE,
            number_of_heads=NUMBER_OF_HEADS,
            number_of_key_value_heads=NUMBER_OF_HEADS,
            sequence_length=SECONDARY_SEQUENCE_LENGTH,
            head_dimension=HEAD_DIMENSION,
        )
        primary = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=PRIMARY_SEQUENCE_LENGTH,
            embedding_dimension=PRIMARY_EMBEDDING_DIMENSION,
        )
        primary_output, secondary_output = layer.forward_with_secondary(
            hidden_states_primary=primary,
            conditioning_cache=conditioning_cache,
        )
        assert isinstance(primary_output, torch.Tensor)
        assert isinstance(secondary_output, torch.Tensor)
        assert primary_output.shape == (
            BATCH_SIZE,
            PRIMARY_SEQUENCE_LENGTH,
            PRIMARY_EMBEDDING_DIMENSION,
        )
        assert secondary_output.shape == (
            BATCH_SIZE,
            SECONDARY_SEQUENCE_LENGTH,
            NUMBER_OF_HEADS * HEAD_DIMENSION,
        )
