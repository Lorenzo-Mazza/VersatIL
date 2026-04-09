"""Tests for versatil.models.layers.transformer.layer.precomputed_dual_stream_layer module."""

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
import torch

from versatil.models.layers.transformer.layer.precomputed_dual_stream_layer import (
    PrecomputedDualStreamLayer,
)

PRIMARY_EMBEDDING_DIMENSION = 32
SECONDARY_EMBEDDING_DIMENSION = 16
NUMBER_OF_HEADS = 2
HEAD_DIMENSION = PRIMARY_EMBEDDING_DIMENSION // NUMBER_OF_HEADS
BATCH_SIZE = 2
PRIMARY_SEQUENCE_LENGTH = 8
SECONDARY_SEQUENCE_LENGTH = 4


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
                secondary_feedforward_dimension=SECONDARY_EMBEDDING_DIMENSION * 4,
                conditioning_dimension=conditioning_dimension,
                use_gating=use_gating,
            )
            mock_block.assert_called_once()
            mock_ff.assert_called_once()
            for call in mock_norm.call_args_list:
                assert call.kwargs["condition_dim"] == conditioning_dimension
                assert call.kwargs["use_gating"] == use_gating


class TestPrecomputedDualStreamLayerForward:
    def test_attention_block_called_then_feedforward(
        self,
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        layer = PrecomputedDualStreamLayer(
            primary_embedding_dimension=PRIMARY_EMBEDDING_DIMENSION,
            secondary_embedding_dimension=SECONDARY_EMBEDDING_DIMENSION,
            number_of_heads=NUMBER_OF_HEADS,
            number_of_key_value_heads=NUMBER_OF_HEADS,
            head_dimension=HEAD_DIMENSION,
            secondary_feedforward_dimension=SECONDARY_EMBEDDING_DIMENSION * 4,
        )
        precomputed = (
            torch.randn(
                BATCH_SIZE, NUMBER_OF_HEADS, PRIMARY_SEQUENCE_LENGTH, HEAD_DIMENSION
            ),
            torch.randn(
                BATCH_SIZE, NUMBER_OF_HEADS, PRIMARY_SEQUENCE_LENGTH, HEAD_DIMENSION
            ),
            torch.randn(
                BATCH_SIZE, NUMBER_OF_HEADS, PRIMARY_SEQUENCE_LENGTH, HEAD_DIMENSION
            ),
        )
        secondary = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=SECONDARY_SEQUENCE_LENGTH,
            embedding_dimension=SECONDARY_EMBEDDING_DIMENSION,
        )
        raw_primary_output = torch.randn(
            BATCH_SIZE, PRIMARY_SEQUENCE_LENGTH, NUMBER_OF_HEADS * HEAD_DIMENSION
        )
        attention_secondary_output = torch.randn_like(secondary)
        feedforward_secondary_output = torch.randn_like(secondary)
        layer.attention_block = MagicMock(
            spec=torch.nn.Module,
            return_value=(raw_primary_output, attention_secondary_output),
        )
        layer.feedforward_block_secondary = MagicMock(
            spec=torch.nn.Module,
            return_value=feedforward_secondary_output,
        )
        primary_out, secondary_out = layer(
            precomputed_primary=precomputed,
            hidden_states_secondary=secondary,
        )
        layer.attention_block.assert_called_once()
        layer.feedforward_block_secondary.assert_called_once_with(
            hidden_states=attention_secondary_output, conditioning=None
        )
        assert torch.equal(primary_out, raw_primary_output)
        assert torch.equal(secondary_out, feedforward_secondary_output)

    def test_conditioning_forwarded_to_all_submodules(
        self,
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        layer = PrecomputedDualStreamLayer(
            primary_embedding_dimension=PRIMARY_EMBEDDING_DIMENSION,
            secondary_embedding_dimension=SECONDARY_EMBEDDING_DIMENSION,
            number_of_heads=NUMBER_OF_HEADS,
            number_of_key_value_heads=NUMBER_OF_HEADS,
            head_dimension=HEAD_DIMENSION,
            secondary_feedforward_dimension=SECONDARY_EMBEDDING_DIMENSION * 4,
            conditioning_dimension=SECONDARY_EMBEDDING_DIMENSION,
        )
        precomputed = (
            torch.randn(
                BATCH_SIZE, NUMBER_OF_HEADS, PRIMARY_SEQUENCE_LENGTH, HEAD_DIMENSION
            ),
            torch.randn(
                BATCH_SIZE, NUMBER_OF_HEADS, PRIMARY_SEQUENCE_LENGTH, HEAD_DIMENSION
            ),
            torch.randn(
                BATCH_SIZE, NUMBER_OF_HEADS, PRIMARY_SEQUENCE_LENGTH, HEAD_DIMENSION
            ),
        )
        secondary = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=SECONDARY_SEQUENCE_LENGTH,
            embedding_dimension=SECONDARY_EMBEDDING_DIMENSION,
        )
        conditioning = condition_factory(
            batch_size=BATCH_SIZE, condition_dim=SECONDARY_EMBEDDING_DIMENSION
        )
        attention_secondary_output = torch.randn_like(secondary)
        layer.attention_block = MagicMock(
            spec=torch.nn.Module,
            return_value=(
                torch.randn(
                    BATCH_SIZE,
                    PRIMARY_SEQUENCE_LENGTH,
                    NUMBER_OF_HEADS * HEAD_DIMENSION,
                ),
                attention_secondary_output,
            ),
        )
        layer.feedforward_block_secondary = MagicMock(
            spec=torch.nn.Module,
            return_value=torch.randn_like(secondary),
        )
        layer(
            precomputed_primary=precomputed,
            hidden_states_secondary=secondary,
            conditioning=conditioning,
        )
        attention_call_kwargs = layer.attention_block.call_args.kwargs
        assert torch.equal(attention_call_kwargs["conditioning"], conditioning)
        feedforward_call_kwargs = layer.feedforward_block_secondary.call_args.kwargs
        assert torch.equal(feedforward_call_kwargs["conditioning"], conditioning)
