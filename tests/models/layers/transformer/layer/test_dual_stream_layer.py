"""Tests for versatil.models.layers.transformer.layer.dual_stream_layer module."""

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
import torch

from versatil.models.layers.transformer.layer.dual_stream_layer import DualStreamLayer

EMBEDDING_DIMENSION = 32
NUMBER_OF_HEADS = 4
BATCH_SIZE = 2
PRIMARY_SEQUENCE_LENGTH = 6
SECONDARY_SEQUENCE_LENGTH = 4


@pytest.fixture
def layer_factory() -> Callable[..., DualStreamLayer]:

    def factory(
        embedding_dimension: int = EMBEDDING_DIMENSION,
        number_of_heads: int = NUMBER_OF_HEADS,
        conditioning_dimension: int | None = None,
        use_gating: bool = False,
    ) -> DualStreamLayer:
        return DualStreamLayer(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            conditioning_dimension=conditioning_dimension,
            use_gating=use_gating,
            dropout=0.0,
            attention_dropout=0.0,
        )

    return factory


class TestDualStreamLayerInitialization:
    @pytest.mark.parametrize(
        "conditioning_dimension, use_gating",
        [(None, False), (16, True)],
        ids=["unconditioned", "conditioned_gated"],
    )
    def test_creates_attention_block_and_feedforward_blocks(
        self,
        conditioning_dimension: int | None,
        use_gating: bool,
    ):
        with (
            patch(
                "versatil.models.layers.transformer.layer.dual_stream_layer.DualStreamAttentionBlock"
            ) as mock_attn,
            patch(
                "versatil.models.layers.transformer.layer.dual_stream_layer.build_feedforward"
            ),
            patch(
                "versatil.models.layers.transformer.layer.dual_stream_layer.FeedforwardBlock"
            ) as mock_ff_block,
            patch(
                "versatil.models.layers.transformer.layer.dual_stream_layer.create_block_normalization"
            ) as mock_norm,
        ):
            DualStreamLayer(
                embedding_dimension=32,
                number_of_heads=4,
                conditioning_dimension=conditioning_dimension,
                use_gating=use_gating,
                dropout=0.0,
                attention_dropout=0.0,
            )
            mock_attn.assert_called_once()
            assert mock_ff_block.call_count == 2
            norm_calls = mock_norm.call_args_list
            for call in norm_calls:
                assert call.kwargs["condition_dim"] == conditioning_dimension
                assert call.kwargs["use_gating"] == use_gating


class TestDualStreamLayerForward:
    def test_attention_block_called_then_both_feedforwards(
        self,
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        layer = DualStreamLayer(
            embedding_dimension=EMBEDDING_DIMENSION,
            number_of_heads=NUMBER_OF_HEADS,
            dropout=0.0,
            attention_dropout=0.0,
        )
        primary = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=PRIMARY_SEQUENCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        secondary = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=SECONDARY_SEQUENCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        attention_output_primary = torch.randn_like(primary)
        attention_output_secondary = torch.randn_like(secondary)
        ff_primary = torch.randn_like(primary)
        ff_secondary = torch.randn_like(secondary)
        layer.attention_block = MagicMock(
            spec=torch.nn.Module,
            return_value=(attention_output_primary, attention_output_secondary),
        )
        layer.feedforward_block_primary = MagicMock(
            spec=torch.nn.Module, return_value=ff_primary
        )
        layer.feedforward_block_secondary = MagicMock(
            spec=torch.nn.Module, return_value=ff_secondary
        )
        primary_out, secondary_out = layer(
            hidden_states_primary=primary,
            hidden_states_secondary=secondary,
        )
        layer.attention_block.assert_called_once()
        layer.feedforward_block_primary.assert_called_once_with(
            hidden_states=attention_output_primary, conditioning=None
        )
        layer.feedforward_block_secondary.assert_called_once_with(
            hidden_states=attention_output_secondary, conditioning=None
        )
        assert torch.equal(primary_out, ff_primary)
        assert torch.equal(secondary_out, ff_secondary)

    def test_conditioning_forwarded_to_all_submodules(
        self,
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        layer = DualStreamLayer(
            embedding_dimension=EMBEDDING_DIMENSION,
            number_of_heads=NUMBER_OF_HEADS,
            conditioning_dimension=EMBEDDING_DIMENSION,
            dropout=0.0,
            attention_dropout=0.0,
        )
        primary = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=PRIMARY_SEQUENCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        secondary = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=SECONDARY_SEQUENCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        conditioning = condition_factory(
            batch_size=BATCH_SIZE, condition_dim=EMBEDDING_DIMENSION
        )
        attention_output_primary = torch.randn_like(primary)
        attention_output_secondary = torch.randn_like(secondary)
        layer.attention_block = MagicMock(
            spec=torch.nn.Module,
            return_value=(attention_output_primary, attention_output_secondary),
        )
        layer.feedforward_block_primary = MagicMock(
            spec=torch.nn.Module, return_value=torch.randn_like(primary)
        )
        layer.feedforward_block_secondary = MagicMock(
            spec=torch.nn.Module, return_value=torch.randn_like(secondary)
        )
        layer(
            hidden_states_primary=primary,
            hidden_states_secondary=secondary,
            conditioning=conditioning,
        )
        attention_call_kwargs = layer.attention_block.call_args.kwargs
        assert torch.equal(attention_call_kwargs["conditioning"], conditioning)
        feedforward_primary_call_kwargs = (
            layer.feedforward_block_primary.call_args.kwargs
        )
        assert torch.equal(
            feedforward_primary_call_kwargs["conditioning"], conditioning
        )
        feedforward_secondary_call_kwargs = (
            layer.feedforward_block_secondary.call_args.kwargs
        )
        assert torch.equal(
            feedforward_secondary_call_kwargs["conditioning"], conditioning
        )
