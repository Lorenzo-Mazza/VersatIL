"""Tests for versatil.models.decoding.action_heads.conditional module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from versatil.models.decoding.action_heads.blocks import ConditionalActionHeadBlock
from versatil.models.decoding.action_heads.conditional import ConditionalActionHead


@pytest.fixture
def conditional_block_factory() -> Callable[..., MagicMock]:
    """Factory for mocked conditional action-head blocks."""

    def factory(output_dim: int = 64) -> MagicMock:
        block = MagicMock(spec=ConditionalActionHeadBlock)
        block.output_dim = output_dim
        return block

    return factory


@pytest.fixture
def mocked_block_conditional_head_factory(
    conditional_block_factory: Callable[..., MagicMock],
) -> Callable[..., ConditionalActionHead]:
    """Factory for conditional action heads with mocked conditional blocks."""

    def factory(
        input_dimension: int = 64,
        conditioning_dimension: int = 16,
        block_count: int = 0,
        block_output_dim: int = 64,
    ) -> ConditionalActionHead:
        blocks = [
            conditional_block_factory(output_dim=block_output_dim)
            for _ in range(block_count)
        ]
        return ConditionalActionHead(
            input_dimension=input_dimension,
            conditioning_dimension=conditioning_dimension,
            blocks=blocks,
        )

    return factory


@pytest.mark.unit
def test_conditional_action_head_stores_configuration(
    mocked_block_conditional_head_factory: Callable[..., ConditionalActionHead],
) -> None:
    input_dimension = 32
    conditioning_dimension = 12
    head = mocked_block_conditional_head_factory(
        input_dimension=input_dimension,
        conditioning_dimension=conditioning_dimension,
        block_count=2,
        block_output_dim=input_dimension,
    )

    assert head.input_dimension == input_dimension
    assert head.conditioning_dimension == conditioning_dimension
    assert len(head.blocks) == 2


@pytest.mark.unit
class TestConditionalActionHeadForward:
    def test_raises_if_output_dim_not_set(
        self,
        mocked_block_conditional_head_factory: Callable[..., ConditionalActionHead],
        embedding_tensor_factory: Callable[..., torch.Tensor],
    ) -> None:
        head = mocked_block_conditional_head_factory(
            input_dimension=64, conditioning_dimension=16
        )
        action_embedding = embedding_tensor_factory(embedding_dimension=64)
        condition = torch.zeros(action_embedding.shape[0], 16)

        with pytest.raises(
            RuntimeError,
            match=re.escape("output_dim not set. Call set_output_dim() first."),
        ):
            head(action_embedding=action_embedding, condition=condition)

    def test_passes_condition_through_blocks_before_projection(
        self,
        conditional_block_factory: Callable[..., MagicMock],
        embedding_tensor_factory: Callable[..., torch.Tensor],
    ) -> None:
        input_dimension = 64
        output_dim = 3
        first_block_output = torch.full((2, 8, input_dimension), fill_value=1.0)
        second_block_output = torch.full((2, 8, input_dimension), fill_value=2.0)
        projected_output = torch.full((2, 8, output_dim), fill_value=3.0)
        first_block = conditional_block_factory(output_dim=input_dimension)
        second_block = conditional_block_factory(output_dim=input_dimension)
        first_block.return_value = first_block_output
        second_block.return_value = second_block_output
        output_projection = MagicMock(spec=nn.Module)
        output_projection.return_value = projected_output
        head = ConditionalActionHead(
            input_dimension=input_dimension,
            conditioning_dimension=16,
            blocks=[first_block, second_block],
        )
        head.output_proj = output_projection
        action_embedding = embedding_tensor_factory(embedding_dimension=input_dimension)
        condition = torch.ones(action_embedding.shape[0], 16)

        result = head(action_embedding=action_embedding, condition=condition)

        first_block.assert_called_once_with(action_embedding, condition)
        second_block.assert_called_once_with(first_block_output, condition)
        output_projection.assert_called_once_with(second_block_output)
        torch.testing.assert_close(result, projected_output)
