"""Tests for versatil.models.decoding.action_heads.base module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from versatil.models.decoding.action_heads.base import BaseActionHead
from versatil.models.decoding.action_heads.blocks import MLPBlock


class ConcreteActionHead(BaseActionHead):
    """Concrete implementation for testing abstract BaseActionHead."""

    def forward(self, action_embedding: torch.Tensor) -> torch.Tensor:
        if self.output_proj is None:
            raise RuntimeError("output_dim not set")
        action_embedding = self._apply_blocks(action_embedding)
        return self.output_proj(action_embedding)


@pytest.fixture
def concrete_action_head_factory() -> Callable[..., ConcreteActionHead]:
    """Factory for ConcreteActionHead instances."""

    def factory(
        input_dimension: int = 64,
        blocks: list | None = None,
    ) -> ConcreteActionHead:
        return ConcreteActionHead(
            input_dimension=input_dimension,
            blocks=blocks,
        )

    return factory


@pytest.mark.unit
class TestBaseActionHeadInitialization:
    @pytest.mark.parametrize("input_dimension", [32, 128])
    @pytest.mark.parametrize("use_blocks", [False, True])
    def test_stores_configuration(
        self,
        concrete_action_head_factory: Callable[..., ConcreteActionHead],
        input_dimension: int,
        use_blocks: bool,
    ):
        blocks = (
            [
                MLPBlock(
                    input_dimension=input_dimension, hidden_dimensions=[input_dimension]
                )
            ]
            if use_blocks
            else None
        )
        head = concrete_action_head_factory(
            input_dimension=input_dimension, blocks=blocks
        )
        assert head.input_dimension == input_dimension
        assert head.output_proj is None
        assert isinstance(head.blocks, nn.ModuleList)
        if blocks is None:
            assert len(head.blocks) == 0
        else:
            assert len(head.blocks) == len(blocks)

    def test_output_dim_raises_when_not_set(
        self,
        concrete_action_head_factory: Callable[..., ConcreteActionHead],
    ):
        head = concrete_action_head_factory()
        with pytest.raises(
            RuntimeError,
            match=re.escape("output_dim not set. Call set_output_dim() first."),
        ):
            _ = head.output_dim


@pytest.mark.unit
class TestBaseActionHeadSetOutputDim:
    @pytest.mark.parametrize("dim", [3, 7])
    def test_sets_output_dim(
        self,
        concrete_action_head_factory: Callable[..., ConcreteActionHead],
        dim: int,
    ):
        head = concrete_action_head_factory()
        head.set_output_dim(dim)
        assert head.output_dim == dim

    def test_creates_output_projection(
        self,
        concrete_action_head_factory: Callable[..., ConcreteActionHead],
    ):
        head = concrete_action_head_factory(input_dimension=64)
        head.set_output_dim(3)
        assert head.output_proj is not None
        assert head.output_proj.in_features == 64
        assert head.output_proj.out_features == 3

    def test_projection_uses_last_block_dim(self):
        blocks = [MLPBlock(input_dimension=64, hidden_dimensions=[32])]
        head = ConcreteActionHead(input_dimension=64, blocks=blocks)
        head.set_output_dim(3)
        assert head.output_proj.in_features == 32


@pytest.mark.unit
class TestBaseActionHeadGetHiddenDim:
    def test_returns_input_dim_without_blocks(
        self,
        concrete_action_head_factory: Callable[..., ConcreteActionHead],
    ):
        head = concrete_action_head_factory(input_dimension=64)
        assert head._get_hidden_dim() == 64

    def test_returns_last_block_output_dim(self):
        blocks = [
            MLPBlock(input_dimension=64, hidden_dimensions=[48]),
            MLPBlock(input_dimension=48, hidden_dimensions=[32]),
        ]
        head = ConcreteActionHead(input_dimension=64, blocks=blocks)
        assert head._get_hidden_dim() == 32


@pytest.mark.unit
def test_apply_blocks_passes_embedding_through_blocks_in_order(
    embedding_tensor_factory: Callable[..., torch.Tensor],
):
    first_block = MLPBlock(input_dimension=64, hidden_dimensions=[64])
    second_block = MLPBlock(input_dimension=64, hidden_dimensions=[64])
    head = ConcreteActionHead(
        input_dimension=64,
        blocks=[first_block, second_block],
    )
    embedding = embedding_tensor_factory(embedding_dimension=64)
    first_output = torch.full_like(embedding, 2.0)
    second_output = torch.full_like(embedding, 3.0)
    first_forward = MagicMock(
        spec=first_block.forward,
        return_value=first_output,
    )
    second_forward = MagicMock(
        spec=second_block.forward,
        return_value=second_output,
    )

    with (
        patch.object(first_block, "forward", first_forward),
        patch.object(second_block, "forward", second_forward),
    ):
        result = head._apply_blocks(embedding)

    first_forward.assert_called_once()
    second_forward.assert_called_once()
    torch.testing.assert_close(first_forward.call_args.args[0], embedding)
    torch.testing.assert_close(second_forward.call_args.args[0], first_output)
    torch.testing.assert_close(result, second_output)


@pytest.mark.integration
class TestBaseActionHeadForward:
    @pytest.mark.parametrize("output_dim", [3, 7])
    def test_output_shape(
        self,
        concrete_action_head_factory: Callable[..., ConcreteActionHead],
        embedding_tensor_factory: Callable[..., torch.Tensor],
        output_dim: int,
    ):
        head = concrete_action_head_factory(input_dimension=64)
        head.set_output_dim(output_dim)
        embedding = embedding_tensor_factory(embedding_dimension=64)
        result = head(embedding)
        assert result.shape == (2, 8, output_dim)

    def test_forward_equals_projection_of_applied_blocks(
        self,
        embedding_tensor_factory: Callable[..., torch.Tensor],
    ):
        blocks = [MLPBlock(input_dimension=64, hidden_dimensions=[32])]
        head = ConcreteActionHead(input_dimension=64, blocks=blocks)
        head.set_output_dim(3)
        head.eval()
        embedding = embedding_tensor_factory(embedding_dimension=64)
        expected = head.output_proj(head._apply_blocks(embedding))
        result = head(embedding)
        assert result.shape == (2, 8, 3)
        torch.testing.assert_close(result, expected)
