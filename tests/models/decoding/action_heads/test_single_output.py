"""Tests for versatil.models.decoding.action_heads.single_output module."""
import re
from collections.abc import Callable

import pytest
import torch

from versatil.models.decoding.action_heads.blocks import MLPBlock
from versatil.models.decoding.action_heads.single_output import ActionHead


class TestActionHeadForward:

    def test_raises_if_output_dim_not_set(
        self,
        action_head_factory: Callable[..., ActionHead],
        embedding_tensor_factory: Callable[..., torch.Tensor],
    ):
        head = action_head_factory()
        embedding = embedding_tensor_factory()
        with pytest.raises(
            RuntimeError,
            match=re.escape("output_dim not set. Call set_output_dim() first."),
        ):
            head(embedding)

    @pytest.mark.parametrize("output_dim", [3, 7])
    def test_output_shape(
        self,
        action_head_factory: Callable[..., ActionHead],
        embedding_tensor_factory: Callable[..., torch.Tensor],
        output_dim: int,
    ):
        head = action_head_factory(input_dim=64)
        head.set_output_dim(output_dim)
        embedding = embedding_tensor_factory(embedding_dimension=64)
        result = head(embedding)
        assert result.shape == (2, 8, output_dim)

    def test_forward_with_blocks(
        self,
        embedding_tensor_factory: Callable[..., torch.Tensor],
    ):
        blocks = [MLPBlock(input_dim=64, hidden_dims=[32])]
        head = ActionHead(input_dim=64, blocks=blocks)
        head.set_output_dim(3)
        embedding = embedding_tensor_factory(embedding_dimension=64)
        result = head(embedding)
        assert result.shape == (2, 8, 3)
