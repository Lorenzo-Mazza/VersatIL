"""Tests for versatil.models.decoding.action_heads.blocks module."""
import re
from collections.abc import Callable

import pytest
import torch
import torch.nn as nn

from versatil.models.decoding.action_heads.blocks import (
    AttentionBlock,
    MLPBlock,
    ResidualBlock,
)
from versatil.models.layers.activation import ActivationFunction


@pytest.fixture
def mlp_block_factory() -> Callable[..., MLPBlock]:
    """Factory for MLPBlock instances."""
    def factory(
        input_dim: int = 64,
        hidden_dims: list[int] | None = None,
        output_dim: int | None = None,
        activation: str = ActivationFunction.GELU.value,
        dropout: float = 0.0,
        normalization: bool = True,
    ) -> MLPBlock:
        if hidden_dims is None:
            hidden_dims = [32]
        return MLPBlock(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            dropout=dropout,
            normalization=normalization,
        )
    return factory


@pytest.fixture
def attention_block_factory() -> Callable[..., AttentionBlock]:
    """Factory for AttentionBlock instances."""
    def factory(
        embedding_dimension: int = 64,
        num_heads: int = 4,
        dropout: float = 0.0,
        normalization: bool = True,
    ) -> AttentionBlock:
        return AttentionBlock(
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
            dropout=dropout,
            normalization=normalization,
        )
    return factory


class TestMLPBlockInitialization:

    @pytest.mark.parametrize("input_dim", [32, 128])
    @pytest.mark.parametrize("hidden_dims, output_dim, expected_output_dim", [
        ([32], None, 32),
        ([64, 32], None, 32),
        (None, 16, 16),
    ])
    def test_stores_configuration(
        self,
        mlp_block_factory: Callable[..., MLPBlock],
        input_dim: int,
        hidden_dims: list[int] | None,
        output_dim: int | None,
        expected_output_dim: int,
    ):
        block = mlp_block_factory(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
        )
        assert block.input_dim == input_dim
        assert block.output_dim == expected_output_dim

    def test_raises_without_hidden_dims_or_output_dim(self):
        with pytest.raises(
            ValueError,
            match=re.escape("Either output_dim or hidden_dims must be specified."),
        ):
            MLPBlock(input_dim=64, hidden_dims=None, output_dim=None)

    @pytest.mark.parametrize("normalization, expected_type", [
        (True, nn.LayerNorm),
        (False, nn.Identity),
    ])
    def test_normalization_layer(
        self,
        mlp_block_factory: Callable[..., MLPBlock],
        normalization: bool,
        expected_type: type,
    ):
        block = mlp_block_factory(normalization=normalization)
        assert isinstance(block.norm, expected_type)


class TestMLPBlockForward:

    @pytest.mark.parametrize("hidden_dims, output_dim", [
        ([32], None),
        ([64, 16], None),
        (None, 48),
    ])
    def test_output_shape(
        self,
        mlp_block_factory: Callable[..., MLPBlock],
        embedding_tensor_factory: Callable[..., torch.Tensor],
        hidden_dims: list[int] | None,
        output_dim: int | None,
    ):
        block = mlp_block_factory(
            input_dim=64,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
        )
        embedding = embedding_tensor_factory(embedding_dim=64)
        result = block(embedding)
        expected_last_dim = output_dim if output_dim is not None else hidden_dims[-1]
        assert result.shape == (2, 8, expected_last_dim)

    def test_different_activations_produce_different_outputs(
        self,
        mlp_block_factory: Callable[..., MLPBlock],
        embedding_tensor_factory: Callable[..., torch.Tensor],
    ):
        block_gelu = mlp_block_factory(activation=ActivationFunction.GELU.value)
        block_relu = mlp_block_factory(activation=ActivationFunction.RELU.value)
        # Use same weights so only activation differs
        block_relu.load_state_dict(block_gelu.state_dict())
        embedding = embedding_tensor_factory(embedding_dim=64)
        result_gelu = block_gelu(embedding)
        result_relu = block_relu(embedding)
        assert not torch.allclose(result_gelu, result_relu)


class TestAttentionBlockInitialization:

    @pytest.mark.parametrize("embedding_dimension", [32, 128])
    def test_stores_dimensions(
        self,
        attention_block_factory: Callable[..., AttentionBlock],
        embedding_dimension: int,
    ):
        block = attention_block_factory(embedding_dimension=embedding_dimension)
        assert block.input_dim == embedding_dimension
        assert block.output_dim == embedding_dimension

    @pytest.mark.parametrize("normalization, expected_type", [
        (True, nn.LayerNorm),
        (False, nn.Identity),
    ])
    def test_normalization_layer(
        self,
        attention_block_factory: Callable[..., AttentionBlock],
        normalization: bool,
        expected_type: type,
    ):
        block = attention_block_factory(normalization=normalization)
        assert isinstance(block.norm, expected_type)


class TestAttentionBlockForward:

    def test_output_shape_preserves_input_shape(
        self,
        attention_block_factory: Callable[..., AttentionBlock],
        embedding_tensor_factory: Callable[..., torch.Tensor],
    ):
        block = attention_block_factory(embedding_dimension=64)
        embedding = embedding_tensor_factory(embedding_dim=64)
        result = block(embedding)
        assert result.shape == embedding.shape

    def test_residual_adds_attention_output_to_input(
        self,
        attention_block_factory: Callable[..., AttentionBlock],
        embedding_tensor_factory: Callable[..., torch.Tensor],
    ):
        block = attention_block_factory(embedding_dimension=64, dropout=0.0)
        embedding = embedding_tensor_factory(embedding_dim=64)
        normalized = block.norm(embedding)
        attention_output, _ = block.attention(normalized, normalized, normalized)
        result = block(embedding)
        assert torch.allclose(result, embedding + attention_output)


class TestResidualBlockInitialization:

    @pytest.mark.parametrize("dim", [32, 64])
    def test_stores_configuration(self, dim: int):
        inner = MLPBlock(input_dim=dim, hidden_dims=[dim])
        block = ResidualBlock(block=inner)
        assert block.block is inner
        assert block.input_dim == dim
        assert block.output_dim == dim

    def test_raises_if_inner_dims_mismatch(self):
        inner = MLPBlock(input_dim=64, hidden_dims=[32])
        with pytest.raises(
            ValueError,
            match=re.escape("Input and output dimensions must match for ResidualBlock."),
        ):
            ResidualBlock(block=inner)


class TestResidualBlockForward:

    def test_output_shape_preserves_input_shape(
        self,
        embedding_tensor_factory: Callable[..., torch.Tensor],
    ):
        inner = MLPBlock(input_dim=64, hidden_dims=[64])
        block = ResidualBlock(block=inner)
        embedding = embedding_tensor_factory(embedding_dim=64)
        result = block(embedding)
        assert result.shape == embedding.shape

    def test_residual_adds_block_output_to_input(
        self,
        embedding_tensor_factory: Callable[..., torch.Tensor],
    ):
        inner = MLPBlock(input_dim=64, hidden_dims=[64])
        block = ResidualBlock(block=inner)
        embedding = embedding_tensor_factory(embedding_dim=64)
        inner_output = inner(embedding)
        result = block(embedding)
        assert torch.allclose(result, embedding + inner_output)
