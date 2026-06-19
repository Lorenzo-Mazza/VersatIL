"""Tests for versatil.models.decoding.action_heads.blocks module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from versatil.models.decoding.action_heads.blocks import (
    AdaNormBlock,
    AttentionBlock,
    LayerNormBlock,
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


@pytest.fixture
def adanorm_block_factory() -> Callable[..., AdaNormBlock]:
    def factory(
        input_dim: int = 64,
        condition_dim: int = 16,
        activation: str = ActivationFunction.SILU.value,
    ) -> AdaNormBlock:
        return AdaNormBlock(
            input_dim=input_dim,
            condition_dim=condition_dim,
            activation=activation,
        )

    return factory


@pytest.fixture
def layer_norm_block_factory() -> Callable[..., LayerNormBlock]:
    def factory(input_dim: int = 64) -> LayerNormBlock:
        return LayerNormBlock(input_dim=input_dim)

    return factory


@pytest.mark.unit
class TestMLPBlockInitialization:
    @pytest.mark.parametrize("input_dim", [32, 128])
    @pytest.mark.parametrize(
        "hidden_dims, output_dim, expected_output_dim",
        [
            ([32], None, 32),
            ([64, 32], None, 32),
            (None, 16, 16),
        ],
    )
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

    @pytest.mark.parametrize(
        "normalization, expected_type",
        [
            (True, nn.LayerNorm),
            (False, nn.Identity),
        ],
    )
    def test_normalization_layer(
        self,
        mlp_block_factory: Callable[..., MLPBlock],
        normalization: bool,
        expected_type: type,
    ):
        block = mlp_block_factory(normalization=normalization)
        assert isinstance(block.norm, expected_type)


@pytest.mark.unit
class TestMLPBlockForward:
    def test_passes_normalized_embedding_to_mlp(
        self,
        mlp_block_factory: Callable[..., MLPBlock],
        embedding_tensor_factory: Callable[..., torch.Tensor],
    ):
        block = mlp_block_factory(input_dim=64, hidden_dims=[32])
        embedding = embedding_tensor_factory(embedding_dimension=64)
        normalized_embedding = torch.full_like(embedding, 2.0)
        expected_output = torch.ones(embedding.shape[0], embedding.shape[1], 32)
        norm_forward = MagicMock(
            spec=block.norm.forward,
            return_value=normalized_embedding,
        )
        mlp_forward = MagicMock(
            spec=block.mlp.forward,
            return_value=expected_output,
        )

        with (
            patch.object(block.norm, "forward", norm_forward),
            patch.object(block.mlp, "forward", mlp_forward),
        ):
            result = block(embedding)

        norm_forward.assert_called_once()
        mlp_forward.assert_called_once()
        torch.testing.assert_close(norm_forward.call_args.args[0], embedding)
        torch.testing.assert_close(mlp_forward.call_args.args[0], normalized_embedding)
        torch.testing.assert_close(result, expected_output)


@pytest.mark.integration
def test_mlp_block_different_activations_produce_different_outputs(
    mlp_block_factory: Callable[..., MLPBlock],
    embedding_tensor_factory: Callable[..., torch.Tensor],
):
    block_gelu = mlp_block_factory(activation=ActivationFunction.GELU.value)
    block_relu = mlp_block_factory(activation=ActivationFunction.RELU.value)
    # Use same weights so only activation differs
    block_relu.load_state_dict(block_gelu.state_dict())
    embedding = embedding_tensor_factory(embedding_dimension=64)
    result_gelu = block_gelu(embedding)
    result_relu = block_relu(embedding)
    assert not torch.allclose(result_gelu, result_relu)


@pytest.mark.unit
@pytest.mark.parametrize("input_dim", [32, 128])
def test_layer_norm_block_stores_dimensions(
    layer_norm_block_factory: Callable[..., LayerNormBlock],
    input_dim: int,
) -> None:
    block = layer_norm_block_factory(input_dim=input_dim)

    assert block.input_dim == input_dim
    assert block.output_dim == input_dim
    assert isinstance(block.norm, nn.LayerNorm)


@pytest.mark.unit
def test_layer_norm_block_passes_embedding_to_layer_norm(
    layer_norm_block_factory: Callable[..., LayerNormBlock],
    embedding_tensor_factory: Callable[..., torch.Tensor],
) -> None:
    block = layer_norm_block_factory(input_dim=64)
    embedding = embedding_tensor_factory(embedding_dimension=64)
    expected_output = torch.ones_like(embedding)
    norm_forward = MagicMock(
        spec=block.norm.forward,
        return_value=expected_output,
    )

    with patch.object(block.norm, "forward", norm_forward):
        result = block(embedding)

    norm_forward.assert_called_once()
    torch.testing.assert_close(norm_forward.call_args.args[0], embedding)
    torch.testing.assert_close(result, expected_output)


@pytest.mark.unit
@pytest.mark.parametrize("input_dim", [32, 128])
@pytest.mark.parametrize("condition_dim", [16, 64])
def test_adanorm_block_stores_dimensions(
    adanorm_block_factory: Callable[..., AdaNormBlock],
    input_dim: int,
    condition_dim: int,
) -> None:
    block = adanorm_block_factory(
        input_dim=input_dim,
        condition_dim=condition_dim,
    )

    assert block.input_dim == input_dim
    assert block.output_dim == input_dim


@pytest.mark.unit
def test_adanorm_block_passes_action_embedding_and_condition_to_adaptive_norm(
    adanorm_block_factory: Callable[..., AdaNormBlock],
    embedding_tensor_factory: Callable[..., torch.Tensor],
) -> None:
    condition_dim = 16
    block = adanorm_block_factory(input_dim=64, condition_dim=condition_dim)
    embedding = embedding_tensor_factory(embedding_dimension=64)
    condition = torch.ones(embedding.shape[0], condition_dim)
    expected_output = torch.ones_like(embedding)
    ada_norm_forward = MagicMock(
        spec=block.ada_norm.forward,
        return_value=(expected_output, None),
    )

    with patch.object(
        block.ada_norm,
        "forward",
        ada_norm_forward,
    ):
        result = block(
            action_embedding=embedding,
            condition=condition,
        )

    ada_norm_forward.assert_called_once()
    torch.testing.assert_close(ada_norm_forward.call_args.args[0], embedding)
    torch.testing.assert_close(ada_norm_forward.call_args.args[1], condition)
    torch.testing.assert_close(result, expected_output)


@pytest.mark.integration
def test_adanorm_block_condition_modulates_output(
    adanorm_block_factory: Callable[..., AdaNormBlock],
    embedding_tensor_factory: Callable[..., torch.Tensor],
) -> None:
    condition_dim = 16
    block = adanorm_block_factory(input_dim=64, condition_dim=condition_dim)
    # Break zero-init modulation so conditioning has an effect.
    for parameter in block.parameters():
        nn.init.normal_(parameter, std=0.5)
    block.eval()
    embedding = embedding_tensor_factory(embedding_dimension=64)
    batch_size = embedding.shape[0]
    first_condition = torch.zeros(batch_size, condition_dim)
    second_condition = torch.ones(batch_size, condition_dim)
    first_output = block(action_embedding=embedding, condition=first_condition)
    second_output = block(action_embedding=embedding, condition=second_condition)
    assert not torch.allclose(first_output, second_output)


@pytest.mark.unit
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

    @pytest.mark.parametrize(
        "normalization, expected_type",
        [
            (True, nn.LayerNorm),
            (False, nn.Identity),
        ],
    )
    def test_normalization_layer(
        self,
        attention_block_factory: Callable[..., AttentionBlock],
        normalization: bool,
        expected_type: type,
    ):
        block = attention_block_factory(normalization=normalization)
        assert isinstance(block.norm, expected_type)


@pytest.mark.unit
def test_attention_block_passes_normalized_embedding_to_attention_and_dropout(
    attention_block_factory: Callable[..., AttentionBlock],
    embedding_tensor_factory: Callable[..., torch.Tensor],
):
    block = attention_block_factory(embedding_dimension=64, dropout=0.0)
    embedding = embedding_tensor_factory(embedding_dimension=64)
    normalized_embedding = torch.full_like(embedding, 2.0)
    attention_output = torch.ones_like(embedding)
    dropout_output = torch.full_like(embedding, 3.0)
    norm_forward = MagicMock(
        spec=block.norm.forward,
        return_value=normalized_embedding,
    )
    attention_forward = MagicMock(
        spec=block.attention.forward,
        return_value=(attention_output, None),
    )
    dropout_forward = MagicMock(
        spec=block.dropout.forward,
        return_value=dropout_output,
    )

    with (
        patch.object(block.norm, "forward", norm_forward),
        patch.object(block.attention, "forward", attention_forward),
        patch.object(block.dropout, "forward", dropout_forward),
    ):
        result = block(embedding)

    norm_forward.assert_called_once()
    attention_forward.assert_called_once()
    dropout_forward.assert_called_once()
    torch.testing.assert_close(norm_forward.call_args.args[0], embedding)
    for attention_argument in attention_forward.call_args.args:
        torch.testing.assert_close(attention_argument, normalized_embedding)
    torch.testing.assert_close(dropout_forward.call_args.args[0], attention_output)
    torch.testing.assert_close(result, embedding + dropout_output)


@pytest.mark.integration
def test_attention_block_residual_adds_attention_output_to_input(
    attention_block_factory: Callable[..., AttentionBlock],
    embedding_tensor_factory: Callable[..., torch.Tensor],
):
    block = attention_block_factory(embedding_dimension=64, dropout=0.0)
    block.eval()
    embedding = embedding_tensor_factory(embedding_dimension=64)
    normalized = block.norm(embedding)
    attention_output, _ = block.attention(normalized, normalized, normalized)
    result = block(embedding)
    torch.testing.assert_close(result, embedding + attention_output)


@pytest.mark.unit
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
            match=re.escape(
                "Input and output dimensions must match for ResidualBlock."
            ),
        ):
            ResidualBlock(block=inner)

    @pytest.mark.parametrize(
        "dropout, expected_type",
        [
            (0.0, nn.Identity),
            (0.5, nn.Dropout),
        ],
    )
    def test_dropout_layer_selected_by_rate(
        self,
        dropout: float,
        expected_type: type,
    ):
        inner = MLPBlock(input_dim=64, hidden_dims=[64])
        block = ResidualBlock(block=inner, dropout=dropout)
        assert isinstance(block.dropout, expected_type)


@pytest.mark.unit
def test_residual_block_passes_embedding_to_wrapped_block_and_dropout(
    embedding_tensor_factory: Callable[..., torch.Tensor],
):
    inner = MLPBlock(input_dim=64, hidden_dims=[64])
    block = ResidualBlock(block=inner, dropout=0.5)
    embedding = embedding_tensor_factory(embedding_dimension=64)
    inner_output = torch.ones_like(embedding)
    dropout_output = torch.full_like(embedding, 2.0)
    inner_forward = MagicMock(
        spec=inner.forward,
        return_value=inner_output,
    )
    dropout_forward = MagicMock(
        spec=block.dropout.forward,
        return_value=dropout_output,
    )

    with (
        patch.object(inner, "forward", inner_forward),
        patch.object(block.dropout, "forward", dropout_forward),
    ):
        result = block(embedding)

    inner_forward.assert_called_once()
    dropout_forward.assert_called_once()
    torch.testing.assert_close(inner_forward.call_args.args[0], embedding)
    torch.testing.assert_close(dropout_forward.call_args.args[0], inner_output)
    torch.testing.assert_close(result, embedding + dropout_output)


@pytest.mark.integration
def test_residual_block_adds_block_output_to_input(
    embedding_tensor_factory: Callable[..., torch.Tensor],
):
    inner = MLPBlock(input_dim=64, hidden_dims=[64])
    block = ResidualBlock(block=inner)
    block.eval()
    embedding = embedding_tensor_factory(embedding_dimension=64)
    inner_output = inner(embedding)
    result = block(embedding)
    torch.testing.assert_close(result, embedding + inner_output)
