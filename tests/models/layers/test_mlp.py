"""Tests for versatil.models.layers.mlp module."""

from collections.abc import Callable

import pytest
import torch
import torch.nn as nn

from versatil.models.layers.gated_linear_unit import SwiGLU
from versatil.models.layers.mlp import MLP


@pytest.fixture
def mlp_factory() -> Callable[..., MLP]:
    """Factory for MLP instances."""

    def factory(
        input_dimension: int = 32,
        hidden_dimensions: list[int] | None = None,
        output_dim: int | None = None,
        activation_function: type[nn.Module] = nn.GELU,
        dropout: float = 0.0,
    ) -> MLP:
        return MLP(
            input_dimension=input_dimension,
            hidden_dimensions=hidden_dimensions,
            output_dim=output_dim,
            activation_function=activation_function,
            dropout=dropout,
        )

    return factory


class TestMLPInitialization:
    def test_layers_are_iterable_and_ordered(
        self,
        mlp_factory: Callable[..., MLP],
    ):
        mlp = mlp_factory(input_dimension=32, hidden_dimensions=[64], output_dim=16)
        # Functional: layers can be iterated and produce the expected sequence
        layer_types = [type(layer).__name__ for layer in mlp.layers]
        assert layer_types[0] == "Linear"
        assert layer_types[-1] == "Linear"

    @pytest.mark.parametrize(
        "hidden_dimensions, expected_linear_count",
        [
            (None, 0),
            ([64], 1),
            ([64, 128], 2),
        ],
    )
    def test_creates_correct_number_of_hidden_layers(
        self,
        mlp_factory: Callable[..., MLP],
        hidden_dimensions: list[int] | None,
        expected_linear_count: int,
    ):
        mlp = mlp_factory(input_dimension=32, hidden_dimensions=hidden_dimensions)
        linear_layers = [layer for layer in mlp.layers if isinstance(layer, nn.Linear)]
        assert len(linear_layers) == expected_linear_count

    @pytest.mark.parametrize(
        "output_dim, expected_extra_linear",
        [
            (None, 0),
            (16, 1),
        ],
    )
    def test_output_layer_added_when_output_dim_specified(
        self,
        mlp_factory: Callable[..., MLP],
        output_dim: int | None,
        expected_extra_linear: int,
    ):
        mlp = mlp_factory(
            input_dimension=32,
            hidden_dimensions=[64],
            output_dim=output_dim,
        )
        linear_layers = [layer for layer in mlp.layers if isinstance(layer, nn.Linear)]
        # 1 for hidden + expected_extra_linear for output
        assert len(linear_layers) == 1 + expected_extra_linear

    def test_dropout_layers_added_when_nonzero(
        self,
        mlp_factory: Callable[..., MLP],
    ):
        mlp = mlp_factory(
            input_dimension=32,
            hidden_dimensions=[64, 128],
            dropout=0.1,
        )
        dropout_layers = [
            layer for layer in mlp.layers if isinstance(layer, nn.Dropout)
        ]
        assert len(dropout_layers) == 2

    def test_no_dropout_layers_when_zero(
        self,
        mlp_factory: Callable[..., MLP],
    ):
        mlp = mlp_factory(
            input_dimension=32,
            hidden_dimensions=[64, 128],
            dropout=0.0,
        )
        dropout_layers = [
            layer for layer in mlp.layers if isinstance(layer, nn.Dropout)
        ]
        assert len(dropout_layers) == 0


class TestMLPForward:
    @pytest.mark.parametrize(
        "input_dimension, hidden_dimensions, output_dim, expected_output_dim",
        [
            (32, [64], 16, 16),
            (32, [64, 128], 8, 8),
            (32, [64], None, 64),
            (32, None, 16, 16),
        ],
    )
    def test_output_shape(
        self,
        mlp_factory: Callable[..., MLP],
        flat_tensor_factory: Callable[..., torch.Tensor],
        input_dimension: int,
        hidden_dimensions: list[int] | None,
        output_dim: int | None,
        expected_output_dim: int,
    ):
        mlp = mlp_factory(
            input_dimension=input_dimension,
            hidden_dimensions=hidden_dimensions,
            output_dim=output_dim,
        )
        tensor = flat_tensor_factory(batch_size=4, feature_dimension=input_dimension)
        output = mlp(tensor)
        assert output.shape == (4, expected_output_dim)

    def test_no_hidden_no_output_is_identity(
        self,
        mlp_factory: Callable[..., MLP],
        flat_tensor_factory: Callable[..., torch.Tensor],
    ):
        mlp = mlp_factory(input_dimension=32, hidden_dimensions=None, output_dim=None)
        tensor = flat_tensor_factory(batch_size=4, feature_dimension=32)
        output = mlp(tensor)
        assert output.shape == (4, 32)
        assert torch.equal(output, tensor)

    @pytest.mark.parametrize(
        "activation_function",
        [
            nn.GELU,
            nn.ReLU,
            nn.SiLU,
        ],
    )
    def test_works_with_different_activations(
        self,
        mlp_factory: Callable[..., MLP],
        flat_tensor_factory: Callable[..., torch.Tensor],
        activation_function: type[nn.Module],
    ):
        mlp = mlp_factory(
            input_dimension=32,
            hidden_dimensions=[64],
            output_dim=16,
            activation_function=activation_function,
        )
        tensor = flat_tensor_factory(batch_size=4, feature_dimension=32)
        output = mlp(tensor)
        assert output.shape == (4, 16)

    def test_dropout_does_not_affect_eval_output_shape(
        self,
        mlp_factory: Callable[..., MLP],
        flat_tensor_factory: Callable[..., torch.Tensor],
    ):
        mlp = mlp_factory(
            input_dimension=32,
            hidden_dimensions=[64],
            output_dim=16,
            dropout=0.5,
        )
        mlp.eval()
        tensor = flat_tensor_factory(batch_size=4, feature_dimension=32)
        output = mlp(tensor)
        assert output.shape == (4, 16)


class TestMLPWithSwiGLU:
    def test_forward_with_swiglu_activation(
        self,
        mlp_factory: Callable[..., MLP],
        flat_tensor_factory: Callable[..., torch.Tensor],
    ):
        mlp = mlp_factory(
            input_dimension=32,
            hidden_dimensions=[64],
            output_dim=16,
            activation_function=SwiGLU,
        )
        tensor = flat_tensor_factory(batch_size=4, feature_dimension=32)
        output = mlp(tensor)
        assert output.shape == (4, 16)

    def test_swiglu_used_instead_of_linear_plus_activation(
        self,
        mlp_factory: Callable[..., MLP],
    ):
        mlp = mlp_factory(
            input_dimension=32,
            hidden_dimensions=[64],
            activation_function=SwiGLU,
        )
        # SwiGLU replaces the Linear+Activation pair, so no separate nn.Linear for hidden
        swiglu_layers = [layer for layer in mlp.layers if isinstance(layer, SwiGLU)]
        standalone_linear_layers = [
            layer
            for layer in mlp.layers
            if isinstance(layer, nn.Linear) and not isinstance(layer, SwiGLU)
        ]
        assert len(swiglu_layers) == 1
        assert len(standalone_linear_layers) == 0

    def test_swiglu_with_multiple_hidden_dims(
        self,
        mlp_factory: Callable[..., MLP],
        flat_tensor_factory: Callable[..., torch.Tensor],
    ):
        mlp = mlp_factory(
            input_dimension=32,
            hidden_dimensions=[64, 32],
            output_dim=16,
            activation_function=SwiGLU,
        )
        tensor = flat_tensor_factory(batch_size=4, feature_dimension=32)
        output = mlp(tensor)
        assert output.shape == (4, 16)
