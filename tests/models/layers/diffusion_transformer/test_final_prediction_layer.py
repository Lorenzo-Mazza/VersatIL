"""Tests for versatil.models.layers.diffusion_transformer.final_prediction_layer module."""

from collections.abc import Callable

import pytest
import torch

from tests.models.layers.conftest import reinit_modulation_layers
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.diffusion_transformer.final_prediction_layer import (
    FinalPredictionLayer,
)


@pytest.fixture
def final_prediction_layer_factory() -> Callable[..., FinalPredictionLayer]:
    def factory(
        hidden_dimension: int = 32,
        output_dimension: int = 7,
        activation: str = ActivationFunction.SILU.value,
    ) -> FinalPredictionLayer:
        return FinalPredictionLayer(
            hidden_dimension=hidden_dimension,
            output_dimension=output_dimension,
            activation=activation,
        )

    return factory


class TestFinalPredictionLayerInitialization:
    @pytest.mark.parametrize("hidden_dimension", [32, 64])
    @pytest.mark.parametrize("output_dimension", [7, 14])
    def test_stores_configuration(
        self,
        final_prediction_layer_factory: Callable[..., FinalPredictionLayer],
        hidden_dimension: int,
        output_dimension: int,
    ):
        layer = final_prediction_layer_factory(
            hidden_dimension=hidden_dimension,
            output_dimension=output_dimension,
        )
        assert layer.output_linear.in_features == hidden_dimension
        assert layer.output_linear.out_features == output_dimension

    def test_output_linear_initialized_to_zeros(
        self,
        final_prediction_layer_factory: Callable[..., FinalPredictionLayer],
    ):
        layer = final_prediction_layer_factory(
            hidden_dimension=32,
            output_dimension=7,
        )
        assert torch.all(layer.output_linear.weight == 0.0)
        assert torch.all(layer.output_linear.bias == 0.0)

    def test_reset_parameters_zeros_output_linear(
        self,
        final_prediction_layer_factory: Callable[..., FinalPredictionLayer],
    ):
        layer = final_prediction_layer_factory(
            hidden_dimension=32,
            output_dimension=7,
        )
        layer.output_linear.weight.data.fill_(1.0)
        layer.output_linear.bias.data.fill_(1.0)
        layer.reset_parameters()
        assert torch.all(layer.output_linear.weight == 0.0)
        assert torch.all(layer.output_linear.bias == 0.0)


class TestFinalPredictionLayerForward:
    @pytest.mark.parametrize(
        "batch_size, sequence_length, hidden_dimension, output_dimension",
        [
            (2, 4, 32, 7),
            (1, 8, 64, 14),
        ],
    )
    def test_output_shape(
        self,
        final_prediction_layer_factory: Callable[..., FinalPredictionLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
        batch_size: int,
        sequence_length: int,
        hidden_dimension: int,
        output_dimension: int,
    ):
        layer = final_prediction_layer_factory(
            hidden_dimension=hidden_dimension,
            output_dimension=output_dimension,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=hidden_dimension,
        )
        conditioning = condition_factory(
            batch_size=batch_size,
            conditioning_dimension=hidden_dimension,
        )
        output = layer(hidden_states, conditioning)
        assert output.shape == (batch_size, sequence_length, output_dimension)

    def test_zero_initialized_output_is_zero_before_training(
        self,
        final_prediction_layer_factory: Callable[..., FinalPredictionLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        hidden_dimension = 32
        layer = final_prediction_layer_factory(
            hidden_dimension=hidden_dimension,
            output_dimension=7,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=hidden_dimension,
        )
        conditioning = condition_factory(
            batch_size=2,
            conditioning_dimension=hidden_dimension,
        )
        output = layer(hidden_states, conditioning)
        assert torch.allclose(output, torch.zeros_like(output), atol=1e-6)

    def test_different_conditioning_produces_different_outputs(
        self,
        final_prediction_layer_factory: Callable[..., FinalPredictionLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        hidden_dimension = 32
        layer = final_prediction_layer_factory(
            hidden_dimension=hidden_dimension,
            output_dimension=7,
        )
        # Break zero init on both modulation and output layers
        reinit_modulation_layers(layer)
        torch.nn.init.xavier_uniform_(layer.output_linear.weight)
        hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=hidden_dimension,
        )
        conditioning_a = condition_factory(
            batch_size=2,
            conditioning_dimension=hidden_dimension,
        )
        conditioning_b = condition_factory(
            batch_size=2,
            conditioning_dimension=hidden_dimension,
        )
        output_a = layer(hidden_states, conditioning_a)
        output_b = layer(hidden_states, conditioning_b)
        assert not torch.allclose(output_a, output_b)

    def test_gradient_flows_through_layer(
        self,
        final_prediction_layer_factory: Callable[..., FinalPredictionLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        hidden_dimension = 32
        layer = final_prediction_layer_factory(
            hidden_dimension=hidden_dimension,
            output_dimension=7,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=hidden_dimension,
        )
        conditioning = condition_factory(
            batch_size=2,
            conditioning_dimension=hidden_dimension,
        )
        hidden_states.requires_grad_(True)
        output = layer(hidden_states, conditioning)
        output.sum().backward()
        assert hidden_states.grad is not None
        assert torch.all(torch.isfinite(hidden_states.grad))
