"""Tests for versatil.models.layers.diffusion_transformer.dit_decoder_layer module."""
from collections.abc import Callable

import pytest
import torch

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType
from versatil.models.layers.diffusion_transformer.dit_decoder_layer import DecoderLayer
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.swiglu import SwiGLU

from tests.models.layers.diffusion_transformer.conftest import reinit_modulation_layers


@pytest.fixture
def decoder_layer_factory() -> Callable[..., DecoderLayer]:

    def factory(
        embedding_dimension: int = 32,
        timestep_dimension: int = 32,
        number_of_heads: int = 4,
        number_of_key_value_heads: int | None = None,
        feedforward_dimension: int | None = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation: str = ActivationFunction.SILU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        attention_type: str = AttentionType.MULTI_HEAD.value,
        bias: bool = True,
        normalization_epsilon: float = 1e-6,
        use_gating: bool = True,
    ) -> DecoderLayer:
        return DecoderLayer(
            embedding_dimension=embedding_dimension,
            timestep_dimension=timestep_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_key_value_heads,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation=activation,
            normalization_type=normalization_type,
            attention_type=attention_type,
            bias=bias,
            normalization_epsilon=normalization_epsilon,
            use_gating=use_gating,
        )

    return factory


class TestDecoderLayerInitialization:

    @pytest.mark.parametrize("use_gating", [True, False])
    @pytest.mark.parametrize("activation", [ActivationFunction.SILU.value, ActivationFunction.SWIGLU.value])
    def test_stores_configuration(
        self,
        decoder_layer_factory: Callable[..., DecoderLayer],
        use_gating: bool,
        activation: str,
    ):
        layer = decoder_layer_factory(
            use_gating=use_gating,
            activation=activation,
        )
        assert layer.use_gating == use_gating

    def test_swiglu_activation_creates_swiglu_feedforward(
        self,
        decoder_layer_factory: Callable[..., DecoderLayer],
    ):
        layer = decoder_layer_factory(activation=ActivationFunction.SWIGLU.value)
        first_module = layer.feedforward_network[0]
        assert type(first_module) is SwiGLU

    def test_non_swiglu_activation_creates_linear_feedforward(
        self,
        decoder_layer_factory: Callable[..., DecoderLayer],
    ):
        layer = decoder_layer_factory(activation=ActivationFunction.GELU.value)
        first_module = layer.feedforward_network[0]
        assert type(first_module) is torch.nn.Linear


class TestDecoderLayerForward:

    @pytest.mark.parametrize(
        "batch_size, sequence_length, embedding_dimension",
        [
            (2, 4, 32),
            (1, 8, 64),
        ],
    )
    def test_output_shape(
        self,
        decoder_layer_factory: Callable[..., DecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
        batch_size: int,
        sequence_length: int,
        embedding_dimension: int,
    ):
        layer = decoder_layer_factory(
            embedding_dimension=embedding_dimension,
            timestep_dimension=embedding_dimension,
            number_of_heads=4,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=embedding_dimension,
        )
        conditioning = condition_factory(
            batch_size=batch_size,
            condition_dim=embedding_dimension,
        )
        output = layer(
            hidden_states=hidden_states,
            conditioning_embedding=conditioning,
        )
        assert output.shape == (batch_size, sequence_length, embedding_dimension)

    def test_adaln_zero_output_equals_input_at_initialization(
        self,
        decoder_layer_factory: Callable[..., DecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        # AdaLN-Zero initializes gates to zero, so the gated residual path
        # contributes nothing: output = input + 0 * f(input) = input
        embedding_dimension = 32
        layer = decoder_layer_factory(
            embedding_dimension=embedding_dimension,
            timestep_dimension=embedding_dimension,
            use_gating=True,
        )
        layer.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=embedding_dimension,
        )
        conditioning = condition_factory(
            batch_size=2,
            condition_dim=embedding_dimension,
        )
        output = layer(
            hidden_states=hidden_states,
            conditioning_embedding=conditioning,
        )
        assert torch.allclose(output, hidden_states, atol=1e-6)

    def test_different_conditioning_produces_different_outputs(
        self,
        decoder_layer_factory: Callable[..., DecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        layer = decoder_layer_factory(
            embedding_dimension=embedding_dimension,
            timestep_dimension=embedding_dimension,
            use_gating=False,
        )
        reinit_modulation_layers(layer)
        hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=embedding_dimension,
        )
        conditioning_a = condition_factory(
            batch_size=2,
            condition_dim=embedding_dimension,
        )
        conditioning_b = condition_factory(
            batch_size=2,
            condition_dim=embedding_dimension,
        )
        output_a = layer(
            hidden_states=hidden_states,
            conditioning_embedding=conditioning_a,
        )
        output_b = layer(
            hidden_states=hidden_states,
            conditioning_embedding=conditioning_b,
        )
        assert not torch.allclose(output_a, output_b)

    @pytest.mark.parametrize("use_gating", [True, False])
    def test_gating_path_produces_valid_output(
        self,
        decoder_layer_factory: Callable[..., DecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
        use_gating: bool,
    ):
        embedding_dimension = 32
        layer = decoder_layer_factory(
            embedding_dimension=embedding_dimension,
            timestep_dimension=embedding_dimension,
            use_gating=use_gating,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=embedding_dimension,
        )
        conditioning = condition_factory(
            batch_size=2,
            condition_dim=embedding_dimension,
        )
        output = layer(
            hidden_states=hidden_states,
            conditioning_embedding=conditioning,
        )
        assert output.shape == hidden_states.shape
        assert torch.all(torch.isfinite(output))

    def test_gradient_flows_through_layer(
        self,
        decoder_layer_factory: Callable[..., DecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        layer = decoder_layer_factory(
            embedding_dimension=embedding_dimension,
            timestep_dimension=embedding_dimension,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=embedding_dimension,
        )
        conditioning = condition_factory(
            batch_size=2,
            condition_dim=embedding_dimension,
        )
        hidden_states.requires_grad_(True)
        output = layer(
            hidden_states=hidden_states,
            conditioning_embedding=conditioning,
        )
        output.sum().backward()
        assert hidden_states.grad is not None
        assert torch.all(torch.isfinite(hidden_states.grad))
