"""Tests for versatil.models.layers.diffusion_transformer.cross_attention_dit_layer module."""
from collections.abc import Callable

import pytest
import torch

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType
from versatil.models.layers.diffusion_transformer.cross_attention_dit_layer import (
    CrossConditioningDecoderLayer,
)
from versatil.models.layers.normalization.constants import NormalizationType

from tests.models.layers.diffusion_transformer.conftest import reinit_modulation_layers


@pytest.fixture
def cross_decoder_layer_factory() -> Callable[..., CrossConditioningDecoderLayer]:

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
    ) -> CrossConditioningDecoderLayer:
        return CrossConditioningDecoderLayer(
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


class TestCrossConditioningDecoderLayerInitialization:

    @pytest.mark.parametrize("use_gating", [True, False])
    @pytest.mark.parametrize("activation", [ActivationFunction.SILU.value, ActivationFunction.SWIGLU.value])
    def test_stores_configuration(
        self,
        cross_decoder_layer_factory: Callable[..., CrossConditioningDecoderLayer],
        use_gating: bool,
        activation: str,
    ):
        layer = cross_decoder_layer_factory(
            use_gating=use_gating,
            activation=activation,
        )
        assert layer.use_gating == use_gating

    def test_self_and_cross_attention_have_independent_weights(
        self,
        cross_decoder_layer_factory: Callable[..., CrossConditioningDecoderLayer],
    ):
        layer = cross_decoder_layer_factory(embedding_dimension=32)
        # Mutate self-attention weights and verify cross-attention is unaffected
        original_cross_weight = layer.cross_attention.query_projection.weight.data.clone()
        layer.self_attention.query_projection.weight.data.fill_(999.0)
        assert torch.allclose(
            layer.cross_attention.query_projection.weight.data, original_cross_weight
        )


class TestCrossConditioningDecoderLayerForward:

    @pytest.mark.parametrize(
        "batch_size, decoder_sequence_length, encoder_sequence_length, embedding_dimension",
        [
            (2, 4, 6, 32),
            (1, 8, 10, 64),
        ],
    )
    def test_output_shape(
        self,
        cross_decoder_layer_factory: Callable[..., CrossConditioningDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
        batch_size: int,
        decoder_sequence_length: int,
        encoder_sequence_length: int,
        embedding_dimension: int,
    ):
        layer = cross_decoder_layer_factory(
            embedding_dimension=embedding_dimension,
            timestep_dimension=embedding_dimension,
            number_of_heads=4,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=decoder_sequence_length,
            embedding_dimension=embedding_dimension,
        )
        encoder_hidden = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=encoder_sequence_length,
            embedding_dimension=embedding_dimension,
        )
        conditioning = condition_factory(
            batch_size=batch_size,
            condition_dim=embedding_dimension,
        )
        output = layer(
            hidden_states=hidden_states,
            conditioning_embedding=conditioning,
            encoder_hidden_states=encoder_hidden,
        )
        assert output.shape == (batch_size, decoder_sequence_length, embedding_dimension)

    def test_cross_attention_with_different_context_produces_different_outputs(
        self,
        cross_decoder_layer_factory: Callable[..., CrossConditioningDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        layer = cross_decoder_layer_factory(
            embedding_dimension=embedding_dimension,
            timestep_dimension=embedding_dimension,
            use_gating=False,
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
        encoder_hidden_a = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=embedding_dimension,
        )
        encoder_hidden_b = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=embedding_dimension,
        )
        output_a = layer(
            hidden_states=hidden_states,
            conditioning_embedding=conditioning,
            encoder_hidden_states=encoder_hidden_a,
        )
        output_b = layer(
            hidden_states=hidden_states,
            conditioning_embedding=conditioning,
            encoder_hidden_states=encoder_hidden_b,
        )
        assert not torch.allclose(output_a, output_b)

    def test_different_conditioning_produces_different_outputs(
        self,
        cross_decoder_layer_factory: Callable[..., CrossConditioningDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        layer = cross_decoder_layer_factory(
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
        encoder_hidden = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
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
            encoder_hidden_states=encoder_hidden,
        )
        output_b = layer(
            hidden_states=hidden_states,
            conditioning_embedding=conditioning_b,
            encoder_hidden_states=encoder_hidden,
        )
        assert not torch.allclose(output_a, output_b)

    @pytest.mark.parametrize("use_gating", [True, False])
    def test_gating_path_produces_valid_output(
        self,
        cross_decoder_layer_factory: Callable[..., CrossConditioningDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
        use_gating: bool,
    ):
        embedding_dimension = 32
        layer = cross_decoder_layer_factory(
            embedding_dimension=embedding_dimension,
            timestep_dimension=embedding_dimension,
            use_gating=use_gating,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=embedding_dimension,
        )
        encoder_hidden = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=embedding_dimension,
        )
        conditioning = condition_factory(
            batch_size=2,
            condition_dim=embedding_dimension,
        )
        output = layer(
            hidden_states=hidden_states,
            conditioning_embedding=conditioning,
            encoder_hidden_states=encoder_hidden,
        )
        assert output.shape == hidden_states.shape
        assert torch.all(torch.isfinite(output))

    def test_adaln_zero_gates_self_attention_and_ffn_but_not_cross_attention(
        self,
        cross_decoder_layer_factory: Callable[..., CrossConditioningDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        # AdaLN-Zero gates control self-attention and FFN residual paths,
        # but cross-attention has NO gating -- it always contributes.
        # So with use_gating=True at init, output != input (cross-attention still active).
        embedding_dimension = 32
        layer = cross_decoder_layer_factory(
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
        encoder_hidden = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=embedding_dimension,
        )
        conditioning = condition_factory(
            batch_size=2,
            condition_dim=embedding_dimension,
        )
        output = layer(
            hidden_states=hidden_states,
            conditioning_embedding=conditioning,
            encoder_hidden_states=encoder_hidden,
        )
        # Cross-attention is ungated, so output differs from input
        assert not torch.allclose(output, hidden_states, atol=1e-6)

    def test_gradient_flows_through_cross_attention(
        self,
        cross_decoder_layer_factory: Callable[..., CrossConditioningDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        layer = cross_decoder_layer_factory(
            embedding_dimension=embedding_dimension,
            timestep_dimension=embedding_dimension,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=embedding_dimension,
        )
        encoder_hidden = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=embedding_dimension,
        )
        conditioning = condition_factory(
            batch_size=2,
            condition_dim=embedding_dimension,
        )
        hidden_states.requires_grad_(True)
        encoder_hidden.requires_grad_(True)
        output = layer(
            hidden_states=hidden_states,
            conditioning_embedding=conditioning,
            encoder_hidden_states=encoder_hidden,
        )
        output.sum().backward()
        assert hidden_states.grad is not None
        assert encoder_hidden.grad is not None
        assert torch.all(torch.isfinite(hidden_states.grad))
        assert torch.all(torch.isfinite(encoder_hidden.grad))
