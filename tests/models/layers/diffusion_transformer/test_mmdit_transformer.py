"""Tests for versatil.models.layers.diffusion_transformer.mmdit_transformer module."""

from collections.abc import Callable

import pytest
import torch

from tests.models.layers.conftest import reinit_modulation_layers
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import PositionalEncodingType
from versatil.models.layers.diffusion_transformer.mmdit_transformer import (
    MMDiTTransformer,
)
from versatil.models.layers.normalization.constants import NormalizationType


@pytest.fixture
def mmdit_transformer_factory() -> Callable[..., MMDiTTransformer]:
    def factory(
        number_of_layers: int = 1,
        embedding_dimension: int = 32,
        number_of_heads: int = 4,
        feedforward_dimension: int | None = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation: str = ActivationFunction.SILU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        positional_encoding_type: str | None = None,
        maximum_sequence_length: int = 256,
        maximum_decoder_length: int = 64,
        timestep_embedding_dimension: int = 32,
        use_query_key_norm: bool = True,
        use_gating: bool = True,
        bias: bool = True,
        normalization_epsilon: float = 1e-6,
        initializer_range: float = 0.02,
    ) -> MMDiTTransformer:
        return MMDiTTransformer(
            number_of_layers=number_of_layers,
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation=activation,
            normalization_type=normalization_type,
            positional_encoding_type=positional_encoding_type,
            maximum_sequence_length=maximum_sequence_length,
            maximum_decoder_length=maximum_decoder_length,
            timestep_embedding_dimension=timestep_embedding_dimension,
            use_query_key_norm=use_query_key_norm,
            use_gating=use_gating,
            bias=bias,
            normalization_epsilon=normalization_epsilon,
            initializer_range=initializer_range,
        )

    return factory


class TestMMDiTTransformerInitialization:
    @pytest.mark.parametrize("number_of_layers", [1, 2])
    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    def test_stores_configuration(
        self,
        mmdit_transformer_factory: Callable[..., MMDiTTransformer],
        number_of_layers: int,
        embedding_dimension: int,
    ):
        model = mmdit_transformer_factory(
            number_of_layers=number_of_layers,
            embedding_dimension=embedding_dimension,
            number_of_heads=4,
        )
        assert model.embedding_dimension == embedding_dimension
        assert model.number_of_layers == number_of_layers

    def test_decoder_has_correct_number_of_layers(
        self,
        mmdit_transformer_factory: Callable[..., MMDiTTransformer],
    ):
        number_of_layers = 3
        model = mmdit_transformer_factory(number_of_layers=number_of_layers)
        assert len(model.decoder.layers) == number_of_layers


class TestMMDiTTransformerForward:
    @pytest.mark.parametrize(
        "batch_size, encoder_sequence_length, decoder_sequence_length, embedding_dimension",
        [
            (2, 6, 4, 32),
            (1, 8, 4, 32),
        ],
    )
    def test_hidden_state_and_condition_shape(
        self,
        mmdit_transformer_factory: Callable[..., MMDiTTransformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        continuous_timestep_factory: Callable[..., torch.Tensor],
        batch_size: int,
        encoder_sequence_length: int,
        decoder_sequence_length: int,
        embedding_dimension: int,
    ):
        model = mmdit_transformer_factory(
            embedding_dimension=embedding_dimension,
        )
        encoder_hidden = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=encoder_sequence_length,
            embedding_dimension=embedding_dimension,
        )
        decoder_hidden = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=decoder_sequence_length,
            embedding_dimension=embedding_dimension,
        )
        timesteps = continuous_timestep_factory(batch_size=batch_size)
        output, conditioning = model(
            decoder_hidden_states=decoder_hidden,
            timesteps=timesteps,
            encoder_hidden_states=encoder_hidden,
        )
        assert output.shape == (
            batch_size,
            decoder_sequence_length,
            embedding_dimension,
        )
        assert conditioning.shape == (batch_size, embedding_dimension)

    def test_returns_action_hidden_states_and_conditioning(
        self,
        mmdit_transformer_factory: Callable[..., MMDiTTransformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        continuous_timestep_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        model = mmdit_transformer_factory(
            embedding_dimension=embedding_dimension,
        )
        encoder_hidden = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=embedding_dimension,
        )
        decoder_hidden = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=embedding_dimension,
        )
        timesteps = continuous_timestep_factory(batch_size=2)
        output, conditioning = model(
            decoder_hidden_states=decoder_hidden,
            timesteps=timesteps,
            encoder_hidden_states=encoder_hidden,
        )
        assert output.shape == (2, 4, embedding_dimension)
        assert conditioning.shape == (2, embedding_dimension)

    def test_different_timesteps_produce_different_outputs(
        self,
        mmdit_transformer_factory: Callable[..., MMDiTTransformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        model = mmdit_transformer_factory(
            embedding_dimension=embedding_dimension,
            use_gating=False,
        )
        reinit_modulation_layers(model)
        encoder_hidden = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=embedding_dimension,
        )
        decoder_hidden = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=embedding_dimension,
        )
        timesteps_low = torch.tensor([0.1, 0.1])
        timesteps_high = torch.tensor([0.9, 0.9])
        output_low, condition_low = model(
            decoder_hidden_states=decoder_hidden,
            timesteps=timesteps_low,
            encoder_hidden_states=encoder_hidden,
        )
        output_high, condition_high = model(
            decoder_hidden_states=decoder_hidden,
            timesteps=timesteps_high,
            encoder_hidden_states=encoder_hidden,
        )
        assert not torch.allclose(output_low, output_high)
        assert not torch.allclose(condition_low, condition_high)

    def test_different_encoder_context_produces_different_outputs(
        self,
        mmdit_transformer_factory: Callable[..., MMDiTTransformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        continuous_timestep_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        model = mmdit_transformer_factory(
            embedding_dimension=embedding_dimension,
            use_gating=False,
        )
        reinit_modulation_layers(model)
        decoder_hidden = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=embedding_dimension,
        )
        timesteps = continuous_timestep_factory(batch_size=2)
        encoder_a = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=embedding_dimension,
        )
        encoder_b = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=embedding_dimension,
        )
        output_a, _ = model(
            decoder_hidden_states=decoder_hidden,
            timesteps=timesteps,
            encoder_hidden_states=encoder_a,
        )
        output_b, _ = model(
            decoder_hidden_states=decoder_hidden,
            timesteps=timesteps,
            encoder_hidden_states=encoder_b,
        )
        assert not torch.allclose(output_a, output_b)

    def test_different_decoder_input_produces_different_outputs(
        self,
        mmdit_transformer_factory: Callable[..., MMDiTTransformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        continuous_timestep_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        model = mmdit_transformer_factory(
            embedding_dimension=embedding_dimension,
            use_gating=False,
        )
        reinit_modulation_layers(model)
        encoder_hidden = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=embedding_dimension,
        )
        timesteps = continuous_timestep_factory(batch_size=2)
        decoder_a = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=embedding_dimension,
        )
        decoder_b = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=embedding_dimension,
        )
        output_a, _ = model(
            decoder_hidden_states=decoder_a,
            timesteps=timesteps,
            encoder_hidden_states=encoder_hidden,
        )
        output_b, _ = model(
            decoder_hidden_states=decoder_b,
            timesteps=timesteps,
            encoder_hidden_states=encoder_hidden,
        )
        assert not torch.allclose(output_a, output_b)

    def test_padding_masks_affect_output(
        self,
        mmdit_transformer_factory: Callable[..., MMDiTTransformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        continuous_timestep_factory: Callable[..., torch.Tensor],
        padding_mask_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        model = mmdit_transformer_factory(
            embedding_dimension=embedding_dimension,
            use_gating=False,
        )
        reinit_modulation_layers(model)
        encoder_hidden = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=embedding_dimension,
        )
        decoder_hidden = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=embedding_dimension,
        )
        timesteps = continuous_timestep_factory(batch_size=2)
        output_no_mask, _ = model(
            decoder_hidden_states=decoder_hidden,
            timesteps=timesteps,
            encoder_hidden_states=encoder_hidden,
        )
        encoder_mask = padding_mask_factory(
            batch_size=2,
            sequence_length=6,
            mask_last_n=2,
        )
        output_with_mask, _ = model(
            decoder_hidden_states=decoder_hidden,
            timesteps=timesteps,
            encoder_hidden_states=encoder_hidden,
            encoder_padding_mask=encoder_mask,
        )
        assert not torch.allclose(output_no_mask, output_with_mask)

    @pytest.mark.parametrize(
        "positional_encoding_type",
        [
            None,
            PositionalEncodingType.SINUSOIDAL.value,
            PositionalEncodingType.ROPE.value,
        ],
    )
    def test_positional_encoding_path_produces_valid_output(
        self,
        mmdit_transformer_factory: Callable[..., MMDiTTransformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        continuous_timestep_factory: Callable[..., torch.Tensor],
        positional_encoding_type: str | None,
    ):
        embedding_dimension = 32
        model = mmdit_transformer_factory(
            embedding_dimension=embedding_dimension,
            positional_encoding_type=positional_encoding_type,
        )
        encoder_hidden = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=embedding_dimension,
        )
        decoder_hidden = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=embedding_dimension,
        )
        timesteps = continuous_timestep_factory(batch_size=2)
        output, conditioning = model(
            decoder_hidden_states=decoder_hidden,
            timesteps=timesteps,
            encoder_hidden_states=encoder_hidden,
        )
        assert torch.all(torch.isfinite(output))
        assert torch.all(torch.isfinite(conditioning))

    def test_gradient_flows_through_full_model(
        self,
        mmdit_transformer_factory: Callable[..., MMDiTTransformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        continuous_timestep_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        model = mmdit_transformer_factory(embedding_dimension=embedding_dimension)
        encoder_hidden = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=embedding_dimension,
        )
        decoder_hidden = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=embedding_dimension,
        )
        timesteps = continuous_timestep_factory(batch_size=2)
        encoder_hidden.requires_grad_(True)
        decoder_hidden.requires_grad_(True)
        output, conditioning = model(
            decoder_hidden_states=decoder_hidden,
            timesteps=timesteps,
            encoder_hidden_states=encoder_hidden,
        )
        (output.sum() + conditioning.sum()).backward()
        assert encoder_hidden.grad is not None
        assert decoder_hidden.grad is not None
        assert torch.all(torch.isfinite(encoder_hidden.grad))
        assert torch.all(torch.isfinite(decoder_hidden.grad))
