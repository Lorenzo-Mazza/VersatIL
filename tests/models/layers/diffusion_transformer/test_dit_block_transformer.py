"""Tests for versatil.models.layers.diffusion_transformer.dit_block_transformer module."""

from collections.abc import Callable

import pytest
import torch

from tests.models.layers.conftest import reinit_modulation_layers
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType
from versatil.models.layers.diffusion_transformer.dit_block_transformer import DiTBlock
from versatil.models.layers.normalization.constants import NormalizationType


@pytest.fixture
def dit_block_factory() -> Callable[..., DiTBlock]:
    def factory(
        number_of_encoder_layers: int = 1,
        number_of_decoder_layers: int = 1,
        embedding_dimension: int = 32,
        number_of_heads: int = 4,
        output_dimension: int | None = None,
        number_of_key_value_heads: int | None = None,
        feedforward_dimension: int | None = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation: str = ActivationFunction.SILU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        attention_type: str = AttentionType.MULTI_HEAD.value,
        positional_encoding_type: str | None = None,
        maximum_sequence_length: int = 256,
        maximum_decoder_length: int = 64,
        timestep_embedding_dimension: int = 32,
        bias: bool = True,
        normalization_epsilon: float = 1e-6,
        use_gating: bool = True,
        initializer_range: float = 0.02,
    ) -> DiTBlock:
        return DiTBlock(
            number_of_encoder_layers=number_of_encoder_layers,
            number_of_decoder_layers=number_of_decoder_layers,
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            output_dimension=output_dimension,
            number_of_key_value_heads=number_of_key_value_heads,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation=activation,
            normalization_type=normalization_type,
            attention_type=attention_type,
            positional_encoding_type=positional_encoding_type,
            maximum_sequence_length=maximum_sequence_length,
            maximum_decoder_length=maximum_decoder_length,
            timestep_embedding_dimension=timestep_embedding_dimension,
            bias=bias,
            normalization_epsilon=normalization_epsilon,
            use_gating=use_gating,
            initializer_range=initializer_range,
        )

    return factory


class TestDiTBlockInitialization:
    @pytest.mark.parametrize("number_of_encoder_layers", [1, 2])
    @pytest.mark.parametrize("number_of_decoder_layers", [1, 3])
    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    def test_stores_configuration(
        self,
        dit_block_factory: Callable[..., DiTBlock],
        number_of_encoder_layers: int,
        number_of_decoder_layers: int,
        embedding_dimension: int,
    ):
        block = dit_block_factory(
            number_of_encoder_layers=number_of_encoder_layers,
            number_of_decoder_layers=number_of_decoder_layers,
            embedding_dimension=embedding_dimension,
            number_of_heads=4,
        )
        assert block.embedding_dimension == embedding_dimension
        assert block.number_of_encoder_layers == number_of_encoder_layers
        assert block.number_of_decoder_layers == number_of_decoder_layers

    def test_output_dimension_defaults_to_embedding_dimension(
        self,
        dit_block_factory: Callable[..., DiTBlock],
    ):
        embedding_dimension = 32
        block = dit_block_factory(
            embedding_dimension=embedding_dimension,
            output_dimension=None,
        )
        assert block.output_dimension == embedding_dimension

    def test_explicit_output_dimension(
        self,
        dit_block_factory: Callable[..., DiTBlock],
    ):
        block = dit_block_factory(
            embedding_dimension=32,
            output_dimension=7,
        )
        assert block.output_dimension == 7


class TestDiTBlockForward:
    @pytest.mark.parametrize(
        "batch_size, encoder_sequence_length, decoder_sequence_length, embedding_dimension, output_dimension",
        [
            (2, 6, 4, 32, None),
            (1, 8, 4, 32, 7),
        ],
    )
    def test_output_shape(
        self,
        dit_block_factory: Callable[..., DiTBlock],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        continuous_timestep_factory: Callable[..., torch.Tensor],
        batch_size: int,
        encoder_sequence_length: int,
        decoder_sequence_length: int,
        embedding_dimension: int,
        output_dimension: int | None,
    ):
        block = dit_block_factory(
            embedding_dimension=embedding_dimension,
            output_dimension=output_dimension,
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
        encoder_output_mean, decoder_output = block(
            decoder_hidden_states=decoder_hidden,
            timesteps=timesteps,
            encoder_hidden_states=encoder_hidden,
        )
        expected_output_dim = output_dimension or embedding_dimension
        assert encoder_output_mean.shape == (batch_size, embedding_dimension)
        assert decoder_output.shape == (
            batch_size,
            decoder_sequence_length,
            expected_output_dim,
        )

    def test_encoder_cache_bypasses_encoder(
        self,
        dit_block_factory: Callable[..., DiTBlock],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        continuous_timestep_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        block = dit_block_factory(embedding_dimension=embedding_dimension)
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
        encoder_cache = condition_factory(
            batch_size=2,
            condition_dim=embedding_dimension,
        )
        encoder_output_mean, decoder_output = block(
            decoder_hidden_states=decoder_hidden,
            timesteps=timesteps,
            encoder_hidden_states=encoder_hidden,
            encoder_cache=encoder_cache,
        )
        # When cache is provided, encoder_output_mean should be the cache itself
        assert torch.allclose(encoder_output_mean, encoder_cache)

    def test_different_timesteps_produce_different_outputs(
        self,
        dit_block_factory: Callable[..., DiTBlock],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        block = dit_block_factory(
            embedding_dimension=embedding_dimension,
            use_gating=False,
        )
        # Break zero init on modulation and prediction layers
        reinit_modulation_layers(block)
        torch.nn.init.xavier_uniform_(block.epsilon_network.output_linear.weight)
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
        _, output_low = block(
            decoder_hidden_states=decoder_hidden,
            timesteps=timesteps_low,
            encoder_hidden_states=encoder_hidden,
        )
        _, output_high = block(
            decoder_hidden_states=decoder_hidden,
            timesteps=timesteps_high,
            encoder_hidden_states=encoder_hidden,
        )
        assert not torch.allclose(output_low, output_high)

    def test_encoder_output_mean_is_mean_pooled(
        self,
        dit_block_factory: Callable[..., DiTBlock],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        continuous_timestep_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        block = dit_block_factory(embedding_dimension=embedding_dimension)
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
        encoder_output_mean, _ = block(
            decoder_hidden_states=decoder_hidden,
            timesteps=timesteps,
            encoder_hidden_states=encoder_hidden,
        )
        # Encoder mean should be a (B, D) vector
        assert encoder_output_mean.shape == (2, embedding_dimension)
        # Verify forward_encoder produces the same result
        direct_mean = block.forward_encoder(encoder_hidden)
        assert torch.allclose(encoder_output_mean, direct_mean)

    def test_gradient_flows_through_full_model(
        self,
        dit_block_factory: Callable[..., DiTBlock],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        continuous_timestep_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        block = dit_block_factory(embedding_dimension=embedding_dimension)
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
        _, decoder_output = block(
            decoder_hidden_states=decoder_hidden,
            timesteps=timesteps,
            encoder_hidden_states=encoder_hidden,
        )
        decoder_output.sum().backward()
        assert encoder_hidden.grad is not None
        assert decoder_hidden.grad is not None
        assert torch.all(torch.isfinite(encoder_hidden.grad))
        assert torch.all(torch.isfinite(decoder_hidden.grad))
