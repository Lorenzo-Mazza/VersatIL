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


class _IdentityEncoder(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return hidden_states


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


class TestDiTBlockForward:
    @pytest.mark.parametrize(
        "batch_size, encoder_sequence_length, decoder_sequence_length, embedding_dimension",
        [
            (2, 6, 4, 32),
            (1, 8, 4, 32),
        ],
    )
    def test_hidden_state_and_condition_shape(
        self,
        dit_block_factory: Callable[..., DiTBlock],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        continuous_timestep_factory: Callable[..., torch.Tensor],
        batch_size: int,
        encoder_sequence_length: int,
        decoder_sequence_length: int,
        embedding_dimension: int,
    ):
        block = dit_block_factory(
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
        encoder_output_mean, decoder_output, conditioning = block(
            decoder_hidden_states=decoder_hidden,
            timesteps=timesteps,
            encoder_hidden_states=encoder_hidden,
        )
        assert encoder_output_mean.shape == (batch_size, embedding_dimension)
        assert decoder_output.shape == (
            batch_size,
            decoder_sequence_length,
            embedding_dimension,
        )
        assert conditioning.shape == (batch_size, embedding_dimension)

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
        encoder_output_mean, decoder_output, conditioning = block(
            decoder_hidden_states=decoder_hidden,
            timesteps=timesteps,
            encoder_hidden_states=encoder_hidden,
            encoder_cache=encoder_cache,
        )
        assert torch.allclose(encoder_output_mean, encoder_cache)
        assert decoder_output.shape == (2, 4, embedding_dimension)
        assert conditioning.shape == (2, embedding_dimension)

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
        reinit_modulation_layers(block)
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
        _, output_low, condition_low = block(
            decoder_hidden_states=decoder_hidden,
            timesteps=timesteps_low,
            encoder_hidden_states=encoder_hidden,
        )
        _, output_high, condition_high = block(
            decoder_hidden_states=decoder_hidden,
            timesteps=timesteps_high,
            encoder_hidden_states=encoder_hidden,
        )
        assert not torch.allclose(output_low, output_high)
        assert not torch.allclose(condition_low, condition_high)

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
        encoder_output_mean, _, _ = block(
            decoder_hidden_states=decoder_hidden,
            timesteps=timesteps,
            encoder_hidden_states=encoder_hidden,
        )
        # Encoder mean should be a (B, D) vector
        assert encoder_output_mean.shape == (2, embedding_dimension)
        # Verify forward_encoder produces the same result
        direct_mean = block.forward_encoder(encoder_hidden)
        assert torch.allclose(encoder_output_mean, direct_mean)

    def test_forward_encoder_excludes_padded_tokens_from_mean(
        self,
        dit_block_factory: Callable[..., DiTBlock],
    ):
        block = dit_block_factory(embedding_dimension=4, number_of_heads=2)
        block.encoder = _IdentityEncoder()
        encoder_hidden = torch.tensor(
            [
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [3.0, 3.0, 3.0, 3.0],
                    [100.0, 100.0, 100.0, 100.0],
                ],
                [[2.0, 0.0, 2.0, 0.0], [6.0, 0.0, 6.0, 0.0], [10.0, 0.0, 10.0, 0.0]],
            ]
        )
        padding_mask = torch.tensor(
            [
                [False, False, True],
                [False, True, True],
            ]
        )
        encoder_output_mean = block.forward_encoder(
            hidden_states=encoder_hidden,
            padding_mask=padding_mask,
        )
        expected = torch.tensor(
            [
                [2.0, 2.0, 2.0, 2.0],
                [2.0, 0.0, 2.0, 0.0],
            ]
        )
        torch.testing.assert_close(encoder_output_mean, expected)

    def test_forward_encoder_all_padded_tokens_returns_zero_mean(
        self,
        dit_block_factory: Callable[..., DiTBlock],
    ):
        block = dit_block_factory(embedding_dimension=4, number_of_heads=2)
        block.encoder = _IdentityEncoder()
        encoder_hidden = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
        padding_mask = torch.tensor([[True, True]])
        encoder_output_mean = block.forward_encoder(
            hidden_states=encoder_hidden,
            padding_mask=padding_mask,
        )
        torch.testing.assert_close(
            encoder_output_mean,
            torch.zeros(1, 4),
        )

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
        _, decoder_output, conditioning = block(
            decoder_hidden_states=decoder_hidden,
            timesteps=timesteps,
            encoder_hidden_states=encoder_hidden,
        )
        (decoder_output.sum() + conditioning.sum()).backward()
        assert encoder_hidden.grad is not None
        assert decoder_hidden.grad is not None
        assert torch.all(torch.isfinite(encoder_hidden.grad))
        assert torch.all(torch.isfinite(decoder_hidden.grad))
