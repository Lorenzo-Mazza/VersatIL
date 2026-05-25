"""Tests for versatil.models.layers.diffusion_transformer.cross_attention_dit module."""

from collections.abc import Callable

import pytest
import torch

from tests.models.layers.conftest import reinit_modulation_layers
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType
from versatil.models.layers.diffusion_transformer.cross_attention_dit import (
    CrossAttentionDiT,
)
from versatil.models.layers.normalization.constants import NormalizationType


@pytest.fixture
def cross_attention_dit_factory() -> Callable[..., CrossAttentionDiT]:
    def factory(
        number_of_layers: int = 1,
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
        timestep_embedding_dimension: int = 32,
        bias: bool = True,
        normalization_epsilon: float = 1e-6,
        use_gating: bool = True,
        initializer_range: float = 0.02,
    ) -> CrossAttentionDiT:
        return CrossAttentionDiT(
            number_of_layers=number_of_layers,
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
            timestep_embedding_dimension=timestep_embedding_dimension,
            bias=bias,
            normalization_epsilon=normalization_epsilon,
            use_gating=use_gating,
            initializer_range=initializer_range,
        )

    return factory


class TestCrossAttentionDiTInitialization:
    @pytest.mark.parametrize("number_of_layers", [1, 2])
    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    def test_stores_configuration(
        self,
        cross_attention_dit_factory: Callable[..., CrossAttentionDiT],
        number_of_layers: int,
        embedding_dimension: int,
    ):
        model = cross_attention_dit_factory(
            number_of_layers=number_of_layers,
            embedding_dimension=embedding_dimension,
            number_of_heads=4,
        )
        assert model.embedding_dimension == embedding_dimension
        assert model.number_of_layers == number_of_layers


class TestCrossAttentionDiTForward:
    @pytest.mark.parametrize(
        "batch_size, decoder_sequence_length, encoder_sequence_length, embedding_dimension",
        [
            (2, 4, 6, 32),
            (1, 8, 10, 32),
        ],
    )
    def test_hidden_state_and_condition_shape(
        self,
        cross_attention_dit_factory: Callable[..., CrossAttentionDiT],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        continuous_timestep_factory: Callable[..., torch.Tensor],
        batch_size: int,
        decoder_sequence_length: int,
        encoder_sequence_length: int,
        embedding_dimension: int,
    ):
        model = cross_attention_dit_factory(
            embedding_dimension=embedding_dimension,
        )
        decoder_hidden = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=decoder_sequence_length,
            embedding_dimension=embedding_dimension,
        )
        encoder_hidden = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=encoder_sequence_length,
            embedding_dimension=embedding_dimension,
        )
        timesteps = continuous_timestep_factory(batch_size=batch_size)
        hidden_states, conditioning = model(
            decoder_hidden_states=decoder_hidden,
            timesteps=timesteps,
            encoder_hidden_states=encoder_hidden,
        )
        assert hidden_states.shape == (
            batch_size,
            decoder_sequence_length,
            embedding_dimension,
        )
        assert conditioning.shape == (batch_size, embedding_dimension)

    def test_different_timesteps_produce_different_outputs(
        self,
        cross_attention_dit_factory: Callable[..., CrossAttentionDiT],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        model = cross_attention_dit_factory(
            embedding_dimension=embedding_dimension,
            use_gating=False,
        )
        reinit_modulation_layers(model)
        decoder_hidden = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=embedding_dimension,
        )
        encoder_hidden = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
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
        cross_attention_dit_factory: Callable[..., CrossAttentionDiT],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        continuous_timestep_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        model = cross_attention_dit_factory(
            embedding_dimension=embedding_dimension,
            use_gating=False,
        )
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

    def test_cached_forward_matches_uncached(
        self,
        cross_attention_dit_factory: Callable[..., CrossAttentionDiT],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        continuous_timestep_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        model = cross_attention_dit_factory(
            embedding_dimension=embedding_dimension,
            use_gating=False,
        )
        reinit_modulation_layers(model)
        model.eval()
        decoder_hidden = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=embedding_dimension,
        )
        encoder_hidden = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=embedding_dimension,
        )
        timesteps = continuous_timestep_factory(batch_size=2)
        output_uncached, condition_uncached = model(
            decoder_hidden_states=decoder_hidden,
            timesteps=timesteps,
            encoder_hidden_states=encoder_hidden,
        )
        conditioning_cache = model.precompute_conditioning_kv(
            encoder_hidden_states=encoder_hidden,
        )
        output_cached, condition_cached = model(
            decoder_hidden_states=decoder_hidden,
            timesteps=timesteps,
            conditioning_cache=conditioning_cache,
        )
        assert torch.allclose(output_uncached, output_cached, atol=1e-5)
        assert torch.allclose(condition_uncached, condition_cached, atol=1e-5)

    def test_cached_forward_with_padding_mask_matches_uncached(
        self,
        cross_attention_dit_factory: Callable[..., CrossAttentionDiT],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        continuous_timestep_factory: Callable[..., torch.Tensor],
        padding_mask_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        model = cross_attention_dit_factory(
            embedding_dimension=embedding_dimension,
            use_gating=False,
        )
        reinit_modulation_layers(model)
        model.eval()
        decoder_hidden = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=embedding_dimension,
        )
        encoder_hidden = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=embedding_dimension,
        )
        encoder_padding_mask = padding_mask_factory(
            batch_size=2,
            sequence_length=6,
            mask_last_n=2,
        )
        timesteps = continuous_timestep_factory(batch_size=2)
        output_uncached, condition_uncached = model(
            decoder_hidden_states=decoder_hidden,
            timesteps=timesteps,
            encoder_hidden_states=encoder_hidden,
            encoder_padding_mask=encoder_padding_mask,
        )
        conditioning_cache = model.precompute_conditioning_kv(
            encoder_hidden_states=encoder_hidden,
        )
        output_cached, condition_cached = model(
            decoder_hidden_states=decoder_hidden,
            timesteps=timesteps,
            conditioning_cache=conditioning_cache,
            encoder_padding_mask=encoder_padding_mask,
        )
        assert torch.allclose(output_uncached, output_cached, atol=1e-5)
        assert torch.allclose(condition_uncached, condition_cached, atol=1e-5)

    def test_precompute_returns_one_cache_per_layer(
        self,
        cross_attention_dit_factory: Callable[..., CrossAttentionDiT],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        number_of_layers = 3
        model = cross_attention_dit_factory(
            number_of_layers=number_of_layers,
            embedding_dimension=32,
        )
        encoder_hidden = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=32,
        )
        conditioning_cache = model.precompute_conditioning_kv(
            encoder_hidden_states=encoder_hidden,
        )
        assert len(conditioning_cache.layers) == number_of_layers

    def test_gradient_flows_through_full_model(
        self,
        cross_attention_dit_factory: Callable[..., CrossAttentionDiT],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        continuous_timestep_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        model = cross_attention_dit_factory(embedding_dimension=embedding_dimension)
        decoder_hidden = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=embedding_dimension,
        )
        encoder_hidden = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=embedding_dimension,
        )
        timesteps = continuous_timestep_factory(batch_size=2)
        decoder_hidden.requires_grad_(True)
        encoder_hidden.requires_grad_(True)
        output, conditioning = model(
            decoder_hidden_states=decoder_hidden,
            timesteps=timesteps,
            encoder_hidden_states=encoder_hidden,
        )
        (output.sum() + conditioning.sum()).backward()
        assert decoder_hidden.grad is not None
        assert encoder_hidden.grad is not None
        assert torch.all(torch.isfinite(decoder_hidden.grad))
        assert torch.all(torch.isfinite(encoder_hidden.grad))
