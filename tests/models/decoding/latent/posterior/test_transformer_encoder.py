"""Tests for versatil.models.decoding.latent.posterior.transformer_encoder module."""
import logging
from collections.abc import Callable
from unittest.mock import patch

import pytest
import torch

from versatil.data.constants import SampleKey
from versatil.models.decoding.constants import LatentKey
from versatil.models.decoding.latent.posterior.base_posterior import (
    PosteriorLatentEncoder,
)
from versatil.models.decoding.latent.posterior.transformer_encoder import (
    VAETransformerEncoder,
)


@pytest.fixture
def vae_encoder_factory() -> Callable[..., VAETransformerEncoder]:
    """Factory for VAETransformerEncoder instances."""
    def factory(
        embedding_dimension: int = 64,
        latent_dimension: int = 16,
        prediction_horizon: int = 8,
        observation_horizon: int = 1,
        device: str = "cpu",
        number_of_heads: int = 4,
        feedforward_dimension: int = 128,
        number_of_encoder_layers: int = 2,
        deterministic: bool = False,
        min_logvar: float | None = None,
        exclude_keys: list[str] | None = None,
    ) -> VAETransformerEncoder:
        return VAETransformerEncoder(
            embedding_dimension=embedding_dimension,
            latent_dimension=latent_dimension,
            prediction_horizon=prediction_horizon,
            observation_horizon=observation_horizon,
            device=device,
            number_of_heads=number_of_heads,
            feedforward_dimension=feedforward_dimension,
            number_of_encoder_layers=number_of_encoder_layers,
            deterministic=deterministic,
            min_logvar=min_logvar,
            exclude_keys=exclude_keys,
        )
    return factory


class TestVAETransformerEncoderInitialization:

    def test_inherits_from_posterior_latent_encoder(
        self,
        vae_encoder_factory: Callable[..., VAETransformerEncoder],
    ):
        encoder = vae_encoder_factory()
        assert isinstance(encoder, PosteriorLatentEncoder)

    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("latent_dimension", [8, 16])
    @pytest.mark.parametrize("deterministic", [True, False])
    def test_stores_configuration(
        self,
        vae_encoder_factory: Callable[..., VAETransformerEncoder],
        embedding_dimension: int,
        latent_dimension: int,
        deterministic: bool,
    ):
        encoder = vae_encoder_factory(
            embedding_dimension=embedding_dimension,
            latent_dimension=latent_dimension,
            deterministic=deterministic,
        )
        assert encoder.embedding_dimension == embedding_dimension
        assert encoder.latent_dimension == latent_dimension
        assert encoder.deterministic is deterministic

    @pytest.mark.parametrize("deterministic, expected_multiplier", [
        (True, 1),
        (False, 2),
    ])
    def test_projection_dim_depends_on_deterministic(
        self,
        vae_encoder_factory: Callable[..., VAETransformerEncoder],
        deterministic: bool,
        expected_multiplier: int,
    ):
        latent_dimension = 16
        encoder = vae_encoder_factory(
            latent_dimension=latent_dimension,
            deterministic=deterministic,
        )
        assert encoder.latent_projection.out_features == latent_dimension * expected_multiplier


class TestVAETransformerEncoderEncode:

    def test_deterministic_returns_exact_keys(
        self,
        vae_encoder_factory: Callable[..., VAETransformerEncoder],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        encoder = vae_encoder_factory(deterministic=True)
        actions = action_dictionary_factory(prediction_horizon=8)
        features = feature_dictionary_factory()
        result = encoder.encode(actions=actions, observations=features)
        assert isinstance(result, dict)
        assert set(result.keys()) == {
            LatentKey.POSTERIOR_LATENT.value,
            LatentKey.POSTERIOR_MU.value,
        }
        assert isinstance(result[LatentKey.POSTERIOR_LATENT.value], torch.Tensor)
        assert isinstance(result[LatentKey.POSTERIOR_MU.value], torch.Tensor)
        assert torch.equal(
            result[LatentKey.POSTERIOR_LATENT.value],
            result[LatentKey.POSTERIOR_MU.value],
        )

    def test_stochastic_returns_exact_keys(
        self,
        vae_encoder_factory: Callable[..., VAETransformerEncoder],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        encoder = vae_encoder_factory(deterministic=False)
        actions = action_dictionary_factory(prediction_horizon=8)
        features = feature_dictionary_factory()
        result = encoder.encode(actions=actions, observations=features)
        assert isinstance(result, dict)
        assert set(result.keys()) == {
            LatentKey.POSTERIOR_LATENT.value,
            LatentKey.POSTERIOR_MU.value,
            LatentKey.POSTERIOR_LOGVAR.value,
        }
        assert isinstance(result[LatentKey.POSTERIOR_LATENT.value], torch.Tensor)
        assert isinstance(result[LatentKey.POSTERIOR_MU.value], torch.Tensor)
        assert isinstance(result[LatentKey.POSTERIOR_LOGVAR.value], torch.Tensor)
        assert not torch.equal(
            result[LatentKey.POSTERIOR_LATENT.value],
            result[LatentKey.POSTERIOR_MU.value],
        )

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("latent_dimension", [8, 32])
    def test_output_shapes(
        self,
        vae_encoder_factory: Callable[..., VAETransformerEncoder],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        batch_size: int,
        latent_dimension: int,
    ):
        embedding_dimension = 64
        encoder = vae_encoder_factory(
            embedding_dimension=embedding_dimension,
            latent_dimension=latent_dimension,
            deterministic=False,
        )
        actions = action_dictionary_factory(
            batch_size=batch_size,
            prediction_horizon=8,
        )
        features = feature_dictionary_factory(
            batch_size=batch_size,
            feature_dimension=embedding_dimension,
        )
        result = encoder.encode(actions=actions, observations=features)
        assert result[LatentKey.POSTERIOR_LATENT.value].shape == (batch_size, latent_dimension)
        assert result[LatentKey.POSTERIOR_MU.value].shape == (batch_size, latent_dimension)
        assert result[LatentKey.POSTERIOR_LOGVAR.value].shape == (batch_size, latent_dimension)

    def test_min_logvar_clamps_logvar(
        self,
        vae_encoder_factory: Callable[..., VAETransformerEncoder],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        min_logvar = -1.0
        encoder = vae_encoder_factory(
            deterministic=False,
            min_logvar=min_logvar,
        )
        actions = action_dictionary_factory(prediction_horizon=8)
        features = feature_dictionary_factory()
        result = encoder.encode(actions=actions, observations=features)
        logvar = result[LatentKey.POSTERIOR_LOGVAR.value]
        assert torch.all(logvar >= min_logvar)

    def test_excludes_keys_from_observations(
        self,
        vae_encoder_factory: Callable[..., VAETransformerEncoder],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        excluded_key = "heavy_feature"
        encoder = vae_encoder_factory(exclude_keys=[excluded_key])
        actions = action_dictionary_factory(prediction_horizon=8)
        features = feature_dictionary_factory(
            feature_keys=["rgb_features", excluded_key],
        )
        builder = encoder.input_sequence_builder
        original_forward = builder.forward
        captured_observations = {}

        def capturing_forward(observations):
            captured_observations.update(observations)
            return original_forward(observations)

        builder.forward = capturing_forward
        encoder.encode(actions=actions, observations=features)
        assert excluded_key not in captured_observations
        assert "rgb_features" in captured_observations

    def test_encode_without_observations(
        self,
        vae_encoder_factory: Callable[..., VAETransformerEncoder],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        encoder = vae_encoder_factory(deterministic=False)
        actions = action_dictionary_factory(prediction_horizon=8)
        result = encoder.encode(actions=actions, observations=None)
        assert isinstance(result, dict)
        assert set(result.keys()) == {
            LatentKey.POSTERIOR_LATENT.value,
            LatentKey.POSTERIOR_MU.value,
            LatentKey.POSTERIOR_LOGVAR.value,
        }

    def test_encode_without_padding_mask_injects_zero_padding(
        self,
        vae_encoder_factory: Callable[..., VAETransformerEncoder],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        caplog: pytest.LogCaptureFixture,
    ):
        prediction_horizon = 8
        encoder = vae_encoder_factory(
            deterministic=False,
            prediction_horizon=prediction_horizon,
        )
        actions = action_dictionary_factory(
            prediction_horizon=prediction_horizon,
            include_padding_mask=False,
        )
        features = feature_dictionary_factory()
        builder = encoder.input_sequence_builder
        original_forward = builder.forward
        captured_observations = {}

        def capturing_forward(observations):
            captured_observations.update(observations)
            return original_forward(observations)

        builder.forward = capturing_forward
        with caplog.at_level(logging.WARNING):
            encoder.encode(actions=actions, observations=features)
        pad_key = SampleKey.IS_PAD_ACTION.value
        assert pad_key in captured_observations
        assert captured_observations[pad_key].shape == (2, prediction_horizon)
        assert not captured_observations[pad_key].any()
        assert "No padding key found in actions" in caplog.text


class TestVAETransformerEncoderForward:

    def test_forward_delegates_to_encode(
        self,
        vae_encoder_factory: Callable[..., VAETransformerEncoder],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        encoder = vae_encoder_factory(deterministic=False)
        actions = action_dictionary_factory(prediction_horizon=8)
        features = feature_dictionary_factory()
        with patch.object(
            encoder,
            "encode",
            wraps=encoder.encode,
        ) as mock_encode:
            result = encoder.forward(actions=actions, observations=features)
            mock_encode.assert_called_once_with(actions, features)
        assert isinstance(result, dict)
        assert set(result.keys()) == {
            LatentKey.POSTERIOR_LATENT.value,
            LatentKey.POSTERIOR_MU.value,
            LatentKey.POSTERIOR_LOGVAR.value,
        }