"""Tests for versatil.models.decoding.latent.prior.transformer_encoder module."""

from collections.abc import Callable

import pytest
import torch

from versatil.models.decoding.constants import LatentKey
from versatil.models.decoding.latent.prior.base_prior import PriorLatentEncoder
from versatil.models.decoding.latent.prior.transformer_encoder import (
    PriorTransformerEncoder,
)


@pytest.fixture
def prior_transformer_factory() -> Callable[..., PriorTransformerEncoder]:
    """Factory for PriorTransformerEncoder instances."""

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
        learn_variance: bool = True,
        min_logvar: float | None = None,
        exclude_keys: list[str] | None = None,
    ) -> PriorTransformerEncoder:
        return PriorTransformerEncoder(
            embedding_dimension=embedding_dimension,
            latent_dimension=latent_dimension,
            prediction_horizon=prediction_horizon,
            observation_horizon=observation_horizon,
            device=device,
            number_of_heads=number_of_heads,
            feedforward_dimension=feedforward_dimension,
            number_of_encoder_layers=number_of_encoder_layers,
            deterministic=deterministic,
            learn_variance=learn_variance,
            min_logvar=min_logvar,
            exclude_keys=exclude_keys,
        )

    return factory


class TestPriorTransformerEncoderInitialization:
    def test_inherits_from_prior_latent_encoder(
        self,
        prior_transformer_factory: Callable[..., PriorTransformerEncoder],
    ):
        encoder = prior_transformer_factory(
            embedding_dimension=64,
            latent_dimension=16,
        )
        assert isinstance(encoder, PriorLatentEncoder)

    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("latent_dimension", [8, 16])
    @pytest.mark.parametrize("deterministic", [True, False])
    @pytest.mark.parametrize("learn_variance", [True, False])
    def test_stores_configuration(
        self,
        prior_transformer_factory: Callable[..., PriorTransformerEncoder],
        embedding_dimension: int,
        latent_dimension: int,
        deterministic: bool,
        learn_variance: bool,
    ):
        encoder = prior_transformer_factory(
            embedding_dimension=embedding_dimension,
            latent_dimension=latent_dimension,
            deterministic=deterministic,
            learn_variance=learn_variance,
        )
        assert encoder.embedding_dimension == embedding_dimension
        assert encoder.latent_dimension == latent_dimension
        assert encoder.deterministic is deterministic
        assert encoder.learn_variance is learn_variance

    @pytest.mark.parametrize(
        "deterministic, learn_variance, expected_multiplier",
        [
            (True, True, 1),
            (True, False, 1),
            (False, True, 2),
            (False, False, 1),
        ],
    )
    def test_projection_dim_depends_on_deterministic_and_learn_variance(
        self,
        prior_transformer_factory: Callable[..., PriorTransformerEncoder],
        deterministic: bool,
        learn_variance: bool,
        expected_multiplier: int,
    ):
        latent_dimension = 16
        encoder = prior_transformer_factory(
            latent_dimension=latent_dimension,
            deterministic=deterministic,
            learn_variance=learn_variance,
        )
        assert (
            encoder.latent_projection.out_features
            == latent_dimension * expected_multiplier
        )


class TestPriorTransformerEncoderGetAuxiliaryOutputKeys:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "deterministic, expect_logvar",
        [(False, True), (True, False)],
        ids=["stochastic_includes_logvar", "deterministic_excludes_logvar"],
    )
    def test_logvar_presence_depends_on_deterministic(
        self,
        prior_transformer_factory: Callable[..., PriorTransformerEncoder],
        deterministic: bool,
        expect_logvar: bool,
    ) -> None:
        prior = prior_transformer_factory(
            deterministic=deterministic, latent_dimension=8
        )
        keys = prior.get_auxiliary_output_keys()
        assert LatentKey.PRIOR_LATENT.value in keys
        assert LatentKey.PRIOR_MU.value in keys
        assert (LatentKey.PRIOR_LOGVAR.value in keys) == expect_logvar


class TestPriorTransformerEncoderForward:
    def test_deterministic_returns_exact_keys(
        self,
        prior_transformer_factory: Callable[..., PriorTransformerEncoder],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        encoder = prior_transformer_factory(deterministic=True)
        features = feature_dictionary_factory()
        result = encoder.forward(
            target_latents=torch.zeros(2, encoder.latent_dimension),
            observations=features,
        )
        assert isinstance(result, dict)
        assert set(result.keys()) == {
            LatentKey.PRIOR_LATENT.value,
            LatentKey.PRIOR_MU.value,
        }
        assert isinstance(result[LatentKey.PRIOR_LATENT.value], torch.Tensor)
        assert isinstance(result[LatentKey.PRIOR_MU.value], torch.Tensor)

    def test_learn_variance_returns_exact_keys(
        self,
        prior_transformer_factory: Callable[..., PriorTransformerEncoder],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        encoder = prior_transformer_factory(
            deterministic=False,
            learn_variance=True,
        )
        features = feature_dictionary_factory()
        result = encoder.forward(
            target_latents=torch.zeros(2, encoder.latent_dimension),
            observations=features,
        )
        assert isinstance(result, dict)
        assert set(result.keys()) == {
            LatentKey.PRIOR_LATENT.value,
            LatentKey.PRIOR_MU.value,
            LatentKey.PRIOR_LOGVAR.value,
        }
        assert isinstance(result[LatentKey.PRIOR_LATENT.value], torch.Tensor)
        assert isinstance(result[LatentKey.PRIOR_MU.value], torch.Tensor)
        assert isinstance(result[LatentKey.PRIOR_LOGVAR.value], torch.Tensor)

    def test_fixed_variance_returns_zero_logvar(
        self,
        prior_transformer_factory: Callable[..., PriorTransformerEncoder],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        encoder = prior_transformer_factory(
            deterministic=False,
            learn_variance=False,
        )
        features = feature_dictionary_factory()
        result = encoder.forward(
            target_latents=torch.zeros(2, encoder.latent_dimension),
            observations=features,
        )
        assert isinstance(result, dict)
        assert set(result.keys()) == {
            LatentKey.PRIOR_LATENT.value,
            LatentKey.PRIOR_MU.value,
            LatentKey.PRIOR_LOGVAR.value,
        }
        assert isinstance(result[LatentKey.PRIOR_LOGVAR.value], torch.Tensor)
        assert torch.all(result[LatentKey.PRIOR_LOGVAR.value] == 0.0)

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("latent_dimension", [8, 32])
    def test_output_shapes(
        self,
        prior_transformer_factory: Callable[..., PriorTransformerEncoder],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        batch_size: int,
        latent_dimension: int,
    ):
        embedding_dimension = 64
        encoder = prior_transformer_factory(
            embedding_dimension=embedding_dimension,
            latent_dimension=latent_dimension,
            deterministic=False,
            learn_variance=True,
        )
        features = feature_dictionary_factory(
            batch_size=batch_size,
            feature_dimension=embedding_dimension,
        )
        result = encoder.forward(
            target_latents=torch.zeros(2, encoder.latent_dimension),
            observations=features,
        )
        assert result[LatentKey.PRIOR_LATENT.value].shape == (
            batch_size,
            latent_dimension,
        )
        assert result[LatentKey.PRIOR_MU.value].shape == (batch_size, latent_dimension)
        assert result[LatentKey.PRIOR_LOGVAR.value].shape == (
            batch_size,
            latent_dimension,
        )

    def test_min_logvar_clamps_logvar(
        self,
        prior_transformer_factory: Callable[..., PriorTransformerEncoder],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        min_logvar = -1.0
        encoder = prior_transformer_factory(
            deterministic=False,
            learn_variance=True,
            min_logvar=min_logvar,
        )
        features = feature_dictionary_factory()
        result = encoder.forward(
            target_latents=torch.zeros(2, encoder.latent_dimension),
            observations=features,
        )
        logvar = result[LatentKey.PRIOR_LOGVAR.value]
        assert torch.all(logvar >= min_logvar)

    def test_excludes_keys_from_observations(
        self,
        prior_transformer_factory: Callable[..., PriorTransformerEncoder],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        excluded_key = "palantir_vision"
        encoder = prior_transformer_factory(
            exclude_keys=[excluded_key],
        )
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
        encoder.forward(
            target_latents=torch.zeros(2, encoder.latent_dimension),
            observations=features,
        )
        assert excluded_key not in captured_observations
        assert "rgb_features" in captured_observations


class TestPriorTransformerEncoderSamplePrior:
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("latent_dimension", [8, 32])
    def test_output_shape(
        self,
        prior_transformer_factory: Callable[..., PriorTransformerEncoder],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        batch_size: int,
        latent_dimension: int,
    ):
        embedding_dimension = 64
        encoder = prior_transformer_factory(
            embedding_dimension=embedding_dimension,
            latent_dimension=latent_dimension,
        )
        features = feature_dictionary_factory(
            batch_size=batch_size,
            feature_dimension=embedding_dimension,
        )
        result = encoder.sample_prior(
            batch_size=batch_size,
            observations=features,
        )
        assert isinstance(result, torch.Tensor)
        assert result.shape == (batch_size, latent_dimension)
