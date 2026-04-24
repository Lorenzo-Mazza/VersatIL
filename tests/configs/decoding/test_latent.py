"""Tests for versatil.configs.decoding.latent module."""

import pytest
from hydra.utils import instantiate
from omegaconf import MISSING

from versatil.configs.decoding.latent import (
    DiTPriorConfig,
    GaussianPriorConfig,
    PosteriorLatentEncoderConfig,
    PriorLatentEncoderConfig,
    PriorTransformerEncoderConfig,
    VAETransformerEncoderConfig,
    VampPriorConfig,
)
from versatil.models.decoding.constants import (
    BetaSchedule,
    DenoisingAlgorithm,
    LatentKey,
    ODESolver,
    PredictionType,
)
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType
from versatil.models.layers.denoising.diffusion_process import SchedulerType
from versatil.models.layers.denoising.timestep_sampling import TimestepSampler
from versatil.models.layers.normalization.constants import NormalizationType


@pytest.mark.unit
class TestPosteriorLatentEncoderConfig:
    def test_target_defaults_to_missing(self):
        config = PosteriorLatentEncoderConfig()
        assert config._target_ == MISSING

    def test_latent_dimension_required(self):
        config = PosteriorLatentEncoderConfig()
        assert config.latent_dimension == MISSING


@pytest.mark.unit
class TestPriorLatentEncoderConfig:
    def test_target_defaults_to_missing(self):
        config = PriorLatentEncoderConfig()
        assert config._target_ == MISSING

    def test_latent_dimension_required(self):
        config = PriorLatentEncoderConfig()
        assert config.latent_dimension == MISSING


@pytest.mark.unit
class TestVAETransformerEncoderConfig:
    def test_target_points_to_vae_transformer_encoder(self):
        config = VAETransformerEncoderConfig(
            latent_dimension=32, embedding_dimension=256
        )
        assert (
            config._target_
            == "versatil.models.decoding.latent.posterior.transformer_encoder.VAETransformerEncoder"
        )

    @pytest.mark.parametrize("latent_dimension", [16, 64])
    @pytest.mark.parametrize("embedding_dimension", [128, 512])
    def test_stores_dimensions(self, latent_dimension, embedding_dimension):
        config = VAETransformerEncoderConfig(
            latent_dimension=latent_dimension,
            embedding_dimension=embedding_dimension,
        )
        assert config.latent_dimension == latent_dimension
        assert config.embedding_dimension == embedding_dimension

    def test_activation_default_is_swiglu_string(self):
        config = VAETransformerEncoderConfig(
            latent_dimension=32, embedding_dimension=256
        )
        assert config.activation == ActivationFunction.SWIGLU.value

    @pytest.mark.parametrize("deterministic", [True, False])
    def test_stores_deterministic_flag(self, deterministic):
        config = VAETransformerEncoderConfig(
            latent_dimension=32,
            embedding_dimension=256,
            deterministic=deterministic,
        )
        assert config.deterministic == deterministic

    def test_interpolation_references(self):
        config = VAETransformerEncoderConfig(
            latent_dimension=32, embedding_dimension=256
        )
        assert config.prediction_horizon == "${policy.prediction_horizon}"
        assert config.observation_horizon == "${policy.observation_horizon}"
        assert config.device == "${policy.device}"

    def test_inherits_from_posterior_config(self):
        config = VAETransformerEncoderConfig(
            latent_dimension=32, embedding_dimension=256
        )
        assert isinstance(config, PosteriorLatentEncoderConfig)

    def test_attention_dropout_default(self):
        config = VAETransformerEncoderConfig(
            latent_dimension=32, embedding_dimension=256
        )
        assert config.attention_dropout == 0.0

    def test_normalization_type_default_is_rms_norm(self):
        config = VAETransformerEncoderConfig(
            latent_dimension=32, embedding_dimension=256
        )
        assert config.normalization_type == NormalizationType.RMS_NORM.value

    def test_attention_type_default_is_multi_head(self):
        config = VAETransformerEncoderConfig(
            latent_dimension=32, embedding_dimension=256
        )
        assert config.attention_type == AttentionType.MULTI_HEAD.value

    def test_positional_encoding_type_default_is_none(self):
        config = VAETransformerEncoderConfig(
            latent_dimension=32, embedding_dimension=256
        )
        assert config.positional_encoding_type is None

    @pytest.mark.parametrize("positional_encoding_type", [None, "rope", "sinusoidal"])
    def test_stores_positional_encoding_type(self, positional_encoding_type):
        config = VAETransformerEncoderConfig(
            latent_dimension=32,
            embedding_dimension=256,
            positional_encoding_type=positional_encoding_type,
        )
        assert config.positional_encoding_type == positional_encoding_type


@pytest.mark.unit
class TestGaussianPriorConfig:
    def test_target_points_to_gaussian_prior(self):
        config = GaussianPriorConfig(latent_dimension=32)
        assert (
            config._target_
            == "versatil.models.decoding.latent.prior.gaussian_prior.GaussianPrior"
        )

    @pytest.mark.parametrize("latent_dimension", [16, 64])
    def test_stores_latent_dimension(self, latent_dimension):
        config = GaussianPriorConfig(latent_dimension=latent_dimension)
        assert config.latent_dimension == latent_dimension

    def test_inherits_from_prior_config(self):
        config = GaussianPriorConfig(latent_dimension=32)
        assert isinstance(config, PriorLatentEncoderConfig)


@pytest.mark.unit
class TestPriorTransformerEncoderConfig:
    def test_target_points_to_prior_transformer_encoder(self):
        config = PriorTransformerEncoderConfig(
            latent_dimension=32, embedding_dimension=256
        )
        assert (
            config._target_
            == "versatil.models.decoding.latent.prior.transformer_encoder.PriorTransformerEncoder"
        )

    @pytest.mark.parametrize("learn_variance", [True, False])
    def test_stores_learn_variance(self, learn_variance):
        config = PriorTransformerEncoderConfig(
            latent_dimension=32,
            embedding_dimension=256,
            learn_variance=learn_variance,
        )
        assert config.learn_variance == learn_variance

    def test_inherits_from_prior_config(self):
        config = PriorTransformerEncoderConfig(
            latent_dimension=32, embedding_dimension=256
        )
        assert isinstance(config, PriorLatentEncoderConfig)

    def test_attention_dropout_default(self):
        config = PriorTransformerEncoderConfig(
            latent_dimension=32, embedding_dimension=256
        )
        assert config.attention_dropout == 0.0

    def test_normalization_type_default_is_rms_norm(self):
        config = PriorTransformerEncoderConfig(
            latent_dimension=32, embedding_dimension=256
        )
        assert config.normalization_type == NormalizationType.RMS_NORM.value

    def test_attention_type_default_is_multi_head(self):
        config = PriorTransformerEncoderConfig(
            latent_dimension=32, embedding_dimension=256
        )
        assert config.attention_type == AttentionType.MULTI_HEAD.value

    def test_positional_encoding_type_default_is_none(self):
        config = PriorTransformerEncoderConfig(
            latent_dimension=32, embedding_dimension=256
        )
        assert config.positional_encoding_type is None

    @pytest.mark.parametrize("positional_encoding_type", [None, "rope", "sinusoidal"])
    def test_stores_positional_encoding_type(self, positional_encoding_type):
        config = PriorTransformerEncoderConfig(
            latent_dimension=32,
            embedding_dimension=256,
            positional_encoding_type=positional_encoding_type,
        )
        assert config.positional_encoding_type == positional_encoding_type


@pytest.mark.unit
class TestVampPriorConfig:
    def test_target_points_to_vamp_prior(self):
        config = VampPriorConfig(latent_dimension=32)
        assert (
            config._target_
            == "versatil.models.decoding.latent.prior.vamp_prior.VampPrior"
        )

    @pytest.mark.parametrize("num_components", [20, 100])
    def test_stores_num_components(self, num_components):
        config = VampPriorConfig(latent_dimension=32, num_components=num_components)
        assert config.num_components == num_components

    def test_inherits_from_prior_config(self):
        config = VampPriorConfig(latent_dimension=32)
        assert isinstance(config, PriorLatentEncoderConfig)


@pytest.mark.unit
class TestDiTPriorConfig:
    def test_target_points_to_dit_prior(self):
        config = DiTPriorConfig(latent_dimension=32)
        assert (
            config._target_
            == "versatil.models.decoding.latent.prior.dit_prior.DiTPrior"
        )

    def test_algorithm_type_default_is_flow_matching_string(self):
        config = DiTPriorConfig(latent_dimension=32)
        assert config.algorithm_type == DenoisingAlgorithm.FLOW_MATCHING.value

    def test_ode_solver_default_is_euler_string(self):
        config = DiTPriorConfig(latent_dimension=32)
        assert config.ode_solver == ODESolver.EULER.value

    def test_timestep_sampler_default_is_beta_string(self):
        config = DiTPriorConfig(latent_dimension=32)
        assert config.timestep_sampler == TimestepSampler.BETA.value

    def test_stores_timestep_sampling_fields(self):
        config = DiTPriorConfig(
            latent_dimension=32,
            timestep_sampler=TimestepSampler.LOGIT_NORMAL.value,
            logit_mean=0.25,
            logit_std=0.5,
            beta_alpha=1.25,
            beta_beta=0.75,
            max_timestep=0.9,
        )
        assert config.timestep_sampler == TimestepSampler.LOGIT_NORMAL.value
        assert config.logit_mean == 0.25
        assert config.logit_std == 0.5
        assert config.beta_alpha == 1.25
        assert config.beta_beta == 0.75
        assert config.max_timestep == 0.9

    def test_prior_target_default_is_posterior_mu(self):
        config = DiTPriorConfig(latent_dimension=32)
        assert config.prior_target_key == LatentKey.POSTERIOR_MU.value

    def test_stores_latent_standardization_fields(self):
        config = DiTPriorConfig(
            latent_dimension=32,
            latent_standardization_enabled=False,
            latent_standardization_eps=1e-5,
            latent_standardization_max_batches=7,
            require_fitted_latent_standardization=True,
        )
        assert config.latent_standardization_enabled is False
        assert config.latent_standardization_eps == pytest.approx(1e-5)
        assert config.latent_standardization_max_batches == 7
        assert config.require_fitted_latent_standardization is True

    def test_beta_schedule_default_is_squaredcos_string(self):
        config = DiTPriorConfig(latent_dimension=32)
        assert config.beta_schedule == BetaSchedule.SQUAREDCOS_CAP_V2.value

    def test_scheduler_type_default_is_ddim_string(self):
        config = DiTPriorConfig(latent_dimension=32)
        assert config.scheduler_type == SchedulerType.DDIM.value

    def test_prediction_type_default_is_epsilon_string(self):
        config = DiTPriorConfig(latent_dimension=32)
        assert config.prediction_type == PredictionType.EPSILON.value

    def test_activation_default_is_silu_string(self):
        config = DiTPriorConfig(latent_dimension=32)
        assert config.activation == ActivationFunction.SILU.value

    @pytest.mark.parametrize("use_gating", [True, False])
    def test_stores_gating_option(self, use_gating):
        config = DiTPriorConfig(latent_dimension=32, use_gating=use_gating)
        assert config.use_gating == use_gating

    def test_inherits_from_prior_config(self):
        config = DiTPriorConfig(latent_dimension=32)
        assert isinstance(config, PriorLatentEncoderConfig)


@pytest.mark.unit
class TestLatentInstantiation:
    def test_gaussian_prior_instantiates(self):
        config = GaussianPriorConfig(latent_dimension=32, device="cpu")
        instance = instantiate(config)
        assert instance.latent_dimension == 32

    def test_vae_transformer_encoder_instantiates(self):
        config = VAETransformerEncoderConfig(
            latent_dimension=32,
            embedding_dimension=128,
            prediction_horizon=16,
            observation_horizon=1,
            device="cpu",
            number_of_heads=4,
            feedforward_dimension=256,
            number_of_encoder_layers=2,
            dropout_rate=0.1,
        )
        instance = instantiate(config)
        assert instance.latent_dimension == 32
        assert instance.embedding_dimension == 128
