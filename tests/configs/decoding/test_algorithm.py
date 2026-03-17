"""Tests for versatil.configs.decoding.algorithm module."""

import pytest
from hydra.utils import instantiate
from omegaconf import MISSING

from versatil.configs.decoding.algorithm import (
    BehavioralCloningConfig,
    DecodingAlgorithmConfig,
    DiffusionConfig,
    FlowMatchingConfig,
    VariationalAlgorithmConfig,
)
from versatil.configs.decoding.latent import (
    GaussianPriorConfig,
    VAETransformerEncoderConfig,
)
from versatil.models.decoding.constants import (
    BetaSchedule,
    ODESolver,
    PredictionType,
    VarianceType,
)
from versatil.models.layers.denoising.diffusion_process import SchedulerType
from versatil.models.layers.denoising.timestep_sampling import TimestepSampler


@pytest.mark.unit
class TestDecodingAlgorithmConfig:
    def test_target_defaults_to_missing(self):
        config = DecodingAlgorithmConfig()
        assert config._target_ == MISSING


@pytest.mark.unit
class TestBehavioralCloningConfig:
    def test_target_points_to_behavioral_cloning(self):
        config = BehavioralCloningConfig()
        assert (
            config._target_
            == "versatil.models.decoding.algorithm.behavior_cloning.BehavioralCloning"
        )

    def test_inherits_from_decoding_algorithm_config(self):
        config = BehavioralCloningConfig()
        assert isinstance(config, DecodingAlgorithmConfig)


@pytest.mark.unit
class TestDiffusionConfig:
    def test_target_points_to_diffusion(self):
        config = DiffusionConfig()
        assert (
            config._target_ == "versatil.models.decoding.algorithm.diffusion.Diffusion"
        )

    def test_scheduler_type_default_is_ddim_string(self):
        config = DiffusionConfig()
        assert config.scheduler_type == SchedulerType.DDIM.value

    def test_beta_schedule_default_is_squaredcos_string(self):
        config = DiffusionConfig()
        assert config.beta_schedule == BetaSchedule.SQUAREDCOS_CAP_V2.value

    def test_prediction_type_default_is_epsilon_string(self):
        config = DiffusionConfig()
        assert config.prediction_type == PredictionType.EPSILON.value

    def test_scheduler_variance_type_default_is_fixed_small_string(self):
        config = DiffusionConfig()
        assert config.scheduler_variance_type == VarianceType.FIXED_SMALL.value

    @pytest.mark.parametrize("num_train_timesteps", [50, 200])
    @pytest.mark.parametrize("num_inference_steps", [5, 20])
    def test_stores_timestep_configuration(
        self, num_train_timesteps, num_inference_steps
    ):
        config = DiffusionConfig(
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=num_inference_steps,
        )
        assert config.num_train_timesteps == num_train_timesteps
        assert config.num_inference_steps == num_inference_steps

    @pytest.mark.parametrize("clip_sample", [True, False])
    @pytest.mark.parametrize("set_alpha_to_one", [True, False])
    def test_stores_scheduler_options(self, clip_sample, set_alpha_to_one):
        config = DiffusionConfig(
            clip_sample=clip_sample, set_alpha_to_one=set_alpha_to_one
        )
        assert config.clip_sample == clip_sample
        assert config.set_alpha_to_one == set_alpha_to_one

    def test_inherits_from_decoding_algorithm_config(self):
        config = DiffusionConfig()
        assert isinstance(config, DecodingAlgorithmConfig)


@pytest.mark.unit
class TestFlowMatchingConfig:
    def test_target_points_to_flow_matching(self):
        config = FlowMatchingConfig()
        assert (
            config._target_
            == "versatil.models.decoding.algorithm.flow_matching.FlowMatching"
        )

    def test_ode_solver_default_is_euler_string(self):
        config = FlowMatchingConfig()
        assert config.ode_solver == ODESolver.EULER.value

    def test_timestep_sampler_default_is_beta_string(self):
        config = FlowMatchingConfig()
        assert config.timestep_sampler == TimestepSampler.BETA.value

    @pytest.mark.parametrize("sigma", [0.0, 0.1])
    @pytest.mark.parametrize("num_inference_steps", [5, 20])
    @pytest.mark.parametrize("max_timestep", [0.999, 0.99])
    def test_stores_configuration(self, sigma, num_inference_steps, max_timestep):
        config = FlowMatchingConfig(
            sigma=sigma,
            num_inference_steps=num_inference_steps,
            max_timestep=max_timestep,
        )
        assert config.sigma == sigma
        assert config.num_inference_steps == num_inference_steps
        assert config.max_timestep == max_timestep

    def test_inherits_from_decoding_algorithm_config(self):
        config = FlowMatchingConfig()
        assert isinstance(config, DecodingAlgorithmConfig)


@pytest.mark.unit
class TestVariationalAlgorithmConfig:
    def test_target_points_to_variational_algorithm(self):
        config = VariationalAlgorithmConfig(
            base_algorithm=BehavioralCloningConfig(),
            posterior_encoder=VAETransformerEncoderConfig(
                latent_dimension=32, embedding_dimension=256
            ),
            prior=GaussianPriorConfig(latent_dimension=32),
        )
        assert (
            config._target_
            == "versatil.models.decoding.algorithm.variational.VariationalAlgorithm"
        )

    def test_base_algorithm_required(self):
        config = VariationalAlgorithmConfig()
        assert config.base_algorithm == MISSING

    def test_posterior_encoder_required(self):
        config = VariationalAlgorithmConfig()
        assert config.posterior_encoder == MISSING

    def test_prior_required(self):
        config = VariationalAlgorithmConfig()
        assert config.prior == MISSING

    @pytest.mark.parametrize("sampling_probability", [0.0, 0.5, 1.0])
    def test_stores_sampling_from_prior_probability(self, sampling_probability):
        config = VariationalAlgorithmConfig(
            base_algorithm=BehavioralCloningConfig(),
            posterior_encoder=VAETransformerEncoderConfig(
                latent_dimension=32, embedding_dimension=256
            ),
            prior=GaussianPriorConfig(latent_dimension=32),
            sampling_from_prior_probability=sampling_probability,
        )
        assert config.sampling_from_prior_probability == sampling_probability

    def test_inherits_from_decoding_algorithm_config(self):
        config = VariationalAlgorithmConfig()
        assert isinstance(config, DecodingAlgorithmConfig)


@pytest.mark.unit
class TestAlgorithmInstantiation:
    def test_behavioral_cloning_instantiates(self):
        config = BehavioralCloningConfig()
        instance = instantiate(config)
        assert type(instance).__name__ == "BehavioralCloning"

    def test_diffusion_instantiates_with_parameter_passthrough(self):
        config = DiffusionConfig(
            num_train_timesteps=50,
            num_inference_steps=5,
        )
        instance = instantiate(config)
        assert instance.num_train_timesteps == 50
        assert instance.num_inference_steps == 5

    def test_flow_matching_instantiates_with_parameter_passthrough(self):
        config = FlowMatchingConfig(
            num_inference_steps=20,
        )
        instance = instantiate(config)
        assert instance.num_inference_steps == 20
