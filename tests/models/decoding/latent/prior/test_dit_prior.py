"""Tests for versatil.models.decoding.latent.prior.dit_prior module."""

import re
from collections.abc import Callable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from versatil.models.decoding.constants import (
    DenoisingAlgorithm,
    LatentKey,
    PredictionType,
)
from versatil.models.decoding.latent.prior.base_prior import PriorLatentEncoder
from versatil.models.decoding.latent.prior.dit_prior import DiTPrior
from versatil.models.layers.denoising.timestep_sampling import TimestepSampler

LATENT_DIMENSION = 8
EMBEDDING_DIMENSION = 16
NUMBER_OF_HEADS = 2
NUMBER_OF_LAYERS = 1
FEEDFORWARD_DIMENSION = 32
NUM_TRAIN_TIMESTEPS = 20
NUM_INFERENCE_STEPS = 4


@pytest.fixture
def dit_prior_factory() -> Callable[..., DiTPrior]:
    """Factory for DiTPrior instances."""

    def factory(
        latent_dimension: int = LATENT_DIMENSION,
        embedding_dimension: int = EMBEDDING_DIMENSION,
        number_of_heads: int = NUMBER_OF_HEADS,
        number_of_layers: int = NUMBER_OF_LAYERS,
        feedforward_dimension: int = FEEDFORWARD_DIMENSION,
        device: str = "cpu",
        observation_horizon: int = 1,
        algorithm_type: str = DenoisingAlgorithm.FLOW_MATCHING.value,
        timestep_sampler: str = TimestepSampler.BETA.value,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        beta_alpha: float = 1.5,
        beta_beta: float = 1.0,
        max_timestep: float = 0.999,
        num_train_timesteps: int = NUM_TRAIN_TIMESTEPS,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
        exclude_keys: list[str] | None = None,
        prediction_type: str = PredictionType.EPSILON.value,
        prior_target_key: str = LatentKey.POSTERIOR_MU.value,
        latent_standardization_enabled: bool = True,
        latent_standardization_eps: float = 1e-6,
        latent_standardization_max_batches: int | None = None,
        require_fitted_latent_standardization: bool = False,
    ) -> DiTPrior:
        return DiTPrior(
            latent_dimension=latent_dimension,
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_layers=number_of_layers,
            feedforward_dimension=feedforward_dimension,
            device=device,
            observation_horizon=observation_horizon,
            algorithm_type=algorithm_type,
            timestep_sampler=timestep_sampler,
            logit_mean=logit_mean,
            logit_std=logit_std,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
            max_timestep=max_timestep,
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=num_inference_steps,
            exclude_keys=exclude_keys,
            prediction_type=prediction_type,
            prior_target_key=prior_target_key,
            latent_standardization_enabled=latent_standardization_enabled,
            latent_standardization_eps=latent_standardization_eps,
            latent_standardization_max_batches=latent_standardization_max_batches,
            require_fitted_latent_standardization=require_fitted_latent_standardization,
        )

    return factory


@pytest.fixture
def latent_tensor_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for latent tensors (B, latent_dim)."""

    def factory(
        batch_size: int = 2,
        latent_dimension: int = LATENT_DIMENSION,
    ) -> torch.Tensor:
        return torch.from_numpy(
            rng.standard_normal((batch_size, latent_dimension)).astype(np.float32)
        )

    return factory


class TestDiTPriorInitialization:
    @pytest.mark.unit
    def test_inherits_from_prior_latent_encoder(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
    ):
        prior = dit_prior_factory()
        assert isinstance(prior, PriorLatentEncoder)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "algorithm_type",
        [
            DenoisingAlgorithm.FLOW_MATCHING.value,
            DenoisingAlgorithm.DIFFUSION.value,
        ],
    )
    @pytest.mark.parametrize("latent_dimension", [8, 16])
    @pytest.mark.parametrize("num_train_timesteps", [20, 50])
    def test_stores_configuration(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
        algorithm_type: str,
        latent_dimension: int,
        num_train_timesteps: int,
    ):
        prior = dit_prior_factory(
            algorithm_type=algorithm_type,
            latent_dimension=latent_dimension,
            num_train_timesteps=num_train_timesteps,
        )
        assert prior.algorithm_type == algorithm_type
        assert prior.latent_dimension == latent_dimension
        assert prior.num_train_timesteps == num_train_timesteps
        assert prior.embedding_dimension == EMBEDDING_DIMENSION

    @pytest.mark.unit
    def test_flow_matching_has_flow_matcher_and_no_scheduler(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
    ):
        prior = dit_prior_factory(
            algorithm_type=DenoisingAlgorithm.FLOW_MATCHING.value,
        )
        assert prior.flow_matcher is not None
        assert prior.noise_scheduler is None

    @pytest.mark.unit
    def test_diffusion_has_scheduler_and_no_flow_matcher(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
    ):
        prior = dit_prior_factory(
            algorithm_type=DenoisingAlgorithm.DIFFUSION.value,
        )
        assert prior.noise_scheduler is not None
        assert prior.flow_matcher is None

    @pytest.mark.unit
    def test_invalid_algorithm_type_raises(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
    ):
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Unknown algorithm_type: invalid_type. "
                f"Expected one of {[e.value for e in DenoisingAlgorithm]}"
            ),
        ):
            dit_prior_factory(algorithm_type="invalid_type")

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "timestep_sampler, expectation",
        [
            (TimestepSampler.BETA.value, does_not_raise()),
            (TimestepSampler.UNIFORM.value, does_not_raise()),
            (
                "invalid_sampler",
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "Unknown timestep sampler: invalid_sampler. "
                        f"Expected one of {[e.value for e in TimestepSampler]}"
                    ),
                ),
            ),
        ],
    )
    def test_timestep_sampler_validation(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
        timestep_sampler: str,
        expectation: AbstractContextManager,
    ):
        with expectation:
            prior = dit_prior_factory(timestep_sampler=timestep_sampler)
            assert prior.timestep_sampler == timestep_sampler

    @pytest.mark.unit
    def test_stores_timestep_sampling_configuration(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
    ):
        prior = dit_prior_factory(
            timestep_sampler=TimestepSampler.LOGIT_NORMAL.value,
            logit_mean=0.25,
            logit_std=0.5,
            beta_alpha=1.25,
            beta_beta=0.75,
            max_timestep=0.9,
        )

        assert prior.timestep_sampler == TimestepSampler.LOGIT_NORMAL.value
        assert prior.logit_mean == 0.25
        assert prior.logit_std == 0.5
        assert prior.beta_alpha == 1.25
        assert prior.beta_beta == 0.75
        assert prior.max_timestep == 0.9

    @pytest.mark.unit
    def test_stores_prior_target_and_standardization_configuration(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
    ):
        prior = dit_prior_factory(
            prior_target_key=LatentKey.POSTERIOR_LATENT.value,
            latent_standardization_enabled=False,
            latent_standardization_eps=1e-5,
            latent_standardization_max_batches=3,
            require_fitted_latent_standardization=True,
        )

        assert prior.prior_target_key == LatentKey.POSTERIOR_LATENT.value
        assert prior.latent_standardizer.enabled is False
        assert prior.latent_standardizer.epsilon == pytest.approx(1e-5)
        assert prior.latent_standardization_max_batches == 3
        assert prior.latent_standardizer.require_fitted is True

    @pytest.mark.unit
    def test_invalid_latent_standardization_max_batches_raises(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
    ):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "latent_standardization_max_batches must be positive when set, got 0."
            ),
        ):
            dit_prior_factory(latent_standardization_max_batches=0)

    @pytest.mark.unit
    def test_invalid_prior_target_key_raises(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
    ):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Unsupported DiTPrior prior_target_key: wrong_key. "
                f"Expected one of "
                f"{[LatentKey.POSTERIOR_MU.value, LatentKey.POSTERIOR_LATENT.value]}"
            ),
        ):
            dit_prior_factory(prior_target_key="wrong_key")

    @pytest.mark.unit
    def test_latent_standardizer_buffers_are_registered_at_init(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
    ):
        prior = dit_prior_factory()
        state_dict = prior.state_dict()

        assert "latent_standardizer.mean" in state_dict
        assert "latent_standardizer.std" in state_dict
        assert "latent_standardizer.is_fitted" in state_dict

    @pytest.mark.unit
    def test_temporal_positional_encoding_created_for_multi_horizon(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
    ):
        prior = dit_prior_factory(observation_horizon=3)
        assert prior.input_builder.temporal_positional_encoding_layer is not None

    @pytest.mark.unit
    def test_no_temporal_positional_encoding_for_single_horizon(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
    ):
        prior = dit_prior_factory(observation_horizon=1)
        assert prior.input_builder.temporal_positional_encoding_layer is None

    @pytest.mark.unit
    def test_top_level_projection_layers_use_dit_init(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
    ):
        prior = dit_prior_factory(latent_dimension=4, embedding_dimension=32)

        top_level_linears = [
            prior.latent_input_proj,
            prior.timestep_mlp[0],
            prior.timestep_mlp[2],
        ]
        for layer in top_level_linears:
            assert 0.005 < layer.weight.std().item() < 0.05
            assert torch.all(layer.bias == 0.0)

    @pytest.mark.unit
    def test_final_prediction_layer_remains_zero_initialized(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
    ):
        prior = dit_prior_factory()
        output_linear = prior.final_layer.output_linear
        assert torch.all(output_linear.weight == 0.0)
        assert torch.all(output_linear.bias == 0.0)

    @pytest.mark.unit
    def test_timestep_embedding_depends_on_discrete_timestep(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
    ):
        prior = dit_prior_factory(num_train_timesteps=100)

        embeddings = prior._get_timestep_embedding(torch.tensor([0, 1, 50, 99]))

        assert not torch.allclose(embeddings[0], embeddings[1])
        assert not torch.allclose(embeddings[0], embeddings[2])
        assert not torch.allclose(embeddings[0], embeddings[3])

    @pytest.mark.unit
    def test_timestep_embedding_depends_on_continuous_time(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
    ):
        prior = dit_prior_factory(num_train_timesteps=100)

        embeddings = prior._get_timestep_embedding_continuous(
            torch.tensor([0.0, 0.25, 0.5, 1.0])
        )

        assert not torch.allclose(embeddings[0], embeddings[1])
        assert not torch.allclose(embeddings[0], embeddings[2])
        assert not torch.allclose(embeddings[0], embeddings[3])


class TestDiTPriorGetAuxiliaryOutputKeys:
    @pytest.mark.unit
    def test_returns_denoising_specific_keys(
        self,
        dit_prior_factory: Callable,
    ) -> None:
        prior = dit_prior_factory(algorithm_type=DenoisingAlgorithm.FLOW_MATCHING.value)
        keys = prior.get_auxiliary_output_keys()
        assert keys == {
            LatentKey.PRIOR_LATENT.value,
            LatentKey.PRIOR_PREDICTION.value,
            LatentKey.PRIOR_TARGET.value,
        }

    @pytest.mark.unit
    def test_does_not_contain_gaussian_keys(
        self,
        dit_prior_factory: Callable,
    ) -> None:
        prior = dit_prior_factory(algorithm_type=DenoisingAlgorithm.FLOW_MATCHING.value)
        keys = prior.get_auxiliary_output_keys()
        assert LatentKey.PRIOR_MU.value not in keys
        assert LatentKey.PRIOR_LOGVAR.value not in keys


class TestDiTPriorBuildTrainingTarget:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "prior_target_key",
        [
            LatentKey.POSTERIOR_MU.value,
            LatentKey.POSTERIOR_LATENT.value,
        ],
    )
    def test_selects_configured_detached_target(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
        prior_target_key: str,
    ):
        prior = dit_prior_factory(prior_target_key=prior_target_key)
        posterior_output = {
            LatentKey.POSTERIOR_LATENT.value: torch.full(
                (2, LATENT_DIMENSION), fill_value=1.0, requires_grad=True
            ),
            LatentKey.POSTERIOR_MU.value: torch.full(
                (2, LATENT_DIMENSION), fill_value=2.0, requires_grad=True
            ),
        }

        target = prior.build_training_target(posterior_output)

        torch.testing.assert_close(target, posterior_output[prior_target_key].detach())
        assert target.requires_grad is False


class TestDiTPriorFilterObservations:
    @pytest.mark.unit
    def test_excludes_specified_keys(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
    ):
        prior = dit_prior_factory(exclude_keys=["excluded_key"])
        observations = {
            "kept_key": torch.zeros(2, 16),
            "excluded_key": torch.zeros(2, 16),
        }
        filtered = prior._filter_observations(observations)
        assert "kept_key" in filtered
        assert "excluded_key" not in filtered

    @pytest.mark.unit
    def test_returns_all_keys_when_no_exclusions(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
    ):
        prior = dit_prior_factory(exclude_keys=None)
        observations = {"key_a": torch.zeros(2, 16), "key_b": torch.zeros(2, 16)}
        filtered = prior._filter_observations(observations)
        assert set(filtered.keys()) == {"key_a", "key_b"}


class TestDiTPriorForwardFlowMatching:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "use_none_target, expectation",
        [
            (
                False,
                does_not_raise(),
            ),
            (
                True,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "DiTPrior.forward() requires target_latents for "
                        "denoising training. Use sample_prior() for inference."
                    ),
                ),
            ),
        ],
    )
    def test_target_latents_validation(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
        latent_tensor_factory: Callable[..., torch.Tensor],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
        use_none_target: bool,
        expectation: AbstractContextManager,
    ):
        prior = dit_prior_factory(
            algorithm_type=DenoisingAlgorithm.FLOW_MATCHING.value,
        )
        observations = flat_feature_factory(
            feature_dim=EMBEDDING_DIMENSION,
            feature_keys=["obs_feature"],
        )
        target_latents = None if use_none_target else latent_tensor_factory()
        with expectation:
            prior.forward(target_latents=target_latents, observations=observations)

    @pytest.mark.unit
    def test_returns_prediction_and_target_keys(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
        latent_tensor_factory: Callable[..., torch.Tensor],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        prior = dit_prior_factory(
            algorithm_type=DenoisingAlgorithm.FLOW_MATCHING.value,
        )
        target_latents = latent_tensor_factory()
        observations = flat_feature_factory(
            feature_dim=EMBEDDING_DIMENSION,
            feature_keys=["obs_feature"],
        )
        result = prior.forward(
            target_latents=target_latents,
            observations=observations,
        )
        assert set(result.keys()) == {
            LatentKey.PRIOR_PREDICTION.value,
            LatentKey.PRIOR_TARGET.value,
        }

    @pytest.mark.unit
    def test_output_shapes_match_latent_dimension(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
        latent_tensor_factory: Callable[..., torch.Tensor],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 3
        prior = dit_prior_factory(
            algorithm_type=DenoisingAlgorithm.FLOW_MATCHING.value,
        )
        target_latents = latent_tensor_factory(batch_size=batch_size)
        observations = flat_feature_factory(
            batch_size=batch_size,
            feature_dim=EMBEDDING_DIMENSION,
            feature_keys=["obs_feature"],
        )
        result = prior.forward(
            target_latents=target_latents,
            observations=observations,
        )
        assert result[LatentKey.PRIOR_PREDICTION.value].shape == (
            batch_size,
            LATENT_DIMENSION,
        )
        assert result[LatentKey.PRIOR_TARGET.value].shape == (
            batch_size,
            LATENT_DIMENSION,
        )

    @pytest.mark.unit
    def test_samples_configured_flow_timestep(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
        latent_tensor_factory: Callable[..., torch.Tensor],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 3
        prior = dit_prior_factory(
            algorithm_type=DenoisingAlgorithm.FLOW_MATCHING.value,
            timestep_sampler=TimestepSampler.UNIFORM.value,
        )
        target_latents = latent_tensor_factory(batch_size=batch_size)
        observations = flat_feature_factory(
            batch_size=batch_size,
            feature_dim=EMBEDDING_DIMENSION,
            feature_keys=["obs_feature"],
        )
        sampled_time = torch.tensor([0.1, 0.3, 0.7])
        interpolated_latent = torch.ones_like(target_latents)
        target_velocity = torch.full_like(target_latents, fill_value=2.0)
        predicted_velocity = torch.full_like(target_latents, fill_value=3.0)

        with (
            patch(
                "versatil.models.decoding.latent.prior.dit_prior.sample_timesteps_from_config",
                return_value=sampled_time,
            ) as sample_timesteps_mock,
            patch.object(
                prior.flow_matcher,
                "sample_location_and_conditional_flow",
                return_value=(sampled_time, interpolated_latent, target_velocity),
            ) as flow_matcher_mock,
            patch.object(
                prior,
                "_predict_from_tokens",
                return_value=predicted_velocity,
            ),
        ):
            result = prior.forward(
                target_latents=target_latents,
                observations=observations,
            )

        sample_timesteps_mock.assert_called_once_with(
            config=prior.timestep_sampling_config,
            batch_size=batch_size,
            device=target_latents.device,
        )
        flow_matcher_mock.assert_called_once()
        torch.testing.assert_close(
            flow_matcher_mock.call_args.kwargs["t"],
            sampled_time,
        )
        torch.testing.assert_close(
            result[LatentKey.PRIOR_PREDICTION.value],
            predicted_velocity,
        )
        torch.testing.assert_close(
            result[LatentKey.PRIOR_TARGET.value],
            target_velocity,
        )

    @pytest.mark.unit
    def test_flow_matching_standardizes_training_target(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
        latent_tensor_factory: Callable[..., torch.Tensor],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        prior = dit_prior_factory(
            algorithm_type=DenoisingAlgorithm.FLOW_MATCHING.value,
        )
        prior.latent_standardizer.set_stats(
            mean=torch.arange(LATENT_DIMENSION, dtype=torch.float32),
            std=2.0 * torch.ones(LATENT_DIMENSION),
        )
        target_latents = latent_tensor_factory(batch_size=batch_size)
        observations = flat_feature_factory(
            batch_size=batch_size,
            feature_dim=EMBEDDING_DIMENSION,
            feature_keys=["obs_feature"],
        )
        sampled_time = torch.tensor([0.1, 0.7])
        interpolated_latent = torch.ones_like(target_latents)
        target_velocity = torch.full_like(target_latents, fill_value=2.0)
        predicted_velocity = torch.full_like(target_latents, fill_value=3.0)

        with (
            patch(
                "versatil.models.decoding.latent.prior.dit_prior.sample_timesteps_from_config",
                return_value=sampled_time,
            ),
            patch.object(
                prior.flow_matcher,
                "sample_location_and_conditional_flow",
                return_value=(sampled_time, interpolated_latent, target_velocity),
            ) as flow_matcher_mock,
            patch.object(
                prior,
                "_predict_from_tokens",
                return_value=predicted_velocity,
            ),
        ):
            prior.forward(target_latents=target_latents, observations=observations)

        expected_target = prior.latent_standardizer.standardize(target_latents)
        torch.testing.assert_close(
            flow_matcher_mock.call_args.kwargs["x1"],
            expected_target,
        )


class TestDiTPriorForwardDiffusion:
    @pytest.mark.unit
    def test_returns_prediction_and_target_keys(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
        latent_tensor_factory: Callable[..., torch.Tensor],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        prior = dit_prior_factory(
            algorithm_type=DenoisingAlgorithm.DIFFUSION.value,
        )
        target_latents = latent_tensor_factory()
        observations = flat_feature_factory(
            feature_dim=EMBEDDING_DIMENSION,
            feature_keys=["obs_feature"],
        )
        result = prior.forward(
            target_latents=target_latents,
            observations=observations,
        )
        assert set(result.keys()) == {
            LatentKey.PRIOR_PREDICTION.value,
            LatentKey.PRIOR_TARGET.value,
        }

    @pytest.mark.unit
    def test_output_shapes_match_latent_dimension(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
        latent_tensor_factory: Callable[..., torch.Tensor],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 3
        prior = dit_prior_factory(
            algorithm_type=DenoisingAlgorithm.DIFFUSION.value,
        )
        target_latents = latent_tensor_factory(batch_size=batch_size)
        observations = flat_feature_factory(
            batch_size=batch_size,
            feature_dim=EMBEDDING_DIMENSION,
            feature_keys=["obs_feature"],
        )
        result = prior.forward(
            target_latents=target_latents,
            observations=observations,
        )
        assert result[LatentKey.PRIOR_PREDICTION.value].shape == (
            batch_size,
            LATENT_DIMENSION,
        )
        assert result[LatentKey.PRIOR_TARGET.value].shape == (
            batch_size,
            LATENT_DIMENSION,
        )

    @pytest.mark.unit
    def test_epsilon_target_is_noise(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
        latent_tensor_factory: Callable[..., torch.Tensor],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        prior = dit_prior_factory(
            algorithm_type=DenoisingAlgorithm.DIFFUSION.value,
            prediction_type=PredictionType.EPSILON.value,
        )
        target_latents = latent_tensor_factory()
        observations = flat_feature_factory(
            feature_dim=EMBEDDING_DIMENSION,
            feature_keys=["obs_feature"],
        )
        result = prior.forward(
            target_latents=target_latents,
            observations=observations,
        )
        # Epsilon target is random noise, so it should differ from the clean latents
        assert not torch.equal(result[LatentKey.PRIOR_TARGET.value], target_latents)

    @pytest.mark.unit
    def test_sample_target_equals_clean_latents(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
        latent_tensor_factory: Callable[..., torch.Tensor],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        prior = dit_prior_factory(
            algorithm_type=DenoisingAlgorithm.DIFFUSION.value,
            prediction_type=PredictionType.SAMPLE.value,
        )
        target_latents = latent_tensor_factory()
        observations = flat_feature_factory(
            feature_dim=EMBEDDING_DIMENSION,
            feature_keys=["obs_feature"],
        )
        result = prior.forward(
            target_latents=target_latents,
            observations=observations,
        )
        # Sample prediction type targets the clean data directly
        torch.testing.assert_close(result[LatentKey.PRIOR_TARGET.value], target_latents)

    @pytest.mark.unit
    def test_sample_target_uses_standardized_clean_latents(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
        latent_tensor_factory: Callable[..., torch.Tensor],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        prior = dit_prior_factory(
            algorithm_type=DenoisingAlgorithm.DIFFUSION.value,
            prediction_type=PredictionType.SAMPLE.value,
        )
        prior.latent_standardizer.set_stats(
            mean=torch.arange(LATENT_DIMENSION, dtype=torch.float32),
            std=2.0 * torch.ones(LATENT_DIMENSION),
        )
        target_latents = latent_tensor_factory()
        observations = flat_feature_factory(
            feature_dim=EMBEDDING_DIMENSION,
            feature_keys=["obs_feature"],
        )

        result = prior.forward(
            target_latents=target_latents,
            observations=observations,
        )

        torch.testing.assert_close(
            result[LatentKey.PRIOR_TARGET.value],
            prior.latent_standardizer.standardize(target_latents),
        )

    @pytest.mark.unit
    def test_velocity_target_differs_from_noise_and_clean(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
        latent_tensor_factory: Callable[..., torch.Tensor],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        prior = dit_prior_factory(
            algorithm_type=DenoisingAlgorithm.DIFFUSION.value,
            prediction_type=PredictionType.VELOCITY.value,
        )
        target_latents = latent_tensor_factory()
        observations = flat_feature_factory(
            feature_dim=EMBEDDING_DIMENSION,
            feature_keys=["obs_feature"],
        )
        result = prior.forward(
            target_latents=target_latents,
            observations=observations,
        )
        target = result[LatentKey.PRIOR_TARGET.value]
        # Velocity is a linear combination of noise and clean data — differs from both
        assert not torch.equal(target, target_latents)
        assert target.shape == target_latents.shape

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "prediction_type",
        [
            PredictionType.EPSILON.value,
            PredictionType.SAMPLE.value,
            PredictionType.VELOCITY.value,
        ],
    )
    def test_all_prediction_types_produce_correct_shapes(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
        latent_tensor_factory: Callable[..., torch.Tensor],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
        prediction_type: str,
    ):
        batch_size = 3
        prior = dit_prior_factory(
            algorithm_type=DenoisingAlgorithm.DIFFUSION.value,
            prediction_type=prediction_type,
        )
        target_latents = latent_tensor_factory(batch_size=batch_size)
        observations = flat_feature_factory(
            batch_size=batch_size,
            feature_dim=EMBEDDING_DIMENSION,
            feature_keys=["obs_feature"],
        )
        result = prior.forward(
            target_latents=target_latents,
            observations=observations,
        )
        assert result[LatentKey.PRIOR_PREDICTION.value].shape == (
            batch_size,
            LATENT_DIMENSION,
        )
        assert result[LatentKey.PRIOR_TARGET.value].shape == (
            batch_size,
            LATENT_DIMENSION,
        )


class TestDiTPriorSamplePriorFlowMatching:
    @pytest.mark.unit
    def test_output_shape(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 3
        prior = dit_prior_factory(
            algorithm_type=DenoisingAlgorithm.FLOW_MATCHING.value,
        )
        observations = flat_feature_factory(
            batch_size=batch_size,
            feature_dim=EMBEDDING_DIMENSION,
            feature_keys=["obs_feature"],
        )
        sample = prior.sample_prior(
            batch_size=batch_size,
            observations=observations,
        )
        assert sample.shape == (batch_size, LATENT_DIMENSION)

    @pytest.mark.unit
    def test_uses_ode_integration(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        prior = dit_prior_factory(
            algorithm_type=DenoisingAlgorithm.FLOW_MATCHING.value,
        )
        observations = flat_feature_factory(
            batch_size=batch_size,
            feature_dim=EMBEDDING_DIMENSION,
            feature_keys=["obs_feature"],
        )
        with patch(
            "versatil.models.decoding.latent.prior.dit_prior.integrate_ode",
        ) as mock_integrate:
            mock_integrate.return_value = torch.zeros(batch_size, LATENT_DIMENSION)
            prior.sample_prior(
                batch_size=batch_size,
                observations=observations,
            )
        mock_integrate.assert_called_once()

    @pytest.mark.unit
    def test_unstandardizes_integrated_sample(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        prior = dit_prior_factory(
            algorithm_type=DenoisingAlgorithm.FLOW_MATCHING.value,
        )
        prior.latent_standardizer.set_stats(
            mean=torch.arange(LATENT_DIMENSION, dtype=torch.float32),
            std=2.0 * torch.ones(LATENT_DIMENSION),
        )
        standardized_sample = torch.full((batch_size, LATENT_DIMENSION), fill_value=3.0)
        observations = flat_feature_factory(
            batch_size=batch_size,
            feature_dim=EMBEDDING_DIMENSION,
            feature_keys=["obs_feature"],
        )

        with patch(
            "versatil.models.decoding.latent.prior.dit_prior.integrate_ode",
            return_value=standardized_sample,
        ):
            sample = prior.sample_prior(
                batch_size=batch_size, observations=observations
            )

        expected_sample = prior.latent_standardizer.unstandardize(standardized_sample)
        torch.testing.assert_close(sample, expected_sample)

    @pytest.mark.unit
    def test_sample_without_observations(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
    ):
        batch_size = 2
        prior = dit_prior_factory(
            algorithm_type=DenoisingAlgorithm.FLOW_MATCHING.value,
        )
        sample = prior.sample_prior(
            batch_size=batch_size,
            observations=None,
        )
        assert sample.shape == (batch_size, LATENT_DIMENSION)


class TestDiTPriorSamplePriorDiffusion:
    @pytest.mark.unit
    def test_output_shape(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 3
        prior = dit_prior_factory(
            algorithm_type=DenoisingAlgorithm.DIFFUSION.value,
        )
        observations = flat_feature_factory(
            batch_size=batch_size,
            feature_dim=EMBEDDING_DIMENSION,
            feature_keys=["obs_feature"],
        )
        sample = prior.sample_prior(
            batch_size=batch_size,
            observations=observations,
        )
        assert sample.shape == (batch_size, LATENT_DIMENSION)

    @pytest.mark.unit
    def test_iterates_through_scheduler_timesteps(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        prior = dit_prior_factory(
            algorithm_type=DenoisingAlgorithm.DIFFUSION.value,
            num_inference_steps=NUM_INFERENCE_STEPS,
        )
        observations = flat_feature_factory(
            batch_size=batch_size,
            feature_dim=EMBEDDING_DIMENSION,
            feature_keys=["obs_feature"],
        )
        # Just confirm it runs and produces valid output
        sample = prior.sample_prior(
            batch_size=batch_size,
            observations=observations,
        )
        assert sample.shape == (batch_size, LATENT_DIMENSION)
        assert not torch.any(torch.isnan(sample))

    @pytest.mark.unit
    def test_unstandardizes_diffusion_sample(
        self,
        dit_prior_factory: Callable[..., DiTPrior],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        prior = dit_prior_factory(
            algorithm_type=DenoisingAlgorithm.DIFFUSION.value,
            num_inference_steps=1,
        )
        prior.latent_standardizer.set_stats(
            mean=torch.arange(LATENT_DIMENSION, dtype=torch.float32),
            std=2.0 * torch.ones(LATENT_DIMENSION),
        )
        standardized_sample = torch.full((batch_size, LATENT_DIMENSION), fill_value=3.0)
        observations = flat_feature_factory(
            batch_size=batch_size,
            feature_dim=EMBEDDING_DIMENSION,
            feature_keys=["obs_feature"],
        )
        prior.noise_scheduler.timesteps = torch.tensor([0])

        with (
            patch(
                "versatil.models.decoding.latent.prior.dit_prior.setup_inference_timesteps",
            ),
            patch.object(
                prior,
                "_predict_from_tokens",
                return_value=torch.zeros_like(standardized_sample),
            ),
            patch.object(
                prior.noise_scheduler,
                "step",
                return_value=MagicMock(prev_sample=standardized_sample),
            ),
        ):
            sample = prior.sample_prior(
                batch_size=batch_size, observations=observations
            )

        expected_sample = prior.latent_standardizer.unstandardize(standardized_sample)
        torch.testing.assert_close(sample, expected_sample)
