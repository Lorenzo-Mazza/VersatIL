"""Tests for versatil.models.decoding.latent.prior.dit_prior module."""

import re
from collections.abc import Callable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from unittest.mock import patch

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
        num_train_timesteps: int = NUM_TRAIN_TIMESTEPS,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
        exclude_keys: list[str] | None = None,
        prediction_type: str = PredictionType.EPSILON.value,
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
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=num_inference_steps,
            exclude_keys=exclude_keys,
            prediction_type=prediction_type,
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
