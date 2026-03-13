"""Tests for versatil.models.decoding.algorithm.diffusion module."""
import re
from collections.abc import Callable
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
import torch
from diffusers import DDIMScheduler, DDPMScheduler

from versatil.data.constants import SampleKey
from versatil.models.decoding.algorithm.base import DecodingAlgorithm
from versatil.models.decoding.algorithm.diffusion import Diffusion
from versatil.models.decoding.constants import (
    BetaSchedule,
    DecoderOutputKey,
    PredictionType,
    VarianceType,
)
from versatil.models.layers.denoising.diffusion_process import SchedulerType


@dataclass
class StepOutput:
    """Mock output for noise scheduler step."""

    prev_sample: torch.Tensor


@pytest.fixture
def diffusion_factory() -> Callable[..., Diffusion]:
    """Factory for Diffusion instances."""
    def factory(
        scheduler_type: str = SchedulerType.DDIM.value,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 10,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = BetaSchedule.SQUAREDCOS_CAP_V2.value,
        prediction_type: str = PredictionType.EPSILON.value,
        scheduler_variance_type: str = VarianceType.FIXED_SMALL.value,
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
    ) -> Diffusion:
        return Diffusion(
            scheduler_type=scheduler_type,
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=num_inference_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            scheduler_variance_type=scheduler_variance_type,
            clip_sample=clip_sample,
            set_alpha_to_one=set_alpha_to_one,
            steps_offset=steps_offset,
        )
    return factory


class TestDiffusionInitialization:

    def test_inherits_from_decoding_algorithm(
        self,
        diffusion_factory: Callable[..., Diffusion],
    ):
        diff = diffusion_factory()
        assert isinstance(diff, DecodingAlgorithm)

    @pytest.mark.parametrize("num_train_timesteps", [50, 200])
    @pytest.mark.parametrize("num_inference_steps", [5, 20])
    @pytest.mark.parametrize("prediction_type", [
        PredictionType.EPSILON.value,
        PredictionType.VELOCITY.value,
    ])
    @pytest.mark.parametrize("scheduler_type, expected_scheduler_class", [
        (SchedulerType.DDPM.value, DDPMScheduler),
        (SchedulerType.DDIM.value, DDIMScheduler),
    ])
    def test_stores_configuration(
        self,
        diffusion_factory: Callable[..., Diffusion],
        num_train_timesteps: int,
        num_inference_steps: int,
        prediction_type: str,
        scheduler_type: str,
        expected_scheduler_class: type,
    ):
        diff = diffusion_factory(
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=num_inference_steps,
            prediction_type=prediction_type,
            scheduler_type=scheduler_type,
        )
        assert diff.num_train_timesteps == num_train_timesteps
        assert diff.num_inference_steps == num_inference_steps
        assert diff.prediction_type == prediction_type
        assert isinstance(diff.noise_scheduler, expected_scheduler_class)


class TestDiffusionForward:

    def test_raises_without_actions(
        self,
        diffusion_factory: Callable[..., Diffusion],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        diff = diffusion_factory()
        mock_network = mock_action_decoder_factory()
        features = feature_dictionary_factory()
        with pytest.raises(
            ValueError,
            match="Diffusion algorithm requires actions during training",
        ):
            diff.forward(network=mock_network, features=features, actions=None)

    @pytest.mark.parametrize("include_padding_mask", [True, False])
    def test_output_contains_exact_keys(
        self,
        diffusion_factory: Callable[..., Diffusion],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        include_padding_mask: bool,
    ):
        diff = diffusion_factory()
        mock_network = mock_action_decoder_factory(action_keys=["position_action"])
        features = feature_dictionary_factory()
        actions = action_dictionary_factory(
            action_keys=["position_action"],
            prediction_horizon=8,
            action_dimension=3,
            include_padding_mask=include_padding_mask,
        )
        result = diff.forward(network=mock_network, features=features, actions=actions)
        expected_keys = {
            "position_action",
            DecoderOutputKey.TARGET_DIFFUSION.value,
            DecoderOutputKey.NOISE.value,
            DecoderOutputKey.TIMESTEP.value,
            SampleKey.IS_PAD_ACTION.value,
        }
        assert set(result.keys()) == expected_keys
        assert set(result[DecoderOutputKey.TARGET_DIFFUSION.value].keys()) == {"position_action"}
        assert set(result[DecoderOutputKey.NOISE.value].keys()) == {"position_action"}
        if include_padding_mask:
            assert result[SampleKey.IS_PAD_ACTION.value] is not None
        else:
            assert result[SampleKey.IS_PAD_ACTION.value] is None

    def test_epsilon_target_equals_noise(
        self,
        diffusion_factory: Callable[..., Diffusion],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        diff = diffusion_factory(prediction_type=PredictionType.EPSILON.value)
        mock_network = mock_action_decoder_factory()
        features = feature_dictionary_factory()
        actions = action_dictionary_factory(
            action_keys=["position_action"], prediction_horizon=8, action_dimension=3,
            include_padding_mask=False,
        )
        mock_network.return_value = {"position_action": torch.zeros(2, 8, 3)}
        result = diff.forward(network=mock_network, features=features, actions=actions)
        target = result[DecoderOutputKey.TARGET_DIFFUSION.value]
        noise = result[DecoderOutputKey.NOISE.value]
        assert torch.equal(target["position_action"], noise["position_action"])

    def test_sample_target_equals_original_actions(
        self,
        diffusion_factory: Callable[..., Diffusion],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        diff = diffusion_factory(prediction_type=PredictionType.SAMPLE.value)
        mock_network = mock_action_decoder_factory()
        features = feature_dictionary_factory()
        actions = action_dictionary_factory(
            action_keys=["position_action"], prediction_horizon=8, action_dimension=3,
            include_padding_mask=False,
        )
        original_actions = actions["position_action"].clone()
        mock_network.return_value = {"position_action": torch.zeros(2, 8, 3)}
        result = diff.forward(network=mock_network, features=features, actions=actions)
        target = result[DecoderOutputKey.TARGET_DIFFUSION.value]
        assert torch.equal(target["position_action"], original_actions)

    def test_network_receives_timestep_in_features(
        self,
        diffusion_factory: Callable[..., Diffusion],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        diff = diffusion_factory()
        mock_network = mock_action_decoder_factory()
        features = feature_dictionary_factory()
        actions = action_dictionary_factory(
            action_keys=["position_action"], prediction_horizon=8, action_dimension=3,
        )
        mock_network.return_value = {"position_action": torch.zeros(2, 8, 3)}
        diff.forward(network=mock_network, features=features, actions=actions)
        features_passed = mock_network.call_args.kwargs["features"]
        assert DecoderOutputKey.TIMESTEP.value in features_passed

    def test_velocity_target_uses_get_velocity(
        self,
        diffusion_factory: Callable[..., Diffusion],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        diff = diffusion_factory(prediction_type=PredictionType.VELOCITY.value)
        mock_network = mock_action_decoder_factory()
        features = feature_dictionary_factory()
        actions = action_dictionary_factory(
            action_keys=["position_action"], prediction_horizon=8, action_dimension=3,
            include_padding_mask=False,
        )
        mock_network.return_value = {"position_action": torch.zeros(2, 8, 3)}
        result = diff.forward(network=mock_network, features=features, actions=actions)
        target = result[DecoderOutputKey.TARGET_DIFFUSION.value]
        noise = result[DecoderOutputKey.NOISE.value]
        assert set(target.keys()) == {"position_action"}
        # Velocity target should differ from both noise and original actions
        assert not torch.equal(target["position_action"], noise["position_action"])
        assert not torch.equal(target["position_action"], actions["position_action"])

    def test_invalid_prediction_type_raises(
        self,
        diffusion_factory: Callable[..., Diffusion],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        diff = diffusion_factory()
        mock_network = mock_action_decoder_factory()
        features = feature_dictionary_factory()
        actions = action_dictionary_factory(
            action_keys=["position_action"], prediction_horizon=8, action_dimension=3,
            include_padding_mask=False,
        )
        mock_network.return_value = {"position_action": torch.zeros(2, 8, 3)}
        diff.prediction_type = "invalid_type"
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Unknown prediction_type: invalid_type. "
                f"Expected one of {[e.value for e in PredictionType]}"
            ),
        ):
            diff.forward(network=mock_network, features=features, actions=actions)


class TestDiffusionPredict:

    def test_predict_returns_exact_action_keys(
        self,
        diffusion_factory: Callable[..., Diffusion],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        diff = diffusion_factory(num_inference_steps=2, num_train_timesteps=10)
        mock_network = mock_action_decoder_factory(action_keys=["position_action"])
        features = feature_dictionary_factory()
        diff.noise_scheduler.step = MagicMock(
            return_value=StepOutput(prev_sample=torch.zeros(2, 8, 3))
        )
        result = diff.predict(network=mock_network, features=features)
        assert set(result.keys()) == {"position_action"}

    def test_predict_output_shape(
        self,
        diffusion_factory: Callable[..., Diffusion],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        diff = diffusion_factory(num_inference_steps=2, num_train_timesteps=10)
        mock_network = mock_action_decoder_factory(action_keys=["position_action"])
        features = feature_dictionary_factory()
        diff.noise_scheduler.step = MagicMock(
            return_value=StepOutput(prev_sample=torch.zeros(2, 8, 3))
        )
        result = diff.predict(network=mock_network, features=features)
        assert result["position_action"].shape == (2, 8, 3)
