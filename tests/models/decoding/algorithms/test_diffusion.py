"""Tests for versatil.models.decoding.algorithm.diffusion module."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch
from diffusers import DDIMScheduler, DDPMScheduler

from versatil.data.constants import SampleKey
from versatil.models.decoding.algorithm.base import DecodingAlgorithm
from versatil.models.decoding.algorithm.diffusion import Diffusion
from versatil.models.decoding.constants import (
    AlgorithmContextKey,
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

    def test_auxiliary_output_keys_are_empty(
        self,
        diffusion_factory: Callable[..., Diffusion],
    ):
        diff = diffusion_factory()
        assert diff.get_auxiliary_output_keys() == set()

    @pytest.mark.parametrize("num_train_timesteps", [50, 200])
    @pytest.mark.parametrize("num_inference_steps", [5, 20])
    @pytest.mark.parametrize(
        "prediction_type, expected_in_action_space",
        [
            (PredictionType.EPSILON.value, False),
            (PredictionType.VELOCITY.value, False),
            (PredictionType.SAMPLE.value, True),
        ],
    )
    @pytest.mark.parametrize(
        "scheduler_type, expected_scheduler_class",
        [
            (SchedulerType.DDPM.value, DDPMScheduler),
            (SchedulerType.DDIM.value, DDIMScheduler),
        ],
    )
    def test_stores_configuration(
        self,
        diffusion_factory: Callable[..., Diffusion],
        num_train_timesteps: int,
        num_inference_steps: int,
        prediction_type: str,
        expected_in_action_space: bool,
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
        assert diff.predicts_in_action_space is expected_in_action_space
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
            match=re.escape("Diffusion algorithm requires actions during training"),
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
            AlgorithmContextKey.TIMESTEP.value,
        }
        if include_padding_mask:
            expected_keys.add(SampleKey.IS_PAD_ACTION.value)
        assert set(result.keys()) == expected_keys
        assert set(result[DecoderOutputKey.TARGET_DIFFUSION.value].keys()) == {
            "position_action"
        }
        assert set(result[DecoderOutputKey.NOISE.value].keys()) == {"position_action"}
        if include_padding_mask:
            padding_mask = action_dictionary_factory(
                action_keys=["position_action"],
                prediction_horizon=8,
                action_dimension=3,
                include_padding_mask=True,
            )[SampleKey.IS_PAD_ACTION.value]
            assert result[SampleKey.IS_PAD_ACTION.value].shape == padding_mask.shape
        else:
            assert SampleKey.IS_PAD_ACTION.value not in result

    def test_padding_mask_is_passed_through_unchanged(
        self,
        diffusion_factory: Callable[..., Diffusion],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        diff = diffusion_factory()
        mock_network = mock_action_decoder_factory(action_keys=["position_action"])
        features = feature_dictionary_factory()
        actions = action_dictionary_factory(
            action_keys=["position_action"],
            prediction_horizon=8,
            action_dimension=3,
            include_padding_mask=True,
        )
        original_mask = actions[SampleKey.IS_PAD_ACTION.value].clone()
        result = diff.forward(network=mock_network, features=features, actions=actions)
        assert torch.equal(result[SampleKey.IS_PAD_ACTION.value], original_mask)
        assert (
            SampleKey.IS_PAD_ACTION.value
            not in mock_network.call_args.kwargs["actions"]
        )

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
            action_keys=["position_action"],
            prediction_horizon=8,
            action_dimension=3,
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
            action_keys=["position_action"],
            prediction_horizon=8,
            action_dimension=3,
            include_padding_mask=False,
        )
        original_actions = actions["position_action"].clone()
        mock_network.return_value = {"position_action": torch.zeros(2, 8, 3)}
        result = diff.forward(network=mock_network, features=features, actions=actions)
        target = result[DecoderOutputKey.TARGET_DIFFUSION.value]
        assert torch.equal(target["position_action"], original_actions)

    def test_network_receives_sampled_timestep_in_features(
        self,
        diffusion_factory: Callable[..., Diffusion],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        diff = diffusion_factory()
        mock_network = mock_action_decoder_factory()
        features = feature_dictionary_factory(batch_size=2)
        actions = action_dictionary_factory(
            action_keys=["position_action"],
            prediction_horizon=8,
            action_dimension=3,
        )
        mock_network.return_value = {"position_action": torch.zeros(2, 8, 3)}
        result = diff.forward(network=mock_network, features=features, actions=actions)
        features_passed = mock_network.call_args.kwargs["features"]
        timestep_in_features = features_passed[AlgorithmContextKey.TIMESTEP.value]
        assert timestep_in_features.shape == (2,)
        assert torch.equal(
            timestep_in_features, result[AlgorithmContextKey.TIMESTEP.value]
        )

    def test_velocity_target_uses_scheduler_get_velocity(
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
            action_keys=["position_action"],
            prediction_horizon=8,
            action_dimension=3,
            include_padding_mask=False,
        )
        mock_network.return_value = {"position_action": torch.zeros(2, 8, 3)}
        with patch.object(
            diff.noise_scheduler,
            "get_velocity",
            wraps=diff.noise_scheduler.get_velocity,
        ) as get_velocity_spy:
            result = diff.forward(
                network=mock_network, features=features, actions=actions
            )
        target = result[DecoderOutputKey.TARGET_DIFFUSION.value]
        noise = result[DecoderOutputKey.NOISE.value]
        timesteps = result[AlgorithmContextKey.TIMESTEP.value]
        get_velocity_spy.assert_called_once()
        call_kwargs = get_velocity_spy.call_args.kwargs
        assert torch.equal(call_kwargs["sample"], actions["position_action"])
        assert torch.equal(call_kwargs["noise"], noise["position_action"])
        assert torch.equal(call_kwargs["timesteps"], timesteps)
        expected_velocity = diff.noise_scheduler.get_velocity(
            sample=actions["position_action"],
            noise=noise["position_action"],
            timesteps=timesteps,
        )
        assert torch.equal(target["position_action"], expected_velocity)

    def test_forward_with_multiple_action_keys(
        self,
        diffusion_factory: Callable[..., Diffusion],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        diff = diffusion_factory(prediction_type=PredictionType.EPSILON.value)
        action_keys = ["gripper_action", "position_action"]
        mock_network = mock_action_decoder_factory(
            action_keys=action_keys,
            prediction_dimension=3,
        )
        features = feature_dictionary_factory()
        actions = action_dictionary_factory(
            action_keys=action_keys,
            prediction_horizon=8,
            action_dimension=3,
            include_padding_mask=True,
        )
        mock_network.return_value = {key: torch.zeros(2, 8, 3) for key in action_keys}
        result = diff.forward(network=mock_network, features=features, actions=actions)
        target = result[DecoderOutputKey.TARGET_DIFFUSION.value]
        noise = result[DecoderOutputKey.NOISE.value]
        for key in action_keys:
            assert key in target
            assert key in noise
            assert torch.equal(target[key], noise[key])

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
            action_keys=["position_action"],
            prediction_horizon=8,
            action_dimension=3,
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


class TestDiffusionGetTargets:
    def test_returns_algorithm_target_not_raw_actions(
        self,
        diffusion_factory: Callable[..., Diffusion],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        diffusion = diffusion_factory(prediction_type=PredictionType.EPSILON.value)
        mock_network = mock_action_decoder_factory(action_keys=["position_action"])
        features = feature_dictionary_factory()
        actions = action_dictionary_factory(
            action_keys=["position_action"],
            prediction_horizon=8,
            action_dimension=3,
        )
        output = diffusion.forward(
            network=mock_network,
            features=features,
            actions=actions,
        )
        targets = diffusion.get_targets(
            algorithm_output=output,
            ground_truth_actions=actions,
        )
        assert targets is output[DecoderOutputKey.TARGET_DIFFUSION.value]
        # epsilon mode trains on noise, which differs from the clean actions
        assert not torch.equal(targets["position_action"], actions["position_action"])

    def test_epsilon_target_is_noise(
        self,
        diffusion_factory: Callable[..., Diffusion],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        diffusion = diffusion_factory(prediction_type=PredictionType.EPSILON.value)
        mock_network = mock_action_decoder_factory(action_keys=["position_action"])
        features = feature_dictionary_factory()
        actions = action_dictionary_factory(
            action_keys=["position_action"],
            prediction_horizon=8,
            action_dimension=3,
        )
        output = diffusion.forward(
            network=mock_network,
            features=features,
            actions=actions,
        )
        targets = diffusion.get_targets(
            algorithm_output=output,
            ground_truth_actions=actions,
        )
        noise = output[DecoderOutputKey.NOISE.value]["position_action"]
        assert torch.equal(targets["position_action"], noise)

    def test_sample_target_is_raw_actions(
        self,
        diffusion_factory: Callable[..., Diffusion],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        diffusion = diffusion_factory(prediction_type=PredictionType.SAMPLE.value)
        mock_network = mock_action_decoder_factory(action_keys=["position_action"])
        features = feature_dictionary_factory()
        actions = action_dictionary_factory(
            action_keys=["position_action"],
            prediction_horizon=8,
            action_dimension=3,
        )
        output = diffusion.forward(
            network=mock_network,
            features=features,
            actions=actions,
        )
        targets = diffusion.get_targets(
            algorithm_output=output,
            ground_truth_actions=actions,
        )
        assert torch.equal(
            targets["position_action"],
            actions["position_action"],
        )


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

    def test_predict_passes_batch_expanded_timestep_to_network(
        self,
        diffusion_factory: Callable[..., Diffusion],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        diff = diffusion_factory(num_inference_steps=2, num_train_timesteps=10)
        mock_network = mock_action_decoder_factory(action_keys=["position_action"])
        features = feature_dictionary_factory(batch_size=2)
        diff.noise_scheduler.step = MagicMock(
            return_value=StepOutput(prev_sample=torch.zeros(2, 8, 3))
        )
        diff.predict(network=mock_network, features=features)
        timestep_passed = mock_network.call_args.kwargs["features"][
            AlgorithmContextKey.TIMESTEP.value
        ]
        assert timestep_passed.shape == (2,)
        assert torch.equal(timestep_passed, timestep_passed[0].expand(2))

    def test_predict_enables_and_disables_encoder_cache(
        self,
        diffusion_factory: Callable[..., Diffusion],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        diff = diffusion_factory(num_inference_steps=2, num_train_timesteps=10)
        mock_network = mock_action_decoder_factory(action_keys=["position_action"])
        diff.noise_scheduler.step = MagicMock(
            return_value=StepOutput(prev_sample=torch.zeros(2, 8, 3))
        )
        features = feature_dictionary_factory()
        diff.predict(network=mock_network, features=features)
        mock_network.enable_encoder_cache.assert_called_once()
        mock_network.disable_encoder_cache.assert_called_once()

    def test_predict_with_multiple_action_keys(
        self,
        diffusion_factory: Callable[..., Diffusion],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        diff = diffusion_factory(num_inference_steps=2, num_train_timesteps=10)
        action_keys = ["gripper_action", "position_action"]
        mock_network = mock_action_decoder_factory(
            action_keys=action_keys,
            prediction_dimension=3,
        )
        features = feature_dictionary_factory()
        diff.noise_scheduler.step = MagicMock(
            return_value=StepOutput(prev_sample=torch.zeros(2, 8, 3))
        )
        result = diff.predict(network=mock_network, features=features)
        assert set(result.keys()) == set(action_keys)
        for key in action_keys:
            assert result[key].shape == (2, 8, 3)
