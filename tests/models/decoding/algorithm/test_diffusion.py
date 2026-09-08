"""Tests for versatil.models.decoding.algorithm.diffusion module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock, call, patch

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
    def test_changing_step_count_rebuilds_schedule_without_checkpoint_state(
        self,
        diffusion_factory: Callable[..., Diffusion],
    ) -> None:
        diffusion = diffusion_factory(num_train_timesteps=12, num_inference_steps=3)
        assert diffusion.inference_schedule.timesteps == (8, 4, 0)
        diffusion.num_inference_steps = 2
        assert diffusion.num_inference_steps == 2
        assert diffusion.inference_schedule.timesteps == (6, 0)
        assert len(diffusion.inference_schedule.coefficients) == 2
        assert diffusion.state_dict() == {}

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
    @pytest.mark.parametrize(
        "scheduler_type", [member.value for member in SchedulerType]
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_samples_noise_and_delegates_denoising(
        self,
        diffusion_factory: Callable[..., Diffusion],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        scheduler_type: str,
        dtype: torch.dtype,
    ) -> None:
        diffusion = diffusion_factory(
            scheduler_type=scheduler_type, num_inference_steps=2, num_train_timesteps=10
        )
        network = mock_action_decoder_factory(
            action_keys=["position_action", "unused_action"], prediction_horizon=8
        )
        network.action_space.actions_metadata[
            "unused_action"
        ].requires_prediction_head = False
        features = feature_dictionary_factory(batch_size=2)
        initial_noise = torch.ones(2, 8, 3, dtype=dtype)  # (batch, horizon, action_dim)
        step_noise = torch.zeros(
            2, 2, 8, 3, dtype=dtype
        )  # (batch, steps, horizon, action_dim)
        expected = {"position_action": initial_noise}
        device = torch.device("cpu")
        with (
            patch(
                "versatil.models.decoding.algorithm.diffusion.resolve_feature_reference",
                return_value=(2, device, dtype),
            ) as resolve_reference,
            patch(
                "versatil.models.decoding.algorithm.diffusion.torch.randn",
                side_effect=[initial_noise, step_noise],
            ) as sample_noise,
            patch.object(
                diffusion, "predict_from_noise", return_value=expected
            ) as denoise,
        ):
            actual = diffusion.predict(network=network, features=features)
        resolve_reference.assert_called_once_with(features=features)
        stochastic = scheduler_type == SchedulerType.DDPM.value
        denoise.assert_called_once_with(
            network=network,
            features=features,
            initial_noise={"position_action": initial_noise},
            step_noise={"position_action": step_noise} if stochastic else None,
        )
        expected_calls = [call(2, 8, 3, device=device, dtype=dtype)]
        if stochastic:
            expected_calls.append(call(2, 2, 8, 3, device=device, dtype=dtype))
        assert sample_noise.call_args_list == expected_calls
        torch.testing.assert_close(actual, expected)


class TestDiffusionPredictFromNoise:
    @pytest.mark.parametrize(
        "scheduler_type", [member.value for member in SchedulerType]
    )
    def test_advances_each_action_with_its_own_noise_and_timestep(
        self,
        diffusion_factory: Callable[..., Diffusion],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        scheduler_type: str,
    ) -> None:
        diffusion = diffusion_factory(
            scheduler_type=scheduler_type, num_inference_steps=2, num_train_timesteps=10
        )
        action_keys = ["gripper_action", "position_action"]
        initial_noise, prediction, first_update, second_update = (
            action_dictionary_factory(
                action_keys=action_keys,
                prediction_horizon=8,
                action_dimension=3,
                include_padding_mask=False,
            )
            for _ in range(4)
        )
        stochastic = scheduler_type == SchedulerType.DDPM.value
        step_noise = (
            {
                key: torch.stack(
                    (initial_noise[key], prediction[key]), dim=1
                )  # two (batch, horizon, action_dim) -> (batch, steps, horizon, action_dim)
                for key in action_keys
            }
            if stochastic
            else None
        )
        network = mock_action_decoder_factory(
            action_keys=action_keys, return_value=prediction
        )
        features = feature_dictionary_factory(batch_size=2)
        with patch.object(
            diffusion.inference_schedule,
            "forward",
            side_effect=[*first_update.values(), *second_update.values()],
        ) as update:
            actual = diffusion.predict_from_noise(
                network=network,
                features=features,
                initial_noise=initial_noise,
                step_noise=step_noise,
            )
        torch.testing.assert_close(actual, second_update)
        assert network.call_count == 2
        assert update.call_count == 4
        for step_index, previous in enumerate((initial_noise, first_update)):
            network_call = network.call_args_list[step_index].kwargs
            torch.testing.assert_close(network_call["actions"], previous)
            expected_timestep = torch.full(
                (2,),
                diffusion.inference_schedule.timesteps[step_index],
                dtype=torch.long,
            )  # (batch,)
            torch.testing.assert_close(
                network_call["features"],
                {**features, AlgorithmContextKey.TIMESTEP.value: expected_timestep},
            )
            for action_index, key in enumerate(action_keys):
                update_call = update.call_args_list[
                    2 * step_index + action_index
                ].kwargs
                assert update_call["step_index"] == step_index
                torch.testing.assert_close(update_call["model_output"], prediction[key])
                torch.testing.assert_close(update_call["sample"], previous[key])
                if stochastic:
                    torch.testing.assert_close(
                        update_call["noise"], step_noise[key][:, step_index]
                    )  # (batch, steps, horizon, action_dim) -> (batch, horizon, action_dim)
                else:
                    assert update_call["noise"] is None
        network.enable_encoder_cache.assert_called_once_with()
        network.disable_encoder_cache.assert_called_once_with()

    def test_disables_encoder_cache_when_decoder_fails(
        self,
        diffusion_factory: Callable[..., Diffusion],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ) -> None:
        diffusion = diffusion_factory(
            scheduler_type=SchedulerType.DDIM.value,
            num_inference_steps=2,
            num_train_timesteps=10,
        )
        network = mock_action_decoder_factory(action_keys=["position_action"])
        network.side_effect = RuntimeError("Decoder failed.")
        features = feature_dictionary_factory(batch_size=2)
        noise = action_dictionary_factory(
            action_keys=["position_action"],
            prediction_horizon=8,
            action_dimension=3,
            include_padding_mask=False,
        )
        with pytest.raises(RuntimeError, match=re.escape("Decoder failed.")):
            diffusion.predict_from_noise(
                network=network, features=features, initial_noise=noise
            )
        network.enable_encoder_cache.assert_called_once_with()
        network.disable_encoder_cache.assert_called_once_with()

    @pytest.mark.parametrize("noise_case", ["missing", "wrong_key", "wrong_steps"])
    def test_rejects_invalid_ddpm_noise_before_calling_decoder(
        self,
        diffusion_factory: Callable[..., Diffusion],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        noise_case: str,
    ) -> None:
        diffusion = diffusion_factory(
            scheduler_type=SchedulerType.DDPM.value,
            num_inference_steps=2,
            num_train_timesteps=10,
        )
        network = mock_action_decoder_factory(action_keys=["position_action"])
        features = feature_dictionary_factory(batch_size=2)
        initial_noise = {
            "position_action": torch.zeros(2, 8, 3)
        }  # (batch, horizon, action_dim)
        step_noise = None
        message = "DDPM inference requires step noise for every initial-noise action component."
        if noise_case == "wrong_key":
            step_noise = {
                "gripper_action": torch.zeros(2, 2, 8, 3)
            }  # (batch, steps, horizon, action_dim)
        elif noise_case == "wrong_steps":
            step_noise = {
                "position_action": torch.zeros(2, 1, 8, 3)
            }  # (batch, steps, horizon, action_dim)
            message = (
                "DDPM step noise for 'position_action' must have shape (2, 2, 8, 3), "
                "got (2, 1, 8, 3)."
            )
        with pytest.raises(ValueError, match=re.escape(message)):
            diffusion.predict_from_noise(
                network=network,
                features=features,
                initial_noise=initial_noise,
                step_noise=step_noise,
            )
        network.assert_not_called()
        network.enable_encoder_cache.assert_not_called()
