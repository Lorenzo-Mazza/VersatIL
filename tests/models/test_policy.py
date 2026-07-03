"""Tests for versatil.models.policy module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from versatil.configs import register_resolvers
from versatil.data.constants import (
    Cameras,
    MetadataPassthroughSource,
    SampleKey,
    SyntheticObsKey,
)
from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.tokenization import Tokenizer
from versatil.metrics.base import BaseLoss, LossOutput
from versatil.metrics.constants import MetadataKey
from versatil.metrics.losses.gripper import GripperLoss
from versatil.models.decoding.algorithm.base import DecodingAlgorithm
from versatil.models.decoding.constants import DecoderOutputKey
from versatil.models.decoding.decoders.base import ActionDecoder, DecoderInput
from versatil.models.encoding.encoders.constants import EncoderOutputKeys
from versatil.models.policy import Policy

register_resolvers()


class TestPolicyInitialization:
    @pytest.mark.parametrize("prediction_horizon", [4, 16])
    @pytest.mark.parametrize("observation_horizon", [1, 3])
    def test_stores_configuration(
        self,
        policy_factory: Callable[..., Policy],
        encoding_pipeline_factory: Callable[..., MagicMock],
        prediction_horizon: int,
        observation_horizon: int,
    ):
        pipeline = encoding_pipeline_factory()
        algorithm = MagicMock(spec=DecodingAlgorithm)
        decoder = MagicMock(
            spec=ActionDecoder,
            decoder_input=DecoderInput(keys=[]),
        )
        observation_space = MagicMock(spec=ObservationSpace)
        action_space = MagicMock(spec=ActionSpace)
        loss = MagicMock(spec=BaseLoss)
        policy = policy_factory(
            encoding_pipeline=pipeline,
            algorithm=algorithm,
            decoder=decoder,
            observation_space=observation_space,
            action_space=action_space,
            prediction_horizon=prediction_horizon,
            observation_horizon=observation_horizon,
            loss=loss,
        )
        assert policy.encoding_pipeline is pipeline
        assert policy.algorithm is algorithm
        assert policy.decoder is decoder
        assert policy.observation_space is observation_space
        assert policy.action_space is action_space
        assert policy.prediction_horizon == prediction_horizon
        assert policy.observation_horizon == observation_horizon
        assert policy.loss_module is loss

    def test_converts_device_string_to_torch_device(
        self, policy_factory: Callable[..., Policy]
    ):
        policy = policy_factory(device="cpu")
        assert policy.device.type == "cpu"

    def test_normalizer_starts_empty(self, policy_factory: Callable[..., Policy]):
        policy = policy_factory()
        assert len(list(policy.normalizer.parameters())) == 0

    def test_tokenizer_starts_unset(self, policy_factory: Callable[..., Policy]):
        policy = policy_factory()
        assert policy.tokenizer is None


class TestInputOutputKeys:
    def test_input_keys_from_single_encoder(
        self,
        policy_factory: Callable[..., Policy],
        encoding_pipeline_factory: Callable[..., MagicMock],
        vision_encoder_factory: Callable[..., MagicMock],
    ):
        encoder = vision_encoder_factory(input_keys=["right", "left"])
        pipeline = encoding_pipeline_factory(encoders={"rgb": encoder})
        policy = policy_factory(encoding_pipeline=pipeline)
        assert policy.input_keys == ["left", "right"]

    def test_input_keys_from_multiple_encoders(
        self,
        policy_factory: Callable[..., Policy],
        encoding_pipeline_factory: Callable[..., MagicMock],
        vision_encoder_factory: Callable[..., MagicMock],
    ):
        rgb_encoder = vision_encoder_factory(input_keys=["left"])
        proprio_encoder = vision_encoder_factory(input_keys=["proprio_robot_frame"])
        pipeline = encoding_pipeline_factory(
            encoders={"rgb": rgb_encoder, "proprio": proprio_encoder}
        )
        policy = policy_factory(encoding_pipeline=pipeline)
        assert policy.input_keys == ["left", "proprio_robot_frame"]

    def test_input_keys_includes_conditional_encoders(
        self,
        policy_factory: Callable[..., Policy],
        encoding_pipeline_factory: Callable[..., MagicMock],
        vision_encoder_factory: Callable[..., MagicMock],
    ):
        encoder = vision_encoder_factory(input_keys=["left"])
        conditional_encoder = vision_encoder_factory(input_keys=["depth"])
        pipeline = encoding_pipeline_factory(
            encoders={"rgb": encoder},
            conditional_encoders={"film": conditional_encoder},
        )
        policy = policy_factory(encoding_pipeline=pipeline)
        assert policy.input_keys == ["depth", "left"]

    def test_input_keys_deduplicates_shared_keys(
        self,
        policy_factory: Callable[..., Policy],
        encoding_pipeline_factory: Callable[..., MagicMock],
        vision_encoder_factory: Callable[..., MagicMock],
    ):
        encoder_a = vision_encoder_factory(input_keys=["left", "right"])
        encoder_b = vision_encoder_factory(input_keys=["left", "depth"])
        pipeline = encoding_pipeline_factory(encoders={"a": encoder_a, "b": encoder_b})
        policy = policy_factory(encoding_pipeline=pipeline)
        assert policy.input_keys == ["depth", "left", "right"]

    def test_input_keys_adds_tokenized_keys_when_tokenizer_has_observation_tokenizer(
        self,
        policy_factory: Callable[..., Policy],
        encoding_pipeline_factory: Callable[..., MagicMock],
        vision_encoder_factory: Callable[..., MagicMock],
    ):
        encoder = vision_encoder_factory(input_keys=["left"])
        pipeline = encoding_pipeline_factory(encoders={"rgb": encoder})
        policy = policy_factory(encoding_pipeline=pipeline)
        tokenizer = MagicMock(spec=Tokenizer)
        tokenizer.observation_tokenizer = MagicMock()
        policy.tokenizer = tokenizer
        result = policy.input_keys
        assert SampleKey.TOKENIZED_OBSERVATIONS.value in result
        assert SampleKey.IS_PAD_OBSERVATION.value in result

    def test_input_keys_omits_tokenized_keys_when_no_tokenizer(
        self,
        policy_factory: Callable[..., Policy],
        encoding_pipeline_factory: Callable[..., MagicMock],
        vision_encoder_factory: Callable[..., MagicMock],
    ):
        encoder = vision_encoder_factory(input_keys=["left"])
        pipeline = encoding_pipeline_factory(encoders={"rgb": encoder})
        policy = policy_factory(encoding_pipeline=pipeline)
        policy.tokenizer = None
        result = policy.input_keys
        assert SampleKey.TOKENIZED_OBSERVATIONS.value not in result
        assert SampleKey.IS_PAD_OBSERVATION.value not in result

    def test_input_keys_omits_tokenized_keys_when_observation_tokenizer_is_none(
        self,
        policy_factory: Callable[..., Policy],
        encoding_pipeline_factory: Callable[..., MagicMock],
        vision_encoder_factory: Callable[..., MagicMock],
    ):
        encoder = vision_encoder_factory(input_keys=["left"])
        pipeline = encoding_pipeline_factory(encoders={"rgb": encoder})
        policy = policy_factory(encoding_pipeline=pipeline)
        tokenizer = MagicMock(spec=Tokenizer)
        tokenizer.observation_tokenizer = None
        policy.tokenizer = tokenizer
        result = policy.input_keys
        assert SampleKey.TOKENIZED_OBSERVATIONS.value not in result
        assert SampleKey.IS_PAD_OBSERVATION.value not in result

    def test_input_keys_includes_decoder_owned_vlm_observation_keys(
        self,
        policy_factory: Callable[..., Policy],
    ):
        observation_space = MagicMock(
            spec=ObservationSpace,
            observations_metadata={Cameras.LEFT.value: MagicMock()},
        )
        decoder = MagicMock(
            spec=ActionDecoder,
            decoder_input=DecoderInput(
                keys=[
                    Cameras.LEFT.value,
                    SampleKey.TOKENIZED_OBSERVATIONS.value,
                    SampleKey.IS_PAD_OBSERVATION.value,
                ],
                needs_raw_observations=True,
            ),
        )
        decoder.action_heads = {}
        policy = policy_factory(decoder=decoder, observation_space=observation_space)
        assert policy.input_keys == [
            SampleKey.IS_PAD_OBSERVATION.value,
            Cameras.LEFT.value,
            SampleKey.TOKENIZED_OBSERVATIONS.value,
        ]

    def test_output_keys_returns_sorted_decoder_prediction_keys(
        self,
        policy_factory: Callable[..., Policy],
    ):
        decoder = MagicMock(
            spec=ActionDecoder,
            decoder_input=DecoderInput(keys=[]),
        )
        decoder.action_heads = {
            "position": MagicMock(),
            "gripper": MagicMock(),
            "orientation": MagicMock(),
        }
        decoder.get_prediction_output_keys.return_value = [
            "position",
            "gripper",
            "orientation",
        ]
        policy = policy_factory(decoder=decoder)
        assert policy.output_keys == ["position", "gripper", "orientation"]


class TestSetNormalizer:
    def test_loads_normalizer_state_dict(self, policy_factory: Callable[..., Policy]):
        policy = policy_factory()
        normalizer = MagicMock(spec=LinearNormalizer)
        normalizer.state_dict.return_value = {}
        with patch.object(policy.normalizer, "load_state_dict") as mock_load:
            policy.set_normalizer(normalizer=normalizer)
            mock_load.assert_called_once_with(normalizer.state_dict())

    def test_moves_normalizer_to_device(self, policy_factory: Callable[..., Policy]):
        policy = policy_factory(device="cpu")
        normalizer = MagicMock(spec=LinearNormalizer)
        normalizer.state_dict.return_value = {}
        with (
            patch.object(policy.normalizer, "load_state_dict"),
            patch.object(policy.normalizer, "to") as mock_to,
        ):
            policy.set_normalizer(normalizer=normalizer)
            mock_to.assert_called_once_with(policy.device)

    def test_propagates_normalizer_to_decoder(
        self, policy_factory: Callable[..., Policy]
    ):
        decoder = MagicMock(
            spec=ActionDecoder,
            decoder_input=DecoderInput(keys=[]),
        )
        policy = policy_factory(decoder=decoder)
        normalizer = MagicMock(spec=LinearNormalizer)
        normalizer.state_dict.return_value = {}
        with (
            patch.object(policy.normalizer, "load_state_dict"),
            patch.object(policy.normalizer, "to"),
        ):
            policy.set_normalizer(normalizer=normalizer)
        decoder.set_normalizer.assert_called_once_with(policy.normalizer)


class TestSetTokenizer:
    def test_stores_tokenizer(self, policy_factory: Callable[..., Policy]):
        tokenizer = MagicMock(spec=Tokenizer)
        policy = policy_factory()
        policy.set_tokenizer(tokenizer=tokenizer)
        assert policy.tokenizer is tokenizer

    def test_propagates_to_encoding_pipeline(
        self, policy_factory: Callable[..., Policy]
    ):
        tokenizer = MagicMock(spec=Tokenizer)
        policy = policy_factory()
        policy.set_tokenizer(tokenizer=tokenizer)
        policy.encoding_pipeline.set_tokenizer.assert_called_once_with(tokenizer)

    def test_propagates_to_decoder(self, policy_factory: Callable[..., Policy]):
        tokenizer = MagicMock(spec=Tokenizer)
        policy = policy_factory()
        policy.set_tokenizer(tokenizer=tokenizer)
        policy.decoder.set_tokenizer.assert_called_once_with(tokenizer)

    def test_accepts_none_tokenizer(self, policy_factory: Callable[..., Policy]):
        policy = policy_factory()
        policy.set_tokenizer(tokenizer=None)
        assert policy.tokenizer is None
        policy.encoding_pipeline.set_tokenizer.assert_called_once_with(None)
        policy.decoder.set_tokenizer.assert_called_once_with(None)


class TestSetDenoisingThresholds:
    def test_stores_thresholds_as_parameters(
        self, policy_factory: Callable[..., Policy]
    ):
        policy = policy_factory()
        thresholds = {"position": 0.05, "orientation": 0.02}
        policy.set_denoising_thresholds(thresholds=thresholds)
        assert "position" in policy.denoising_thresholds.params_dict
        assert "orientation" in policy.denoising_thresholds.params_dict
        assert policy.denoising_thresholds.params_dict[
            "position"
        ].item() == pytest.approx(0.05)
        assert policy.denoising_thresholds.params_dict[
            "orientation"
        ].item() == pytest.approx(0.02)

    def test_parameters_have_no_gradient(self, policy_factory: Callable[..., Policy]):
        policy = policy_factory()
        policy.set_denoising_thresholds(thresholds={"key": 1.0})
        assert policy.denoising_thresholds.params_dict["key"].requires_grad is False

    def test_empty_thresholds_dict(self, policy_factory: Callable[..., Policy]):
        policy = policy_factory()
        policy.set_denoising_thresholds(thresholds={})
        assert len(policy.denoising_thresholds.params_dict) == 0


class TestSetGripperClassWeights:
    def test_sets_pos_weight_on_gripper_loss_modules(
        self,
        policy_factory: Callable[..., Policy],
        rng: np.random.Generator,
    ):
        gripper_loss = MagicMock(spec=GripperLoss)
        gripper_loss.__class__ = GripperLoss
        loss_module = MagicMock(spec=BaseLoss)
        loss_module.modules.return_value = [loss_module, gripper_loss]
        policy = policy_factory(loss=loss_module)
        pos_weight = torch.from_numpy(rng.standard_normal(1).astype(np.float32))
        policy.set_gripper_class_weights(pos_weight=pos_weight)
        assert gripper_loss.pos_weight is pos_weight

    def test_ignores_non_gripper_loss_modules(
        self,
        policy_factory: Callable[..., Policy],
        rng: np.random.Generator,
    ):
        other_module = MagicMock(spec=BaseLoss)
        original_pos_weight = torch.from_numpy(
            rng.standard_normal(1).astype(np.float32)
        )
        other_module.pos_weight = original_pos_weight
        loss_module = MagicMock(spec=BaseLoss)
        loss_module.modules.return_value = [loss_module, other_module]
        policy = policy_factory(loss=loss_module)
        new_pos_weight = torch.from_numpy(rng.standard_normal(1).astype(np.float32))
        policy.set_gripper_class_weights(pos_weight=new_pos_weight)
        # Non-GripperLoss modules should retain their original pos_weight
        assert torch.equal(other_module.pos_weight, original_pos_weight)

    def test_none_pos_weight_clears_weight(
        self,
        policy_factory: Callable[..., Policy],
        rng: np.random.Generator,
    ):
        gripper_loss = MagicMock(spec=GripperLoss)
        gripper_loss.__class__ = GripperLoss
        gripper_loss.pos_weight = torch.from_numpy(
            rng.standard_normal(1).astype(np.float32)
        )
        loss_module = MagicMock(spec=BaseLoss)
        loss_module.modules.return_value = [loss_module, gripper_loss]
        policy = policy_factory(loss=loss_module)
        policy.set_gripper_class_weights(pos_weight=None)
        assert gripper_loss.pos_weight is None


class TestForward:
    def test_extracts_observations_and_calls_pipeline(
        self,
        policy_factory: Callable[..., Policy],
        batch_dictionary_factory: Callable[..., dict[str, dict[str, torch.Tensor]]],
    ) -> None:
        policy = policy_factory()
        batch = batch_dictionary_factory()
        policy.forward(batch=batch)
        policy.encoding_pipeline.assert_called_once()
        actual_obs = policy.encoding_pipeline.call_args[0][0]
        expected_obs = batch[SampleKey.OBSERVATION.value]
        for key in expected_obs:
            assert torch.equal(actual_obs[key], expected_obs[key])

    def test_strips_metadata_passthrough_observations_before_pipeline(
        self,
        policy_factory: Callable[..., Policy],
        batch_dictionary_factory: Callable[..., dict[str, dict[str, torch.Tensor]]],
    ) -> None:
        policy = policy_factory(
            metadata_passthrough={
                MetadataPassthroughSource.OBSERVATION.value: {
                    SyntheticObsKey.MODE_ID.value: MetadataKey.LATENT_COLOR_LABEL.value
                }
            },
        )
        batch = batch_dictionary_factory()
        mode_id = torch.tensor([[[0]], [[1]]], dtype=torch.long)
        batch[SampleKey.OBSERVATION.value][SyntheticObsKey.MODE_ID.value] = mode_id

        policy.forward(batch=batch)

        actual_obs = policy.encoding_pipeline.call_args.args[0]
        assert SyntheticObsKey.MODE_ID.value not in actual_obs
        assert SyntheticObsKey.MODE_ID.value in batch[SampleKey.OBSERVATION.value]

    def test_preserves_metadata_passthrough_observations_used_by_encoder(
        self,
        policy_factory: Callable[..., Policy],
        batch_dictionary_factory: Callable[..., dict[str, dict[str, torch.Tensor]]],
        encoding_pipeline_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ) -> None:
        encoder = MagicMock()
        encoder.input_specification.keys = [SyntheticObsKey.MODE_ID.value]
        pipeline = encoding_pipeline_factory(encoders={"mode": encoder})
        pipeline.return_value = feature_dictionary_factory()
        policy = policy_factory(
            encoding_pipeline=pipeline,
            metadata_passthrough={
                MetadataPassthroughSource.OBSERVATION.value: {
                    SyntheticObsKey.MODE_ID.value: MetadataKey.LATENT_COLOR_LABEL.value
                }
            },
        )
        batch = batch_dictionary_factory()
        mode_id = torch.tensor([[[0]], [[1]]], dtype=torch.long)
        batch[SampleKey.OBSERVATION.value][SyntheticObsKey.MODE_ID.value] = mode_id

        policy.forward(batch=batch)

        actual_obs = policy.encoding_pipeline.call_args.args[0]
        assert torch.equal(actual_obs[SyntheticObsKey.MODE_ID.value], mode_id)

    def test_passes_features_and_actions_to_algorithm(
        self,
        policy_factory: Callable[..., Policy],
        batch_dictionary_factory: Callable[..., dict[str, dict[str, torch.Tensor]]],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        encoding_pipeline_factory: Callable[..., MagicMock],
    ) -> None:
        features = feature_dictionary_factory()
        pipeline = encoding_pipeline_factory()
        pipeline.return_value = features
        policy = policy_factory(encoding_pipeline=pipeline)
        batch = batch_dictionary_factory()
        policy.forward(batch=batch)
        policy.algorithm.forward.assert_called_once_with(
            features=features,
            actions=batch[SampleKey.ACTION.value],
            network=policy.decoder,
        )

    def test_filters_algorithm_features_to_decoder_input_keys(
        self,
        policy_factory: Callable[..., Policy],
        batch_dictionary_factory: Callable[..., dict[str, dict[str, torch.Tensor]]],
        encoding_pipeline_factory: Callable[..., MagicMock],
    ) -> None:
        kept_feature = torch.ones(2, 4)
        dropped_feature = torch.zeros(2, 4)
        kept_mask = torch.zeros(2, 4, dtype=torch.bool)
        raw_camera = torch.ones(2, 3, 8, 8)
        dropped_raw_camera = torch.zeros(2, 3, 8, 8)
        feature_key = "rgb_features"
        mask_key = f"{feature_key}_{EncoderOutputKeys.PADDING_MASK.value}"
        pipeline = encoding_pipeline_factory()
        pipeline.return_value = {
            feature_key: kept_feature,
            "unused_features": dropped_feature,
            mask_key: kept_mask,
        }
        decoder = MagicMock(
            spec=ActionDecoder,
            decoder_input=DecoderInput(
                keys=[feature_key, Cameras.LEFT.value],
            ),
        )
        decoder.action_heads = {}
        policy = policy_factory(encoding_pipeline=pipeline, decoder=decoder)
        batch = batch_dictionary_factory()
        batch[SampleKey.OBSERVATION.value][Cameras.LEFT.value] = raw_camera
        batch[SampleKey.OBSERVATION.value][Cameras.RIGHT.value] = dropped_raw_camera

        policy.forward(batch=batch)

        actual_features = policy.algorithm.forward.call_args.kwargs["features"]
        assert set(actual_features) == {feature_key, mask_key, Cameras.LEFT.value}
        assert torch.equal(actual_features[feature_key], kept_feature)
        assert torch.equal(actual_features[mask_key], kept_mask)
        assert torch.equal(actual_features[Cameras.LEFT.value], raw_camera)

    def test_passes_raw_observations_to_algorithm_for_decoder_owned_vlm(
        self,
        policy_factory: Callable[..., Policy],
        batch_dictionary_factory: Callable[..., dict[str, dict[str, torch.Tensor]]],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        encoding_pipeline_factory: Callable[..., MagicMock],
    ) -> None:
        features = feature_dictionary_factory(feature_keys=["robot_state_proprio"])
        pipeline = encoding_pipeline_factory()
        pipeline.return_value = features
        decoder = MagicMock(
            spec=ActionDecoder,
            decoder_input=DecoderInput(
                keys=[
                    "robot_state_proprio",
                    Cameras.LEFT.value,
                    SampleKey.TOKENIZED_OBSERVATIONS.value,
                ],
                needs_raw_observations=True,
            ),
        )
        decoder.action_heads = {}
        policy = policy_factory(encoding_pipeline=pipeline, decoder=decoder)
        batch = batch_dictionary_factory()
        batch[SampleKey.OBSERVATION.value][Cameras.LEFT.value] = torch.ones(2, 3, 8, 8)
        batch[SampleKey.OBSERVATION.value][SampleKey.TOKENIZED_OBSERVATIONS.value] = (
            torch.ones(2, 4, dtype=torch.long)
        )

        policy.forward(batch=batch)

        actual_features = policy.algorithm.forward.call_args.kwargs["features"]
        assert torch.equal(
            actual_features["robot_state_proprio"], features["robot_state_proprio"]
        )
        assert torch.equal(
            actual_features[Cameras.LEFT.value],
            batch[SampleKey.OBSERVATION.value][Cameras.LEFT.value],
        )
        assert torch.equal(
            actual_features[SampleKey.TOKENIZED_OBSERVATIONS.value],
            batch[SampleKey.OBSERVATION.value][SampleKey.TOKENIZED_OBSERVATIONS.value],
        )

    def test_returns_algorithm_output(
        self,
        policy_factory: Callable[..., Policy],
        batch_dictionary_factory: Callable[..., dict[str, dict[str, torch.Tensor]]],
    ) -> None:
        expected_output = {"prediction": torch.zeros(2, 4, 7)}
        policy = policy_factory(algorithm_forward_return=expected_output)
        batch = batch_dictionary_factory()
        result = policy.forward(batch=batch)
        assert result is expected_output

    def test_passes_none_actions_when_action_key_absent(
        self,
        policy_factory: Callable[..., Policy],
        observation_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ) -> None:
        policy = policy_factory()
        batch = {SampleKey.OBSERVATION.value: observation_dictionary_factory()}
        policy.forward(batch=batch)
        call_kwargs = policy.algorithm.forward.call_args.kwargs
        assert call_kwargs["actions"] is None


class TestComputeLoss:
    def test_calls_forward_then_loss_module(
        self,
        policy_factory: Callable[..., Policy],
        batch_dictionary_factory: Callable[..., dict[str, dict[str, torch.Tensor]]],
    ):
        forward_output = {"prediction": torch.zeros(2, 4, 7)}
        loss_output = LossOutput(total_loss=torch.tensor(0.5))
        loss_module = MagicMock(spec=BaseLoss)
        loss_module.return_value = loss_output
        policy = policy_factory(
            loss=loss_module, algorithm_forward_return=forward_output
        )
        batch = batch_dictionary_factory()
        result = policy.compute_loss(batch=batch)
        assert result is loss_output
        loss_module.assert_called_once()

    def test_passes_predictions_and_algorithm_targets_to_loss_module(
        self,
        policy_factory: Callable[..., Policy],
        batch_dictionary_factory: Callable[..., dict[str, dict[str, torch.Tensor]]],
    ):
        forward_output = {"prediction": torch.zeros(2, 4, 7)}
        algorithm_targets = {"prediction": torch.ones(2, 4, 7)}
        loss_module = MagicMock(spec=BaseLoss)
        loss_module.return_value = LossOutput(total_loss=torch.tensor(0.1))
        algorithm = MagicMock(spec=DecodingAlgorithm)
        algorithm.forward.return_value = forward_output
        algorithm.get_targets.return_value = algorithm_targets
        policy = policy_factory(loss=loss_module, algorithm=algorithm)
        batch = batch_dictionary_factory()
        policy.compute_loss(batch=batch)
        call_kwargs = loss_module.call_args.kwargs
        assert call_kwargs["predictions"] is forward_output
        assert call_kwargs["targets"] is algorithm_targets
        algorithm.get_targets.assert_called_once_with(
            algorithm_output=forward_output,
            ground_truth_actions=batch[SampleKey.ACTION.value],
        )

    def test_extracts_is_pad_from_action_dictionary(
        self,
        policy_factory: Callable[..., Policy],
        batch_dictionary_factory: Callable[..., dict[str, dict[str, torch.Tensor]]],
    ):
        forward_output = {"prediction": torch.zeros(2, 4, 7)}
        loss_module = MagicMock(spec=BaseLoss)
        loss_module.return_value = LossOutput(total_loss=torch.tensor(0.1))
        policy = policy_factory(
            loss=loss_module, algorithm_forward_return=forward_output
        )
        batch = batch_dictionary_factory()
        is_pad = batch[SampleKey.ACTION.value][SampleKey.IS_PAD_ACTION.value]
        policy.compute_loss(batch=batch)
        call_kwargs = loss_module.call_args.kwargs
        assert torch.equal(call_kwargs["is_pad"], is_pad)

    def test_adds_configured_observation_metadata_to_loss_output(
        self,
        policy_factory: Callable[..., Policy],
        batch_dictionary_factory: Callable[..., dict[str, dict[str, torch.Tensor]]],
    ) -> None:
        loss_output = LossOutput(total_loss=torch.tensor(0.1))
        loss_module = MagicMock(spec=BaseLoss)
        loss_module.return_value = loss_output
        policy = policy_factory(
            loss=loss_module,
            metadata_passthrough={
                MetadataPassthroughSource.OBSERVATION.value: {
                    SyntheticObsKey.MODE_ID.value: MetadataKey.LATENT_COLOR_LABEL.value
                }
            },
        )
        batch = batch_dictionary_factory()
        mode_id = torch.tensor([[[0]], [[1]]], dtype=torch.long)
        batch[SampleKey.OBSERVATION.value][SyntheticObsKey.MODE_ID.value] = mode_id

        result = policy.compute_loss(batch=batch)

        assert torch.equal(
            result.metadata[MetadataKey.LATENT_COLOR_LABEL.value], mode_id
        )

    def test_adds_hydra_configured_observation_metadata_to_loss_output(
        self,
        policy_factory: Callable[..., Policy],
        batch_dictionary_factory: Callable[..., dict[str, dict[str, torch.Tensor]]],
    ) -> None:
        loss_output = LossOutput(total_loss=torch.tensor(0.1))
        loss_module = MagicMock(spec=BaseLoss)
        loss_module.return_value = loss_output
        policy = policy_factory(
            loss=loss_module,
            metadata_passthrough=OmegaConf.create(
                {
                    "${metadata_passthrough_source:OBSERVATION}": {
                        "${synthetic_obs_key:MODE_ID}": (
                            MetadataKey.LATENT_COLOR_LABEL.value
                        )
                    }
                }
            ),
        )
        batch = batch_dictionary_factory()
        mode_id = torch.tensor([[[0]], [[1]]], dtype=torch.long)
        batch[SampleKey.OBSERVATION.value][SyntheticObsKey.MODE_ID.value] = mode_id

        result = policy.compute_loss(batch=batch)

        assert torch.equal(
            result.metadata[MetadataKey.LATENT_COLOR_LABEL.value], mode_id
        )

    def test_adds_configured_prediction_metadata_to_loss_output(
        self,
        policy_factory: Callable[..., Policy],
        batch_dictionary_factory: Callable[..., dict[str, dict[str, torch.Tensor]]],
    ) -> None:
        prediction_metadata = torch.tensor([0, 1])
        forward_output = {
            "prediction": torch.zeros(2, 4, 7),
            "predicted_label": prediction_metadata,
        }
        loss_module = MagicMock(spec=BaseLoss)
        loss_module.return_value = LossOutput(total_loss=torch.tensor(0.1))
        policy = policy_factory(
            loss=loss_module,
            algorithm_forward_return=forward_output,
            metadata_passthrough={
                MetadataPassthroughSource.PREDICTION.value: {
                    "predicted_label": MetadataKey.LATENT_COLOR_LABEL.value
                }
            },
        )
        batch = batch_dictionary_factory()

        result = policy.compute_loss(batch=batch)

        assert torch.equal(
            result.metadata[MetadataKey.LATENT_COLOR_LABEL.value],
            prediction_metadata,
        )

    def test_rejects_unknown_metadata_passthrough_source(
        self,
        policy_factory: Callable[..., Policy],
    ) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Unknown metadata passthrough source 'targets'. "
                "Valid sources: ['action', 'observation', 'prediction']."
            ),
        ):
            policy_factory(metadata_passthrough={"targets": {"x": "y"}})


class TestPredictAction:
    @patch("versatil.models.policy.unnormalize_actions")
    @patch("versatil.models.policy.normalize_observation")
    @patch("versatil.models.policy.to_device")
    def test_calls_normalize_encode_predict_unnormalize(
        self,
        mock_to_device,
        mock_normalize,
        mock_unnormalize,
        policy_factory: Callable[..., Policy],
        observation_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        observation = observation_dictionary_factory()
        normalized_observation = observation_dictionary_factory()
        features = feature_dictionary_factory()
        predictions = {"position": torch.zeros(2, 4, 3)}
        unnormalized = {"position": torch.ones(2, 4, 3)}
        mock_to_device.side_effect = lambda x, device: x
        mock_normalize.return_value = normalized_observation
        mock_unnormalize.return_value = unnormalized
        algorithm = MagicMock(spec=DecodingAlgorithm)
        algorithm.predict.return_value = predictions
        policy = policy_factory(
            feature_return_value=features,
            algorithm=algorithm,
        )

        result = policy.predict_action(obs_dict=observation)

        mock_normalize.assert_called_once()
        policy.encoding_pipeline.assert_called_once()
        actual_obs = policy.encoding_pipeline.call_args[0][0]
        for key in normalized_observation:
            assert torch.equal(actual_obs[key], normalized_observation[key])
        algorithm.predict.assert_called_once_with(
            features=features, network=policy.decoder
        )
        mock_unnormalize.assert_called_once()
        assert result is unnormalized

    @patch("versatil.models.policy.unnormalize_actions")
    @patch("versatil.models.policy.normalize_observation")
    @patch("versatil.models.policy.to_device")
    def test_passes_raw_observations_to_predict_for_decoder_owned_vlm(
        self,
        mock_to_device: MagicMock,
        mock_normalize: MagicMock,
        mock_unnormalize: MagicMock,
        policy_factory: Callable[..., Policy],
        observation_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        encoding_pipeline_factory: Callable[..., MagicMock],
    ) -> None:
        observation = observation_dictionary_factory()
        normalized_observation = observation_dictionary_factory()
        normalized_observation[Cameras.LEFT.value] = torch.ones(2, 3, 8, 8)
        normalized_observation[SampleKey.TOKENIZED_OBSERVATIONS.value] = torch.ones(
            2, 4, dtype=torch.long
        )
        features = feature_dictionary_factory(feature_keys=["robot_state_proprio"])
        pipeline = encoding_pipeline_factory()
        pipeline.return_value = features
        mock_to_device.side_effect = lambda x, device: x
        mock_normalize.return_value = normalized_observation
        mock_unnormalize.return_value = {}
        decoder = MagicMock(
            spec=ActionDecoder,
            decoder_input=DecoderInput(
                keys=[
                    "robot_state_proprio",
                    Cameras.LEFT.value,
                    SampleKey.TOKENIZED_OBSERVATIONS.value,
                ],
                needs_raw_observations=True,
            ),
        )
        decoder.action_heads = {}
        policy = policy_factory(encoding_pipeline=pipeline, decoder=decoder)
        policy.algorithm.predict.return_value = {}

        policy.predict_action(obs_dict=observation)

        actual_features = policy.algorithm.predict.call_args.kwargs["features"]
        assert torch.equal(
            actual_features["robot_state_proprio"], features["robot_state_proprio"]
        )
        assert torch.equal(
            actual_features[Cameras.LEFT.value],
            normalized_observation[Cameras.LEFT.value],
        )
        assert torch.equal(
            actual_features[SampleKey.TOKENIZED_OBSERVATIONS.value],
            normalized_observation[SampleKey.TOKENIZED_OBSERVATIONS.value],
        )

    @patch("versatil.models.policy.unnormalize_actions")
    @patch("versatil.models.policy.normalize_observation")
    @patch("versatil.models.policy.to_device")
    def test_strips_metadata_passthrough_observations_before_predict_pipeline(
        self,
        mock_to_device: MagicMock,
        mock_normalize: MagicMock,
        mock_unnormalize: MagicMock,
        policy_factory: Callable[..., Policy],
        observation_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ) -> None:
        observation = observation_dictionary_factory()
        normalized_observation = observation_dictionary_factory()
        normalized_observation[SyntheticObsKey.MODE_ID.value] = torch.tensor(
            [[[0]], [[1]]], dtype=torch.long
        )
        mock_to_device.side_effect = lambda x, device: x
        mock_normalize.return_value = normalized_observation
        mock_unnormalize.return_value = {}
        policy = policy_factory(
            metadata_passthrough={
                MetadataPassthroughSource.OBSERVATION.value: {
                    SyntheticObsKey.MODE_ID.value: MetadataKey.LATENT_COLOR_LABEL.value
                }
            },
        )
        policy.algorithm.predict.return_value = {}

        policy.predict_action(obs_dict=observation)

        actual_obs = policy.encoding_pipeline.call_args.args[0]
        assert SyntheticObsKey.MODE_ID.value not in actual_obs
        assert SyntheticObsKey.MODE_ID.value in normalized_observation

    @patch("versatil.models.policy.unnormalize_actions")
    @patch("versatil.models.policy.tokenize_observation")
    @patch("versatil.models.policy.normalize_observation")
    @patch("versatil.models.policy.to_device")
    def test_tokenizes_observation_when_tokenizer_set(
        self,
        mock_to_device,
        mock_normalize,
        mock_tokenize,
        mock_unnormalize,
        policy_factory: Callable[..., Policy],
        observation_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        observation = observation_dictionary_factory()
        normalized_observation = observation_dictionary_factory()
        tokenized_observation = observation_dictionary_factory()
        mock_to_device.side_effect = lambda x, device: x
        mock_normalize.return_value = normalized_observation
        mock_tokenize.return_value = tokenized_observation
        mock_unnormalize.return_value = {}
        policy = policy_factory()
        tokenizer = MagicMock(spec=Tokenizer)
        tokenizer.observation_tokenizer = MagicMock()
        tokenizer.action_tokenizer = None
        policy.tokenizer = tokenizer
        policy.algorithm.predict.return_value = {}

        policy.predict_action(obs_dict=observation)

        mock_tokenize.assert_called_once_with(
            observation=normalized_observation,
            obs_tokenizer=tokenizer.observation_tokenizer,
        )

    @patch("versatil.models.policy.unnormalize_actions")
    @patch("versatil.models.policy.detokenize_actions")
    @patch("versatil.models.policy.to_device")
    @patch("versatil.models.policy.normalize_observation")
    def test_detokenizes_actions_when_tokenized_key_present(
        self,
        mock_normalize,
        mock_to_device_function,
        mock_detokenize,
        mock_unnormalize,
        policy_factory: Callable[..., Policy],
        observation_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        observation = observation_dictionary_factory()
        mock_normalize.return_value = observation
        mock_to_device_function.side_effect = lambda x, device: x
        detokenized = {"position": torch.zeros(2, 4, 3)}
        mock_detokenize.return_value = detokenized
        mock_unnormalize.return_value = {}
        policy = policy_factory()
        tokenizer = MagicMock(spec=Tokenizer)
        tokenizer.observation_tokenizer = None
        tokenizer.action_tokenizer = MagicMock()
        policy.tokenizer = tokenizer
        policy.algorithm.predict.return_value = {
            DecoderOutputKey.PREDICTED_ACTION_TOKENS.value: torch.zeros(2, 4),
        }

        policy.predict_action(obs_dict=observation)

        mock_detokenize.assert_called_once()

    @patch("versatil.models.policy.unnormalize_actions")
    @patch("versatil.models.policy.normalize_observation")
    @patch("versatil.models.policy.to_device")
    def test_skips_tokenization_when_observation_tokenizer_is_none(
        self,
        mock_to_device,
        mock_normalize,
        mock_unnormalize,
        policy_factory: Callable[..., Policy],
        observation_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        observation = observation_dictionary_factory()
        normalized_observation = observation_dictionary_factory()
        mock_to_device.side_effect = lambda x, device: x
        mock_normalize.return_value = normalized_observation
        mock_unnormalize.return_value = {}
        policy = policy_factory()
        tokenizer = MagicMock(spec=Tokenizer)
        tokenizer.observation_tokenizer = None
        tokenizer.action_tokenizer = None
        policy.tokenizer = tokenizer
        policy.algorithm.predict.return_value = {}
        with patch("versatil.models.policy.tokenize_observation") as mock_tokenize:
            policy.predict_action(obs_dict=observation)
            mock_tokenize.assert_not_called()

    @patch("versatil.models.policy.unnormalize_actions")
    @patch("versatil.models.policy.normalize_observation")
    @patch("versatil.models.policy.to_device")
    def test_raises_when_tokenized_actions_without_action_tokenizer(
        self,
        mock_to_device,
        mock_normalize,
        mock_unnormalize,
        policy_factory: Callable[..., Policy],
        observation_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        observation = observation_dictionary_factory()
        mock_to_device.side_effect = lambda x, device: x
        mock_normalize.return_value = observation
        policy = policy_factory()
        tokenizer = MagicMock(spec=Tokenizer)
        tokenizer.observation_tokenizer = None
        tokenizer.action_tokenizer = None
        policy.tokenizer = tokenizer
        policy.algorithm.predict.return_value = {
            DecoderOutputKey.PREDICTED_ACTION_TOKENS.value: torch.zeros(2, 4),
        }
        with pytest.raises(
            RuntimeError,
            match=re.escape("Action tokenizer not set. Cannot detokenize actions."),
        ):
            policy.predict_action(obs_dict=observation)

    @patch("versatil.models.policy.unnormalize_actions")
    @patch("versatil.models.policy.normalize_observation")
    @patch("versatil.models.policy.to_device")
    def test_raises_when_tokenized_actions_without_tokenizer(
        self,
        mock_to_device,
        mock_normalize,
        mock_unnormalize,
        policy_factory: Callable[..., Policy],
        observation_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        observation = observation_dictionary_factory()
        mock_to_device.side_effect = lambda x, device: x
        mock_normalize.return_value = observation
        policy = policy_factory()
        policy.tokenizer = None
        policy.algorithm.predict.return_value = {
            DecoderOutputKey.PREDICTED_ACTION_TOKENS.value: torch.zeros(2, 4),
        }
        with pytest.raises(
            RuntimeError,
            match=re.escape("Action tokenizer not set. Cannot detokenize actions."),
        ):
            policy.predict_action(obs_dict=observation)
