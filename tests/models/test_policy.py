"""Tests for versatil.models.policy module."""
import re
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from versatil.data.constants import Cameras, SampleKey
from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.tokenization import Tokenizer
from versatil.metrics.base import BaseLoss, LossOutput
from versatil.metrics.components import GripperLoss
from versatil.models.decoding.algorithm.base import DecodingAlgorithm
from versatil.models.decoding.constants import DecoderOutputKey
from versatil.models.decoding.decoders.base import ActionDecoder
from versatil.models.policy import Policy


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
        decoder = MagicMock(spec=ActionDecoder)
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

    def test_converts_device_string_to_torch_device(self, policy_factory: Callable[..., Policy]):
        policy = policy_factory(device="cpu")
        assert policy.device.type == "cpu"

    def test_normalizer_starts_empty(self, policy_factory: Callable[..., Policy]):
        policy = policy_factory()
        assert len(list(policy.normalizer.parameters())) == 0

    def test_tokenizer_starts_unset(self, policy_factory: Callable[..., Policy]):
        policy = policy_factory()
        assert policy.tokenizer is None


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
        with patch.object(policy.normalizer, "load_state_dict"):
            with patch.object(policy.normalizer, "to") as mock_to:
                policy.set_normalizer(normalizer=normalizer)
                mock_to.assert_called_once_with(policy.device)

    def test_propagates_normalizer_to_decoder(self, policy_factory: Callable[..., Policy]):
        decoder = MagicMock(spec=ActionDecoder)
        policy = policy_factory(decoder=decoder)
        normalizer = MagicMock(spec=LinearNormalizer)
        normalizer.state_dict.return_value = {}
        with patch.object(policy.normalizer, "load_state_dict"):
            with patch.object(policy.normalizer, "to"):
                policy.set_normalizer(normalizer=normalizer)
        decoder.set_normalizer.assert_called_once_with(policy.normalizer)


class TestSetTokenizer:

    def test_stores_tokenizer(self, policy_factory: Callable[..., Policy]):
        tokenizer = MagicMock(spec=Tokenizer)
        policy = policy_factory()
        policy.set_tokenizer(tokenizer=tokenizer)
        assert policy.tokenizer is tokenizer

    def test_propagates_to_encoding_pipeline(self, policy_factory: Callable[..., Policy]):
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

    def test_stores_thresholds_as_parameters(self, policy_factory: Callable[..., Policy]):
        policy = policy_factory()
        thresholds = {"position": 0.05, "orientation": 0.02}
        policy.set_denoising_thresholds(thresholds=thresholds)
        assert "position" in policy.denoising_thresholds.params_dict
        assert "orientation" in policy.denoising_thresholds.params_dict
        assert policy.denoising_thresholds.params_dict["position"].item() == pytest.approx(0.05)
        assert policy.denoising_thresholds.params_dict["orientation"].item() == pytest.approx(0.02)

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
        original_pos_weight = torch.from_numpy(rng.standard_normal(1).astype(np.float32))
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
        gripper_loss.pos_weight = torch.from_numpy(rng.standard_normal(1).astype(np.float32))
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
    ):
        policy = policy_factory()
        batch = batch_dictionary_factory()
        policy.forward(batch=batch)
        policy.encoding_pipeline.assert_called_once_with(
            batch[SampleKey.OBSERVATION.value],
        )

    def test_passes_features_and_actions_to_algorithm(
        self,
        policy_factory: Callable[..., Policy],
        batch_dictionary_factory: Callable[..., dict[str, dict[str, torch.Tensor]]],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        encoding_pipeline_factory: Callable[..., MagicMock],
    ):
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

    def test_returns_algorithm_output(
        self,
        policy_factory: Callable[..., Policy],
        batch_dictionary_factory: Callable[..., dict[str, dict[str, torch.Tensor]]],
    ):
        expected_output = {"prediction": torch.zeros(2, 4, 7)}
        policy = policy_factory(algorithm_forward_return=expected_output)
        batch = batch_dictionary_factory()
        result = policy.forward(batch=batch)
        assert result is expected_output

    def test_passes_none_actions_when_action_key_absent(
        self,
        policy_factory: Callable[..., Policy],
        observation_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
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
        policy = policy_factory(loss=loss_module, algorithm_forward_return=forward_output)
        batch = batch_dictionary_factory()
        result = policy.compute_loss(batch=batch)
        assert result is loss_output
        loss_module.assert_called_once()

    def test_passes_predictions_and_targets_to_loss_module(
        self,
        policy_factory: Callable[..., Policy],
        batch_dictionary_factory: Callable[..., dict[str, dict[str, torch.Tensor]]],
    ):
        forward_output = {"prediction": torch.zeros(2, 4, 7)}
        loss_module = MagicMock(spec=BaseLoss)
        loss_module.return_value = LossOutput(total_loss=torch.tensor(0.1))
        policy = policy_factory(loss=loss_module, algorithm_forward_return=forward_output)
        batch = batch_dictionary_factory()
        policy.compute_loss(batch=batch)
        call_kwargs = loss_module.call_args.kwargs
        assert call_kwargs["predictions"] is forward_output
        assert call_kwargs["targets"] is batch[SampleKey.ACTION.value]

    def test_extracts_is_pad_from_action_dictionary(
        self,
        policy_factory: Callable[..., Policy],
        batch_dictionary_factory: Callable[..., dict[str, dict[str, torch.Tensor]]],
    ):
        forward_output = {"prediction": torch.zeros(2, 4, 7)}
        loss_module = MagicMock(spec=BaseLoss)
        loss_module.return_value = LossOutput(total_loss=torch.tensor(0.1))
        policy = policy_factory(loss=loss_module, algorithm_forward_return=forward_output)
        batch = batch_dictionary_factory()
        is_pad = batch[SampleKey.ACTION.value][SampleKey.IS_PAD_ACTION.value]
        policy.compute_loss(batch=batch)
        call_kwargs = loss_module.call_args.kwargs
        assert torch.equal(call_kwargs["is_pad"], is_pad)


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
        policy.encoding_pipeline.assert_called_once_with(normalized_observation)
        algorithm.predict.assert_called_once_with(features=features, network=policy.decoder)
        mock_unnormalize.assert_called_once()
        assert result is unnormalized

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


class TestGetVisionEncoderModules:

    def test_finds_encoder_with_backbone(
        self,
        policy_factory: Callable[..., Policy],
        vision_encoder_factory: Callable[..., MagicMock],
        encoding_pipeline_factory: Callable[..., MagicMock],
    ):
        encoder = vision_encoder_factory(has_backbone=True)
        pipeline = encoding_pipeline_factory(encoders={"rgb_encoder": encoder})
        policy = policy_factory(encoding_pipeline=pipeline)
        result = policy.get_vision_encoder_modules()
        assert "rgb_encoder" in result
        assert result["rgb_encoder"] is encoder

    def test_finds_encoder_with_stages(
        self,
        policy_factory: Callable[..., Policy],
        vision_encoder_factory: Callable[..., MagicMock],
        encoding_pipeline_factory: Callable[..., MagicMock],
    ):
        encoder = vision_encoder_factory(has_stages=True)
        encoder.stages = [MagicMock()]
        pipeline = encoding_pipeline_factory(encoders={"dformer_encoder": encoder})
        policy = policy_factory(encoding_pipeline=pipeline)
        result = policy.get_vision_encoder_modules()
        assert "dformer_encoder" in result

    def test_finds_encoder_with_layer4(
        self,
        policy_factory: Callable[..., Policy],
        vision_encoder_factory: Callable[..., MagicMock],
        encoding_pipeline_factory: Callable[..., MagicMock],
    ):
        encoder = vision_encoder_factory(has_layer4=True)
        pipeline = encoding_pipeline_factory(conditional_encoders={"film_encoder": encoder})
        policy = policy_factory(encoding_pipeline=pipeline)
        result = policy.get_vision_encoder_modules()
        assert "film_encoder" in result

    def test_finds_encoder_with_attention_block(
        self,
        policy_factory: Callable[..., Policy],
        vision_encoder_factory: Callable[..., MagicMock],
        encoding_pipeline_factory: Callable[..., MagicMock],
    ):
        encoder = vision_encoder_factory(has_attention_block=True)
        pipeline = encoding_pipeline_factory(encoders={"light_geo": encoder})
        policy = policy_factory(encoding_pipeline=pipeline)
        result = policy.get_vision_encoder_modules()
        assert "light_geo" in result

    def test_raises_when_no_vision_encoders(
        self,
        policy_factory: Callable[..., Policy],
        vision_encoder_factory: Callable[..., MagicMock],
        encoding_pipeline_factory: Callable[..., MagicMock],
    ):
        encoder = vision_encoder_factory()
        pipeline = encoding_pipeline_factory(encoders={"proprio_encoder": encoder})
        policy = policy_factory(encoding_pipeline=pipeline)
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "No compatible vision encoders found in the encoding pipeline. "
                "Explainer requires encoders that produce spatial feature maps "
                "(CNNEncoder, DepthCNNEncoder, ConditionalCNNEncoder, DFormerEncoder, LightGeometricEncoder). "
                "Available encoders: ['proprio_encoder']"
            ),
        ):
            policy.get_vision_encoder_modules()


class TestGetGradcamTargetLayers:

    def test_returns_layer4_for_timm_resnet_backbone(
        self,
        policy_factory: Callable[..., Policy],
        vision_encoder_factory: Callable[..., MagicMock],
        encoding_pipeline_factory: Callable[..., MagicMock],
    ):
        actual_backbone = MagicMock()
        actual_backbone.layer4 = MagicMock()
        backbone = MagicMock()
        backbone._backbone = actual_backbone
        encoder = vision_encoder_factory(has_backbone=True)
        encoder.backbone = backbone
        pipeline = encoding_pipeline_factory(encoders={"rgb": encoder})
        policy = policy_factory(encoding_pipeline=pipeline)
        result = policy.get_gradcam_target_layers(encoder_name="rgb")
        assert result == [actual_backbone.layer4]

    def test_returns_last_stage_for_timm_stages_backbone(
        self,
        policy_factory: Callable[..., Policy],
        vision_encoder_factory: Callable[..., MagicMock],
        encoding_pipeline_factory: Callable[..., MagicMock],
    ):
        actual_backbone = MagicMock()
        del actual_backbone.layer4
        actual_backbone.stages = [MagicMock(), MagicMock()]
        backbone = MagicMock()
        backbone._backbone = actual_backbone
        encoder = vision_encoder_factory(has_backbone=True)
        encoder.backbone = backbone
        pipeline = encoding_pipeline_factory(encoders={"rgb": encoder})
        policy = policy_factory(encoding_pipeline=pipeline)
        result = policy.get_gradcam_target_layers(encoder_name="rgb")
        assert result == [actual_backbone.stages[-1]]

    def test_returns_last_stage_for_dformer_encoder(
        self,
        policy_factory: Callable[..., Policy],
        vision_encoder_factory: Callable[..., MagicMock],
        encoding_pipeline_factory: Callable[..., MagicMock],
    ):
        encoder = vision_encoder_factory(has_stages=True)
        encoder.stages = [MagicMock(), MagicMock(), MagicMock()]
        pipeline = encoding_pipeline_factory(encoders={"dformer": encoder})
        policy = policy_factory(encoding_pipeline=pipeline)
        result = policy.get_gradcam_target_layers(encoder_name="dformer")
        assert result == [encoder.stages[-1]]

    def test_returns_last_block_for_film_encoder(
        self,
        policy_factory: Callable[..., Policy],
        vision_encoder_factory: Callable[..., MagicMock],
        encoding_pipeline_factory: Callable[..., MagicMock],
    ):
        encoder = vision_encoder_factory(has_layer4=True)
        encoder.layer4 = [MagicMock(), MagicMock()]
        pipeline = encoding_pipeline_factory(conditional_encoders={"film": encoder})
        policy = policy_factory(encoding_pipeline=pipeline)
        result = policy.get_gradcam_target_layers(encoder_name="film")
        assert result == [encoder.layer4[-1]]

    def test_returns_attention_block_for_light_geometric(
        self,
        policy_factory: Callable[..., Policy],
        vision_encoder_factory: Callable[..., MagicMock],
        encoding_pipeline_factory: Callable[..., MagicMock],
    ):
        encoder = vision_encoder_factory(has_attention_block=True)
        pipeline = encoding_pipeline_factory(encoders={"light_geo": encoder})
        policy = policy_factory(encoding_pipeline=pipeline)
        result = policy.get_gradcam_target_layers(encoder_name="light_geo")
        assert result == [encoder.attention_block]

    def test_returns_last_stage_for_direct_backbone_stages(
        self,
        policy_factory: Callable[..., Policy],
        vision_encoder_factory: Callable[..., MagicMock],
        encoding_pipeline_factory: Callable[..., MagicMock],
    ):
        backbone = MagicMock()
        del backbone._backbone
        backbone.stages = [MagicMock(), MagicMock()]
        encoder = vision_encoder_factory(has_backbone=True)
        encoder.backbone = backbone
        pipeline = encoding_pipeline_factory(encoders={"rgb": encoder})
        policy = policy_factory(encoding_pipeline=pipeline)
        result = policy.get_gradcam_target_layers(encoder_name="rgb")
        assert result == [backbone.stages[-1]]

    def test_raises_for_unrecognized_timm_backbone_with_wrapper(
        self,
        policy_factory: Callable[..., Policy],
        vision_encoder_factory: Callable[..., MagicMock],
        encoding_pipeline_factory: Callable[..., MagicMock],
    ):
        actual_backbone = MagicMock()
        del actual_backbone.layer4
        del actual_backbone.stages
        backbone = MagicMock()
        backbone._backbone = actual_backbone
        encoder = vision_encoder_factory(has_backbone=True)
        encoder.backbone = backbone
        pipeline = encoding_pipeline_factory(encoders={"rgb": encoder})
        policy = policy_factory(encoding_pipeline=pipeline)
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                f"Encoder 'rgb' backbone structure not recognized. "
                f"Backbone type: {type(actual_backbone).__name__}"
            ),
        ):
            policy.get_gradcam_target_layers(encoder_name="rgb")

    def test_raises_for_unrecognized_direct_backbone(
        self,
        policy_factory: Callable[..., Policy],
        vision_encoder_factory: Callable[..., MagicMock],
        encoding_pipeline_factory: Callable[..., MagicMock],
    ):
        backbone = MagicMock()
        del backbone._backbone
        del backbone.stages
        encoder = vision_encoder_factory(has_backbone=True)
        encoder.backbone = backbone
        pipeline = encoding_pipeline_factory(encoders={"rgb": encoder})
        policy = policy_factory(encoding_pipeline=pipeline)
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                f"Encoder 'rgb' has backbone but structure not recognized. "
                f"Backbone type: {type(backbone).__name__}"
            ),
        ):
            policy.get_gradcam_target_layers(encoder_name="rgb")

    def test_raises_for_unknown_encoder_name(
        self,
        policy_factory: Callable[..., Policy],
        vision_encoder_factory: Callable[..., MagicMock],
        encoding_pipeline_factory: Callable[..., MagicMock],
    ):
        encoder = vision_encoder_factory(has_backbone=True)
        pipeline = encoding_pipeline_factory(encoders={"rgb": encoder})
        policy = policy_factory(encoding_pipeline=pipeline)
        vision_encoder_keys = list(policy.get_vision_encoder_modules().keys())
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Encoder 'nonexistent' not found or not a vision encoder. "
                f"Available vision encoders: {vision_encoder_keys}"
            ),
        ):
            policy.get_gradcam_target_layers(encoder_name="nonexistent")


class TestGetCameraToEncoderMapping:

    def test_maps_camera_keys_to_encoder_names(
        self,
        policy_factory: Callable[..., Policy],
        vision_encoder_factory: Callable[..., MagicMock],
        encoding_pipeline_factory: Callable[..., MagicMock],
    ):
        encoder = vision_encoder_factory(
            has_backbone=True, input_keys=[Cameras.LEFT.value],
        )
        pipeline = encoding_pipeline_factory(encoders={"rgb_encoder": encoder})
        policy = policy_factory(encoding_pipeline=pipeline)
        result = policy.get_camera_to_encoder_mapping()
        assert result == {Cameras.LEFT.value: "rgb_encoder"}

    def test_excludes_non_camera_keys(
        self,
        policy_factory: Callable[..., Policy],
        vision_encoder_factory: Callable[..., MagicMock],
        encoding_pipeline_factory: Callable[..., MagicMock],
    ):
        encoder = vision_encoder_factory(
            has_backbone=True, input_keys=[Cameras.LEFT.value, "proprio_robot_frame"],
        )
        pipeline = encoding_pipeline_factory(encoders={"rgb_encoder": encoder})
        policy = policy_factory(encoding_pipeline=pipeline)
        result = policy.get_camera_to_encoder_mapping()
        assert "proprio_robot_frame" not in result
        assert Cameras.LEFT.value in result

    def test_skips_encoders_without_input_specification(
        self,
        policy_factory: Callable[..., Policy],
        vision_encoder_factory: Callable[..., MagicMock],
        encoding_pipeline_factory: Callable[..., MagicMock],
    ):
        encoder_with_spec = vision_encoder_factory(
            has_backbone=True, input_keys=[Cameras.RIGHT.value],
        )
        encoder_without_spec = vision_encoder_factory(has_backbone=True)
        del encoder_without_spec.input_specification
        pipeline = encoding_pipeline_factory(
            encoders={"with_spec": encoder_with_spec, "no_spec": encoder_without_spec},
        )
        policy = policy_factory(encoding_pipeline=pipeline)
        result = policy.get_camera_to_encoder_mapping()
        assert result == {Cameras.RIGHT.value: "with_spec"}
        assert "no_spec" not in result.values()

    def test_raises_when_no_camera_mappings(
        self,
        policy_factory: Callable[..., Policy],
        vision_encoder_factory: Callable[..., MagicMock],
        encoding_pipeline_factory: Callable[..., MagicMock],
    ):
        encoder = vision_encoder_factory(
            has_backbone=True, input_keys=["proprio_robot_frame"],
        )
        pipeline = encoding_pipeline_factory(encoders={"proprio_encoder": encoder})
        policy = policy_factory(encoding_pipeline=pipeline)
        valid_camera_keys = {cam.value for cam in Cameras}
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                f"No camera-to-encoder mappings found. "
                f"Valid camera keys: {valid_camera_keys}. "
                f"Vision encoders: ['proprio_encoder']"
            ),
        ):
            policy.get_camera_to_encoder_mapping()