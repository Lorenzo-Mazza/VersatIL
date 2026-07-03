"""Tests for versatil.models.decoding.decoders.factory.openvla_oft module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import PretrainedConfig

from versatil.data.constants import Cameras, SampleKey
from versatil.models.decoding.action_heads.single_output import ActionHead
from versatil.models.decoding.constants import AlgorithmContextKey
from versatil.models.decoding.decoders.factory.openvla_oft import (
    JOINT_ACTION_HEAD_KEY,
    OpenVLAOFTDecoder,
)
from versatil.models.decoding.generative_language_models.base import (
    CausalLanguageModelOutput,
)
from versatil.models.decoding.generative_language_models.vision_language.base import (
    GenerativeVLM,
)
from versatil.models.decoding.generative_language_models.vision_language.prismatic import (
    PrismaticVLM,
)
from versatil.models.input_specification import InputSpecification

BATCH_SIZE = 2
PREFIX_TOKEN_LENGTH = 5
ACTION_DIMENSION = 3
PREDICTION_HORIZON = 4
LANGUAGE_HIDDEN_DIMENSION = 16
CAMERA_KEY = "agentview"


@pytest.fixture
def vlm_backbone_factory() -> Callable[..., MagicMock]:
    def factory() -> MagicMock:
        vlm_backbone = MagicMock(spec=GenerativeVLM)
        vlm_backbone.hidden_dimension = LANGUAGE_HIDDEN_DIMENSION
        vlm_backbone.total_image_tokens = 2
        vlm_backbone.max_text_length = PREFIX_TOKEN_LENGTH
        vlm_backbone.get_text_config.return_value = PretrainedConfig(
            max_position_embeddings=128,
        )
        vlm_backbone.input_specification = InputSpecification(
            keys=[
                CAMERA_KEY,
                SampleKey.TOKENIZED_OBSERVATIONS.value,
                SampleKey.IS_PAD_OBSERVATION.value,
            ],
            required=[SampleKey.TOKENIZED_OBSERVATIONS.value],
            requires_tokenized=True,
        )
        vlm_backbone.build_prefix.return_value = (
            torch.zeros(BATCH_SIZE, PREFIX_TOKEN_LENGTH, LANGUAGE_HIDDEN_DIMENSION),
            torch.zeros(BATCH_SIZE, PREFIX_TOKEN_LENGTH, dtype=torch.bool),
        )
        language_output = MagicMock(spec=CausalLanguageModelOutput)
        language_output.hidden_states = (
            torch.ones(BATCH_SIZE, PREFIX_TOKEN_LENGTH, LANGUAGE_HIDDEN_DIMENSION),
        )
        vlm_backbone.forward_language_model.return_value = language_output
        return vlm_backbone

    return factory


@pytest.fixture
def openvla_oft_decoder_factory(
    mock_action_space_factory: Callable[..., MagicMock],
    mock_observation_space_factory: Callable[..., MagicMock],
    action_head_factory: Callable[..., ActionHead],
    vlm_backbone_factory: Callable[..., MagicMock],
) -> Callable[..., OpenVLAOFTDecoder]:
    def factory(
        input_keys: list[str] | None = None,
        vlm_backbone: MagicMock | None = None,
        slots_per_action_dimension: bool = True,
        action_head_input_dim: int | None = None,
        has_orientation: bool = False,
        orientation_dim: int = 0,
        has_gripper: bool = False,
        gripper_dim: int = 0,
        causal_action_slots: bool = True,
    ) -> OpenVLAOFTDecoder:
        if input_keys is None:
            input_keys = []
        if vlm_backbone is None:
            vlm_backbone = vlm_backbone_factory()
        action_space = mock_action_space_factory(
            position_dim=ACTION_DIMENSION,
            has_orientation=has_orientation,
            orientation_dim=orientation_dim,
            has_gripper=has_gripper,
            gripper_dim=gripper_dim,
        )
        if action_head_input_dim is None:
            action_head_input_dim = (
                action_space.get_total_action_dim() * LANGUAGE_HIDDEN_DIMENSION
                if slots_per_action_dimension
                else LANGUAGE_HIDDEN_DIMENSION
            )
        action_heads = {
            JOINT_ACTION_HEAD_KEY: action_head_factory(
                input_dimension=action_head_input_dim
            )
        }
        return OpenVLAOFTDecoder(
            action_heads=action_heads,
            input_keys=input_keys,
            action_space=action_space,
            observation_space=mock_observation_space_factory(),
            observation_horizon=1,
            prediction_horizon=PREDICTION_HORIZON,
            device="cpu",
            vlm_backbone=vlm_backbone,
            slots_per_action_dimension=slots_per_action_dimension,
            causal_action_slots=causal_action_slots,
            min_period=0.004,
            max_period=4.0,
        )

    return factory


@pytest.mark.unit
class TestOpenVLAOFTDecoderInitialization:
    def test_requests_raw_vlm_observations(
        self,
        openvla_oft_decoder_factory: Callable[..., OpenVLAOFTDecoder],
    ) -> None:
        decoder = openvla_oft_decoder_factory(input_keys=[])

        assert decoder.decoder_input.needs_raw_observations is True
        assert decoder.decoder_input.requires_actions is False
        assert decoder.decoder_input.keys == [
            CAMERA_KEY,
            SampleKey.TOKENIZED_OBSERVATIONS.value,
            SampleKey.IS_PAD_OBSERVATION.value,
        ]

    def test_rejects_extra_input_keys(
        self,
        openvla_oft_decoder_factory: Callable[..., OpenVLAOFTDecoder],
    ) -> None:
        expected_message = (
            "OpenVLAOFTDecoder builds its prefix from vlm_backbone inputs. "
            "Set input_keys to an empty list, got ['encoded_feature']."
        )
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            openvla_oft_decoder_factory(input_keys=["encoded_feature"])

    def test_rejects_mismatched_oft_action_head_input_dimension(
        self,
        openvla_oft_decoder_factory: Callable[..., OpenVLAOFTDecoder],
    ) -> None:
        expected_message = (
            "OpenVLAOFTDecoder action head input_dimension mismatch. "
            "slots_per_action_dimension=True uses one slot per action scalar, "
            "so the joint action head input_dimension must equal action_dim * "
            "language_hidden_dimension (3 * 16 = 48). "
            "Got {'joint_action': 16}."
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            openvla_oft_decoder_factory(action_head_input_dim=16)

    def test_rejects_mismatched_timestep_slot_action_head_input_dimension(
        self,
        openvla_oft_decoder_factory: Callable[..., OpenVLAOFTDecoder],
    ) -> None:
        expected_message = (
            "OpenVLAOFTDecoder action head input_dimension mismatch. "
            "slots_per_action_dimension=False uses one slot per timestep, "
            "so the joint action head input_dimension must equal "
            "language_hidden_dimension (16). "
            "Got {'joint_action': 48}."
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            openvla_oft_decoder_factory(
                slots_per_action_dimension=False,
                action_head_input_dim=ACTION_DIMENSION * LANGUAGE_HIDDEN_DIMENSION,
            )

    @pytest.mark.parametrize(
        "slots_per_action_dimension, expected_slots_per_step, expected_projection_input",
        [
            (True, ACTION_DIMENSION, 1),
            (False, 1, ACTION_DIMENSION),
        ],
    )
    def test_builds_action_slot_components_from_slot_layout(
        self,
        openvla_oft_decoder_factory: Callable[..., OpenVLAOFTDecoder],
        slots_per_action_dimension: bool,
        expected_slots_per_step: int,
        expected_projection_input: int,
    ) -> None:
        decoder = openvla_oft_decoder_factory(
            slots_per_action_dimension=slots_per_action_dimension,
        )

        assert decoder.action_slots_per_step == expected_slots_per_step
        assert decoder.action_slot_count == PREDICTION_HORIZON * expected_slots_per_step
        assert (
            decoder.action_slot_embeddings.num_embeddings == decoder.action_slot_count
        )
        assert decoder.action_slot_embeddings.embedding_dim == LANGUAGE_HIDDEN_DIMENSION
        assert decoder.noisy_action_projection.in_features == expected_projection_input
        assert decoder.noisy_action_projection.out_features == LANGUAGE_HIDDEN_DIMENSION

    def test_rejects_vlm_context_that_cannot_hold_oft_sequence(
        self,
        openvla_oft_decoder_factory: Callable[..., OpenVLAOFTDecoder],
        vlm_backbone_factory: Callable[..., MagicMock],
    ) -> None:
        vlm_backbone = vlm_backbone_factory()
        vlm_backbone.get_text_config.return_value = PretrainedConfig(
            max_position_embeddings=18,
        )
        expected_message = (
            "OpenVLAOFTDecoder sequence length exceeds the VLM language "
            "context. Required total_image_tokens + max_text_length + "
            "denoising_timestep_tokens + action_slots = 2 + 5 + 0 + "
            "12 = 19, but max_position_embeddings=18."
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            openvla_oft_decoder_factory(vlm_backbone=vlm_backbone)

    def test_accepts_context_that_only_lacks_denoising_timestep_at_init(
        self,
        openvla_oft_decoder_factory: Callable[..., OpenVLAOFTDecoder],
        vlm_backbone_factory: Callable[..., MagicMock],
    ) -> None:
        vlm_backbone = vlm_backbone_factory()
        vlm_backbone.get_text_config.return_value = PretrainedConfig(
            max_position_embeddings=19,
        )

        decoder = openvla_oft_decoder_factory(vlm_backbone=vlm_backbone)

        assert decoder.action_slot_count == PREDICTION_HORIZON * ACTION_DIMENSION

    def test_rejects_denoising_context_when_timestep_would_exceed_capacity(
        self,
        openvla_oft_decoder_factory: Callable[..., OpenVLAOFTDecoder],
        vlm_backbone_factory: Callable[..., MagicMock],
    ) -> None:
        vlm_backbone = vlm_backbone_factory()
        vlm_backbone.get_text_config.return_value = PretrainedConfig(
            max_position_embeddings=19,
        )
        decoder = openvla_oft_decoder_factory(vlm_backbone=vlm_backbone)
        expected_message = (
            "OpenVLAOFTDecoder sequence length exceeds the VLM language "
            "context. Required total_image_tokens + max_text_length + "
            "denoising_timestep_tokens + action_slots = 2 + 5 + 1 + "
            "12 = 20, but max_position_embeddings=19."
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            decoder._validate_context_capacity(
                vlm_backbone=vlm_backbone,
                includes_denoising_timestep=True,
            )


@pytest.mark.unit
class TestOpenVLAOFTDecoderPrefix:
    def test_build_prefix_uses_vlm_backbone(
        self,
        openvla_oft_decoder_factory: Callable[..., OpenVLAOFTDecoder],
    ) -> None:
        decoder = openvla_oft_decoder_factory(input_keys=[])
        features = {CAMERA_KEY: torch.zeros(BATCH_SIZE, 3, 16, 16)}

        prefix_tokens, prefix_mask = decoder._build_prefix(features=features)

        decoder.vlm_backbone.build_prefix.assert_called_once_with(inputs=features)
        assert torch.equal(
            prefix_tokens,
            decoder.vlm_backbone.build_prefix.return_value[0],
        )
        assert torch.equal(
            prefix_mask, decoder.vlm_backbone.build_prefix.return_value[1]
        )

    def test_build_denoising_prefix_filters_timestep_before_vlm_prefix(
        self,
        openvla_oft_decoder_factory: Callable[..., OpenVLAOFTDecoder],
    ) -> None:
        decoder = openvla_oft_decoder_factory(input_keys=[])
        features = {
            CAMERA_KEY: torch.zeros(BATCH_SIZE, 3, 16, 16),
            AlgorithmContextKey.TIMESTEP.value: torch.zeros(BATCH_SIZE),
        }
        appended_tokens = torch.ones(
            BATCH_SIZE,
            PREFIX_TOKEN_LENGTH + 1,
            LANGUAGE_HIDDEN_DIMENSION,
        )
        appended_mask = torch.zeros(
            BATCH_SIZE, PREFIX_TOKEN_LENGTH + 1, dtype=torch.bool
        )
        with patch.object(
            decoder,
            "_append_timestep_feature_token",
            return_value=(appended_tokens, appended_mask),
        ) as append_spy:
            prefix_tokens, prefix_mask = decoder._build_denoising_prefix(
                features=features,
                timestep=features[AlgorithmContextKey.TIMESTEP.value],
            )

        expected_features = {CAMERA_KEY: features[CAMERA_KEY]}
        decoder.vlm_backbone.build_prefix.assert_called_once_with(
            inputs=expected_features
        )
        append_spy.assert_called_once()
        assert torch.equal(prefix_tokens, appended_tokens)
        assert prefix_mask is None

    def test_append_timestep_feature_token_adds_unmasked_prefix_token(
        self,
        openvla_oft_decoder_factory: Callable[..., OpenVLAOFTDecoder],
    ) -> None:
        decoder = openvla_oft_decoder_factory(input_keys=[])
        feature_tokens = torch.zeros(
            BATCH_SIZE,
            PREFIX_TOKEN_LENGTH,
            LANGUAGE_HIDDEN_DIMENSION,
        )
        feature_mask = torch.zeros(BATCH_SIZE, PREFIX_TOKEN_LENGTH, dtype=torch.bool)
        timestep = torch.ones(BATCH_SIZE)
        timestep_embedding = torch.full(
            (BATCH_SIZE, LANGUAGE_HIDDEN_DIMENSION),
            fill_value=2.0,
        )

        with patch.object(
            decoder.timestep_embedding,
            "forward",
            return_value=timestep_embedding,
        ) as timestep_spy:
            prefix_tokens, prefix_mask = decoder._append_timestep_feature_token(
                feature_tokens=feature_tokens,
                feature_mask=feature_mask,
                timestep=timestep,
            )

        timestep_spy.assert_called_once()
        torch.testing.assert_close(
            prefix_tokens[:, :-1, :],
            feature_tokens,
        )
        torch.testing.assert_close(prefix_tokens[:, -1, :], timestep_embedding)
        torch.testing.assert_close(
            prefix_mask,
            torch.zeros(BATCH_SIZE, PREFIX_TOKEN_LENGTH + 1, dtype=torch.bool),
        )

    def test_run_language_model_uses_vlm_language_model(
        self,
        openvla_oft_decoder_factory: Callable[..., OpenVLAOFTDecoder],
    ) -> None:
        decoder = openvla_oft_decoder_factory(input_keys=[])
        tokens = torch.zeros(BATCH_SIZE, PREFIX_TOKEN_LENGTH, LANGUAGE_HIDDEN_DIMENSION)
        attention_mask = torch.ones(
            BATCH_SIZE,
            1,
            PREFIX_TOKEN_LENGTH,
            PREFIX_TOKEN_LENGTH,
            dtype=torch.bool,
        )

        hidden_states = decoder._run_language_model(
            tokens=tokens,
            attention_mask=attention_mask,
        )

        decoder.vlm_backbone.forward_language_model.assert_called_once_with(
            inputs_embeds=tokens,
            attention_mask=attention_mask,
            use_cache=False,
        )
        assert torch.equal(
            hidden_states,
            decoder.vlm_backbone.forward_language_model.return_value.hidden_states[-1],
        )

    def test_run_language_model_rejects_missing_hidden_states(
        self,
        openvla_oft_decoder_factory: Callable[..., OpenVLAOFTDecoder],
    ) -> None:
        decoder = openvla_oft_decoder_factory(input_keys=[])
        output = MagicMock(spec=CausalLanguageModelOutput)
        output.hidden_states = None
        decoder.vlm_backbone.forward_language_model.return_value = output
        tokens = torch.zeros(BATCH_SIZE, PREFIX_TOKEN_LENGTH, LANGUAGE_HIDDEN_DIMENSION)
        attention_mask = torch.ones(
            BATCH_SIZE,
            1,
            PREFIX_TOKEN_LENGTH,
            PREFIX_TOKEN_LENGTH,
            dtype=torch.bool,
        )
        expected_message = (
            "OpenVLAOFTDecoder requires VLM language-model hidden states."
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            decoder._run_language_model(tokens=tokens, attention_mask=attention_mask)

    def test_forward_uses_vlm_prefix_before_action_slots(
        self,
        openvla_oft_decoder_factory: Callable[..., OpenVLAOFTDecoder],
    ) -> None:
        decoder = openvla_oft_decoder_factory(input_keys=[])
        features = {CAMERA_KEY: torch.zeros(BATCH_SIZE, 3, 16, 16)}
        action_slots = torch.ones(
            BATCH_SIZE,
            PREDICTION_HORIZON * ACTION_DIMENSION,
            LANGUAGE_HIDDEN_DIMENSION,
        )
        sequence_output = torch.ones(
            BATCH_SIZE,
            PREFIX_TOKEN_LENGTH + PREDICTION_HORIZON * ACTION_DIMENSION,
            LANGUAGE_HIDDEN_DIMENSION,
        )
        action_embeddings = torch.ones(
            BATCH_SIZE,
            PREDICTION_HORIZON,
            ACTION_DIMENSION * LANGUAGE_HIDDEN_DIMENSION,
        )
        predictions = {
            "position_action": torch.ones(
                BATCH_SIZE,
                PREDICTION_HORIZON,
                ACTION_DIMENSION,
            )
        }
        with (
            patch.object(
                decoder,
                "_build_action_slots",
                return_value=action_slots,
            ) as slots_spy,
            patch.object(
                decoder,
                "_run_language_model",
                return_value=sequence_output,
            ) as language_spy,
            patch.object(
                decoder,
                "_reshape_action_slot_states",
                return_value=action_embeddings,
            ) as reshape_spy,
            patch.object(
                decoder,
                "_project_action_output",
                return_value=predictions,
            ) as project_spy,
        ):
            output = decoder(features=features, actions=None)

        decoder.vlm_backbone.build_prefix.assert_called_once_with(inputs=features)
        slots_spy.assert_called_once()
        language_spy.assert_called_once()
        reshape_spy.assert_called_once()
        project_spy.assert_called_once_with(action_embeddings)
        assert torch.equal(output["position_action"], predictions["position_action"])

    def test_forward_passes_configured_action_slot_attention_causality(
        self,
        openvla_oft_decoder_factory: Callable[..., OpenVLAOFTDecoder],
    ) -> None:
        decoder = openvla_oft_decoder_factory(
            input_keys=[],
            causal_action_slots=False,
        )
        features = {CAMERA_KEY: torch.zeros(BATCH_SIZE, 3, 16, 16)}
        sequence_output = torch.ones(
            BATCH_SIZE,
            PREFIX_TOKEN_LENGTH + PREDICTION_HORIZON * ACTION_DIMENSION,
            LANGUAGE_HIDDEN_DIMENSION,
        )
        predictions = {
            "position_action": torch.ones(
                BATCH_SIZE,
                PREDICTION_HORIZON,
                ACTION_DIMENSION,
            )
        }

        with (
            patch.object(
                decoder,
                "_build_prefix_suffix_inputs",
                wraps=decoder._build_prefix_suffix_inputs,
            ) as prefix_suffix_spy,
            patch.object(
                decoder,
                "_run_language_model",
                return_value=sequence_output,
            ),
            patch.object(
                decoder,
                "_project_action_output",
                return_value=predictions,
            ),
        ):
            output = decoder(features=features, actions=None)

        assert prefix_suffix_spy.call_args.kwargs["causal_suffix"] is False
        assert torch.equal(output["position_action"], predictions["position_action"])

    def test_forward_raises_for_denoising_without_actions(
        self,
        openvla_oft_decoder_factory: Callable[..., OpenVLAOFTDecoder],
    ) -> None:
        decoder = openvla_oft_decoder_factory(input_keys=[])
        features = {
            CAMERA_KEY: torch.zeros(BATCH_SIZE, 3, 16, 16),
            AlgorithmContextKey.TIMESTEP.value: torch.zeros(BATCH_SIZE),
        }
        expected_message = (
            "OpenVLAOFTDecoder with denoising algorithm requires "
            "ground truth actions during training."
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            decoder(features=features, actions=None)

    def test_forward_uses_denoising_actions_as_action_slots(
        self,
        openvla_oft_decoder_factory: Callable[..., OpenVLAOFTDecoder],
    ) -> None:
        decoder = openvla_oft_decoder_factory(input_keys=[])
        features = {
            CAMERA_KEY: torch.zeros(BATCH_SIZE, 3, 16, 16),
            AlgorithmContextKey.TIMESTEP.value: torch.ones(BATCH_SIZE),
        }
        actions = {
            "position_action": torch.zeros(
                BATCH_SIZE,
                PREDICTION_HORIZON,
                ACTION_DIMENSION,
            )
        }
        prefix_tokens = torch.zeros(
            BATCH_SIZE,
            PREFIX_TOKEN_LENGTH + 1,
            LANGUAGE_HIDDEN_DIMENSION,
        )
        prefix_mask = torch.zeros(BATCH_SIZE, PREFIX_TOKEN_LENGTH + 1, dtype=torch.bool)
        action_slots = torch.ones(
            BATCH_SIZE,
            PREDICTION_HORIZON * ACTION_DIMENSION,
            LANGUAGE_HIDDEN_DIMENSION,
        )
        sequence_output = torch.ones(
            BATCH_SIZE,
            PREFIX_TOKEN_LENGTH + 1 + PREDICTION_HORIZON * ACTION_DIMENSION,
            LANGUAGE_HIDDEN_DIMENSION,
        )
        action_embeddings = torch.ones(
            BATCH_SIZE,
            PREDICTION_HORIZON,
            ACTION_DIMENSION * LANGUAGE_HIDDEN_DIMENSION,
        )
        predictions = {
            "position_action": torch.ones(
                BATCH_SIZE,
                PREDICTION_HORIZON,
                ACTION_DIMENSION,
            )
        }
        timestep = torch.full((BATCH_SIZE,), fill_value=0.5)

        with (
            patch(
                "versatil.models.decoding.decoders.factory.openvla_oft.validate_action_tensors_against_dimensions",
                autospec=True,
                return_value=(BATCH_SIZE, torch.device("cpu")),
            ) as validate_spy,
            patch(
                "versatil.models.decoding.decoders.factory.openvla_oft.extract_timestep_conditioning",
                autospec=True,
                return_value=timestep,
            ) as timestep_spy,
            patch.object(
                decoder,
                "_build_denoising_prefix",
                return_value=(prefix_tokens, prefix_mask),
            ) as prefix_spy,
            patch.object(
                decoder,
                "_build_denoising_action_slots",
                return_value=action_slots,
            ) as slots_spy,
            patch.object(
                decoder,
                "_run_language_model",
                return_value=sequence_output,
            ),
            patch.object(
                decoder,
                "_reshape_action_slot_states",
                return_value=action_embeddings,
            ),
            patch.object(
                decoder,
                "_project_action_output",
                return_value=predictions,
            ),
        ):
            output = decoder(features=features, actions=actions)

        validate_spy.assert_called_once_with(
            actions=actions,
            action_dimensions=decoder._predicted_action_dimensions(),
            prediction_horizon=PREDICTION_HORIZON,
            decoder_name="OpenVLAOFTDecoder",
        )
        timestep_spy.assert_called_once_with(
            features=features,
            batch_size=BATCH_SIZE,
            action_device=torch.device("cpu"),
        )
        prefix_spy.assert_called_once_with(features=features, timestep=timestep)
        slots_spy.assert_called_once_with(
            actions=actions,
            dtype=prefix_tokens.dtype,
            device=prefix_tokens.device,
        )
        assert torch.equal(output["position_action"], predictions["position_action"])


@pytest.mark.unit
class TestOpenVLAOFTDecoderProjection:
    @pytest.mark.parametrize(
        "slots_per_action_dimension, expected_shape",
        [
            (True, (BATCH_SIZE, PREDICTION_HORIZON * ACTION_DIMENSION, 1)),
            (False, (BATCH_SIZE, PREDICTION_HORIZON, ACTION_DIMENSION)),
        ],
    )
    def test_build_denoising_action_slots_projects_layout_specific_noisy_actions(
        self,
        openvla_oft_decoder_factory: Callable[..., OpenVLAOFTDecoder],
        slots_per_action_dimension: bool,
        expected_shape: tuple[int, int, int],
    ) -> None:
        decoder = openvla_oft_decoder_factory(
            input_keys=[],
            slots_per_action_dimension=slots_per_action_dimension,
        )
        actions = {
            "position_action": torch.ones(
                BATCH_SIZE,
                PREDICTION_HORIZON,
                ACTION_DIMENSION,
            )
        }

        with patch.object(
            decoder.noisy_action_projection,
            "forward",
            wraps=decoder.noisy_action_projection.forward,
        ) as projection_spy:
            action_slots = decoder._build_denoising_action_slots(
                actions=actions,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )

        assert projection_spy.call_args.args[0].shape == expected_shape
        assert action_slots.shape == (
            BATCH_SIZE,
            PREDICTION_HORIZON * decoder.action_slots_per_step,
            LANGUAGE_HIDDEN_DIMENSION,
        )

    @pytest.mark.parametrize(
        "slots_per_action_dimension, input_shape, expected_shape",
        [
            (
                True,
                (
                    BATCH_SIZE,
                    PREDICTION_HORIZON * ACTION_DIMENSION,
                    LANGUAGE_HIDDEN_DIMENSION,
                ),
                (
                    BATCH_SIZE,
                    PREDICTION_HORIZON,
                    ACTION_DIMENSION * LANGUAGE_HIDDEN_DIMENSION,
                ),
            ),
            (
                False,
                (BATCH_SIZE, PREDICTION_HORIZON, LANGUAGE_HIDDEN_DIMENSION),
                (BATCH_SIZE, PREDICTION_HORIZON, LANGUAGE_HIDDEN_DIMENSION),
            ),
        ],
    )
    def test_reshape_action_slot_states_matches_slot_layout(
        self,
        openvla_oft_decoder_factory: Callable[..., OpenVLAOFTDecoder],
        slots_per_action_dimension: bool,
        input_shape: tuple[int, int, int],
        expected_shape: tuple[int, int, int],
    ) -> None:
        decoder = openvla_oft_decoder_factory(
            input_keys=[],
            slots_per_action_dimension=slots_per_action_dimension,
        )
        slot_states = torch.zeros(input_shape)

        action_embeddings = decoder._reshape_action_slot_states(slot_states)

        assert action_embeddings.shape == expected_shape

    def test_project_action_output_uses_configured_action_heads(
        self,
        openvla_oft_decoder_factory: Callable[..., OpenVLAOFTDecoder],
    ) -> None:
        decoder = openvla_oft_decoder_factory(input_keys=[])
        action_embeddings = torch.ones(
            BATCH_SIZE,
            PREDICTION_HORIZON,
            ACTION_DIMENSION * LANGUAGE_HIDDEN_DIMENSION,
        )
        expected_output = torch.full(
            (BATCH_SIZE, PREDICTION_HORIZON, ACTION_DIMENSION),
            fill_value=2.0,
        )

        with patch.object(
            decoder.action_heads[JOINT_ACTION_HEAD_KEY],
            "forward",
            return_value=expected_output,
        ) as action_head_spy:
            output = decoder._project_action_output(action_embeddings)

        action_head_spy.assert_called_once_with(action_embeddings)
        assert torch.equal(output["position_action"], expected_output)

    def test_project_action_output_splits_joint_head_output_by_action_space(
        self,
        openvla_oft_decoder_factory: Callable[..., OpenVLAOFTDecoder],
    ) -> None:
        orientation_dimension = 2
        decoder = openvla_oft_decoder_factory(
            input_keys=[],
            has_orientation=True,
            orientation_dim=orientation_dimension,
        )
        action_embeddings = torch.ones(
            BATCH_SIZE,
            PREDICTION_HORIZON,
            (ACTION_DIMENSION + orientation_dimension) * LANGUAGE_HIDDEN_DIMENSION,
        )
        joint_action_output = torch.arange(
            BATCH_SIZE
            * PREDICTION_HORIZON
            * (ACTION_DIMENSION + orientation_dimension),
            dtype=torch.float32,
        ).reshape(
            BATCH_SIZE,
            PREDICTION_HORIZON,
            ACTION_DIMENSION + orientation_dimension,
        )

        with patch.object(
            decoder.action_heads[JOINT_ACTION_HEAD_KEY],
            "forward",
            return_value=joint_action_output,
        ):
            output = decoder._project_action_output(action_embeddings)

        torch.testing.assert_close(
            output["position_action"],
            joint_action_output[:, :, :ACTION_DIMENSION],
        )
        torch.testing.assert_close(
            output["orientation_action"],
            joint_action_output[:, :, ACTION_DIMENSION:],
        )


@pytest.mark.integration
def test_forward_runs_real_tiny_vlm_and_joint_oft_head(
    tiny_prismatic_vlm_factory: Callable[..., PrismaticVLM],
    mock_action_space_factory: Callable[..., MagicMock],
    mock_observation_space_factory: Callable[..., MagicMock],
    action_head_factory: Callable[..., ActionHead],
) -> None:
    vlm_backbone = tiny_prismatic_vlm_factory()
    action_space = mock_action_space_factory(position_dim=ACTION_DIMENSION)
    action_head = action_head_factory(
        input_dimension=ACTION_DIMENSION * LANGUAGE_HIDDEN_DIMENSION
    )
    decoder = OpenVLAOFTDecoder(
        action_heads={JOINT_ACTION_HEAD_KEY: action_head},
        input_keys=[],
        action_space=action_space,
        observation_space=mock_observation_space_factory(),
        observation_horizon=1,
        prediction_horizon=PREDICTION_HORIZON,
        device="cpu",
        vlm_backbone=vlm_backbone,
        slots_per_action_dimension=True,
        causal_action_slots=True,
    )
    features = {
        Cameras.LEFT.value: torch.zeros(
            BATCH_SIZE,
            3,
            vlm_backbone.image_size,
            vlm_backbone.image_size,
        ),
        SampleKey.TOKENIZED_OBSERVATIONS.value: torch.arange(
            BATCH_SIZE * PREFIX_TOKEN_LENGTH,
            dtype=torch.long,
        ).reshape(BATCH_SIZE, PREFIX_TOKEN_LENGTH),
        SampleKey.IS_PAD_OBSERVATION.value: torch.zeros(
            BATCH_SIZE,
            PREFIX_TOKEN_LENGTH,
            dtype=torch.bool,
        ),
    }

    with (
        torch.no_grad(),
        patch.object(
            vlm_backbone,
            "forward_language_model",
            wraps=vlm_backbone.forward_language_model,
        ) as forward_language_model_spy,
        patch.object(
            action_head,
            "forward",
            wraps=action_head.forward,
        ) as action_head_spy,
    ):
        output = decoder(features=features)

    forward_language_model_spy.assert_called_once()
    language_model_call = forward_language_model_spy.call_args
    assert language_model_call.kwargs["use_cache"] is False
    assert language_model_call.kwargs["inputs_embeds"].shape[0] == BATCH_SIZE
    assert language_model_call.kwargs["attention_mask"].shape[-1] == (
        vlm_backbone.total_image_tokens
        + vlm_backbone.max_text_length
        + PREDICTION_HORIZON * ACTION_DIMENSION
    )
    action_head_spy.assert_called_once()
    assert action_head_spy.call_args.args[0].shape == (
        BATCH_SIZE,
        PREDICTION_HORIZON,
        ACTION_DIMENSION * LANGUAGE_HIDDEN_DIMENSION,
    )
    assert output["position_action"].shape == (
        BATCH_SIZE,
        PREDICTION_HORIZON,
        ACTION_DIMENSION,
    )
