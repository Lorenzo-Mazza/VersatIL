"""Tests for versatil.models.decoding.generative_language_models.vision_language.paligemma module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from transformers import (
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
    PretrainedConfig,
)
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.gemma2.modeling_gemma2 import Gemma2Model

from versatil.data.constants import Cameras, SampleKey
from versatil.data.metadata import BaseMetadata, CameraMetadata, RGBCameraMetadata
from versatil.models.adaptation.constants import LoRATargetModulePreset
from versatil.models.adaptation.lora import LoRAAdaptation
from versatil.models.decoding.generative_language_models.constants import (
    PaliGemmaModelType,
)
from versatil.models.decoding.generative_language_models.vision_language.paligemma import (
    PaliGemmaVLM,
)
from versatil.models.encoding.encoders.constants import EncoderOutputKeys
from versatil.models.encoding.explainability import (
    ActivationLayout,
    ExplanationTargetKind,
)
from versatil.training.constants import PrecisionType

HIDDEN_DIM = 64
IMAGE_SIZE = 56
NUM_IMAGE_TOKENS = 16  # (56/14)^2
MAX_TEXT_LENGTH = 128
VOCAB_SIZE = 32000
DEFAULT_INPUT_KEYS = [Cameras.LEFT.value]


def _create_mock_config() -> MagicMock:
    config = MagicMock(spec=PaliGemmaConfig)
    config.vision_config = MagicMock(spec=PretrainedConfig)
    config.text_config = MagicMock(spec=PretrainedConfig)
    config.image_token_id = 99999
    config.vision_config.image_size = IMAGE_SIZE
    config.vision_config.num_image_tokens = NUM_IMAGE_TOKENS
    config.text_config.hidden_size = HIDDEN_DIM
    config.text_config.max_position_embeddings = MAX_TEXT_LENGTH
    config.text_config.vocab_size = VOCAB_SIZE
    return config


@pytest.fixture
def mock_vlm_factory() -> Callable[..., MagicMock]:
    def factory(
        batch_size: int = 2,
        num_cameras: int = 1,
    ) -> MagicMock:
        mock_vlm = MagicMock(spec=PaliGemmaForConditionalGeneration)
        language_model = MagicMock(spec=Gemma2Model)
        language_model.config = MagicMock(spec=PretrainedConfig)
        language_model.config.vocab_size = VOCAB_SIZE
        language_model.layers = [MagicMock(spec=nn.Module)]
        language_model.rotary_emb = MagicMock(spec=nn.Module)
        mock_vlm.model = MagicMock(spec=nn.Module)
        mock_vlm.model.language_model = language_model
        mock_vlm.model.multi_modal_projector = nn.Identity()

        total_image_tokens = NUM_IMAGE_TOKENS * num_cameras
        mock_image_output = MagicMock(spec=BaseModelOutputWithPooling)
        mock_image_output.pooler_output = torch.zeros(
            batch_size, NUM_IMAGE_TOKENS, HIDDEN_DIM
        )
        mock_vlm.get_image_features.return_value = mock_image_output

        mock_embed = MagicMock(spec=nn.Embedding)
        mock_embed.return_value = torch.zeros(batch_size, MAX_TEXT_LENGTH, HIDDEN_DIM)
        mock_vlm.model.language_model.get_input_embeddings.return_value = mock_embed

        total_seq = total_image_tokens + MAX_TEXT_LENGTH
        mock_lm_output = MagicMock(spec=BaseModelOutput)
        mock_lm_output.last_hidden_state = torch.zeros(
            batch_size, total_seq, HIDDEN_DIM
        )
        mock_vlm.model.language_model.return_value = mock_lm_output

        return mock_vlm

    return factory


@pytest.fixture
def paligemma_backbone_factory(
    mock_vlm_factory: Callable[..., MagicMock],
) -> Callable[..., PaliGemmaVLM]:
    def factory(
        input_keys: str | list[str] | None = None,
        pretrained: bool = False,
        frozen: bool = False,
    ) -> PaliGemmaVLM:
        if input_keys is None:
            input_keys = [Cameras.LEFT.value]
        mock_config = _create_mock_config()
        mock_vlm = mock_vlm_factory()

        with (
            patch(
                "versatil.models.decoding.generative_language_models.vision_language.huggingface.AutoConfig.from_pretrained",
                return_value=mock_config,
            ),
            patch(
                "versatil.models.decoding.generative_language_models.vision_language.huggingface.AutoModelForImageTextToText.from_pretrained",
                return_value=mock_vlm,
            ),
            patch(
                "versatil.models.decoding.generative_language_models.vision_language.huggingface.AutoModelForImageTextToText.from_config",
                return_value=mock_vlm,
            ),
        ):
            return PaliGemmaVLM(
                input_keys=input_keys,
                pretrained=pretrained,
                frozen=frozen,
                model_name=PaliGemmaModelType.PALIGEMMA2_3B_224.value,
            )

    return factory


@pytest.fixture
def paligemma_input_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    def factory(
        camera_key: str = Cameras.LEFT.value,
        batch_size: int = 2,
        channels: int = 3,
        height: int = IMAGE_SIZE,
        width: int = IMAGE_SIZE,
        sequence_length: int = 10,
        time_steps: int = 1,
        include_padding_mask: bool = False,
    ) -> dict[str, torch.Tensor]:
        image_shape = (batch_size, time_steps, channels, height, width)
        text_shape = (batch_size, time_steps, sequence_length)
        images = torch.from_numpy(rng.standard_normal(image_shape).astype(np.float32))
        token_ids = torch.from_numpy(
            rng.integers(low=0, high=VOCAB_SIZE, size=text_shape).astype(np.int64)
        )
        result = {
            camera_key: images,
            SampleKey.TOKENIZED_OBSERVATIONS.value: token_ids,
        }
        if include_padding_mask:
            mask = torch.zeros(text_shape, dtype=torch.bool)
            result[SampleKey.IS_PAD_OBSERVATION.value] = mask
        return result

    return factory


def _setup_mock_vlm_for_batch(
    backbone: PaliGemmaVLM,
    effective_batch_size: int,
) -> None:
    mock_image_output = MagicMock(spec=BaseModelOutputWithPooling)
    mock_image_output.pooler_output = torch.zeros(
        effective_batch_size, NUM_IMAGE_TOKENS, HIDDEN_DIM
    )
    backbone.vlm.get_image_features.return_value = mock_image_output
    mock_embed = MagicMock(spec=nn.Embedding)
    mock_embed.return_value = torch.zeros(
        effective_batch_size,
        MAX_TEXT_LENGTH,
        HIDDEN_DIM,
    )
    backbone.vlm.model.language_model.get_input_embeddings.return_value = mock_embed
    total_seq = backbone.total_image_tokens + MAX_TEXT_LENGTH
    mock_lm_output = MagicMock(spec=BaseModelOutput)
    mock_lm_output.last_hidden_state = torch.zeros(
        effective_batch_size, total_seq, HIDDEN_DIM
    )
    backbone.vlm.model.language_model.return_value = mock_lm_output


@pytest.mark.unit
class TestPaliGemmaVLMInitialization:
    @pytest.mark.parametrize(
        "input_keys, expected_camera_count",
        [
            ([Cameras.LEFT.value], 1),
            ([Cameras.LEFT.value, Cameras.RIGHT.value], 2),
        ],
    )
    @pytest.mark.parametrize("frozen", [True, False])
    def test_stores_configuration(
        self,
        paligemma_backbone_factory: Callable[..., PaliGemmaVLM],
        input_keys: list[str],
        expected_camera_count: int,
        frozen: bool,
    ) -> None:
        backbone = paligemma_backbone_factory(
            input_keys=input_keys,
            pretrained=False,
            frozen=frozen,
        )
        assert backbone.hidden_dimension == HIDDEN_DIM
        assert backbone.image_size == IMAGE_SIZE
        assert backbone.max_text_length == MAX_TEXT_LENGTH
        assert backbone.num_image_tokens_per_camera == NUM_IMAGE_TOKENS
        assert len(backbone.camera_keys) == expected_camera_count
        assert backbone.input_specification.requires_tokenized is True
        assert SampleKey.IS_PAD_OBSERVATION.value in backbone.input_specification.keys
        if frozen:
            for parameter in backbone.parameters():
                assert not parameter.requires_grad


@pytest.mark.unit
class TestPaliGemmaVLMForward:
    @pytest.mark.parametrize("time_steps", [1, 3])
    @pytest.mark.parametrize(
        "input_keys, num_cameras",
        [
            ([Cameras.LEFT.value], 1),
            ([Cameras.LEFT.value, Cameras.RIGHT.value], 2),
        ],
    )
    def test_output_shape_scales_with_cameras_and_time(
        self,
        paligemma_backbone_factory: Callable[..., PaliGemmaVLM],
        paligemma_input_factory: Callable[..., dict[str, torch.Tensor]],
        time_steps: int,
        input_keys: list[str],
        num_cameras: int,
    ) -> None:
        batch_size = 2
        backbone = paligemma_backbone_factory(input_keys=input_keys)
        _setup_mock_vlm_for_batch(backbone, batch_size * time_steps)
        camera_keys = [
            k for k in input_keys if k != SampleKey.TOKENIZED_OBSERVATIONS.value
        ]
        inputs = {}
        for camera_key in camera_keys:
            inputs.update(
                paligemma_input_factory(
                    camera_key=camera_key,
                    batch_size=batch_size,
                    time_steps=time_steps,
                    include_padding_mask=True,
                )
            )
        output = backbone(inputs=inputs)
        fused = output[EncoderOutputKeys.FUSED_RGB_LANGUAGE.value]
        total_seq = num_cameras * NUM_IMAGE_TOKENS + MAX_TEXT_LENGTH
        assert fused.shape == (batch_size, time_steps, total_seq, HIDDEN_DIM)

    def test_output_contains_fused_features_and_padding_mask(
        self,
        paligemma_backbone_factory: Callable[..., PaliGemmaVLM],
        paligemma_input_factory: Callable[..., dict[str, torch.Tensor]],
    ) -> None:
        batch_size = 2
        backbone = paligemma_backbone_factory(
            input_keys=DEFAULT_INPUT_KEYS,
            pretrained=False,
            frozen=False,
        )
        _setup_mock_vlm_for_batch(backbone, batch_size)
        inputs = paligemma_input_factory(
            camera_key=Cameras.LEFT.value,
            batch_size=batch_size,
            channels=3,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            sequence_length=10,
            time_steps=1,
            include_padding_mask=False,
        )
        output = backbone(inputs=inputs)
        assert EncoderOutputKeys.FUSED_RGB_LANGUAGE.value in output
        assert backbone.padding_mask_name in output
        assert len(output) == 2

    def test_forward_language_model_converts_position_ids_to_paligemma_positions(
        self,
        paligemma_backbone_factory: Callable[..., PaliGemmaVLM],
    ) -> None:
        backbone = paligemma_backbone_factory()
        language_model = backbone._get_language_model()
        hidden_states = torch.zeros(2, 3, HIDDEN_DIM)
        language_output = MagicMock(spec=BaseModelOutput)
        language_output.last_hidden_state = hidden_states
        language_output.hidden_states = (hidden_states,)
        language_output.past_key_values = MagicMock()
        language_model.return_value = language_output
        output_head = MagicMock(return_value=torch.zeros(2, 3, VOCAB_SIZE))
        backbone.vlm.get_output_embeddings.return_value = output_head
        inputs_embeds = torch.zeros(2, 3, HIDDEN_DIM)
        position_ids = torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long)

        backbone.forward_language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            output_hidden_states=True,
        )

        call_position_ids = language_model.call_args.kwargs["position_ids"]
        torch.testing.assert_close(call_position_ids, position_ids + 1)

    def test_vision_tower_called_once_per_camera(
        self,
        paligemma_backbone_factory: Callable[..., PaliGemmaVLM],
        paligemma_input_factory: Callable[..., dict[str, torch.Tensor]],
    ) -> None:
        batch_size = 2
        input_keys = [
            Cameras.LEFT.value,
            Cameras.RIGHT.value,
            SampleKey.TOKENIZED_OBSERVATIONS.value,
        ]
        backbone = paligemma_backbone_factory(
            input_keys=input_keys,
            pretrained=False,
            frozen=False,
        )
        _setup_mock_vlm_for_batch(backbone, batch_size)
        left_inputs = paligemma_input_factory(
            camera_key=Cameras.LEFT.value,
            batch_size=batch_size,
            channels=3,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            sequence_length=10,
            time_steps=1,
            include_padding_mask=False,
        )
        right_inputs = paligemma_input_factory(
            camera_key=Cameras.RIGHT.value,
            batch_size=batch_size,
            channels=3,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            sequence_length=10,
            time_steps=1,
            include_padding_mask=False,
        )
        inputs = {**left_inputs, **right_inputs}
        inputs[SampleKey.TOKENIZED_OBSERVATIONS.value] = left_inputs[
            SampleKey.TOKENIZED_OBSERVATIONS.value
        ]

        backbone(inputs=inputs)

        assert backbone.vlm.get_image_features.call_count == 2
        left_call = backbone.vlm.get_image_features.call_args_list[0].args[0]
        right_call = backbone.vlm.get_image_features.call_args_list[1].args[0]
        torch.testing.assert_close(left_call, inputs[Cameras.LEFT.value].squeeze(1))
        torch.testing.assert_close(right_call, inputs[Cameras.RIGHT.value].squeeze(1))

    def test_language_model_receives_concatenated_image_and_text_embeddings(
        self,
        paligemma_backbone_factory: Callable[..., PaliGemmaVLM],
        paligemma_input_factory: Callable[..., dict[str, torch.Tensor]],
    ) -> None:
        batch_size = 2
        backbone = paligemma_backbone_factory(
            input_keys=DEFAULT_INPUT_KEYS,
            pretrained=False,
            frozen=False,
        )
        _setup_mock_vlm_for_batch(backbone, batch_size)
        inputs = paligemma_input_factory(
            camera_key=Cameras.LEFT.value,
            batch_size=batch_size,
            channels=3,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            sequence_length=10,
            time_steps=1,
            include_padding_mask=False,
        )
        backbone(inputs=inputs)
        call_kwargs = backbone.vlm.model.language_model.call_args.kwargs
        expected_seq_length = NUM_IMAGE_TOKENS + MAX_TEXT_LENGTH
        assert call_kwargs["inputs_embeds"].shape == (
            batch_size,
            expected_seq_length,
            HIDDEN_DIM,
        )

    def test_language_model_attention_mask_marks_auto_padded_text_tokens(
        self,
        paligemma_backbone_factory: Callable[..., PaliGemmaVLM],
        paligemma_input_factory: Callable[..., dict[str, torch.Tensor]],
    ) -> None:
        batch_size = 2
        sequence_length = 5
        backbone = paligemma_backbone_factory(
            input_keys=DEFAULT_INPUT_KEYS,
            pretrained=False,
            frozen=False,
        )
        _setup_mock_vlm_for_batch(backbone, batch_size)
        inputs = paligemma_input_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        backbone(inputs=inputs)
        attention_mask = backbone.vlm.model.language_model.call_args.kwargs[
            "attention_mask"
        ]
        image_attention = attention_mask[:, :NUM_IMAGE_TOKENS]
        text_attention = attention_mask[:, NUM_IMAGE_TOKENS:]
        assert image_attention.all()
        assert text_attention[:, :sequence_length].all()
        assert not text_attention[:, sequence_length:].any()

    def test_different_images_produce_different_vision_tower_inputs(
        self,
        paligemma_backbone_factory: Callable[..., PaliGemmaVLM],
        rng: np.random.Generator,
    ) -> None:
        batch_size = 1
        backbone = paligemma_backbone_factory(
            input_keys=DEFAULT_INPUT_KEYS,
            pretrained=False,
            frozen=False,
        )
        _setup_mock_vlm_for_batch(backbone, batch_size)
        images_a = torch.from_numpy(
            rng.standard_normal((batch_size, 1, 3, IMAGE_SIZE, IMAGE_SIZE)).astype(
                np.float32
            )
        )
        images_b = torch.from_numpy(
            rng.standard_normal((batch_size, 1, 3, IMAGE_SIZE, IMAGE_SIZE)).astype(
                np.float32
            )
        )
        tokens = torch.zeros(batch_size, 1, 10, dtype=torch.long)

        backbone(
            inputs={
                Cameras.LEFT.value: images_a,
                SampleKey.TOKENIZED_OBSERVATIONS.value: tokens,
            }
        )
        first_call_pixel = backbone.vlm.get_image_features.call_args[0][0]

        backbone(
            inputs={
                Cameras.LEFT.value: images_b,
                SampleKey.TOKENIZED_OBSERVATIONS.value: tokens,
            }
        )
        second_call_pixel = backbone.vlm.get_image_features.call_args[0][0]

        assert not torch.equal(first_call_pixel, second_call_pixel)

    def test_padding_mask_image_portion_is_never_padded(
        self,
        paligemma_backbone_factory: Callable[..., PaliGemmaVLM],
        paligemma_input_factory: Callable[..., dict[str, torch.Tensor]],
    ) -> None:
        batch_size = 2
        backbone = paligemma_backbone_factory(
            input_keys=DEFAULT_INPUT_KEYS,
            pretrained=False,
            frozen=False,
        )
        _setup_mock_vlm_for_batch(backbone, batch_size)
        inputs = paligemma_input_factory(
            batch_size=batch_size, include_padding_mask=True
        )
        output = backbone(inputs=inputs)
        padding_mask = output[backbone.padding_mask_name]
        # Temporal dim: (B, T=1, total_seq)
        image_portion = padding_mask[:, :, :NUM_IMAGE_TOKENS]
        assert not image_portion.any()

    def test_language_model_called_for_fused_vision_language_features(
        self,
        paligemma_backbone_factory: Callable[..., PaliGemmaVLM],
        paligemma_input_factory: Callable[..., dict[str, torch.Tensor]],
    ) -> None:
        batch_size = 2
        backbone = paligemma_backbone_factory(
            input_keys=DEFAULT_INPUT_KEYS,
            pretrained=False,
            frozen=False,
        )
        _setup_mock_vlm_for_batch(backbone, batch_size)
        inputs = paligemma_input_factory(
            batch_size=batch_size, include_padding_mask=True
        )
        output = backbone(inputs=inputs)
        fused = output[EncoderOutputKeys.FUSED_RGB_LANGUAGE.value]
        total_seq = NUM_IMAGE_TOKENS + MAX_TEXT_LENGTH
        assert fused.shape == (batch_size, 1, total_seq, HIDDEN_DIM)
        assert backbone.padding_mask_name in output
        backbone.vlm.model.language_model.assert_called_once()

    def test_missing_language_key_raises(
        self,
        paligemma_backbone_factory: Callable[..., PaliGemmaVLM],
        rng: np.random.Generator,
    ) -> None:
        backbone = paligemma_backbone_factory(
            input_keys=DEFAULT_INPUT_KEYS,
            pretrained=False,
            frozen=False,
        )
        images = torch.from_numpy(
            rng.standard_normal((2, 1, 3, IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"PaliGemmaVLM expects pre-tokenized input. "
                f"Expected key '{SampleKey.TOKENIZED_OBSERVATIONS.value}' "
                f"not found in inputs. "
                f"Ensure tokenization is enabled in DataloaderConfig."
            ),
        ):
            backbone(inputs={Cameras.LEFT.value: images})


@pytest.mark.unit
class TestPaliGemmaVLMValidateInputMetadata:
    @pytest.mark.parametrize(
        "metadata, expected_error",
        [
            (
                RGBCameraMetadata(
                    camera_key="left",
                    dtype="uint8",
                    image_height=224,
                    image_width=224,
                ),
                None,
            ),
            (
                CameraMetadata(
                    camera_key="depth",
                    dtype="uint8",
                    channels=1,
                    image_height=224,
                    image_width=224,
                ),
                None,
            ),
            (
                MagicMock(spec=BaseMetadata),
                f"Expected CameraMetadata for '{Cameras.LEFT.value}', got MagicMock",
            ),
        ],
    )
    def test_validates_camera_key_metadata(
        self,
        paligemma_backbone_factory: Callable[..., PaliGemmaVLM],
        metadata: BaseMetadata,
        expected_error: str | None,
    ) -> None:
        backbone = paligemma_backbone_factory(
            input_keys=DEFAULT_INPUT_KEYS,
            pretrained=False,
            frozen=False,
        )
        result = backbone.validate_input_metadata(
            key=Cameras.LEFT.value, metadata=metadata
        )
        assert result == expected_error

    @pytest.mark.parametrize(
        "metadata, expected_error",
        [
            (
                RGBCameraMetadata(
                    camera_key="left",
                    dtype="uint8",
                    image_height=224,
                    image_width=224,
                ),
                f"PaliGemmaVLM cannot process image data for "
                f"'{SampleKey.TOKENIZED_OBSERVATIONS.value}'. "
                f"Got CameraMetadata, expected tokenized text input.",
            ),
            (
                MagicMock(spec=BaseMetadata),
                None,
            ),
        ],
    )
    def test_validates_language_key_metadata(
        self,
        paligemma_backbone_factory: Callable[..., PaliGemmaVLM],
        metadata: BaseMetadata,
        expected_error: str | None,
    ) -> None:
        backbone = paligemma_backbone_factory(
            input_keys=DEFAULT_INPUT_KEYS,
            pretrained=False,
            frozen=False,
        )
        result = backbone.validate_input_metadata(
            key=SampleKey.TOKENIZED_OBSERVATIONS.value, metadata=metadata
        )
        assert result == expected_error


@pytest.mark.unit
class TestPaliGemmaVLMGetVocabSize:
    def test_returns_language_model_vocab_size(
        self,
        paligemma_backbone_factory: Callable[..., PaliGemmaVLM],
    ) -> None:
        backbone = paligemma_backbone_factory(
            input_keys=DEFAULT_INPUT_KEYS,
            pretrained=False,
            frozen=False,
        )
        assert backbone.get_vocab_size() == VOCAB_SIZE

    def test_resize_token_embeddings_delegates_to_vlm(
        self,
        paligemma_backbone_factory: Callable[..., PaliGemmaVLM],
    ) -> None:
        backbone = paligemma_backbone_factory(
            input_keys=DEFAULT_INPUT_KEYS,
            pretrained=False,
            frozen=False,
        )

        backbone.resize_token_embeddings(vocabulary_size=VOCAB_SIZE + 1)

        backbone.vlm.resize_token_embeddings.assert_called_once_with(VOCAB_SIZE + 1)
        assert backbone.get_vocab_size() == VOCAB_SIZE + 1


@pytest.mark.unit
class TestPaliGemmaVLMBackboneAccessors:
    def test_get_backbone_layers_accesses_language_model_layers(
        self,
        paligemma_backbone_factory: Callable[..., PaliGemmaVLM],
    ) -> None:
        backbone = paligemma_backbone_factory(
            input_keys=DEFAULT_INPUT_KEYS,
            pretrained=False,
            frozen=False,
        )
        result = backbone.get_backbone_layers()
        assert result == backbone.vlm.model.language_model.layers

    def test_get_rotary_embedding_accesses_language_model_rotary_emb(
        self,
        paligemma_backbone_factory: Callable[..., PaliGemmaVLM],
    ) -> None:
        backbone = paligemma_backbone_factory(
            input_keys=DEFAULT_INPUT_KEYS,
            pretrained=False,
            frozen=False,
        )
        result = backbone.get_rotary_embedding()
        assert result == backbone.vlm.model.language_model.rotary_emb

    def test_get_backbone_hidden_dim_returns_hidden_dim(
        self,
        paligemma_backbone_factory: Callable[..., PaliGemmaVLM],
    ) -> None:
        backbone = paligemma_backbone_factory(
            input_keys=DEFAULT_INPUT_KEYS,
            pretrained=False,
            frozen=False,
        )
        assert backbone.get_backbone_hidden_dim() == HIDDEN_DIM


@pytest.mark.unit
class TestPaliGemmaVLMExplainabilityTargets:
    def test_exposes_multi_modal_projector_token_target(
        self,
        paligemma_backbone_factory: Callable[..., PaliGemmaVLM],
    ) -> None:
        backbone = paligemma_backbone_factory(
            input_keys=[Cameras.LEFT.value, Cameras.RIGHT.value],
            pretrained=False,
            frozen=False,
        )

        targets = backbone.get_explainability_targets()

        assert len(targets) == 1
        assert targets[0].layer is backbone.vlm.model.multi_modal_projector
        assert targets[0].target_kind == ExplanationTargetKind.TOKEN_SEQUENCE.value
        assert targets[0].activation_layout == ActivationLayout.NLC.value
        assert targets[0].patch_grid == (4, 4)
        assert backbone.is_multi_camera is True


class TestPaliGemmaVLMIntegration:
    @pytest.mark.integration
    @pytest.mark.parametrize("lora_enabled", [False, True])
    def test_forward_pass_with_real_model(
        self,
        real_paligemma_backbone: Callable[..., PaliGemmaVLM],
        rng: np.random.Generator,
        lora_enabled: bool,
        parameter_count: Callable[[torch.nn.Module], int],
        trainable_parameter_count: Callable[[torch.nn.Module], int],
    ) -> None:
        batch_size = 1
        lora_config = (
            LoRAAdaptation(
                enabled=True,
                rank=2,
                alpha=4,
                target_modules=(
                    LoRATargetModulePreset.VLM_TEXT_MODEL_ATTENTION_AND_FEEDFORWARD.value
                ),
            )
            if lora_enabled
            else None
        )
        backbone = real_paligemma_backbone(
            model_dtype=PrecisionType.FP32.value,
            frozen=not lora_enabled,
            lora_config=lora_config,
        )
        backbone.eval()
        vocab_size = backbone.get_vocab_size()
        images = torch.from_numpy(
            rng.standard_normal(
                (batch_size, 1, 3, backbone.image_size, backbone.image_size)
            ).astype(np.float32)
        )
        token_ids = torch.from_numpy(
            rng.integers(low=0, high=vocab_size, size=(batch_size, 1, 10)).astype(
                np.int64
            )
        )
        inputs = {
            Cameras.LEFT.value: images,
            SampleKey.TOKENIZED_OBSERVATIONS.value: token_ids,
        }
        with torch.no_grad():
            output = backbone(inputs=inputs)
        fused = output[EncoderOutputKeys.FUSED_RGB_LANGUAGE.value]
        assert fused.shape[0] == batch_size
        assert fused.shape[-1] == backbone.hidden_dimension
        if lora_enabled:
            trainable_parameter_names = [
                name
                for name, parameter in backbone.vlm.named_parameters()
                if parameter.requires_grad
            ]
            trainable_parameters = trainable_parameter_count(backbone.vlm)
            total_parameters = parameter_count(backbone.vlm)
            assert trainable_parameter_names
            assert all("lora_" in name for name in trainable_parameter_names)
            assert all(".language_model." in name for name in trainable_parameter_names)
            assert 0 < trainable_parameters < total_parameters

    @pytest.mark.integration
    def test_forward_language_model_with_real_peft_resized_vocabulary(
        self,
        real_paligemma_backbone: Callable[..., PaliGemmaVLM],
    ) -> None:
        lora_config = LoRAAdaptation(
            enabled=True,
            rank=3,
            alpha=6,
            target_modules=(
                LoRATargetModulePreset.VLM_TEXT_MODEL_QUERY_VALUE_PROJECTIONS.value
            ),
        )
        backbone = real_paligemma_backbone(
            model_dtype=PrecisionType.FP32.value,
            frozen=False,
            lora_config=lora_config,
        )
        backbone.eval()
        resized_vocab_size = backbone.get_vocab_size() + 1
        backbone.resize_token_embeddings(vocabulary_size=resized_vocab_size)
        token_ids = torch.tensor([[0, resized_vocab_size - 1]], dtype=torch.long)
        attention_mask = torch.ones_like(token_ids)
        inputs_embeds = backbone.embed_input_ids(token_ids=token_ids)

        with torch.no_grad():
            output = backbone.forward_language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
            )

        assert output.logits.shape == (
            token_ids.shape[0],
            token_ids.shape[1],
            resized_vocab_size,
        )
        assert output.hidden_states[-1].shape == inputs_embeds.shape
        assert output.past_key_values is not None

    @pytest.mark.integration
    def test_backbone_accessors_return_real_modules(
        self,
        real_paligemma_backbone: Callable[..., PaliGemmaVLM],
    ) -> None:
        backbone = real_paligemma_backbone(model_dtype=PrecisionType.FP32.value)
        layers = backbone.get_backbone_layers()
        assert len(layers) > 0
        hidden = torch.zeros(1, 1, backbone.hidden_dimension)
        position_ids = torch.zeros(1, 1, dtype=torch.long)
        cos, sin = backbone.get_rotary_embedding()(hidden, position_ids)
        assert cos.shape[0] == hidden.shape[0]
        assert sin.shape[0] == hidden.shape[0]
        assert backbone.get_backbone_hidden_dim() == backbone.hidden_dimension

    @pytest.mark.integration
    def test_real_model_explainability_target_captures_image_tokens(
        self,
        real_paligemma_backbone: Callable[..., PaliGemmaVLM],
        rng: np.random.Generator,
    ) -> None:
        batch_size = 1
        backbone = real_paligemma_backbone(model_dtype=PrecisionType.FP32.value)
        target = backbone.get_explainability_targets()[0]
        captured_output = {}

        def capture_output(
            module: nn.Module,
            module_input: tuple[torch.Tensor, ...],
            module_output: torch.Tensor,
        ) -> None:
            del module, module_input
            captured_output["image_tokens"] = module_output

        handle = target.layer.register_forward_hook(capture_output)
        try:
            images = torch.from_numpy(
                rng.standard_normal(
                    (batch_size, 1, 3, backbone.image_size, backbone.image_size)
                ).astype(np.float32)
            )
            token_ids = torch.from_numpy(
                rng.integers(
                    low=0,
                    high=backbone.get_vocab_size(),
                    size=(batch_size, 1, 10),
                ).astype(np.int64)
            )
            with torch.no_grad():
                backbone(
                    inputs={
                        Cameras.LEFT.value: images,
                        SampleKey.TOKENIZED_OBSERVATIONS.value: token_ids,
                    }
                )
        finally:
            handle.remove()

        image_tokens = captured_output["image_tokens"]
        assert image_tokens.shape == (
            batch_size,
            backbone.num_image_tokens_per_camera,
            backbone.hidden_dimension,
        )
        assert target.patch_grid == (4, 4)

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "precision, frozen, expected_dtype",
        [
            (PrecisionType.FP32.value, True, torch.float32),
            (PrecisionType.BF16_MIXED.value, True, torch.bfloat16),
            (PrecisionType.BF16_MIXED.value, False, torch.float32),
        ],
    )
    def test_model_dtype_sets_vlm_parameter_dtype(
        self,
        real_paligemma_backbone: Callable[..., PaliGemmaVLM],
        precision: str,
        frozen: bool,
        expected_dtype: torch.dtype,
    ) -> None:
        backbone = real_paligemma_backbone(model_dtype=precision, frozen=frozen)
        param_dtype = next(backbone.vlm.parameters()).dtype
        assert param_dtype == expected_dtype
