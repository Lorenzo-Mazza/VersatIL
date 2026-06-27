"""Tests for versatil.models.decoding.generative_language_models.vision_language.smolvlm module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from transformers import (
    Idefics3Config,
    Idefics3ForConditionalGeneration,
    PretrainedConfig,
)
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.llama.modeling_llama import LlamaModel

from versatil.data.constants import Cameras, SampleKey
from versatil.data.metadata import BaseMetadata, CameraMetadata, RGBCameraMetadata
from versatil.models.adaptation.constants import LoRATargetModulePreset
from versatil.models.adaptation.lora import LoRAAdaptation
from versatil.models.decoding.generative_language_models.constants import (
    SmolVLMModelType,
)
from versatil.models.decoding.generative_language_models.vision_language.smolvlm import (
    SmolVLM,
)
from versatil.models.encoding.encoders.constants import EncoderOutputKeys
from versatil.models.encoding.explainability import (
    ActivationLayout,
    ExplanationTargetKind,
)
from versatil.training.constants import PrecisionType

HIDDEN_DIM = 64
IMAGE_SIZE = 56
PATCH_SIZE = 14
SCALE_FACTOR = 4
NUM_IMAGE_TOKENS = 1  # (56/14)^2 / 4^2 = 16/16
MAX_TEXT_LENGTH = 128
VOCAB_SIZE = 32000
DEFAULT_INPUT_KEYS = [Cameras.LEFT.value]


def _return_model(
    model: MagicMock,
    lora_config: LoRAAdaptation | None,
    frozen: bool,
) -> MagicMock:
    return model


def _create_mock_config() -> MagicMock:
    config = MagicMock(spec=Idefics3Config)
    config.vision_config = MagicMock(spec=PretrainedConfig)
    config.text_config = MagicMock(spec=PretrainedConfig)
    config.image_token_id = 99999
    config.scale_factor = SCALE_FACTOR
    config.vision_config.image_size = IMAGE_SIZE
    config.vision_config.patch_size = PATCH_SIZE
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
        mock_vlm = MagicMock(spec=Idefics3ForConditionalGeneration)
        text_model = MagicMock(spec=LlamaModel)
        text_model.config = MagicMock(spec=PretrainedConfig)
        text_model.config.vocab_size = VOCAB_SIZE
        text_model.layers = [MagicMock(spec=nn.Module)]
        text_model.rotary_emb = MagicMock(spec=nn.Module)
        mock_vlm.model = MagicMock(spec=nn.Module)
        mock_vlm.model.text_model = text_model
        mock_vlm.model.connector = nn.Identity()

        mock_image_output = MagicMock(spec=BaseModelOutputWithPooling)
        # Real VLM returns (B * num_cameras, tokens_per_camera, hidden_dim)
        mock_image_output.pooler_output = torch.zeros(
            batch_size * num_cameras, NUM_IMAGE_TOKENS, HIDDEN_DIM
        )
        mock_vlm.get_image_features.return_value = mock_image_output

        mock_embed = MagicMock(spec=nn.Embedding)
        mock_embed.return_value = torch.zeros(batch_size, MAX_TEXT_LENGTH, HIDDEN_DIM)
        mock_vlm.model.text_model.get_input_embeddings.return_value = mock_embed

        total_seq = NUM_IMAGE_TOKENS * num_cameras + MAX_TEXT_LENGTH
        mock_lm_output = MagicMock(spec=BaseModelOutput)
        mock_lm_output.last_hidden_state = torch.zeros(
            batch_size, total_seq, HIDDEN_DIM
        )
        mock_vlm.model.text_model.return_value = mock_lm_output

        return mock_vlm

    return factory


@pytest.fixture
def smolvlm_backbone_factory(
    mock_vlm_factory: Callable[..., MagicMock],
) -> Callable[..., SmolVLM]:
    def factory(
        input_keys: str | list[str] | None = None,
        pretrained: bool = False,
        frozen: bool = False,
        lora_config: LoRAAdaptation | None = None,
    ) -> SmolVLM:
        if input_keys is None:
            input_keys = [Cameras.LEFT.value]
        mock_config = _create_mock_config()
        camera_keys = [input_keys] if isinstance(input_keys, str) else input_keys
        camera_count = len(camera_keys)
        mock_vlm = mock_vlm_factory(num_cameras=max(camera_count, 1))

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
            return SmolVLM(
                input_keys=input_keys,
                pretrained=pretrained,
                frozen=frozen,
                model_name=SmolVLMModelType.SMOLVLM_256M.value,
                lora_config=lora_config,
            )

    return factory


@pytest.fixture
def smolvlm_input_factory(
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
    backbone: SmolVLM,
    effective_batch_size: int,
) -> None:
    num_cameras = len(backbone.camera_keys)
    total_image_tokens = backbone.total_image_tokens
    mock_image_output = MagicMock(spec=BaseModelOutputWithPooling)
    # Real VLM returns (B * num_cameras, tokens_per_camera, hidden_dim)
    mock_image_output.pooler_output = torch.zeros(
        effective_batch_size * num_cameras,
        backbone.num_image_tokens_per_camera,
        HIDDEN_DIM,
    )
    backbone.vlm.get_image_features.return_value = mock_image_output

    mock_embed = MagicMock(spec=nn.Embedding)
    mock_embed.return_value = torch.zeros(
        effective_batch_size, MAX_TEXT_LENGTH, HIDDEN_DIM
    )
    backbone.vlm.model.text_model.get_input_embeddings.return_value = mock_embed

    total_seq = total_image_tokens + MAX_TEXT_LENGTH
    mock_lm_output = MagicMock(spec=BaseModelOutput)
    mock_lm_output.last_hidden_state = torch.zeros(
        effective_batch_size, total_seq, HIDDEN_DIM
    )
    backbone.vlm.model.text_model.return_value = mock_lm_output


@pytest.mark.unit
class TestSmolVLMInitialization:
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
        smolvlm_backbone_factory: Callable[..., SmolVLM],
        input_keys: list[str],
        expected_camera_count: int,
        frozen: bool,
    ) -> None:
        backbone = smolvlm_backbone_factory(
            input_keys=input_keys,
            pretrained=False,
            frozen=frozen,
        )
        assert backbone.hidden_dim == HIDDEN_DIM
        assert backbone.image_size == IMAGE_SIZE
        assert backbone.max_text_length == MAX_TEXT_LENGTH
        assert backbone.num_image_tokens_per_camera == NUM_IMAGE_TOKENS
        assert len(backbone.camera_keys) == expected_camera_count
        assert SampleKey.IS_PAD_OBSERVATION.value in backbone.input_specification.keys
        assert backbone.is_stacked_camera_batch
        if frozen:
            for parameter in backbone.parameters():
                assert not parameter.requires_grad

    def test_applies_lora_to_loaded_vlm(
        self,
        smolvlm_backbone_factory: Callable[..., SmolVLM],
    ) -> None:
        lora_config = LoRAAdaptation(
            enabled=True,
            target_modules=LoRATargetModulePreset.ALL_LINEAR.value,
        )

        with patch(
            "versatil.models.decoding.generative_language_models.vision_language.huggingface.apply_lora_config",
            side_effect=_return_model,
        ) as mock_apply_lora:
            backbone = smolvlm_backbone_factory(lora_config=lora_config)

        mock_apply_lora.assert_called_once()
        assert mock_apply_lora.call_args.kwargs["lora_config"] is lora_config
        assert mock_apply_lora.call_args.kwargs["frozen"] is False
        assert backbone.lora_config is lora_config

    @pytest.mark.parametrize(
        "image_size, patch_size, scale_factor, expected_tokens",
        [
            (56, 14, 4, 1),
            (64, 16, 2, 4),
            (512, 16, 4, 64),
        ],
    )
    def test_num_image_tokens_computed_from_patches_and_scale(
        self,
        image_size: int,
        patch_size: int,
        scale_factor: int,
        expected_tokens: int,
        mock_vlm_factory: Callable[..., MagicMock],
    ) -> None:
        mock_config = _create_mock_config()
        mock_config.vision_config.image_size = image_size
        mock_config.vision_config.patch_size = patch_size
        mock_config.scale_factor = scale_factor
        mock_vlm = mock_vlm_factory()

        with (
            patch(
                "versatil.models.decoding.generative_language_models.vision_language.huggingface.AutoConfig.from_pretrained",
                return_value=mock_config,
            ),
            patch(
                "versatil.models.decoding.generative_language_models.vision_language.huggingface.AutoModelForImageTextToText.from_config",
                return_value=mock_vlm,
            ),
        ):
            backbone = SmolVLM(
                input_keys=[
                    Cameras.LEFT.value,
                    SampleKey.TOKENIZED_OBSERVATIONS.value,
                ],
                pretrained=False,
                frozen=False,
                model_name=SmolVLMModelType.SMOLVLM_256M.value,
            )
        assert backbone.num_image_tokens_per_camera == expected_tokens

    def test_exposes_connector_as_stacked_camera_token_target(
        self,
        smolvlm_backbone_factory: Callable[..., SmolVLM],
    ) -> None:
        backbone = smolvlm_backbone_factory(
            input_keys=[Cameras.LEFT.value, Cameras.RIGHT.value],
            pretrained=False,
            frozen=False,
        )

        targets = backbone.get_explainability_targets()

        assert len(targets) == 1
        assert targets[0].layer is backbone.vlm.model.connector
        assert targets[0].target_kind == ExplanationTargetKind.TOKEN_SEQUENCE.value
        assert targets[0].activation_layout == ActivationLayout.NLC.value
        assert targets[0].prefix_token_count == 0
        assert targets[0].patch_grid == (1, 1)


@pytest.mark.unit
class TestSmolVLMForward:
    @pytest.mark.parametrize("time_steps", [1, 3])
    def test_output_shape(
        self,
        smolvlm_backbone_factory: Callable[..., SmolVLM],
        smolvlm_input_factory: Callable[..., dict[str, torch.Tensor]],
        time_steps: int,
    ) -> None:
        batch_size = 2
        backbone = smolvlm_backbone_factory(
            input_keys=DEFAULT_INPUT_KEYS,
            pretrained=False,
            frozen=False,
        )
        _setup_mock_vlm_for_batch(backbone, batch_size * time_steps)
        inputs = smolvlm_input_factory(
            batch_size=batch_size,
            time_steps=time_steps,
            include_padding_mask=True,
        )
        output = backbone(inputs=inputs)
        fused = output[EncoderOutputKeys.FUSED_RGB_LANGUAGE.value]
        total_seq = NUM_IMAGE_TOKENS + MAX_TEXT_LENGTH
        assert fused.shape == (batch_size, time_steps, total_seq, HIDDEN_DIM)

    def test_output_contains_fused_features_and_padding_mask(
        self,
        smolvlm_backbone_factory: Callable[..., SmolVLM],
        smolvlm_input_factory: Callable[..., dict[str, torch.Tensor]],
    ) -> None:
        batch_size = 2
        backbone = smolvlm_backbone_factory(
            input_keys=DEFAULT_INPUT_KEYS,
            pretrained=False,
            frozen=False,
        )
        _setup_mock_vlm_for_batch(backbone, batch_size)
        inputs = smolvlm_input_factory(
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

    def test_image_embeddings_scaled_by_sqrt_hidden_dim(
        self,
        smolvlm_backbone_factory: Callable[..., SmolVLM],
        smolvlm_input_factory: Callable[..., dict[str, torch.Tensor]],
        rng: np.random.Generator,
    ) -> None:
        batch_size = 2
        backbone = smolvlm_backbone_factory(
            input_keys=DEFAULT_INPUT_KEYS,
            pretrained=False,
            frozen=False,
        )
        _setup_mock_vlm_for_batch(backbone, batch_size)
        raw_image_embeddings = torch.from_numpy(
            rng.standard_normal((batch_size, NUM_IMAGE_TOKENS, HIDDEN_DIM)).astype(
                np.float32
            )
        )
        mock_image_output = MagicMock(spec=BaseModelOutputWithPooling)
        mock_image_output.pooler_output = raw_image_embeddings.clone()
        backbone.vlm.get_image_features.return_value = mock_image_output
        mock_embed = MagicMock(spec=nn.Embedding)
        mock_embed.return_value = torch.zeros(batch_size, MAX_TEXT_LENGTH, HIDDEN_DIM)
        backbone.vlm.model.text_model.get_input_embeddings.return_value = mock_embed
        inputs = smolvlm_input_factory(
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
        inputs_embeds = backbone.vlm.model.text_model.call_args.kwargs["inputs_embeds"]
        image_portion = inputs_embeds[:, :NUM_IMAGE_TOKENS, :]
        expected_scale = HIDDEN_DIM**0.5
        assert torch.allclose(
            image_portion, raw_image_embeddings * expected_scale, atol=1e-5
        )

    def test_language_embeddings_scaled_by_sqrt_hidden_dim(
        self,
        smolvlm_backbone_factory: Callable[..., SmolVLM],
        smolvlm_input_factory: Callable[..., dict[str, torch.Tensor]],
        rng: np.random.Generator,
    ) -> None:
        batch_size = 2
        backbone = smolvlm_backbone_factory(
            input_keys=DEFAULT_INPUT_KEYS,
            pretrained=False,
            frozen=False,
        )
        _setup_mock_vlm_for_batch(backbone, batch_size)
        raw_language_embeddings = torch.from_numpy(
            rng.standard_normal((batch_size, MAX_TEXT_LENGTH, HIDDEN_DIM)).astype(
                np.float32
            )
        )
        mock_image_output = MagicMock(spec=BaseModelOutputWithPooling)
        mock_image_output.pooler_output = torch.zeros(
            batch_size, NUM_IMAGE_TOKENS, HIDDEN_DIM
        )
        backbone.vlm.get_image_features.return_value = mock_image_output
        mock_embed = MagicMock(spec=nn.Embedding)
        mock_embed.return_value = raw_language_embeddings.clone()
        backbone.vlm.model.text_model.get_input_embeddings.return_value = mock_embed
        inputs = smolvlm_input_factory(
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
        inputs_embeds = backbone.vlm.model.text_model.call_args.kwargs["inputs_embeds"]
        language_portion = inputs_embeds[:, NUM_IMAGE_TOKENS:, :]
        expected_scale = HIDDEN_DIM**0.5
        assert torch.allclose(
            language_portion, raw_language_embeddings * expected_scale, atol=1e-5
        )

    def test_images_stacked_along_num_images_dim_for_idefics3(
        self,
        smolvlm_backbone_factory: Callable[..., SmolVLM],
        smolvlm_input_factory: Callable[..., dict[str, torch.Tensor]],
    ) -> None:
        batch_size = 2
        backbone = smolvlm_backbone_factory(
            input_keys=DEFAULT_INPUT_KEYS,
            pretrained=False,
            frozen=False,
        )
        _setup_mock_vlm_for_batch(backbone, batch_size)
        inputs = smolvlm_input_factory(
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
        call_args = backbone.vlm.get_image_features.call_args
        pixel_values = call_args[0][0]
        # Single camera → (B, 1, C, H, W)
        assert pixel_values.ndim == 5
        assert pixel_values.shape[1] == 1

    def test_text_model_attention_mask_marks_auto_padded_text_tokens(
        self,
        smolvlm_backbone_factory: Callable[..., SmolVLM],
        smolvlm_input_factory: Callable[..., dict[str, torch.Tensor]],
    ) -> None:
        batch_size = 2
        sequence_length = 5
        backbone = smolvlm_backbone_factory(
            input_keys=DEFAULT_INPUT_KEYS,
            pretrained=False,
            frozen=False,
        )
        _setup_mock_vlm_for_batch(backbone, batch_size)
        inputs = smolvlm_input_factory(
            batch_size=batch_size,
            camera_key=Cameras.LEFT.value,
            channels=3,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            sequence_length=sequence_length,
            time_steps=1,
            include_padding_mask=False,
        )
        backbone(inputs=inputs)
        attention_mask = backbone.vlm.model.text_model.call_args.kwargs[
            "attention_mask"
        ]
        image_attention = attention_mask[:, :NUM_IMAGE_TOKENS]
        text_attention = attention_mask[:, NUM_IMAGE_TOKENS:]
        assert image_attention.all()
        assert text_attention[:, :sequence_length].all()
        assert not text_attention[:, sequence_length:].any()

    def test_multi_camera_stacks_all_cameras_in_single_call(
        self,
        smolvlm_backbone_factory: Callable[..., SmolVLM],
        smolvlm_input_factory: Callable[..., dict[str, torch.Tensor]],
    ) -> None:
        batch_size = 2
        input_keys = [
            Cameras.LEFT.value,
            Cameras.RIGHT.value,
            SampleKey.TOKENIZED_OBSERVATIONS.value,
        ]
        backbone = smolvlm_backbone_factory(
            input_keys=input_keys,
            pretrained=False,
            frozen=False,
        )
        _setup_mock_vlm_for_batch(backbone, batch_size)
        left_inputs = smolvlm_input_factory(
            camera_key=Cameras.LEFT.value,
            batch_size=batch_size,
            channels=3,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            sequence_length=10,
            time_steps=1,
            include_padding_mask=False,
        )
        right_inputs = smolvlm_input_factory(
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

        assert backbone.vlm.get_image_features.call_count == 1
        call_args = backbone.vlm.get_image_features.call_args
        pixel_values = call_args[0][0]
        assert pixel_values.shape[1] == 2
        torch.testing.assert_close(
            pixel_values[:, 0],
            inputs[Cameras.LEFT.value].squeeze(1),
        )
        torch.testing.assert_close(
            pixel_values[:, 1],
            inputs[Cameras.RIGHT.value].squeeze(1),
        )

    def test_padding_mask_image_portion_is_never_padded(
        self,
        smolvlm_backbone_factory: Callable[..., SmolVLM],
        smolvlm_input_factory: Callable[..., dict[str, torch.Tensor]],
    ) -> None:
        batch_size = 2
        backbone = smolvlm_backbone_factory(
            input_keys=DEFAULT_INPUT_KEYS,
            pretrained=False,
            frozen=False,
        )
        _setup_mock_vlm_for_batch(backbone, batch_size)
        inputs = smolvlm_input_factory(
            camera_key=Cameras.LEFT.value,
            batch_size=batch_size,
            channels=3,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            sequence_length=10,
            time_steps=1,
            include_padding_mask=True,
        )
        output = backbone(inputs=inputs)
        padding_mask = output[backbone.padding_mask_name]
        image_portion = padding_mask[:, :, :NUM_IMAGE_TOKENS]
        assert not image_portion.any()

    def test_text_model_called_for_fused_vision_language_features(
        self,
        smolvlm_backbone_factory: Callable[..., SmolVLM],
        smolvlm_input_factory: Callable[..., dict[str, torch.Tensor]],
    ) -> None:
        batch_size = 2
        backbone = smolvlm_backbone_factory(
            input_keys=DEFAULT_INPUT_KEYS,
            pretrained=False,
            frozen=False,
        )
        _setup_mock_vlm_for_batch(backbone, batch_size)
        inputs = smolvlm_input_factory(
            camera_key=Cameras.LEFT.value,
            batch_size=batch_size,
            channels=3,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            sequence_length=10,
            time_steps=1,
            include_padding_mask=True,
        )
        output = backbone(inputs=inputs)
        fused = output[EncoderOutputKeys.FUSED_RGB_LANGUAGE.value]
        total_seq = NUM_IMAGE_TOKENS + MAX_TEXT_LENGTH
        assert fused.shape == (batch_size, 1, total_seq, HIDDEN_DIM)
        assert backbone.padding_mask_name in output
        backbone.vlm.model.text_model.assert_called_once()

    def test_missing_language_key_raises(
        self,
        smolvlm_backbone_factory: Callable[..., SmolVLM],
        rng: np.random.Generator,
    ) -> None:
        backbone = smolvlm_backbone_factory(
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
                f"SmolVLM expects pre-tokenized input. "
                f"Expected key '{SampleKey.TOKENIZED_OBSERVATIONS.value}' "
                f"not found in inputs. "
                f"Ensure tokenization is enabled in DataloaderConfig."
            ),
        ):
            backbone(inputs={Cameras.LEFT.value: images})


@pytest.mark.unit
class TestSmolVLMValidateInputMetadata:
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
        smolvlm_backbone_factory: Callable[..., SmolVLM],
        metadata: BaseMetadata,
        expected_error: str | None,
    ) -> None:
        backbone = smolvlm_backbone_factory(
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
                f"SmolVLM cannot process image data for "
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
        smolvlm_backbone_factory: Callable[..., SmolVLM],
        metadata: BaseMetadata,
        expected_error: str | None,
    ) -> None:
        backbone = smolvlm_backbone_factory(
            input_keys=DEFAULT_INPUT_KEYS,
            pretrained=False,
            frozen=False,
        )
        result = backbone.validate_input_metadata(
            key=SampleKey.TOKENIZED_OBSERVATIONS.value, metadata=metadata
        )
        assert result == expected_error


@pytest.mark.unit
class TestSmolVLMGetVocabSize:
    def test_returns_text_model_vocab_size(
        self,
        smolvlm_backbone_factory: Callable[..., SmolVLM],
    ) -> None:
        backbone = smolvlm_backbone_factory(
            input_keys=DEFAULT_INPUT_KEYS,
            pretrained=False,
            frozen=False,
        )
        assert backbone.get_vocab_size() == VOCAB_SIZE

    def test_resize_token_embeddings_delegates_to_vlm(
        self,
        smolvlm_backbone_factory: Callable[..., SmolVLM],
    ) -> None:
        backbone = smolvlm_backbone_factory(
            input_keys=DEFAULT_INPUT_KEYS,
            pretrained=False,
            frozen=False,
        )

        backbone.resize_token_embeddings(vocabulary_size=VOCAB_SIZE + 1)

        backbone.vlm.resize_token_embeddings.assert_called_once_with(VOCAB_SIZE + 1)
        assert backbone.get_vocab_size() == VOCAB_SIZE + 1


@pytest.mark.unit
class TestSmolVLMBackboneAccessors:
    def test_get_backbone_layers_accesses_text_model_layers(
        self,
        smolvlm_backbone_factory: Callable[..., SmolVLM],
    ) -> None:
        backbone = smolvlm_backbone_factory(
            input_keys=DEFAULT_INPUT_KEYS,
            pretrained=False,
            frozen=False,
        )
        result = backbone.get_backbone_layers()
        assert result == backbone.vlm.model.text_model.layers

    def test_get_rotary_embedding_accesses_text_model_rotary_emb(
        self,
        smolvlm_backbone_factory: Callable[..., SmolVLM],
    ) -> None:
        backbone = smolvlm_backbone_factory(
            input_keys=DEFAULT_INPUT_KEYS,
            pretrained=False,
            frozen=False,
        )
        result = backbone.get_rotary_embedding()
        assert result == backbone.vlm.model.text_model.rotary_emb

    def test_get_backbone_hidden_dim_returns_hidden_dim(
        self,
        smolvlm_backbone_factory: Callable[..., SmolVLM],
    ) -> None:
        backbone = smolvlm_backbone_factory(
            input_keys=DEFAULT_INPUT_KEYS,
            pretrained=False,
            frozen=False,
        )
        assert backbone.get_backbone_hidden_dim() == HIDDEN_DIM


class TestSmolVLMIntegration:
    @pytest.mark.integration
    def test_exposes_real_connector_token_target(
        self,
        real_smolvlm_backbone: Callable[..., SmolVLM],
    ) -> None:
        backbone = real_smolvlm_backbone(model_dtype=PrecisionType.FP32.value)

        targets = backbone.get_explainability_targets()

        assert len(targets) == 1
        assert targets[0].layer is backbone.vlm.model.connector
        assert targets[0].target_kind == ExplanationTargetKind.TOKEN_SEQUENCE.value
        assert targets[0].activation_layout == ActivationLayout.NLC.value
        assert targets[0].prefix_token_count == 0
        assert targets[0].patch_grid == backbone._get_image_token_grid()

    @pytest.mark.integration
    @pytest.mark.parametrize("lora_enabled", [False, True])
    def test_forward_pass_with_real_model(
        self,
        real_smolvlm_backbone: Callable[..., SmolVLM],
        smolvlm_input_factory: Callable[..., dict[str, torch.Tensor]],
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
        backbone = real_smolvlm_backbone(
            model_dtype=PrecisionType.FP32.value,
            lora_config=lora_config,
        )
        backbone.eval()
        inputs = smolvlm_input_factory(
            batch_size=batch_size,
            height=backbone.image_size,
            width=backbone.image_size,
            sequence_length=10,
        )
        with torch.no_grad():
            output = backbone(inputs=inputs)
        fused = output[EncoderOutputKeys.FUSED_RGB_LANGUAGE.value]
        assert fused.shape[0] == batch_size
        assert fused.shape[-1] == backbone.hidden_dim
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
            assert all(".text_model." in name for name in trainable_parameter_names)
            assert 0 < trainable_parameters < total_parameters

    @pytest.mark.integration
    def test_forward_language_model_with_real_peft_resized_vocabulary(
        self,
        real_smolvlm_backbone: Callable[..., SmolVLM],
    ) -> None:
        lora_config = LoRAAdaptation(
            enabled=True,
            rank=3,
            alpha=6,
            target_modules=(
                LoRATargetModulePreset.VLM_TEXT_MODEL_QUERY_VALUE_PROJECTIONS.value
            ),
        )
        backbone = real_smolvlm_backbone(
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
        real_smolvlm_backbone: Callable[..., SmolVLM],
    ) -> None:
        backbone = real_smolvlm_backbone(model_dtype=PrecisionType.FP32.value)
        layers = backbone.get_backbone_layers()
        assert len(layers) > 0
        hidden = torch.zeros(1, 1, backbone.hidden_dim)
        position_ids = torch.zeros(1, 1, dtype=torch.long)
        cos, sin = backbone.get_rotary_embedding()(hidden, position_ids)
        assert cos.shape[0] == hidden.shape[0]
        assert sin.shape[0] == hidden.shape[0]
        assert backbone.get_backbone_hidden_dim() == backbone.hidden_dim

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "precision, expected_dtype",
        [
            (PrecisionType.FP32.value, torch.float32),
            (PrecisionType.BF16_MIXED.value, torch.bfloat16),
        ],
    )
    def test_model_dtype_sets_vlm_parameter_dtype(
        self,
        real_smolvlm_backbone: Callable[..., SmolVLM],
        precision: str,
        expected_dtype: torch.dtype,
    ) -> None:
        backbone = real_smolvlm_backbone(model_dtype=precision)
        param_dtype = next(backbone.vlm.parameters()).dtype
        assert param_dtype == expected_dtype
