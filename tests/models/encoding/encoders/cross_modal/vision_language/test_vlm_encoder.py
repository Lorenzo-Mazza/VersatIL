"""Tests for versatil.models.encoding.encoders.cross_modal.vision_language.vlm_encoder module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling

from versatil.data.constants import Cameras, SampleKey
from versatil.data.metadata import BaseMetadata, CameraMetadata, RGBCameraMetadata
from versatil.models.adaptation.constants import LoRATargetModulePreset
from versatil.models.adaptation.lora import LoRAAdaptation
from versatil.models.encoding.encoders.constants import (
    EncoderOutputKeys,
    ImageTextModelType,
    PoolingMethod,
)
from versatil.models.encoding.encoders.cross_modal.vision_language.vlm_encoder import (
    VLMEncoder,
)

HIDDEN_VISION_DIM = 768
HIDDEN_LANGUAGE_DIM = 512
IMAGE_SIZE = 224
MAX_TEXT_LENGTH = 77
VOCAB_SIZE = 49408


def _return_model(
    model: MagicMock,
    lora_config: LoRAAdaptation | None,
    frozen: bool,
) -> MagicMock:
    return model


def _process_images(
    images: torch.Tensor,
    return_tensors: str,
) -> dict[str, torch.Tensor]:
    del return_tensors
    return {"pixel_values": images}


def _create_mock_encoder() -> MagicMock:
    mock_encoder = MagicMock()
    mock_vision_config = MagicMock()
    mock_vision_config.hidden_size = HIDDEN_VISION_DIM
    mock_vision_config.image_size = IMAGE_SIZE
    mock_encoder.vision_model.config = mock_vision_config
    mock_text_config = MagicMock()
    mock_text_config.hidden_size = HIDDEN_LANGUAGE_DIM
    mock_text_config.max_position_embeddings = MAX_TEXT_LENGTH
    mock_text_config.vocab_size = VOCAB_SIZE
    mock_encoder.text_model.config = mock_text_config
    return mock_encoder


@pytest.fixture
def vlm_encoder_factory() -> Callable[..., VLMEncoder]:
    """Factory for VLMEncoder with mocked HuggingFace model downloads."""

    def factory(
        input_keys: str | list[str] | None = None,
        pretrained: bool = False,
        frozen: bool = False,
        pooling_method: str = PoolingMethod.DEFAULT.value,
        model_name: str = ImageTextModelType.CLIP_VITB32.value,
        lora_config: LoRAAdaptation | None = None,
    ) -> VLMEncoder:
        if input_keys is None:
            input_keys = [Cameras.LEFT.value]
        mock_encoder = _create_mock_encoder()
        mock_config = MagicMock()
        mock_image_processor = MagicMock(side_effect=_process_images)

        with (
            patch(
                "versatil.models.encoding.encoders.cross_modal.vision_language.vlm_encoder.AutoConfig.from_pretrained",
                return_value=mock_config,
            ),
            patch(
                "versatil.models.encoding.encoders.cross_modal.vision_language.vlm_encoder.AutoModel.from_pretrained",
                return_value=mock_encoder,
            ),
            patch(
                "versatil.models.encoding.encoders.cross_modal.vision_language.vlm_encoder.AutoModel.from_config",
                return_value=mock_encoder,
            ),
            patch(
                "versatil.models.encoding.encoders.cross_modal.vision_language.vlm_encoder.AutoImageProcessor.from_pretrained",
                return_value=mock_image_processor,
            ),
        ):
            return VLMEncoder(
                input_keys=input_keys,
                pretrained=pretrained,
                frozen=frozen,
                pooling_method=pooling_method,
                model_name=model_name,
                lora_config=lora_config,
            )

    return factory


@pytest.fixture
def vlm_input_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    """Factory for VLM input tensors with images and tokenized text."""

    def factory(
        camera_key: str = Cameras.LEFT.value,
        batch_size: int = 2,
        channels: int = 3,
        height: int = 224,
        width: int = 224,
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


class TestVLMEncoderInitialization:
    def test_has_encoder_interface(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
    ):
        encoder = vlm_encoder_factory()
        spec = encoder.get_output_specification()
        feature_keys = [m.key for m in spec]
        assert len(feature_keys) == 3
        assert EncoderOutputKeys.RGB.value in feature_keys
        assert EncoderOutputKeys.LANGUAGE.value in feature_keys
        assert encoder.padding_mask_name in feature_keys

    @pytest.mark.parametrize(
        "input_keys",
        [[Cameras.LEFT.value], [Cameras.RIGHT.value]],
    )
    @pytest.mark.parametrize(
        "pooling_method",
        [
            PoolingMethod.DEFAULT.value,
            PoolingMethod.AVERAGE.value,
        ],
    )
    def test_stores_configuration(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
        input_keys: list[str],
        pooling_method: str,
    ):
        encoder = vlm_encoder_factory(
            input_keys=input_keys,
            pooling_method=pooling_method,
        )
        expected_camera_keys = [
            key for key in input_keys if key != SampleKey.TOKENIZED_OBSERVATIONS.value
        ]
        assert encoder.camera_keys == expected_camera_keys
        assert encoder.language_key == SampleKey.TOKENIZED_OBSERVATIONS.value
        assert encoder.pooling_method == pooling_method
        assert encoder.hidden_vision_dim == HIDDEN_VISION_DIM
        assert encoder.hidden_language_dim == HIDDEN_LANGUAGE_DIM
        assert encoder.max_text_length == MAX_TEXT_LENGTH
        assert SampleKey.IS_PAD_OBSERVATION.value in encoder.input_specification.keys

    def test_requires_tokenized_specification(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
    ):
        encoder = vlm_encoder_factory()
        assert encoder.input_specification.requires_tokenized is True

    def test_padding_mask_name_format(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
    ):
        encoder = vlm_encoder_factory()
        expected = (
            f"{EncoderOutputKeys.LANGUAGE.value}_{EncoderOutputKeys.PADDING_MASK.value}"
        )
        assert encoder.padding_mask_name == expected

    def test_applies_lora_to_loaded_model(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
    ) -> None:
        lora_config = LoRAAdaptation(
            enabled=True,
            target_modules=LoRATargetModulePreset.ALL_LINEAR.value,
        )

        with patch(
            "versatil.models.encoding.encoders.cross_modal.vision_language.vlm_encoder.apply_lora_config",
            side_effect=_return_model,
        ) as mock_apply_lora:
            encoder = vlm_encoder_factory(lora_config=lora_config)

        mock_apply_lora.assert_called_once()
        assert mock_apply_lora.call_args.kwargs["lora_config"] is lora_config
        assert mock_apply_lora.call_args.kwargs["frozen"] is False
        assert encoder.lora_config is lora_config


class TestVLMEncoderPadTextInputs:
    def test_truncation_when_longer_than_max_text_length(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
        rng: np.random.Generator,
    ):
        encoder = vlm_encoder_factory()
        longer_sequence_length = MAX_TEXT_LENGTH + 20
        text_ids = torch.from_numpy(
            rng.integers(
                low=0, high=VOCAB_SIZE, size=(2, longer_sequence_length)
            ).astype(np.int64)
        )
        mask = torch.zeros(2, longer_sequence_length, dtype=torch.bool)
        result_ids, result_mask = encoder._pad_text_inputs(
            text_input_ids=text_ids,
            language_mask=mask,
            max_length=MAX_TEXT_LENGTH,
        )
        assert result_ids.shape[1] == MAX_TEXT_LENGTH
        assert result_mask.shape[1] == MAX_TEXT_LENGTH

    def test_padding_when_shorter_than_max_text_length(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
        rng: np.random.Generator,
    ):
        encoder = vlm_encoder_factory()
        shorter_sequence_length = 10
        text_ids = torch.from_numpy(
            rng.integers(
                low=0, high=VOCAB_SIZE, size=(2, shorter_sequence_length)
            ).astype(np.int64)
        )
        mask = torch.zeros(2, shorter_sequence_length, dtype=torch.bool)
        result_ids, result_mask = encoder._pad_text_inputs(
            text_input_ids=text_ids,
            language_mask=mask,
            max_length=MAX_TEXT_LENGTH,
        )
        assert result_ids.shape[1] == MAX_TEXT_LENGTH
        assert result_mask.shape[1] == MAX_TEXT_LENGTH
        assert torch.equal(result_ids[:, :shorter_sequence_length], text_ids)
        assert torch.all(result_ids[:, shorter_sequence_length:] == 0)
        assert torch.all(result_mask[:, shorter_sequence_length:])

    def test_exact_length_unchanged(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
        rng: np.random.Generator,
    ):
        encoder = vlm_encoder_factory()
        text_ids = torch.from_numpy(
            rng.integers(low=0, high=VOCAB_SIZE, size=(2, MAX_TEXT_LENGTH)).astype(
                np.int64
            )
        )
        mask = torch.zeros(2, MAX_TEXT_LENGTH, dtype=torch.bool)
        result_ids, result_mask = encoder._pad_text_inputs(
            text_input_ids=text_ids,
            language_mask=mask,
            max_length=MAX_TEXT_LENGTH,
        )
        assert torch.equal(result_ids, text_ids)
        assert torch.equal(result_mask, mask)


class TestVLMEncoderForward:
    def _setup_encoder_mock_outputs(
        self,
        encoder: VLMEncoder,
        batch_size: int,
    ):
        vision_hidden = torch.zeros(batch_size, 49, HIDDEN_VISION_DIM)
        language_hidden = torch.zeros(batch_size, MAX_TEXT_LENGTH, HIDDEN_LANGUAGE_DIM)

        vision_pooler = torch.zeros(batch_size, HIDDEN_VISION_DIM)
        language_pooler = torch.zeros(batch_size, HIDDEN_LANGUAGE_DIM)

        mock_vision_output = BaseModelOutputWithPooling(
            last_hidden_state=vision_hidden,
            pooler_output=vision_pooler,
        )
        mock_language_output = BaseModelOutputWithPooling(
            last_hidden_state=language_hidden,
            pooler_output=language_pooler,
        )
        encoder.encoder.vision_model.return_value = mock_vision_output
        encoder.encoder.text_model.return_value = mock_language_output

    @pytest.mark.parametrize("time_steps", [1, 3])
    def test_output_shape_with_temporal_dimension(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
        vlm_input_factory: Callable[..., dict[str, torch.Tensor]],
        time_steps: int,
    ):
        batch_size = 2
        encoder = vlm_encoder_factory(pooling_method=PoolingMethod.DEFAULT.value)
        effective_batch = batch_size * time_steps
        self._setup_encoder_mock_outputs(
            encoder=encoder,
            batch_size=effective_batch,
        )
        inputs = vlm_input_factory(
            batch_size=batch_size,
            time_steps=time_steps,
            include_padding_mask=True,
        )
        output = encoder(inputs=inputs)
        image_features = output[EncoderOutputKeys.RGB.value]
        language_features = output[EncoderOutputKeys.LANGUAGE.value]
        assert image_features.shape == (batch_size, time_steps, HIDDEN_VISION_DIM)
        assert language_features.shape == (batch_size, time_steps, HIDDEN_LANGUAGE_DIM)

    def test_missing_language_key_raises(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
        rng: np.random.Generator,
    ):
        encoder = vlm_encoder_factory()
        images = torch.from_numpy(
            rng.standard_normal((2, 3, 224, 224)).astype(np.float32)
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"VLMEncoder expects pre-tokenized input. "
                f"Expected key '{SampleKey.TOKENIZED_OBSERVATIONS.value}' "
                f"not found in inputs. "
                f"Ensure tokenization is enabled in DataloaderConfig."
            ),
        ):
            encoder(inputs={Cameras.LEFT.value: images})

    def test_none_pooling_with_time_reshapes_to_4d(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
        vlm_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        time_steps = 3
        vision_seq_len = 49
        pooling_method = PoolingMethod.NONE.value
        encoder = vlm_encoder_factory(pooling_method=pooling_method)
        effective_batch = batch_size * time_steps
        self._setup_encoder_mock_outputs(
            encoder=encoder,
            batch_size=effective_batch,
        )
        inputs = vlm_input_factory(
            batch_size=batch_size,
            time_steps=time_steps,
            include_padding_mask=True,
        )
        output = encoder(inputs=inputs)
        image_features = output[EncoderOutputKeys.RGB.value]
        language_features = output[EncoderOutputKeys.LANGUAGE.value]
        padding_mask = output[encoder.padding_mask_name]
        # (B, T, vision_seq_len - 1, hidden_vision_dim) — CLS token excluded
        assert image_features.shape == (
            batch_size,
            time_steps,
            vision_seq_len - 1,
            HIDDEN_VISION_DIM,
        )
        # (B, T, max_text_length, hidden_language_dim)
        assert language_features.shape == (
            batch_size,
            time_steps,
            MAX_TEXT_LENGTH,
            HIDDEN_LANGUAGE_DIM,
        )
        # (B, T, max_text_length)
        assert padding_mask.shape == (batch_size, time_steps, MAX_TEXT_LENGTH)

    def test_output_contains_all_expected_keys(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
        vlm_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        encoder = vlm_encoder_factory(pooling_method=PoolingMethod.DEFAULT.value)
        self._setup_encoder_mock_outputs(
            encoder=encoder,
            batch_size=batch_size,
        )
        inputs = vlm_input_factory(batch_size=batch_size)
        output = encoder(inputs=inputs)
        assert EncoderOutputKeys.RGB.value in output
        assert EncoderOutputKeys.LANGUAGE.value in output
        assert encoder.padding_mask_name in output

    def test_pixel_attention_mask_forwarded_to_vision_model(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
        rng: np.random.Generator,
    ) -> None:
        batch_size = 2
        encoder = vlm_encoder_factory(pooling_method=PoolingMethod.DEFAULT.value)
        self._setup_encoder_mock_outputs(
            encoder=encoder,
            batch_size=batch_size,
        )
        images = torch.from_numpy(
            rng.standard_normal((batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)).astype(
                np.float32
            )
        )
        pixel_attention_mask = torch.ones(
            batch_size,
            IMAGE_SIZE,
            IMAGE_SIZE,
            dtype=torch.bool,
        )
        encoder.image_processor = MagicMock(
            return_value={
                "pixel_values": images,
                "pixel_attention_mask": pixel_attention_mask,
            }
        )
        encoder._encode_single_image(images=images)
        vision_call_kwargs = encoder.encoder.vision_model.call_args.kwargs
        assert vision_call_kwargs["pixel_attention_mask"] is pixel_attention_mask
        assert "attention_mask" not in vision_call_kwargs

    def test_average_pooling_ignores_language_padding_mask(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
        vlm_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        encoder = vlm_encoder_factory(pooling_method=PoolingMethod.AVERAGE.value)
        vision_hidden = torch.zeros(batch_size, 49, HIDDEN_VISION_DIM)
        vision_pooler = torch.zeros(batch_size, HIDDEN_VISION_DIM)
        language_hidden = torch.full(
            (batch_size, MAX_TEXT_LENGTH, HIDDEN_LANGUAGE_DIM),
            fill_value=100.0,
        )
        language_hidden[:, :3, :] = 2.0
        language_pooler = torch.zeros(batch_size, HIDDEN_LANGUAGE_DIM)
        encoder.encoder.vision_model.return_value = BaseModelOutputWithPooling(
            last_hidden_state=vision_hidden,
            pooler_output=vision_pooler,
        )
        encoder.encoder.text_model.return_value = BaseModelOutputWithPooling(
            last_hidden_state=language_hidden,
            pooler_output=language_pooler,
        )
        inputs = vlm_input_factory(
            batch_size=batch_size,
            sequence_length=5,
            include_padding_mask=True,
        )
        inputs[SampleKey.IS_PAD_OBSERVATION.value][:, :, 3:] = True
        output = encoder(inputs=inputs)
        language_features = output[EncoderOutputKeys.LANGUAGE.value]
        expected = torch.full(
            (batch_size, 1, HIDDEN_LANGUAGE_DIM),
            fill_value=2.0,
        )
        torch.testing.assert_close(language_features, expected)


class TestVLMEncoderValidateInputMetadata:
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
        vlm_encoder_factory: Callable[..., VLMEncoder],
        metadata,
        expected_error: str | None,
    ):
        encoder = vlm_encoder_factory()
        result = encoder.validate_input_metadata(
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
                f"VLMEncoder cannot process image data for "
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
        vlm_encoder_factory: Callable[..., VLMEncoder],
        metadata,
        expected_error: str | None,
    ):
        encoder = vlm_encoder_factory()
        result = encoder.validate_input_metadata(
            key=SampleKey.TOKENIZED_OBSERVATIONS.value, metadata=metadata
        )
        assert result == expected_error


def test_get_vocab_size_returns_text_model_vocab_size(
    vlm_encoder_factory: Callable[..., VLMEncoder],
) -> None:
    encoder = vlm_encoder_factory()
    assert encoder.get_vocab_size() == VOCAB_SIZE


class TestVLMEncoderGetOutputSpecification:
    @pytest.mark.parametrize(
        "pooling_method, expected_vision_dim, expected_language_dim",
        [
            (
                PoolingMethod.DEFAULT.value,
                HIDDEN_VISION_DIM,
                HIDDEN_LANGUAGE_DIM,
            ),
            (
                PoolingMethod.AVERAGE.value,
                HIDDEN_VISION_DIM,
                HIDDEN_LANGUAGE_DIM,
            ),
            (
                PoolingMethod.NONE.value,
                (-1, HIDDEN_VISION_DIM),
                (MAX_TEXT_LENGTH, HIDDEN_LANGUAGE_DIM),
            ),
        ],
    )
    def test_output_dimensions_match_pooling_method(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
        pooling_method: str,
        expected_vision_dim: int | tuple,
        expected_language_dim: int | tuple,
    ):
        encoder = vlm_encoder_factory(pooling_method=pooling_method)
        specification = encoder.get_output_specification()
        expected_vision = (
            expected_vision_dim
            if isinstance(expected_vision_dim, tuple)
            else (expected_vision_dim,)
        )
        expected_language = (
            expected_language_dim
            if isinstance(expected_language_dim, tuple)
            else (expected_language_dim,)
        )
        assert (
            next(
                m for m in specification if m.key == EncoderOutputKeys.RGB.value
            ).dimension
            == expected_vision
        )
        assert (
            next(
                m for m in specification if m.key == EncoderOutputKeys.LANGUAGE.value
            ).dimension
            == expected_language
        )

    def test_features_include_rgb_language_and_padding_mask(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
    ):
        encoder = vlm_encoder_factory()
        specification = encoder.get_output_specification()
        feature_keys = [m.key for m in specification]
        assert EncoderOutputKeys.RGB.value in feature_keys
        assert EncoderOutputKeys.LANGUAGE.value in feature_keys
        assert encoder.padding_mask_name in feature_keys
        assert len(feature_keys) == 3


@pytest.mark.integration
@pytest.mark.parametrize("lora_enabled", [False, True])
@pytest.mark.parametrize(
    "model_name",
    [model_type.value for model_type in ImageTextModelType],
)
def test_integration_forward_pass_per_model(
    rng: np.random.Generator,
    model_name: str,
    lora_enabled: bool,
    parameter_count: Callable[[torch.nn.Module], int],
    trainable_parameter_count: Callable[[torch.nn.Module], int],
) -> None:
    batch_size = 2
    lora_config = (
        LoRAAdaptation(
            enabled=True,
            rank=2,
            alpha=4,
            target_modules=LoRATargetModulePreset.ALL_LINEAR.value,
        )
        if lora_enabled
        else None
    )
    encoder = VLMEncoder(
        input_keys=[
            Cameras.LEFT.value,
            SampleKey.TOKENIZED_OBSERVATIONS.value,
        ],
        pretrained=False,
        frozen=False,
        pooling_method=PoolingMethod.DEFAULT.value,
        model_name=model_name,
        lora_config=lora_config,
    )
    vocab_size = encoder.get_vocab_size()
    image_shape = (batch_size, 1, 3, 224, 224)
    text_shape = (batch_size, 1, 10)
    inputs = {
        Cameras.LEFT.value: torch.from_numpy(
            rng.standard_normal(image_shape).astype(np.float32)
        ),
        SampleKey.TOKENIZED_OBSERVATIONS.value: torch.from_numpy(
            rng.integers(low=0, high=vocab_size, size=text_shape).astype(np.int64)
        ),
    }
    output = encoder(inputs=inputs)
    assert EncoderOutputKeys.RGB.value in output
    assert EncoderOutputKeys.LANGUAGE.value in output
    if lora_enabled:
        trainable_parameter_names = [
            name
            for name, parameter in encoder.encoder.named_parameters()
            if parameter.requires_grad
        ]
        trainable_parameters = trainable_parameter_count(encoder.encoder)
        total_parameters = parameter_count(encoder.encoder)
        assert trainable_parameter_names
        assert all("lora_" in name for name in trainable_parameter_names)
        assert 0 < trainable_parameters < total_parameters
