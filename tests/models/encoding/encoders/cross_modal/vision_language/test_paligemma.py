"""Tests for versatil.models.encoding.encoders.cross_modal.vision_language.paligemma module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from versatil.data.constants import Cameras, SampleKey
from versatil.data.metadata import BaseMetadata, CameraMetadata
from versatil.models.encoding.encoders.constants import (
    EncoderOutputKeys,
    PaliGemmaModelType,
)
from versatil.models.encoding.encoders.cross_modal.vision_language.paligemma import (
    PaliGemmaEncoder,
)
from versatil.training.constants import PrecisionType

HIDDEN_DIM = 64
IMAGE_SIZE = 56
NUM_IMAGE_TOKENS = 16  # (56/14)^2
MAX_TEXT_LENGTH = 128
VOCAB_SIZE = 32000


def _create_mock_config():
    """Create a mock HuggingFace PaliGemma config."""
    config = MagicMock()
    config.image_token_id = 99999
    config.vision_config.image_size = IMAGE_SIZE
    config.vision_config.num_image_tokens = NUM_IMAGE_TOKENS
    config.text_config.hidden_size = HIDDEN_DIM
    config.text_config.max_position_embeddings = MAX_TEXT_LENGTH
    config.text_config.vocab_size = VOCAB_SIZE
    return config


@pytest.fixture
def mock_vlm_factory() -> Callable[..., MagicMock]:
    """Factory for mock PaliGemma VLM with configurable batch size and camera count."""

    def factory(
        batch_size: int = 2,
        num_cameras: int = 1,
    ) -> MagicMock:
        mock_vlm = MagicMock()
        mock_vlm.language_model.config.vocab_size = VOCAB_SIZE

        total_image_tokens = NUM_IMAGE_TOKENS * num_cameras
        mock_image_output = MagicMock()
        mock_image_output.pooler_output = torch.zeros(
            batch_size, NUM_IMAGE_TOKENS, HIDDEN_DIM
        )
        mock_vlm.get_image_features.return_value = mock_image_output

        mock_embed = MagicMock(
            return_value=torch.zeros(batch_size, MAX_TEXT_LENGTH, HIDDEN_DIM)
        )
        mock_vlm.language_model.get_input_embeddings.return_value = mock_embed

        total_seq = total_image_tokens + MAX_TEXT_LENGTH
        mock_lm_output = MagicMock()
        mock_lm_output.last_hidden_state = torch.zeros(
            batch_size, total_seq, HIDDEN_DIM
        )
        mock_vlm.language_model.return_value = mock_lm_output

        return mock_vlm

    return factory


@pytest.fixture
def paligemma_encoder_factory(
    mock_vlm_factory: Callable[..., MagicMock],
) -> Callable[..., PaliGemmaEncoder]:
    """Factory for PaliGemmaEncoder with mocked HuggingFace downloads."""

    def factory(
        input_keys: str | list[str] | None = None,
        pretrained: bool = False,
        frozen: bool = False,
        use_embeddings_only: bool = False,
    ) -> PaliGemmaEncoder:
        if input_keys is None:
            input_keys = [
                Cameras.LEFT.value,
                SampleKey.TOKENIZED_OBSERVATIONS.value,
            ]
        mock_config = _create_mock_config()
        mock_vlm = mock_vlm_factory()

        with (
            patch(
                "versatil.models.encoding.encoders.cross_modal.vision_language.generative_vlm.AutoConfig.from_pretrained",
                return_value=mock_config,
            ),
            patch(
                "versatil.models.encoding.encoders.cross_modal.vision_language.generative_vlm.AutoModel.from_pretrained",
                return_value=mock_vlm,
            ),
            patch(
                "versatil.models.encoding.encoders.cross_modal.vision_language.generative_vlm.AutoModel.from_config",
                return_value=mock_vlm,
            ),
        ):
            return PaliGemmaEncoder(
                input_keys=input_keys,
                pretrained=pretrained,
                frozen=frozen,
                model_name=PaliGemmaModelType.PALIGEMMA2_3B_224.value,
                use_embeddings_only=use_embeddings_only,
            )

    return factory


@pytest.fixture
def paligemma_input_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    """Factory for PaliGemmaEncoder input tensors with temporal dimension."""

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
    encoder: PaliGemmaEncoder,
    effective_batch_size: int,
):
    """Configure mock VLM outputs for a given effective batch size (B*T)."""
    mock_image_output = MagicMock()
    mock_image_output.pooler_output = torch.zeros(
        effective_batch_size, NUM_IMAGE_TOKENS, HIDDEN_DIM
    )
    encoder.vlm.get_image_features.return_value = mock_image_output
    mock_embed = MagicMock(
        return_value=torch.zeros(effective_batch_size, MAX_TEXT_LENGTH, HIDDEN_DIM)
    )
    encoder.vlm.language_model.get_input_embeddings.return_value = mock_embed
    total_seq = encoder.total_image_tokens + MAX_TEXT_LENGTH
    mock_lm_output = MagicMock()
    mock_lm_output.last_hidden_state = torch.zeros(
        effective_batch_size, total_seq, HIDDEN_DIM
    )
    encoder.vlm.language_model.return_value = mock_lm_output


class TestPaliGemmaEncoderInitialization:
    @pytest.mark.parametrize(
        "input_keys, expected_camera_count",
        [
            ([Cameras.LEFT.value, SampleKey.TOKENIZED_OBSERVATIONS.value], 1),
            (
                [
                    Cameras.LEFT.value,
                    Cameras.RIGHT.value,
                    SampleKey.TOKENIZED_OBSERVATIONS.value,
                ],
                2,
            ),
        ],
    )
    @pytest.mark.parametrize("frozen", [True, False])
    @pytest.mark.parametrize("use_embeddings_only", [True, False])
    def test_stores_configuration(
        self,
        paligemma_encoder_factory: Callable[..., PaliGemmaEncoder],
        input_keys: list[str],
        expected_camera_count: int,
        frozen: bool,
        use_embeddings_only: bool,
    ):
        encoder = paligemma_encoder_factory(
            input_keys=input_keys,
            frozen=frozen,
            use_embeddings_only=use_embeddings_only,
        )
        assert encoder.hidden_dim == HIDDEN_DIM
        assert encoder.image_size == IMAGE_SIZE
        assert encoder.max_text_length == MAX_TEXT_LENGTH
        assert encoder.num_image_tokens_per_camera == NUM_IMAGE_TOKENS
        assert len(encoder.camera_keys) == expected_camera_count
        assert encoder.input_specification.requires_tokenized is True
        assert encoder.use_embeddings_only is use_embeddings_only
        if frozen:
            for parameter in encoder.parameters():
                assert not parameter.requires_grad


class TestPaliGemmaEncoderForward:
    @pytest.mark.parametrize("time_steps", [1, 3])
    @pytest.mark.parametrize(
        "input_keys, num_cameras",
        [
            ([Cameras.LEFT.value, SampleKey.TOKENIZED_OBSERVATIONS.value], 1),
            (
                [
                    Cameras.LEFT.value,
                    Cameras.RIGHT.value,
                    SampleKey.TOKENIZED_OBSERVATIONS.value,
                ],
                2,
            ),
        ],
    )
    def test_output_shape_scales_with_cameras_and_time(
        self,
        paligemma_encoder_factory: Callable[..., PaliGemmaEncoder],
        paligemma_input_factory: Callable[..., dict[str, torch.Tensor]],
        time_steps: int,
        input_keys: list[str],
        num_cameras: int,
    ):
        batch_size = 2
        encoder = paligemma_encoder_factory(input_keys=input_keys)
        _setup_mock_vlm_for_batch(encoder, batch_size * time_steps)
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
        output = encoder(inputs=inputs)
        fused = output[EncoderOutputKeys.FUSED_RGB_LANGUAGE.value]
        total_seq = num_cameras * NUM_IMAGE_TOKENS + MAX_TEXT_LENGTH
        assert fused.shape == (batch_size, time_steps, total_seq, HIDDEN_DIM)

    def test_output_contains_fused_features_and_padding_mask(
        self,
        paligemma_encoder_factory: Callable[..., PaliGemmaEncoder],
        paligemma_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        encoder = paligemma_encoder_factory()
        _setup_mock_vlm_for_batch(encoder, batch_size)
        inputs = paligemma_input_factory(batch_size=batch_size)
        output = encoder(inputs=inputs)
        assert EncoderOutputKeys.FUSED_RGB_LANGUAGE.value in output
        assert encoder.padding_mask_name in output
        assert len(output) == 2

    def test_vision_tower_called_once_per_camera(
        self,
        paligemma_encoder_factory: Callable[..., PaliGemmaEncoder],
        paligemma_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        input_keys = [
            Cameras.LEFT.value,
            Cameras.RIGHT.value,
            SampleKey.TOKENIZED_OBSERVATIONS.value,
        ]
        encoder = paligemma_encoder_factory(input_keys=input_keys)
        _setup_mock_vlm_for_batch(encoder, batch_size)
        inputs = {
            **paligemma_input_factory(
                camera_key=Cameras.LEFT.value, batch_size=batch_size
            ),
            **paligemma_input_factory(
                camera_key=Cameras.RIGHT.value, batch_size=batch_size
            ),
        }
        encoder(inputs=inputs)
        assert encoder.vlm.get_image_features.call_count == 2

    def test_language_model_receives_concatenated_image_and_text_embeddings(
        self,
        paligemma_encoder_factory: Callable[..., PaliGemmaEncoder],
        paligemma_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        encoder = paligemma_encoder_factory()
        _setup_mock_vlm_for_batch(encoder, batch_size)
        inputs = paligemma_input_factory(batch_size=batch_size)
        encoder(inputs=inputs)
        call_kwargs = encoder.vlm.language_model.call_args.kwargs
        expected_seq_length = NUM_IMAGE_TOKENS + MAX_TEXT_LENGTH
        assert call_kwargs["inputs_embeds"].shape == (
            batch_size,
            expected_seq_length,
            HIDDEN_DIM,
        )

    def test_different_images_produce_different_vision_tower_inputs(
        self,
        paligemma_encoder_factory: Callable[..., PaliGemmaEncoder],
        rng: np.random.Generator,
    ):
        batch_size = 1
        encoder = paligemma_encoder_factory()
        _setup_mock_vlm_for_batch(encoder, batch_size)
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

        encoder(
            inputs={
                Cameras.LEFT.value: images_a,
                SampleKey.TOKENIZED_OBSERVATIONS.value: tokens,
            }
        )
        first_call_pixel = encoder.vlm.get_image_features.call_args[0][0]

        encoder(
            inputs={
                Cameras.LEFT.value: images_b,
                SampleKey.TOKENIZED_OBSERVATIONS.value: tokens,
            }
        )
        second_call_pixel = encoder.vlm.get_image_features.call_args[0][0]

        assert not torch.equal(first_call_pixel, second_call_pixel)

    def test_padding_mask_image_portion_is_never_padded(
        self,
        paligemma_encoder_factory: Callable[..., PaliGemmaEncoder],
        paligemma_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        encoder = paligemma_encoder_factory()
        _setup_mock_vlm_for_batch(encoder, batch_size)
        inputs = paligemma_input_factory(
            batch_size=batch_size, include_padding_mask=True
        )
        output = encoder(inputs=inputs)
        padding_mask = output[encoder.padding_mask_name]
        # Temporal dim: (B, T=1, total_seq)
        image_portion = padding_mask[:, :, :NUM_IMAGE_TOKENS]
        assert not image_portion.any()

    @pytest.mark.parametrize("use_embeddings_only", [True, False])
    def test_language_model_called_only_when_not_embeddings_only(
        self,
        paligemma_encoder_factory: Callable[..., PaliGemmaEncoder],
        paligemma_input_factory: Callable[..., dict[str, torch.Tensor]],
        use_embeddings_only: bool,
    ):
        batch_size = 2
        encoder = paligemma_encoder_factory(use_embeddings_only=use_embeddings_only)
        _setup_mock_vlm_for_batch(encoder, batch_size)
        inputs = paligemma_input_factory(
            batch_size=batch_size, include_padding_mask=True
        )
        output = encoder(inputs=inputs)
        fused = output[EncoderOutputKeys.FUSED_RGB_LANGUAGE.value]
        total_seq = NUM_IMAGE_TOKENS + MAX_TEXT_LENGTH
        assert fused.shape == (batch_size, 1, total_seq, HIDDEN_DIM)
        assert encoder.padding_mask_name in output
        if use_embeddings_only:
            encoder.vlm.language_model.assert_not_called()
        else:
            encoder.vlm.language_model.assert_called_once()

    def test_missing_language_key_raises(
        self,
        paligemma_encoder_factory: Callable[..., PaliGemmaEncoder],
        rng: np.random.Generator,
    ):
        encoder = paligemma_encoder_factory()
        images = torch.from_numpy(
            rng.standard_normal((2, 1, 3, IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"PaliGemmaEncoder expects pre-tokenized input. "
                f"Expected key '{SampleKey.TOKENIZED_OBSERVATIONS.value}' "
                f"not found in inputs. "
                f"Ensure tokenization is enabled in DataloaderConfig."
            ),
        ):
            encoder(inputs={Cameras.LEFT.value: images})


class TestPaliGemmaEncoderValidateInputMetadata:
    @pytest.mark.parametrize(
        "metadata, expected_error",
        [
            (
                CameraMetadata(
                    camera_key="left",
                    dtype="uint8",
                    channels=3,
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
                f"Expected 3-channel RGB for '{Cameras.LEFT.value}', got 1 channels",
            ),
            (
                MagicMock(spec=BaseMetadata),
                f"Expected CameraMetadata for '{Cameras.LEFT.value}', got MagicMock",
            ),
        ],
    )
    def test_validates_camera_key_metadata(
        self,
        paligemma_encoder_factory: Callable[..., PaliGemmaEncoder],
        metadata,
        expected_error: str | None,
    ):
        encoder = paligemma_encoder_factory()
        result = encoder.validate_input_metadata(
            key=Cameras.LEFT.value, metadata=metadata
        )
        assert result == expected_error

    @pytest.mark.parametrize(
        "metadata, expected_error",
        [
            (
                CameraMetadata(
                    camera_key="left",
                    dtype="uint8",
                    channels=3,
                    image_height=224,
                    image_width=224,
                ),
                f"PaliGemmaEncoder cannot process image data for "
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
        paligemma_encoder_factory: Callable[..., PaliGemmaEncoder],
        metadata,
        expected_error: str | None,
    ):
        encoder = paligemma_encoder_factory()
        result = encoder.validate_input_metadata(
            key=SampleKey.TOKENIZED_OBSERVATIONS.value, metadata=metadata
        )
        assert result == expected_error


class TestPaliGemmaEncoderGetOutputSpecification:
    @pytest.mark.parametrize(
        "input_keys, expected_total_image_tokens",
        [
            (
                [Cameras.LEFT.value, SampleKey.TOKENIZED_OBSERVATIONS.value],
                NUM_IMAGE_TOKENS,
            ),
            (
                [
                    Cameras.LEFT.value,
                    Cameras.RIGHT.value,
                    SampleKey.TOKENIZED_OBSERVATIONS.value,
                ],
                2 * NUM_IMAGE_TOKENS,
            ),
        ],
    )
    def test_fused_dimension_scales_with_cameras(
        self,
        paligemma_encoder_factory: Callable[..., PaliGemmaEncoder],
        input_keys: list[str],
        expected_total_image_tokens: int,
    ):
        encoder = paligemma_encoder_factory(input_keys=input_keys)
        specification = encoder.get_output_specification()
        fused_dim = next(
            m
            for m in specification
            if m.key == EncoderOutputKeys.FUSED_RGB_LANGUAGE.value
        ).dimension
        expected_total_seq = expected_total_image_tokens + MAX_TEXT_LENGTH
        assert fused_dim == (expected_total_seq, HIDDEN_DIM)
        feature_keys = [m.key for m in specification]
        assert len(feature_keys) == 2


class TestPaliGemmaEncoderGetVocabSize:
    def test_returns_language_model_vocab_size(
        self,
        paligemma_encoder_factory: Callable[..., PaliGemmaEncoder],
    ):
        encoder = paligemma_encoder_factory()
        assert encoder.get_vocab_size() == VOCAB_SIZE


class TestPaliGemmaEncoderBackboneAccessors:
    def test_get_backbone_layers_accesses_language_model_layers(
        self,
        paligemma_encoder_factory: Callable[..., PaliGemmaEncoder],
    ):
        encoder = paligemma_encoder_factory()
        result = encoder.get_backbone_layers()
        assert result is encoder.vlm.language_model.layers

    def test_get_rotary_embedding_accesses_language_model_rotary_emb(
        self,
        paligemma_encoder_factory: Callable[..., PaliGemmaEncoder],
    ):
        encoder = paligemma_encoder_factory()
        result = encoder.get_rotary_embedding()
        assert result is encoder.vlm.language_model.rotary_emb

    def test_get_backbone_hidden_dim_returns_hidden_dim(
        self,
        paligemma_encoder_factory: Callable[..., PaliGemmaEncoder],
    ):
        encoder = paligemma_encoder_factory()
        assert encoder.get_backbone_hidden_dim() == HIDDEN_DIM


class TestPaliGemmaEncoderIntegration:
    @pytest.mark.integration
    def test_forward_pass_with_real_model(
        self,
        real_paligemma_encoder: Callable[..., PaliGemmaEncoder],
        rng: np.random.Generator,
    ):
        batch_size = 1
        encoder = real_paligemma_encoder()
        encoder.eval()
        vocab_size = encoder.get_vocab_size()
        images = torch.from_numpy(
            rng.standard_normal(
                (batch_size, 1, 3, encoder.image_size, encoder.image_size)
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
            output = encoder(inputs=inputs)
        fused = output[EncoderOutputKeys.FUSED_RGB_LANGUAGE.value]
        assert fused.shape[0] == batch_size
        assert fused.shape[-1] == encoder.hidden_dim

    @pytest.mark.integration
    def test_backbone_accessors_return_real_modules(
        self,
        real_paligemma_encoder: Callable[..., PaliGemmaEncoder],
    ):
        encoder = real_paligemma_encoder()
        layers = encoder.get_backbone_layers()
        assert isinstance(layers, nn.ModuleList)
        assert len(layers) > 0
        assert isinstance(encoder.get_rotary_embedding(), nn.Module)
        assert encoder.get_backbone_hidden_dim() == encoder.hidden_dim

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
        real_paligemma_encoder: Callable[..., PaliGemmaEncoder],
        precision: str,
        expected_dtype: torch.dtype,
    ):
        encoder = real_paligemma_encoder(model_dtype=precision)
        param_dtype = next(encoder.vlm.parameters()).dtype
        assert param_dtype == expected_dtype
