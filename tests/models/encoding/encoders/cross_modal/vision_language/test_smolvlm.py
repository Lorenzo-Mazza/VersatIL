"""Tests for versatil.models.encoding.encoders.cross_modal.vision_language.smolvlm module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig

from versatil.data.constants import RGB_CAMERAS, Cameras, SampleKey
from versatil.data.metadata import BaseMetadata, CameraMetadata
from versatil.models.encoding.encoders.constants import (
    EncoderOutputKeys,
    SmolVLMModelType,
)
from versatil.models.encoding.encoders.cross_modal.vision_language.smolvlm import (
    SmolVLMEncoder,
)
from versatil.training.constants import PrecisionType

HIDDEN_DIM = 64
IMAGE_SIZE = 56
PATCH_SIZE = 14
SCALE_FACTOR = 4
NUM_IMAGE_TOKENS = 1  # (56/14)^2 / 4^2 = 16/16
MAX_TEXT_LENGTH = 128
VOCAB_SIZE = 32000


def _create_mock_config():
    """Create a mock HuggingFace SmolVLM/Idefics3 config."""
    config = MagicMock()
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
    """Factory for mock SmolVLM with configurable batch size and camera count."""

    def factory(
        batch_size: int = 2,
        num_cameras: int = 1,
    ) -> MagicMock:
        mock_vlm = MagicMock()
        mock_vlm.text_model.config.vocab_size = VOCAB_SIZE

        mock_image_output = MagicMock()
        # Real VLM returns (B * num_cameras, tokens_per_camera, hidden_dim)
        mock_image_output.pooler_output = torch.zeros(
            batch_size * num_cameras, NUM_IMAGE_TOKENS, HIDDEN_DIM
        )
        mock_vlm.get_image_features.return_value = mock_image_output

        mock_embed = MagicMock()
        mock_embed.return_value = torch.zeros(batch_size, MAX_TEXT_LENGTH, HIDDEN_DIM)
        mock_vlm.text_model.get_input_embeddings.return_value = mock_embed

        total_seq = NUM_IMAGE_TOKENS * num_cameras + MAX_TEXT_LENGTH
        mock_lm_output = MagicMock()
        mock_lm_output.last_hidden_state = torch.zeros(
            batch_size, total_seq, HIDDEN_DIM
        )
        mock_vlm.text_model.return_value = mock_lm_output

        return mock_vlm

    return factory


@pytest.fixture
def smolvlm_encoder_factory(
    mock_vlm_factory: Callable[..., MagicMock],
) -> Callable[..., SmolVLMEncoder]:
    """Factory for SmolVLMEncoder with mocked HuggingFace downloads."""

    def factory(
        input_keys: str | list[str] | None = None,
        pretrained: bool = False,
        frozen: bool = False,
        use_embeddings_only: bool = False,
    ) -> SmolVLMEncoder:
        if input_keys is None:
            input_keys = [
                Cameras.LEFT.value,
                SampleKey.TOKENIZED_OBSERVATIONS.value,
            ]
        mock_config = _create_mock_config()
        camera_count = sum(1 for k in input_keys if k in RGB_CAMERAS)
        mock_vlm = mock_vlm_factory(num_cameras=max(camera_count, 1))

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
            return SmolVLMEncoder(
                input_keys=input_keys,
                pretrained=pretrained,
                frozen=frozen,
                model_name=SmolVLMModelType.SMOLVLM_256M.value,
                use_embeddings_only=use_embeddings_only,
            )

    return factory


@pytest.fixture
def smolvlm_input_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    """Factory for SmolVLMEncoder input tensors with temporal dimension."""

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
    encoder: SmolVLMEncoder,
    effective_batch_size: int,
):
    """Configure mock VLM outputs for a given effective batch size (B*T)."""
    num_cameras = len(encoder.camera_keys)
    total_image_tokens = encoder.total_image_tokens
    mock_image_output = MagicMock()
    # Real VLM returns (B * num_cameras, tokens_per_camera, hidden_dim)
    mock_image_output.pooler_output = torch.zeros(
        effective_batch_size * num_cameras,
        encoder.num_image_tokens_per_camera,
        HIDDEN_DIM,
    )
    encoder.vlm.get_image_features.return_value = mock_image_output

    mock_embed = MagicMock()
    mock_embed.return_value = torch.zeros(
        effective_batch_size, MAX_TEXT_LENGTH, HIDDEN_DIM
    )
    encoder.vlm.text_model.get_input_embeddings.return_value = mock_embed

    total_seq = total_image_tokens + MAX_TEXT_LENGTH
    mock_lm_output = MagicMock()
    mock_lm_output.last_hidden_state = torch.zeros(
        effective_batch_size, total_seq, HIDDEN_DIM
    )
    encoder.vlm.text_model.return_value = mock_lm_output


def _make_tiny_smolvlm_config():
    """Create a real but tiny SmolVLM config for fast integration tests."""
    config = AutoConfig.from_pretrained(SmolVLMModelType.SMOLVLM_256M.value)
    config.text_config.num_hidden_layers = 1
    config.text_config.hidden_size = 32
    config.text_config.intermediate_size = 64
    config.text_config.num_attention_heads = 2
    config.text_config.num_key_value_heads = 1
    config.vision_config.hidden_size = 32
    config.vision_config.intermediate_size = 64
    config.vision_config.num_hidden_layers = 1
    config.vision_config.num_attention_heads = 2
    config.vision_config.image_size = 56
    config.vision_config.patch_size = 14
    return config


@pytest.fixture(scope="session")
def real_smolvlm_encoder() -> Callable[..., SmolVLMEncoder]:
    """Factory for a real but tiny SmolVLM encoder, cached per dtype."""
    tiny_config = _make_tiny_smolvlm_config()
    cache: dict[str, SmolVLMEncoder] = {}

    def factory(
        model_dtype: str = PrecisionType.FP32.value,
    ) -> SmolVLMEncoder:
        if model_dtype not in cache:
            with patch(
                "versatil.models.encoding.encoders.cross_modal.vision_language.generative_vlm.AutoConfig.from_pretrained",
                return_value=tiny_config,
            ):
                cache[model_dtype] = SmolVLMEncoder(
                    input_keys=[
                        Cameras.LEFT.value,
                        SampleKey.TOKENIZED_OBSERVATIONS.value,
                    ],
                    pretrained=False,
                    frozen=False,
                    model_name=SmolVLMModelType.SMOLVLM_256M.value,
                    model_dtype=model_dtype,
                )
        return cache[model_dtype]

    return factory


class TestSmolVLMEncoderInitialization:
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
        smolvlm_encoder_factory: Callable[..., SmolVLMEncoder],
        input_keys: list[str],
        expected_camera_count: int,
        frozen: bool,
        use_embeddings_only: bool,
    ):
        encoder = smolvlm_encoder_factory(
            input_keys=input_keys,
            frozen=frozen,
            use_embeddings_only=use_embeddings_only,
        )
        assert encoder.hidden_dim == HIDDEN_DIM
        assert encoder.image_size == IMAGE_SIZE
        assert encoder.max_text_length == MAX_TEXT_LENGTH
        assert encoder.num_image_tokens_per_camera == NUM_IMAGE_TOKENS
        assert len(encoder.camera_keys) == expected_camera_count
        assert encoder.use_embeddings_only is use_embeddings_only
        if frozen:
            for parameter in encoder.parameters():
                assert not parameter.requires_grad

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
    ):
        mock_config = _create_mock_config()
        mock_config.vision_config.image_size = image_size
        mock_config.vision_config.patch_size = patch_size
        mock_config.scale_factor = scale_factor
        mock_vlm = mock_vlm_factory()

        with (
            patch(
                "versatil.models.encoding.encoders.cross_modal.vision_language.generative_vlm.AutoConfig.from_pretrained",
                return_value=mock_config,
            ),
            patch(
                "versatil.models.encoding.encoders.cross_modal.vision_language.generative_vlm.AutoModel.from_config",
                return_value=mock_vlm,
            ),
        ):
            encoder = SmolVLMEncoder(
                input_keys=[
                    Cameras.LEFT.value,
                    SampleKey.TOKENIZED_OBSERVATIONS.value,
                ],
                pretrained=False,
                frozen=False,
                model_name=SmolVLMModelType.SMOLVLM_256M.value,
            )
        assert encoder.num_image_tokens_per_camera == expected_tokens


class TestSmolVLMEncoderForward:
    @pytest.mark.parametrize("time_steps", [1, 3])
    def test_output_shape(
        self,
        smolvlm_encoder_factory: Callable[..., SmolVLMEncoder],
        smolvlm_input_factory: Callable[..., dict[str, torch.Tensor]],
        time_steps: int,
    ):
        batch_size = 2
        encoder = smolvlm_encoder_factory()
        _setup_mock_vlm_for_batch(encoder, batch_size * time_steps)
        inputs = smolvlm_input_factory(
            batch_size=batch_size,
            time_steps=time_steps,
            include_padding_mask=True,
        )
        output = encoder(inputs=inputs)
        fused = output[EncoderOutputKeys.FUSED_RGB_LANGUAGE.value]
        total_seq = NUM_IMAGE_TOKENS + MAX_TEXT_LENGTH
        assert fused.shape == (batch_size, time_steps, total_seq, HIDDEN_DIM)

    def test_output_contains_fused_features_and_padding_mask(
        self,
        smolvlm_encoder_factory: Callable[..., SmolVLMEncoder],
        smolvlm_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        encoder = smolvlm_encoder_factory()
        _setup_mock_vlm_for_batch(encoder, batch_size)
        inputs = smolvlm_input_factory(batch_size=batch_size)
        output = encoder(inputs=inputs)
        assert EncoderOutputKeys.FUSED_RGB_LANGUAGE.value in output
        assert encoder.padding_mask_name in output
        assert len(output) == 2

    def test_image_embeddings_scaled_by_sqrt_hidden_dim(
        self,
        smolvlm_encoder_factory: Callable[..., SmolVLMEncoder],
        smolvlm_input_factory: Callable[..., dict[str, torch.Tensor]],
        rng: np.random.Generator,
    ):
        batch_size = 2
        encoder = smolvlm_encoder_factory(use_embeddings_only=True)
        raw_image_embeddings = torch.from_numpy(
            rng.standard_normal((batch_size, NUM_IMAGE_TOKENS, HIDDEN_DIM)).astype(
                np.float32
            )
        )
        mock_image_output = MagicMock()
        mock_image_output.pooler_output = raw_image_embeddings.clone()
        encoder.vlm.get_image_features.return_value = mock_image_output
        mock_embed = MagicMock()
        mock_embed.return_value = torch.zeros(batch_size, MAX_TEXT_LENGTH, HIDDEN_DIM)
        encoder.vlm.text_model.get_input_embeddings.return_value = mock_embed
        inputs = smolvlm_input_factory(batch_size=batch_size)
        output = encoder(inputs=inputs)
        fused = output[EncoderOutputKeys.FUSED_RGB_LANGUAGE.value]
        image_portion = fused[:, 0, :NUM_IMAGE_TOKENS, :]
        expected_scale = HIDDEN_DIM**0.5
        assert torch.allclose(
            image_portion, raw_image_embeddings * expected_scale, atol=1e-5
        )

    def test_language_embeddings_scaled_by_sqrt_hidden_dim(
        self,
        smolvlm_encoder_factory: Callable[..., SmolVLMEncoder],
        smolvlm_input_factory: Callable[..., dict[str, torch.Tensor]],
        rng: np.random.Generator,
    ):
        batch_size = 2
        encoder = smolvlm_encoder_factory(use_embeddings_only=True)
        raw_language_embeddings = torch.from_numpy(
            rng.standard_normal((batch_size, MAX_TEXT_LENGTH, HIDDEN_DIM)).astype(
                np.float32
            )
        )
        mock_image_output = MagicMock()
        mock_image_output.pooler_output = torch.zeros(
            batch_size, NUM_IMAGE_TOKENS, HIDDEN_DIM
        )
        encoder.vlm.get_image_features.return_value = mock_image_output
        mock_embed = MagicMock()
        mock_embed.return_value = raw_language_embeddings.clone()
        encoder.vlm.text_model.get_input_embeddings.return_value = mock_embed
        inputs = smolvlm_input_factory(batch_size=batch_size)
        output = encoder(inputs=inputs)
        fused = output[EncoderOutputKeys.FUSED_RGB_LANGUAGE.value]
        language_portion = fused[:, 0, NUM_IMAGE_TOKENS:, :]
        expected_scale = HIDDEN_DIM**0.5
        assert torch.allclose(
            language_portion, raw_language_embeddings * expected_scale, atol=1e-5
        )

    def test_images_stacked_along_num_images_dim_for_idefics3(
        self,
        smolvlm_encoder_factory: Callable[..., SmolVLMEncoder],
        smolvlm_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        encoder = smolvlm_encoder_factory()
        _setup_mock_vlm_for_batch(encoder, batch_size)
        inputs = smolvlm_input_factory(batch_size=batch_size)
        encoder(inputs=inputs)
        call_args = encoder.vlm.get_image_features.call_args
        pixel_values = call_args[0][0]
        # Single camera → (B, 1, C, H, W)
        assert pixel_values.ndim == 5
        assert pixel_values.shape[1] == 1

    def test_multi_camera_stacks_all_cameras_in_single_call(
        self,
        smolvlm_encoder_factory: Callable[..., SmolVLMEncoder],
        smolvlm_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        input_keys = [
            Cameras.LEFT.value,
            Cameras.RIGHT.value,
            SampleKey.TOKENIZED_OBSERVATIONS.value,
        ]
        encoder = smolvlm_encoder_factory(input_keys=input_keys)
        _setup_mock_vlm_for_batch(encoder, batch_size)
        inputs = {
            **smolvlm_input_factory(
                camera_key=Cameras.LEFT.value, batch_size=batch_size
            ),
            **smolvlm_input_factory(
                camera_key=Cameras.RIGHT.value, batch_size=batch_size
            ),
        }
        encoder(inputs=inputs)
        # Idefics3 encodes all cameras in one call via num_images dim
        assert encoder.vlm.get_image_features.call_count == 1
        call_args = encoder.vlm.get_image_features.call_args
        pixel_values = call_args[0][0]
        assert pixel_values.shape[1] == 2  # 2 cameras

    def test_padding_mask_image_portion_is_never_padded(
        self,
        smolvlm_encoder_factory: Callable[..., SmolVLMEncoder],
        smolvlm_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        encoder = smolvlm_encoder_factory()
        _setup_mock_vlm_for_batch(encoder, batch_size)
        inputs = smolvlm_input_factory(batch_size=batch_size, include_padding_mask=True)
        output = encoder(inputs=inputs)
        padding_mask = output[encoder.padding_mask_name]
        image_portion = padding_mask[:, :, :NUM_IMAGE_TOKENS]
        assert not image_portion.any()

    @pytest.mark.parametrize("use_embeddings_only", [True, False])
    def test_text_model_called_only_when_not_embeddings_only(
        self,
        smolvlm_encoder_factory: Callable[..., SmolVLMEncoder],
        smolvlm_input_factory: Callable[..., dict[str, torch.Tensor]],
        use_embeddings_only: bool,
    ):
        batch_size = 2
        encoder = smolvlm_encoder_factory(use_embeddings_only=use_embeddings_only)
        _setup_mock_vlm_for_batch(encoder, batch_size)
        inputs = smolvlm_input_factory(batch_size=batch_size, include_padding_mask=True)
        output = encoder(inputs=inputs)
        fused = output[EncoderOutputKeys.FUSED_RGB_LANGUAGE.value]
        total_seq = NUM_IMAGE_TOKENS + MAX_TEXT_LENGTH
        assert fused.shape == (batch_size, 1, total_seq, HIDDEN_DIM)
        assert encoder.padding_mask_name in output
        if use_embeddings_only:
            encoder.vlm.text_model.assert_not_called()
        else:
            encoder.vlm.text_model.assert_called_once()

    def test_missing_language_key_raises(
        self,
        smolvlm_encoder_factory: Callable[..., SmolVLMEncoder],
        rng: np.random.Generator,
    ):
        encoder = smolvlm_encoder_factory()
        images = torch.from_numpy(
            rng.standard_normal((2, 1, 3, IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"SmolVLMEncoder expects pre-tokenized input. "
                f"Expected key '{SampleKey.TOKENIZED_OBSERVATIONS.value}' "
                f"not found in inputs. "
                f"Ensure tokenization is enabled in DataloaderConfig."
            ),
        ):
            encoder(inputs={Cameras.LEFT.value: images})


class TestSmolVLMEncoderValidateInputMetadata:
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
        smolvlm_encoder_factory: Callable[..., SmolVLMEncoder],
        metadata,
        expected_error: str | None,
    ):
        encoder = smolvlm_encoder_factory()
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
                f"SmolVLMEncoder cannot process image data for "
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
        smolvlm_encoder_factory: Callable[..., SmolVLMEncoder],
        metadata,
        expected_error: str | None,
    ):
        encoder = smolvlm_encoder_factory()
        result = encoder.validate_input_metadata(
            key=SampleKey.TOKENIZED_OBSERVATIONS.value, metadata=metadata
        )
        assert result == expected_error


class TestSmolVLMEncoderGetOutputSpecification:
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
        smolvlm_encoder_factory: Callable[..., SmolVLMEncoder],
        input_keys: list[str],
        expected_total_image_tokens: int,
    ):
        encoder = smolvlm_encoder_factory(input_keys=input_keys)
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


class TestSmolVLMEncoderGetVocabSize:
    def test_returns_text_model_vocab_size(
        self,
        smolvlm_encoder_factory: Callable[..., SmolVLMEncoder],
    ):
        encoder = smolvlm_encoder_factory()
        assert encoder.get_vocab_size() == VOCAB_SIZE


class TestSmolVLMEncoderBackboneAccessors:
    def test_get_backbone_layers_accesses_text_model_layers(
        self,
        smolvlm_encoder_factory: Callable[..., SmolVLMEncoder],
    ):
        encoder = smolvlm_encoder_factory()
        result = encoder.get_backbone_layers()
        assert result is encoder.vlm.text_model.layers

    def test_get_rotary_embedding_accesses_text_model_rotary_emb(
        self,
        smolvlm_encoder_factory: Callable[..., SmolVLMEncoder],
    ):
        encoder = smolvlm_encoder_factory()
        result = encoder.get_rotary_embedding()
        assert result is encoder.vlm.text_model.rotary_emb

    def test_get_backbone_hidden_dim_returns_hidden_dim(
        self,
        smolvlm_encoder_factory: Callable[..., SmolVLMEncoder],
    ):
        encoder = smolvlm_encoder_factory()
        assert encoder.get_backbone_hidden_dim() == HIDDEN_DIM


class TestSmolVLMEncoderIntegration:
    @pytest.mark.integration
    def test_forward_pass_with_real_model(
        self,
        real_smolvlm_encoder: Callable[..., SmolVLMEncoder],
        smolvlm_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 1
        encoder = real_smolvlm_encoder()
        encoder.eval()
        inputs = smolvlm_input_factory(
            batch_size=batch_size,
            height=encoder.image_size,
            width=encoder.image_size,
            sequence_length=10,
        )
        with torch.no_grad():
            output = encoder(inputs=inputs)
        fused = output[EncoderOutputKeys.FUSED_RGB_LANGUAGE.value]
        assert fused.shape[0] == batch_size
        assert fused.shape[-1] == encoder.hidden_dim

    @pytest.mark.integration
    def test_backbone_accessors_return_real_modules(
        self,
        real_smolvlm_encoder: Callable[..., SmolVLMEncoder],
    ):
        encoder = real_smolvlm_encoder()
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
        real_smolvlm_encoder: Callable[..., SmolVLMEncoder],
        precision: str,
        expected_dtype: torch.dtype,
    ):
        encoder = real_smolvlm_encoder(model_dtype=precision)
        param_dtype = next(encoder.vlm.parameters()).dtype
        assert param_dtype == expected_dtype

    @pytest.mark.integration
    def test_extract_query_key_value_applies_rope(
        self,
        real_smolvlm_encoder: Callable[..., SmolVLMEncoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        encoder = real_smolvlm_encoder()
        layer = encoder.get_backbone_layers()[0]
        rotary_emb = encoder.get_rotary_embedding()
        batch_size, sequence_length = 2, 8
        hidden = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=encoder.hidden_dim,
        )
        position_ids = torch.arange(sequence_length).unsqueeze(0).expand(batch_size, -1)
        query, key, value = SmolVLMEncoder.extract_query_key_value(
            vlm_layer=layer,
            hidden_states=hidden,
            rotary_embedding=rotary_emb,
            position_ids=position_ids,
        )
        normalized = layer.input_layernorm(hidden)
        head_dim = layer.self_attn.head_dim
        num_kv_heads = layer.self_attn.config.num_key_value_heads
        raw_key = (
            layer.self_attn.k_proj(normalized)
            .view(batch_size, sequence_length, num_kv_heads, head_dim)
            .transpose(1, 2)
        )
        assert not torch.allclose(key, raw_key, atol=1e-5)

    @pytest.mark.integration
    def test_apply_residual_feedforward_matches_llama_forward(
        self,
        real_smolvlm_encoder: Callable[..., SmolVLMEncoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        encoder = real_smolvlm_encoder()
        layer = encoder.get_backbone_layers()[0]
        attn = layer.self_attn
        attention_output_dim = attn.config.num_attention_heads * attn.head_dim
        batch_size, sequence_length = 2, 8
        residual = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=encoder.hidden_dim,
        )
        attn_output = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=attention_output_dim,
        )
        result = SmolVLMEncoder.apply_residual_feedforward(
            vlm_layer=layer,
            vlm_residual=residual,
            vlm_attention_output=attn_output,
        )
        o_proj_out = layer.self_attn.o_proj(attn_output)
        expected = residual + o_proj_out
        expected_residual = expected
        expected = layer.post_attention_layernorm(expected)
        expected = layer.mlp(expected)
        expected = expected_residual + expected
        assert torch.allclose(result, expected, atol=1e-5)
