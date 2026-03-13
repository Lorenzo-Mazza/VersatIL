"""Tests for versatil.models.encoding.encoders.multimodal.vlm module."""
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling

from versatil.data.constants import Cameras, SampleKey
from versatil.models.encoding.encoders.constants import (
    EncoderOutputKeys,
    ImageTextModelType,
    PoolingMethod,
)
from versatil.models.encoding.encoders.multimodal.vlm import VLMEncoder
from versatil.models.encoding.encoders.unconditional import Encoder


HIDDEN_VISION_DIM = 768
HIDDEN_LANGUAGE_DIM = 512
IMAGE_SIZE = 224
MAX_TEXT_LENGTH = 77
VOCAB_SIZE = 49408


def _create_mock_encoder():
    """Create a mock HuggingFace VLM encoder with expected attributes."""
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


def _create_mock_image_processor():
    """Create a mock image processor."""
    mock_processor = MagicMock()

    def processor_side_effect(images, return_tensors, padding=False):
        result = MagicMock()
        result.to.return_value = {"pixel_values": images}
        result.__iter__ = lambda s: iter(["pixel_values"])
        result.__getitem__ = lambda s, k: images
        result.keys = lambda: ["pixel_values"]
        return result

    mock_processor.side_effect = processor_side_effect
    return mock_processor


@pytest.fixture
def vlm_encoder_factory() -> Callable[..., VLMEncoder]:
    """Factory for VLMEncoder with mocked HuggingFace model downloads."""

    def factory(
        input_keys: str | list[str] | None = None,
        pretrained: bool = False,
        frozen: bool = False,
        pooling_method: str = PoolingMethod.DEFAULT.value,
        model_name: str = ImageTextModelType.CLIP_VITB32.value,
    ) -> VLMEncoder:
        if input_keys is None:
            input_keys = [
                Cameras.LEFT.value,
                SampleKey.TOKENIZED_OBSERVATIONS.value,
            ]
        mock_encoder = _create_mock_encoder()
        mock_config = MagicMock()
        mock_processor = _create_mock_image_processor()

        with (
            patch(
                "versatil.models.encoding.encoders.multimodal.vlm.AutoConfig.from_pretrained",
                return_value=mock_config,
            ),
            patch(
                "versatil.models.encoding.encoders.multimodal.vlm.AutoModel.from_pretrained",
                return_value=mock_encoder,
            ),
            patch(
                "versatil.models.encoding.encoders.multimodal.vlm.AutoModel.from_config",
                return_value=mock_encoder,
            ),
            patch(
                "versatil.models.encoding.encoders.multimodal.vlm.AutoImageProcessor.from_pretrained",
                return_value=mock_processor,
            ),
        ):
            return VLMEncoder(
                input_keys=input_keys,
                pretrained=pretrained,
                frozen=frozen,
                pooling_method=pooling_method,
                model_name=model_name,
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
        time_steps: int | None = None,
        include_padding_mask: bool = False,
    ) -> dict[str, torch.Tensor]:
        if time_steps is not None:
            image_shape = (batch_size, time_steps, channels, height, width)
            text_shape = (batch_size, time_steps, sequence_length)
        else:
            image_shape = (batch_size, channels, height, width)
            text_shape = (batch_size, sequence_length)
        images = torch.from_numpy(
            rng.standard_normal(image_shape).astype(np.float32)
        )
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

    def test_inherits_from_encoder(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
    ):
        encoder = vlm_encoder_factory()
        assert isinstance(encoder, Encoder)

    @pytest.mark.parametrize("input_keys", [
        [Cameras.LEFT.value, SampleKey.TOKENIZED_OBSERVATIONS.value],
        [Cameras.RIGHT.value, SampleKey.TOKENIZED_OBSERVATIONS.value],
    ])
    @pytest.mark.parametrize("pooling_method", [
        PoolingMethod.DEFAULT.value,
        PoolingMethod.AVERAGE.value,
    ])
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
        expected_camera_key = next(
            key for key in input_keys
            if key != SampleKey.TOKENIZED_OBSERVATIONS.value
        )
        assert encoder.camera_key == expected_camera_key
        assert encoder.language_key == SampleKey.TOKENIZED_OBSERVATIONS.value
        assert encoder.pooling_method == pooling_method
        assert encoder.hidden_vision_dim == HIDDEN_VISION_DIM
        assert encoder.hidden_language_dim == HIDDEN_LANGUAGE_DIM
        assert encoder.max_text_length == MAX_TEXT_LENGTH

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
            f"{EncoderOutputKeys.LANGUAGE.value}"
            f"_{EncoderOutputKeys.PADDING_MASK.value}"
        )
        assert encoder.padding_mask_name == expected


class TestVLMEncoderPoolFeatures:

    def test_default_pooling_returns_pooler_output(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
        rng: np.random.Generator,
    ):
        encoder = vlm_encoder_factory(
            pooling_method=PoolingMethod.DEFAULT.value
        )
        batch_size = 2
        pooler_output = torch.from_numpy(
            rng.standard_normal((batch_size, HIDDEN_VISION_DIM)).astype(
                np.float32
            )
        )
        hidden_state = torch.from_numpy(
            rng.standard_normal((batch_size, 50, HIDDEN_VISION_DIM)).astype(
                np.float32
            )
        )
        outputs = BaseModelOutputWithPooling(
            last_hidden_state=hidden_state,
            pooler_output=pooler_output,
        )
        result = encoder._pool_features(
            outputs=outputs, modality=EncoderOutputKeys.RGB.value
        )
        assert torch.allclose(result, pooler_output)

    def test_average_pooling_rgb_excludes_cls_token(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
        rng: np.random.Generator,
    ):
        encoder = vlm_encoder_factory(
            pooling_method=PoolingMethod.AVERAGE.value
        )
        batch_size = 2
        sequence_length = 50
        hidden_state = torch.from_numpy(
            rng.standard_normal(
                (batch_size, sequence_length, HIDDEN_VISION_DIM)
            ).astype(np.float32)
        )
        pooler_output = torch.zeros(batch_size, HIDDEN_VISION_DIM)
        outputs = BaseModelOutputWithPooling(
            last_hidden_state=hidden_state,
            pooler_output=pooler_output,
        )
        result = encoder._pool_features(
            outputs=outputs, modality=EncoderOutputKeys.RGB.value
        )
        expected = hidden_state[:, 1:].mean(dim=1)
        assert torch.allclose(result, expected)

    def test_average_pooling_language_includes_all_tokens(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
        rng: np.random.Generator,
    ):
        encoder = vlm_encoder_factory(
            pooling_method=PoolingMethod.AVERAGE.value
        )
        batch_size = 2
        sequence_length = 20
        hidden_state = torch.from_numpy(
            rng.standard_normal(
                (batch_size, sequence_length, HIDDEN_LANGUAGE_DIM)
            ).astype(np.float32)
        )
        pooler_output = torch.zeros(batch_size, HIDDEN_LANGUAGE_DIM)
        outputs = BaseModelOutputWithPooling(
            last_hidden_state=hidden_state,
            pooler_output=pooler_output,
        )
        result = encoder._pool_features(
            outputs=outputs, modality=EncoderOutputKeys.LANGUAGE.value
        )
        expected = hidden_state.mean(dim=1)
        assert torch.allclose(result, expected)

    def test_none_pooling_returns_full_hidden_state(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
        rng: np.random.Generator,
    ):
        encoder = vlm_encoder_factory(
            pooling_method=PoolingMethod.NONE.value
        )
        batch_size = 2
        sequence_length = 50
        hidden_state = torch.from_numpy(
            rng.standard_normal(
                (batch_size, sequence_length, HIDDEN_VISION_DIM)
            ).astype(np.float32)
        )
        pooler_output = torch.zeros(batch_size, HIDDEN_VISION_DIM)
        outputs = BaseModelOutputWithPooling(
            last_hidden_state=hidden_state,
            pooler_output=pooler_output,
        )
        result = encoder._pool_features(
            outputs=outputs, modality=EncoderOutputKeys.RGB.value
        )
        assert torch.allclose(result, hidden_state)

    def test_learned_aggregation_pooling(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
        rng: np.random.Generator,
    ):
        encoder = vlm_encoder_factory(
            pooling_method=PoolingMethod.LEARNED_AGGREGATION.value
        )
        batch_size = 2
        sequence_length = 50
        mock_head = MagicMock()
        mock_head.return_value = torch.zeros(batch_size, HIDDEN_VISION_DIM)
        mock_pooling_heads = MagicMock()
        mock_pooling_heads.__getitem__ = MagicMock(return_value=mock_head)
        # Bypass nn.Module.__setattr__ which rejects non-Module assignments
        object.__setattr__(encoder, "pooling_heads", mock_pooling_heads)
        hidden_state = torch.from_numpy(
            rng.standard_normal(
                (batch_size, sequence_length, HIDDEN_VISION_DIM)
            ).astype(np.float32)
        )
        pooler_output = torch.zeros(batch_size, HIDDEN_VISION_DIM)
        outputs = BaseModelOutputWithPooling(
            last_hidden_state=hidden_state,
            pooler_output=pooler_output,
        )
        result = encoder._pool_features(
            outputs=outputs, modality=EncoderOutputKeys.RGB.value
        )
        mock_head.assert_called_once()
        assert result.shape == (batch_size, HIDDEN_VISION_DIM)

    def test_invalid_pooling_method_raises(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
        rng: np.random.Generator,
    ):
        encoder = vlm_encoder_factory()
        encoder.pooling_method = "invalid_method"
        hidden_state = torch.from_numpy(
            rng.standard_normal((2, 50, HIDDEN_VISION_DIM)).astype(np.float32)
        )
        pooler_output = torch.zeros(2, HIDDEN_VISION_DIM)
        outputs = BaseModelOutputWithPooling(
            last_hidden_state=hidden_state,
            pooler_output=pooler_output,
        )
        with pytest.raises(ValueError, match="Unsupported feature extraction method"):
            encoder._pool_features(
                outputs=outputs, modality=EncoderOutputKeys.RGB.value
            )

    def test_missing_pooler_output_raises(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
    ):
        encoder = vlm_encoder_factory(
            pooling_method=PoolingMethod.DEFAULT.value
        )
        outputs = BaseModelOutputWithPooling(
            last_hidden_state=torch.zeros(2, 50, HIDDEN_VISION_DIM),
            pooler_output=None,
        )
        with pytest.raises(RuntimeError, match="Encoder outputs are missing"):
            encoder._pool_features(
                outputs=outputs, modality=EncoderOutputKeys.RGB.value
            )

    def test_learned_aggregation_with_none_heads_raises(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
        rng: np.random.Generator,
    ):
        encoder = vlm_encoder_factory(
            pooling_method=PoolingMethod.LEARNED_AGGREGATION.value
        )
        encoder.pooling_heads = None
        hidden_state = torch.from_numpy(
            rng.standard_normal((2, 50, HIDDEN_VISION_DIM)).astype(np.float32)
        )
        pooler_output = torch.zeros(2, HIDDEN_VISION_DIM)
        outputs = BaseModelOutputWithPooling(
            last_hidden_state=hidden_state,
            pooler_output=pooler_output,
        )
        with pytest.raises(RuntimeError, match="pooling_head must be initialized"):
            encoder._pool_features(
                outputs=outputs, modality=EncoderOutputKeys.RGB.value
            )


class TestVLMEncoderResizeImages:

    def test_resizes_to_target_size(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
        rng: np.random.Generator,
    ):
        encoder = vlm_encoder_factory()
        images = torch.from_numpy(
            rng.standard_normal((2, 3, 320, 480)).astype(np.float32)
        )
        result = encoder._resize_images(images=images)
        assert result.shape[2] == IMAGE_SIZE
        assert result.shape[3] == IMAGE_SIZE

    def test_preserves_aspect_ratio_with_padding(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
        rng: np.random.Generator,
    ):
        encoder = vlm_encoder_factory()
        images = torch.from_numpy(
            rng.standard_normal((2, 3, 112, 224)).astype(np.float32)
        )
        result = encoder._resize_images(images=images)
        assert result.shape[2] == IMAGE_SIZE
        assert result.shape[3] == IMAGE_SIZE

    def test_invalid_ndim_raises(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
        rng: np.random.Generator,
    ):
        encoder = vlm_encoder_factory()
        images_3d = torch.from_numpy(
            rng.standard_normal((3, 224, 224)).astype(np.float32)
        )
        with pytest.raises(ValueError, match="expected"):
            encoder._resize_images(images=images_3d)

    def test_already_correct_size(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
        rng: np.random.Generator,
    ):
        encoder = vlm_encoder_factory()
        images = torch.from_numpy(
            rng.standard_normal((2, 3, IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
        )
        result = encoder._resize_images(images=images)
        assert result.shape == (2, 3, IMAGE_SIZE, IMAGE_SIZE)


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
            text_input_ids=text_ids, language_mask=mask
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
            text_input_ids=text_ids, language_mask=mask
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
            rng.integers(
                low=0, high=VOCAB_SIZE, size=(2, MAX_TEXT_LENGTH)
            ).astype(np.int64)
        )
        mask = torch.zeros(2, MAX_TEXT_LENGTH, dtype=torch.bool)
        result_ids, result_mask = encoder._pad_text_inputs(
            text_input_ids=text_ids, language_mask=mask
        )
        assert torch.equal(result_ids, text_ids)
        assert torch.equal(result_mask, mask)


class TestVLMEncoderForward:

    def _setup_encoder_mock_outputs(
        self,
        encoder: VLMEncoder,
        batch_size: int,
        pooling_method: str,
    ):
        """Configure the mock encoder to return expected output structure."""
        if pooling_method == PoolingMethod.NONE.value:
            vision_hidden = torch.zeros(
                batch_size, 49, HIDDEN_VISION_DIM
            )
            language_hidden = torch.zeros(
                batch_size, MAX_TEXT_LENGTH, HIDDEN_LANGUAGE_DIM
            )
        else:
            vision_hidden = torch.zeros(
                batch_size, 49, HIDDEN_VISION_DIM
            )
            language_hidden = torch.zeros(
                batch_size, MAX_TEXT_LENGTH, HIDDEN_LANGUAGE_DIM
            )

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
        mock_full_output = MagicMock()
        mock_full_output.vision_model_output = mock_vision_output
        mock_full_output.text_model_output = mock_language_output
        encoder.encoder.return_value = mock_full_output
        # Mock the image processor to pass through
        encoder.image_processor = MagicMock()
        mock_processed = MagicMock()
        mock_processed.to.return_value = {"pixel_values": torch.zeros(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)}
        mock_processed.__iter__ = lambda s: iter(["pixel_values"])
        mock_processed.__getitem__ = lambda s, k: torch.zeros(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
        mock_processed.keys = lambda: ["pixel_values"]
        encoder.image_processor.return_value = mock_processed

    @pytest.mark.parametrize(
        "time_steps, expected_feature_ndim",
        [
            (None, 2),
            (3, 3),
        ],
    )
    def test_output_shape_with_and_without_time(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
        vlm_input_factory: Callable[..., dict[str, torch.Tensor]],
        time_steps: int | None,
        expected_feature_ndim: int,
    ):
        batch_size = 2
        pooling_method = PoolingMethod.DEFAULT.value
        encoder = vlm_encoder_factory(pooling_method=pooling_method)
        effective_batch = batch_size * (time_steps or 1)
        self._setup_encoder_mock_outputs(
            encoder=encoder,
            batch_size=effective_batch,
            pooling_method=pooling_method,
        )
        inputs = vlm_input_factory(
            batch_size=batch_size,
            time_steps=time_steps,
            include_padding_mask=True,
        )
        output = encoder(inputs=inputs)
        image_features = output[EncoderOutputKeys.RGB.value]
        language_features = output[EncoderOutputKeys.LANGUAGE.value]
        assert image_features.ndim == expected_feature_ndim
        assert language_features.ndim == expected_feature_ndim
        assert image_features.shape[0] == batch_size
        assert language_features.shape[0] == batch_size
        if time_steps is not None:
            assert image_features.shape[1] == time_steps
            assert language_features.shape[1] == time_steps

    def test_missing_language_key_raises(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
        rng: np.random.Generator,
    ):
        encoder = vlm_encoder_factory()
        images = torch.from_numpy(
            rng.standard_normal((2, 3, 224, 224)).astype(np.float32)
        )
        with pytest.raises(ValueError, match="Expected key"):
            encoder(inputs={Cameras.LEFT.value: images})

    def test_non_tensor_text_input_raises(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
        rng: np.random.Generator,
    ):
        encoder = vlm_encoder_factory()
        images = torch.from_numpy(
            rng.standard_normal((2, 3, 224, 224)).astype(np.float32)
        )
        with pytest.raises(
            ValueError, match="tokenized_observations must be a tensor"
        ):
            encoder(
                inputs={
                    Cameras.LEFT.value: images,
                    SampleKey.TOKENIZED_OBSERVATIONS.value: "not a tensor",
                }
            )

    def test_non_tensor_image_input_raises(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
        rng: np.random.Generator,
    ):
        encoder = vlm_encoder_factory()
        token_ids = torch.from_numpy(
            rng.integers(low=0, high=VOCAB_SIZE, size=(2, 10)).astype(np.int64)
        )
        with pytest.raises(ValueError, match="images must be a tensor"):
            encoder(
                inputs={
                    Cameras.LEFT.value: "not a tensor",
                    SampleKey.TOKENIZED_OBSERVATIONS.value: token_ids,
                }
            )

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
            pooling_method=pooling_method,
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
        # (B, T, vision_seq_len, hidden_vision_dim)
        assert image_features.shape == (
            batch_size, time_steps, vision_seq_len, HIDDEN_VISION_DIM,
        )
        # (B, T, max_text_length, hidden_language_dim)
        assert language_features.shape == (
            batch_size, time_steps, MAX_TEXT_LENGTH, HIDDEN_LANGUAGE_DIM,
        )
        # (B, T, max_text_length)
        assert padding_mask.shape == (batch_size, time_steps, MAX_TEXT_LENGTH)

    def test_output_contains_all_expected_keys(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
        vlm_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        encoder = vlm_encoder_factory(
            pooling_method=PoolingMethod.DEFAULT.value
        )
        self._setup_encoder_mock_outputs(
            encoder=encoder,
            batch_size=batch_size,
            pooling_method=PoolingMethod.DEFAULT.value,
        )
        inputs = vlm_input_factory(batch_size=batch_size)
        output = encoder(inputs=inputs)
        assert EncoderOutputKeys.RGB.value in output
        assert EncoderOutputKeys.LANGUAGE.value in output
        assert encoder.padding_mask_name in output


class TestVLMEncoderGetVocabSize:

    def test_returns_text_model_vocab_size(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
    ):
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
        assert (
            specification.dimensions[EncoderOutputKeys.RGB.value]
            == expected_vision_dim
        )
        assert (
            specification.dimensions[EncoderOutputKeys.LANGUAGE.value]
            == expected_language_dim
        )

    def test_features_include_rgb_language_and_padding_mask(
        self,
        vlm_encoder_factory: Callable[..., VLMEncoder],
    ):
        encoder = vlm_encoder_factory()
        specification = encoder.get_output_specification()
        assert EncoderOutputKeys.RGB.value in specification.features
        assert EncoderOutputKeys.LANGUAGE.value in specification.features
        assert encoder.padding_mask_name in specification.features
        assert len(specification.features) == 3


class TestVLMEncoderIntegration:

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "model_name",
        [model_type.value for model_type in ImageTextModelType],
    )
    def test_forward_pass_per_model(
        self,
        vlm_input_factory: Callable[..., dict[str, torch.Tensor]],
        model_name: str,
    ):
        batch_size = 2
        encoder = VLMEncoder(
            input_keys=[
                Cameras.LEFT.value,
                SampleKey.TOKENIZED_OBSERVATIONS.value,
            ],
            pretrained=False,
            frozen=False,
            pooling_method=PoolingMethod.DEFAULT.value,
            model_name=model_name,
        )
        inputs = vlm_input_factory(
            batch_size=batch_size,
            sequence_length=10,
        )
        output = encoder(inputs=inputs)
        assert EncoderOutputKeys.RGB.value in output
        assert EncoderOutputKeys.LANGUAGE.value in output
