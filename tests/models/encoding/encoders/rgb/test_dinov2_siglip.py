"""Tests for versatil.models.encoding.encoders.rgb.dinov2_siglip module."""

import re
from collections.abc import Callable, Iterator
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from versatil.data.constants import CameraModality, Cameras
from versatil.data.metadata import BaseMetadata, RGBCameraMetadata
from versatil.models.adaptation.lora import LoRAAdaptation
from versatil.models.encoding.encoders.constants import (
    DinoV2SigLIPBackboneType,
    EncoderOutputKeys,
    FlatBackboneType,
    PoolingMethod,
)
from versatil.models.encoding.encoders.rgb.dinov2_siglip import (
    DINOV2_SIGLIP_BACKBONE_CONFIGS,
    DinoV2SigLIPBackboneConfig,
    DinoV2SigLIPRGBEncoder,
)
from versatil.models.encoding.encoders.rgb.flat import FlatRGBEncoder

IMAGE_SIZE = 224
TINY_IMAGE_SIZE = 32
NUM_PATCHES = 4
DINO_FEATURE_DIMENSION = 2
SIGLIP_FEATURE_DIMENSION = 3


def _make_flat_encoder_mock(
    num_patches: int = NUM_PATCHES,
    num_features: int = DINO_FEATURE_DIMENSION,
) -> MagicMock:
    encoder = MagicMock(spec=FlatRGBEncoder)
    encoder.backbone = MagicMock(spec=nn.Module)
    encoder.backbone.patch_embed = MagicMock(spec=nn.Module)
    encoder.backbone.patch_embed.num_patches = num_patches
    encoder.feature_dim = num_features
    return encoder


class MockFlatRGBEncoderBackend:
    def __init__(self) -> None:
        self.flat_encoder_mock: MagicMock | None = None
        self.dino_encoder: MagicMock
        self.siglip_encoder: MagicMock
        self.encoder_queue: list[MagicMock]
        self.configure()

    @property
    def constructor_mock(self) -> MagicMock:
        if self.flat_encoder_mock is None:
            raise RuntimeError("FlatRGBEncoder constructor patch is not active.")
        return self.flat_encoder_mock

    def configure(
        self,
        dino_num_patches: int = NUM_PATCHES,
        siglip_num_patches: int = NUM_PATCHES,
        dino_feature_dimension: int = DINO_FEATURE_DIMENSION,
        siglip_feature_dimension: int = SIGLIP_FEATURE_DIMENSION,
    ) -> None:
        self.dino_encoder = _make_flat_encoder_mock(
            num_patches=dino_num_patches,
            num_features=dino_feature_dimension,
        )
        self.siglip_encoder = _make_flat_encoder_mock(
            num_patches=siglip_num_patches,
            num_features=siglip_feature_dimension,
        )
        self.encoder_queue = [self.dino_encoder, self.siglip_encoder]

    def build_encoder(
        self,
        input_keys: list[str],
        pretrained: bool,
        frozen: bool,
        pooling_method: str,
        backbone: str,
        image_size: int,
        intermediate_layer_index: int,
        model_dtype: str | None,
        lora_config: LoRAAdaptation | None,
    ) -> MagicMock:
        del (
            input_keys,
            pretrained,
            frozen,
            pooling_method,
            backbone,
            image_size,
            intermediate_layer_index,
            model_dtype,
            lora_config,
        )
        if len(self.encoder_queue) == 0:
            raise RuntimeError("FlatRGBEncoder was constructed more than twice.")
        return self.encoder_queue.pop(0)


@pytest.fixture
def mock_flat_rgb_encoder_backend() -> Iterator[MockFlatRGBEncoderBackend]:
    backend = MockFlatRGBEncoderBackend()
    patcher = patch(
        "versatil.models.encoding.encoders.rgb.dinov2_siglip.FlatRGBEncoder",
        autospec=True,
        side_effect=backend.build_encoder,
    )
    backend.flat_encoder_mock = patcher.start()
    yield backend
    patcher.stop()


@pytest.fixture
def dinov2_siglip_encoder_factory(
    mock_flat_rgb_encoder_backend: MockFlatRGBEncoderBackend,
) -> Callable[..., DinoV2SigLIPRGBEncoder]:
    def factory(
        input_keys: list[str] | None = None,
        pretrained: bool = False,
        frozen: bool = False,
        dino_num_patches: int = NUM_PATCHES,
        siglip_num_patches: int = NUM_PATCHES,
        dino_feature_dimension: int = DINO_FEATURE_DIMENSION,
        siglip_feature_dimension: int = SIGLIP_FEATURE_DIMENSION,
    ) -> DinoV2SigLIPRGBEncoder:
        if input_keys is None:
            input_keys = [Cameras.LEFT.value]
        mock_flat_rgb_encoder_backend.configure(
            dino_num_patches=dino_num_patches,
            siglip_num_patches=siglip_num_patches,
            dino_feature_dimension=dino_feature_dimension,
            siglip_feature_dimension=siglip_feature_dimension,
        )
        return DinoV2SigLIPRGBEncoder(
            input_keys=input_keys,
            pretrained=pretrained,
            frozen=frozen,
            backbone=DinoV2SigLIPBackboneType.DINOV2_SIGLIP_VIT_SO_224PX.value,
            model_dtype=None,
            lora_config=None,
        )

    return factory


class TestDinoV2SigLIPRGBEncoderInitialization:
    @pytest.mark.unit
    def test_invalid_backbone_raises(self) -> None:
        backbone = "invalid"
        valid_backbones = [model_type.value for model_type in DinoV2SigLIPBackboneType]
        expected_message = (
            f"Invalid DINOv2+SigLIP backbone '{backbone}'. "
            f"Must be one of: {valid_backbones}."
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            DinoV2SigLIPRGBEncoder(
                input_keys=Cameras.LEFT.value,
                pretrained=False,
                frozen=False,
                backbone=backbone,
                model_dtype=None,
                lora_config=None,
            )

    @pytest.mark.unit
    def test_builds_resolved_timm_towers(
        self,
        dinov2_siglip_encoder_factory: Callable[..., DinoV2SigLIPRGBEncoder],
        mock_flat_rgb_encoder_backend: MockFlatRGBEncoderBackend,
    ) -> None:
        encoder = dinov2_siglip_encoder_factory(pretrained=True, frozen=False)

        assert encoder.input_specification.keys == [Cameras.LEFT.value]
        assert encoder.input_specification.required_camera_modalities == [
            CameraModality.RGB
        ]
        assert (
            encoder.backbone_name
            == DinoV2SigLIPBackboneType.DINOV2_SIGLIP_VIT_SO_224PX.value
        )
        assert encoder.dino_model_name == FlatBackboneType.DINOV2_VITL14_REG4.value
        assert encoder.siglip_model_name == FlatBackboneType.SIGLIP_SO400M_224.value
        assert encoder.image_size == IMAGE_SIZE
        assert encoder.num_patches == NUM_PATCHES
        assert encoder.feature_dim == DINO_FEATURE_DIMENSION + SIGLIP_FEATURE_DIMENSION
        assert mock_flat_rgb_encoder_backend.constructor_mock.call_args_list[
            0
        ].kwargs == {
            "input_keys": [Cameras.LEFT.value],
            "pretrained": True,
            "frozen": False,
            "pooling_method": PoolingMethod.NONE.value,
            "backbone": FlatBackboneType.DINOV2_VITL14_REG4.value,
            "image_size": IMAGE_SIZE,
            "intermediate_layer_index": -2,
            "model_dtype": None,
            "lora_config": None,
        }
        assert mock_flat_rgb_encoder_backend.constructor_mock.call_args_list[
            1
        ].kwargs == {
            "input_keys": [Cameras.LEFT.value],
            "pretrained": True,
            "frozen": False,
            "pooling_method": PoolingMethod.NONE.value,
            "backbone": FlatBackboneType.SIGLIP_SO400M_224.value,
            "image_size": IMAGE_SIZE,
            "intermediate_layer_index": -2,
            "model_dtype": None,
            "lora_config": None,
        }

    @pytest.mark.unit
    def test_mismatched_patch_counts_raise(
        self,
        dinov2_siglip_encoder_factory: Callable[..., DinoV2SigLIPRGBEncoder],
    ) -> None:
        expected_message = "DINO and SigLIP patch counts must match, got 4 and 5."

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            dinov2_siglip_encoder_factory(
                dino_num_patches=4,
                siglip_num_patches=5,
            )


class TestDinoV2SigLIPRGBEncoderForward:
    @pytest.mark.unit
    def test_standardize_images_applies_channel_statistics(self) -> None:
        pixel_values = torch.tensor(
            [[[[0.2]], [[0.4]], [[0.6]]]],
            dtype=torch.float32,
        )
        mean = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32).view(1, 3, 1, 1)
        standard_deviation = torch.tensor(
            [0.1, 0.2, 0.3],
            dtype=torch.float32,
        ).view(1, 3, 1, 1)

        standardized = DinoV2SigLIPRGBEncoder._standardize_images(
            pixel_values=pixel_values,
            mean=mean,
            standard_deviation=standard_deviation,
        )

        torch.testing.assert_close(standardized, torch.ones_like(pixel_values))

    @pytest.mark.unit
    def test_encode_image_tokens_resizes_and_routes_standardized_inputs(
        self,
        dinov2_siglip_encoder_factory: Callable[..., DinoV2SigLIPRGBEncoder],
        mock_flat_rgb_encoder_backend: MockFlatRGBEncoderBackend,
    ) -> None:
        encoder = dinov2_siglip_encoder_factory()
        images = torch.full((2, 3, 32, 32), 0.25)
        resized_images = torch.full((2, 3, IMAGE_SIZE, IMAGE_SIZE), 0.25)
        dino_features = torch.ones(2, NUM_PATCHES, DINO_FEATURE_DIMENSION)
        siglip_features = torch.full(
            (2, NUM_PATCHES, SIGLIP_FEATURE_DIMENSION),
            2.0,
        )
        mock_flat_rgb_encoder_backend.dino_encoder._encode_single_image.return_value = (
            dino_features
        )
        mock_flat_rgb_encoder_backend.siglip_encoder._encode_single_image.return_value = siglip_features

        with patch(
            "versatil.models.encoding.encoders.rgb.dinov2_siglip.resize_to_target_size",
            autospec=True,
            return_value=resized_images,
        ) as resize_mock:
            fused_features = encoder.encode_image_tokens(images=images)

        resize_mock.assert_called_once()
        resize_call = resize_mock.call_args
        torch.testing.assert_close(resize_call.kwargs["images"], images)
        assert resize_call.kwargs["target_height"] == IMAGE_SIZE
        assert resize_call.kwargs["target_width"] == IMAGE_SIZE
        expected_dino_pixel_values = (
            resized_images - encoder.dino_standardization_mean
        ) / encoder.dino_standardization_std
        expected_siglip_pixel_values = (
            resized_images - encoder.siglip_standardization_mean
        ) / encoder.siglip_standardization_std
        torch.testing.assert_close(
            mock_flat_rgb_encoder_backend.dino_encoder._encode_single_image.call_args.args[
                0
            ],
            expected_dino_pixel_values,
        )
        torch.testing.assert_close(
            mock_flat_rgb_encoder_backend.siglip_encoder._encode_single_image.call_args.args[
                0
            ],
            expected_siglip_pixel_values,
        )
        torch.testing.assert_close(
            fused_features,
            torch.cat([dino_features, siglip_features], dim=2),
        )

    @pytest.mark.unit
    def test_encode_single_image_delegates_to_patch_token_encoder(
        self,
        dinov2_siglip_encoder_factory: Callable[..., DinoV2SigLIPRGBEncoder],
    ) -> None:
        encoder = dinov2_siglip_encoder_factory()
        images = torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE)
        expected_features = torch.ones(
            2,
            NUM_PATCHES,
            DINO_FEATURE_DIMENSION + SIGLIP_FEATURE_DIMENSION,
        )

        with patch.object(
            encoder,
            "encode_image_tokens",
            return_value=expected_features,
        ) as encode_spy:
            features = encoder._encode_single_image(images)

        encode_spy.assert_called_once_with(images=images)
        torch.testing.assert_close(features, expected_features)

    @pytest.mark.unit
    def test_encode_delegates_to_vision_mixin(
        self,
        dinov2_siglip_encoder_factory: Callable[..., DinoV2SigLIPRGBEncoder],
    ) -> None:
        encoder = dinov2_siglip_encoder_factory()
        inputs = {Cameras.LEFT.value: torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE)}
        expected_output = {
            EncoderOutputKeys.RGB.value: torch.ones(
                2,
                NUM_PATCHES,
                DINO_FEATURE_DIMENSION + SIGLIP_FEATURE_DIMENSION,
            )
        }

        with patch.object(
            encoder,
            "_encode_vision",
            return_value=expected_output,
        ) as encode_spy:
            output = encoder.encode(inputs=inputs)

        encode_spy.assert_called_once_with(inputs)
        assert list(output) == [EncoderOutputKeys.RGB.value]
        torch.testing.assert_close(
            output[EncoderOutputKeys.RGB.value],
            expected_output[EncoderOutputKeys.RGB.value],
        )


class TestDinoV2SigLIPRGBEncoderMetadata:
    @pytest.mark.unit
    def test_validate_input_metadata_accepts_camera_metadata(
        self,
        dinov2_siglip_encoder_factory: Callable[..., DinoV2SigLIPRGBEncoder],
    ) -> None:
        encoder = dinov2_siglip_encoder_factory()
        metadata = RGBCameraMetadata(
            camera_key=Cameras.LEFT.value,
            dtype="uint8",
            image_height=64,
            image_width=64,
        )

        error = encoder.validate_input_metadata(
            key=Cameras.LEFT.value,
            metadata=metadata,
        )

        assert error is None

    @pytest.mark.unit
    def test_validate_input_metadata_rejects_non_camera_metadata(
        self,
        dinov2_siglip_encoder_factory: Callable[..., DinoV2SigLIPRGBEncoder],
    ) -> None:
        encoder = dinov2_siglip_encoder_factory()
        metadata = MagicMock(spec=BaseMetadata)

        error = encoder.validate_input_metadata(
            key=Cameras.LEFT.value,
            metadata=metadata,
        )

        assert error == (
            f"Expected CameraMetadata for '{Cameras.LEFT.value}', got MagicMock"
        )

    @pytest.mark.unit
    def test_output_specification_uses_patch_token_dimension(
        self,
        dinov2_siglip_encoder_factory: Callable[..., DinoV2SigLIPRGBEncoder],
    ) -> None:
        encoder = dinov2_siglip_encoder_factory()

        output_specification = encoder.get_output_specification()

        assert len(output_specification) == 1
        assert output_specification[0].key == EncoderOutputKeys.RGB.value
        assert output_specification[0].dimension == (
            -1,
            DINO_FEATURE_DIMENSION + SIGLIP_FEATURE_DIMENSION,
        )


@pytest.mark.integration
def test_forward_pass_with_real_tiny_timm_towers() -> None:
    with patch.dict(
        DINOV2_SIGLIP_BACKBONE_CONFIGS,
        {
            DinoV2SigLIPBackboneType.DINOV2_SIGLIP_VIT_SO_224PX: (
                DinoV2SigLIPBackboneConfig(
                    dino_backbone=FlatBackboneType.DEIT_TINY,
                    siglip_backbone=FlatBackboneType.DEIT_TINY,
                    image_size=TINY_IMAGE_SIZE,
                )
            )
        },
    ):
        encoder = DinoV2SigLIPRGBEncoder(
            input_keys=Cameras.LEFT.value,
            pretrained=False,
            frozen=False,
            backbone=DinoV2SigLIPBackboneType.DINOV2_SIGLIP_VIT_SO_224PX.value,
            model_dtype=None,
            lora_config=None,
        )
    inputs = {
        Cameras.LEFT.value: torch.zeros(
            2,
            1,
            3,
            TINY_IMAGE_SIZE,
            TINY_IMAGE_SIZE,
        )
    }
    image_conditioned_inputs = {
        Cameras.LEFT.value: torch.ones(
            2,
            1,
            3,
            TINY_IMAGE_SIZE,
            TINY_IMAGE_SIZE,
        )
    }

    with torch.no_grad():
        output = encoder(inputs=inputs)
        image_conditioned_output = encoder(inputs=image_conditioned_inputs)

    features = output[EncoderOutputKeys.RGB.value]
    image_conditioned_features = image_conditioned_output[EncoderOutputKeys.RGB.value]
    assert features.shape == (2, 1, encoder.num_patches, encoder.feature_dim)
    assert torch.isfinite(features).all()
    assert not torch.allclose(features, image_conditioned_features)
