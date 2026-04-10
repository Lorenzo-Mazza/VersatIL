"""Tests for versatil.models.encoding.encoders.image_mixin module."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from versatil.data.constants import DEPTH_CAMERAS, RGB_CAMERAS, Cameras
from versatil.models.encoding.encoders.constants import EncoderOutputKeys
from versatil.models.encoding.encoders.image_mixin import (
    DepthEncoderMixin,
    ImageEncoderMixin,
    RGBDEncoderMixin,
    RGBEncoderMixin,
    resize_to_target_size,
)


class ConcreteRGBEncoder(RGBEncoderMixin):
    def __init__(self, input_keys: list[str]):
        self._setup_camera_keys(input_keys=input_keys)

    def _encode_single_image(self, images: torch.Tensor) -> torch.Tensor:
        return torch.zeros(images.shape[0], 16)


class ConcreteDepthEncoder(DepthEncoderMixin):
    def __init__(self, input_keys: list[str]):
        self._setup_camera_keys(input_keys=input_keys)

    def _encode_single_image(self, images: torch.Tensor) -> torch.Tensor:
        return torch.zeros(images.shape[0], 16)


class ConcreteRGBDEncoder(RGBDEncoderMixin):
    def __init__(self, input_keys: list[str]):
        self._setup_camera_keys(input_keys=input_keys)

    def _encode_single_image(self, images: torch.Tensor) -> torch.Tensor:
        return torch.zeros(images.shape[0], 16)


@dataclass
class MixinTestSpec:
    """Test specification for a modality mixin."""

    encoder_class: type[ImageEncoderMixin]
    single_key: str
    camera_group: list[str]
    output_modality: str


@pytest.fixture(
    params=[
        pytest.param(
            MixinTestSpec(
                encoder_class=ConcreteRGBEncoder,
                single_key=Cameras.LEFT.value,
                camera_group=RGB_CAMERAS,
                output_modality=EncoderOutputKeys.RGB.value,
            ),
            id="rgb",
        ),
        pytest.param(
            MixinTestSpec(
                encoder_class=ConcreteDepthEncoder,
                single_key=Cameras.DEPTH.value,
                camera_group=DEPTH_CAMERAS,
                output_modality=EncoderOutputKeys.DEPTH.value,
            ),
            id="depth",
        ),
        pytest.param(
            MixinTestSpec(
                encoder_class=ConcreteRGBDEncoder,
                single_key=Cameras.LEFT.value,
                camera_group=RGB_CAMERAS + DEPTH_CAMERAS,
                output_modality=EncoderOutputKeys.RGBD.value,
            ),
            id="rgbd",
        ),
    ]
)
def mixin_spec(request) -> MixinTestSpec:
    """Parametrized spec that runs each test for both RGB and Depth mixins."""
    return request.param


@pytest.fixture
def camera_image_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    """Factory for camera image input dicts."""

    def factory(
        camera_keys: list[str] | None = None,
        batch_size: int = 2,
        channels: int = 3,
        height: int = 32,
        width: int = 32,
    ) -> dict[str, torch.Tensor]:
        if camera_keys is None:
            camera_keys = [Cameras.LEFT.value]
        shape = (batch_size, channels, height, width)
        return {
            key: torch.from_numpy(rng.standard_normal(shape).astype(np.float32))
            for key in camera_keys
        }

    return factory


class TestResizeToTargetSize:
    def test_no_op_when_size_matches(
        self,
        rng: np.random.Generator,
    ):
        images = torch.from_numpy(
            rng.standard_normal((2, 3, 224, 224)).astype(np.float32)
        )
        result = resize_to_target_size(
            images=images, target_height=224, target_width=224
        )
        assert result.data_ptr() == images.data_ptr()

    def test_landscape_input_pads_height(
        self,
        rng: np.random.Generator,
    ):
        images = torch.from_numpy(
            rng.standard_normal((2, 3, 480, 640)).astype(np.float32)
        )
        result = resize_to_target_size(
            images=images, target_height=224, target_width=224
        )
        assert result.shape == (2, 3, 224, 224)
        resized_height = int(480 / (640 / 224))
        assert torch.all(result[:, :, resized_height:, :] == 0.0)
        assert not torch.all(result[:, :, :resized_height, :] == 0.0)

    def test_portrait_input_pads_width(
        self,
        rng: np.random.Generator,
    ):
        images = torch.from_numpy(
            rng.standard_normal((2, 3, 640, 480)).astype(np.float32)
        )
        result = resize_to_target_size(
            images=images, target_height=224, target_width=224
        )
        assert result.shape == (2, 3, 224, 224)
        resized_width = int(480 / (640 / 224))
        assert torch.all(result[:, :, :, resized_width:] == 0.0)
        assert not torch.all(result[:, :, :, :resized_width] == 0.0)

    def test_square_input_no_padding_needed(
        self,
        rng: np.random.Generator,
    ):
        images = torch.from_numpy(
            rng.standard_normal((2, 3, 256, 256)).astype(np.float32)
        )
        result = resize_to_target_size(
            images=images, target_height=224, target_width=224
        )
        assert result.shape == (2, 3, 224, 224)
        assert not torch.any(torch.all(result == 0.0, dim=1))


class TestMixinModality:
    def test_output_modality_matches_mixin(self, mixin_spec: MixinTestSpec):
        encoder = mixin_spec.encoder_class(input_keys=[mixin_spec.single_key])
        assert encoder._output_modality == mixin_spec.output_modality

    def test_camera_group_matches_mixin(self, mixin_spec: MixinTestSpec):
        encoder = mixin_spec.encoder_class(input_keys=[mixin_spec.single_key])
        assert encoder._camera_group == mixin_spec.camera_group


class TestSetupCameraKeys:
    def test_extracts_matching_camera_keys(self, mixin_spec: MixinTestSpec):
        key = mixin_spec.single_key
        encoder = mixin_spec.encoder_class(input_keys=[key])
        assert encoder.camera_keys == [key]
        assert encoder.is_multi_camera is False

    def test_filters_non_camera_keys(self, mixin_spec: MixinTestSpec):
        key = mixin_spec.single_key
        encoder = mixin_spec.encoder_class(input_keys=[key, "tokenized_observations"])
        assert encoder.camera_keys == [key]
        assert encoder.is_multi_camera is False


class TestSetupCameraKeysRGBMultiCamera:
    def test_multi_camera_detection(self):
        encoder = ConcreteRGBEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.RIGHT.value]
        )
        assert encoder.camera_keys == [Cameras.LEFT.value, Cameras.RIGHT.value]
        assert encoder.is_multi_camera is True


class TestGetVisionFeatureNames:
    def test_single_camera_uses_modality_key(self, mixin_spec: MixinTestSpec):
        encoder = mixin_spec.encoder_class(input_keys=[mixin_spec.single_key])
        assert encoder._get_vision_feature_names() == [mixin_spec.output_modality]

    def test_multi_camera_prefixes_with_modality(self):
        encoder = ConcreteRGBEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.RIGHT.value]
        )
        expected = [
            f"{EncoderOutputKeys.RGB.value}.{Cameras.LEFT.value}",
            f"{EncoderOutputKeys.RGB.value}.{Cameras.RIGHT.value}",
        ]
        assert encoder._get_vision_feature_names() == expected


class TestEncodeVision:
    def test_single_camera_output_key(
        self,
        mixin_spec: MixinTestSpec,
        camera_image_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        key = mixin_spec.single_key
        encoder = mixin_spec.encoder_class(input_keys=[key])
        inputs = camera_image_factory(camera_keys=[key])
        result = encoder._encode_vision(inputs)
        assert list(result.keys()) == [mixin_spec.output_modality]

    def test_multi_camera_output_keys(
        self,
        camera_image_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        camera_keys = [Cameras.LEFT.value, Cameras.RIGHT.value]
        encoder = ConcreteRGBEncoder(input_keys=camera_keys)
        inputs = camera_image_factory(camera_keys=camera_keys)
        result = encoder._encode_vision(inputs)
        rgb = EncoderOutputKeys.RGB.value
        expected_keys = [f"{rgb}.{Cameras.LEFT.value}", f"{rgb}.{Cameras.RIGHT.value}"]
        assert list(result.keys()) == expected_keys

    def test_encode_produces_correct_shape_per_camera(
        self,
        camera_image_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        camera_keys = [Cameras.LEFT.value, Cameras.RIGHT.value]
        encoder = ConcreteRGBEncoder(input_keys=camera_keys)
        inputs = camera_image_factory(camera_keys=camera_keys, batch_size=3)
        result = encoder._encode_vision(inputs)
        rgb = EncoderOutputKeys.RGB.value
        assert result[f"{rgb}.{Cameras.LEFT.value}"].shape == (3, 16)
        assert result[f"{rgb}.{Cameras.RIGHT.value}"].shape == (3, 16)
