"""Tests for versatil.models.encoding.encoders.image_mixin module."""

from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.data.constants import Cameras
from versatil.models.encoding.encoders.constants import EncoderOutputKeys
from versatil.models.encoding.encoders.image_mixin import (
    ImageEncoderMixin,
    resize_to_target_size,
)


class ConcreteImageEncoder(ImageEncoderMixin):
    def __init__(self, input_keys: list[str]):
        self._setup_camera_keys(input_keys=input_keys)

    def _encode_single_image(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        return torch.zeros(batch_size, 16)


@pytest.fixture
def image_encoder_factory() -> Callable[..., ConcreteImageEncoder]:
    """Factory for ConcreteImageEncoder with configurable camera keys."""

    def factory(
        input_keys: list[str] | None = None,
    ) -> ConcreteImageEncoder:
        if input_keys is None:
            input_keys = [Cameras.LEFT.value]
        return ConcreteImageEncoder(input_keys=input_keys)

    return factory


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
        # No zero-padding regions should exist for square-to-square resize
        assert not torch.any(torch.all(result == 0.0, dim=1))


class TestSetupCameraKeys:
    @pytest.mark.parametrize(
        "input_keys, expected_camera_keys, expected_multi_camera",
        [
            ([Cameras.LEFT.value], [Cameras.LEFT.value], False),
            (
                [Cameras.LEFT.value, Cameras.RIGHT.value],
                [Cameras.LEFT.value, Cameras.RIGHT.value],
                True,
            ),
            (
                [Cameras.LEFT.value, "tokenized_observations"],
                [Cameras.LEFT.value],
                False,
            ),
        ],
    )
    def test_extracts_camera_keys_and_detects_multi_camera(
        self,
        image_encoder_factory: Callable[..., ConcreteImageEncoder],
        input_keys: list[str],
        expected_camera_keys: list[str],
        expected_multi_camera: bool,
    ):
        encoder = image_encoder_factory(input_keys=input_keys)
        assert encoder.camera_keys == expected_camera_keys
        assert encoder.is_multi_camera is expected_multi_camera


class TestGetVisionFeatureNames:
    @pytest.mark.parametrize(
        "input_keys, expected_names",
        [
            (
                [Cameras.LEFT.value],
                [EncoderOutputKeys.RGB.value],
            ),
            (
                [Cameras.LEFT.value, Cameras.RIGHT.value],
                [
                    f"{EncoderOutputKeys.RGB.value}.{Cameras.LEFT.value}",
                    f"{EncoderOutputKeys.RGB.value}.{Cameras.RIGHT.value}",
                ],
            ),
        ],
    )
    def test_feature_names_match_camera_configuration(
        self,
        image_encoder_factory: Callable[..., ConcreteImageEncoder],
        input_keys: list[str],
        expected_names: list[str],
    ):
        encoder = image_encoder_factory(input_keys=input_keys)
        assert encoder._get_vision_feature_names() == expected_names


class TestEncodeVision:
    @pytest.mark.parametrize(
        "input_keys, expected_keys",
        [
            (
                [Cameras.LEFT.value],
                [EncoderOutputKeys.RGB.value],
            ),
            (
                [Cameras.LEFT.value, Cameras.RIGHT.value],
                [
                    f"{EncoderOutputKeys.RGB.value}.{Cameras.LEFT.value}",
                    f"{EncoderOutputKeys.RGB.value}.{Cameras.RIGHT.value}",
                ],
            ),
        ],
    )
    def test_output_keys_match_camera_configuration(
        self,
        image_encoder_factory: Callable[..., ConcreteImageEncoder],
        camera_image_factory: Callable[..., dict[str, torch.Tensor]],
        input_keys: list[str],
        expected_keys: list[str],
    ):
        encoder = image_encoder_factory(input_keys=input_keys)
        inputs = camera_image_factory(camera_keys=input_keys)
        result = encoder._encode_vision(inputs)
        assert list(result.keys()) == expected_keys

    def test_encode_single_image_called_per_camera(
        self,
        image_encoder_factory: Callable[..., ConcreteImageEncoder],
        camera_image_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        camera_keys = [Cameras.LEFT.value, Cameras.RIGHT.value]
        encoder = image_encoder_factory(input_keys=camera_keys)
        inputs = camera_image_factory(camera_keys=camera_keys, batch_size=3)
        result = encoder._encode_vision(inputs)
        rgb = EncoderOutputKeys.RGB.value
        assert result[f"{rgb}.{Cameras.LEFT.value}"].shape == (3, 16)
        assert result[f"{rgb}.{Cameras.RIGHT.value}"].shape == (3, 16)
