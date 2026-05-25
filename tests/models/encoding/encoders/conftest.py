"""Encoder test fixtures."""

from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.data.constants import CameraModality, Cameras
from versatil.data.metadata import CameraMetadata
from versatil.models.encoding.encoders.base import EncoderInput


@pytest.fixture
def encoder_input_factory() -> Callable[..., EncoderInput]:
    """Factory for EncoderInput instances with configurable fields."""

    def factory(
        keys: str | list[str] = "left",
        required: list[str] | None = None,
        exactly_one_camera_modality: list[CameraModality] | None = None,
        required_camera_modalities: list[CameraModality] | None = None,
        conditioning_key: str | None = None,
        conditioning_required: list[str] | None = None,
        conditioning_one_of_groups: list[list[str]] | None = None,
        requires_tokenized: bool = False,
    ) -> EncoderInput:
        return EncoderInput(
            keys=keys,
            required=required if required is not None else [],
            exactly_one_camera_modality=exactly_one_camera_modality
            if exactly_one_camera_modality is not None
            else [],
            required_camera_modalities=required_camera_modalities
            if required_camera_modalities is not None
            else [],
            conditioning_key=conditioning_key,
            conditioning_required=conditioning_required
            if conditioning_required is not None
            else [],
            conditioning_one_of_groups=conditioning_one_of_groups
            if conditioning_one_of_groups is not None
            else [],
            requires_tokenized=requires_tokenized,
        )

    return factory


@pytest.fixture
def rgbd_camera_metadata_factory(
    camera_metadata_factory: Callable[..., CameraMetadata],
) -> Callable[..., dict[str, CameraMetadata]]:
    """Factory for paired RGB and depth camera metadata."""

    def factory(
        rgb_key: str = Cameras.LEFT.value,
        depth_key: str = Cameras.DEPTH.value,
        image_width: int = 224,
        image_height: int = 224,
    ) -> dict[str, CameraMetadata]:
        return {
            rgb_key: camera_metadata_factory(
                camera_key=rgb_key,
                channels=3,
                image_width=image_width,
                image_height=image_height,
            ),
            depth_key: camera_metadata_factory(
                camera_key=depth_key,
                channels=1,
                image_width=image_width,
                image_height=image_height,
            ),
        }

    return factory


@pytest.fixture
def image_input_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    """Factory for image input tensors."""

    def factory(
        key: str = "left",
        batch_size: int = 2,
        channels: int = 3,
        height: int = 224,
        width: int = 224,
        time_steps: int = 1,
    ) -> dict[str, torch.Tensor]:
        shape = (batch_size, time_steps, channels, height, width)
        tensor = torch.from_numpy(rng.standard_normal(shape).astype(np.float32))
        return {key: tensor}

    return factory


@pytest.fixture
def rgbd_input_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    """Factory for paired RGB + depth input tensors."""

    def factory(
        rgb_key: str = Cameras.LEFT.value,
        depth_key: str = Cameras.DEPTH.value,
        batch_size: int = 2,
        rgb_channels: int = 3,
        depth_channels: int = 1,
        height: int = 224,
        width: int = 224,
        time_steps: int = 1,
    ) -> dict[str, torch.Tensor]:
        rgb_shape = (batch_size, time_steps, rgb_channels, height, width)
        depth_shape = (batch_size, time_steps, depth_channels, height, width)
        rgb_tensor = torch.from_numpy(rng.standard_normal(rgb_shape).astype(np.float32))
        depth_tensor = torch.from_numpy(
            rng.standard_normal(depth_shape).astype(np.float32)
        )
        return {rgb_key: rgb_tensor, depth_key: depth_tensor}

    return factory
