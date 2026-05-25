"""Tests for versatil.data.processing.image_processor module."""

from collections.abc import Callable
from unittest.mock import MagicMock

import albumentations as A
import hydra.utils
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from versatil.data.metadata import (
    CameraMetadata,
    DepthCameraMetadata,
    RGBCameraMetadata,
)
from versatil.data.processing.image_processor import ImageProcessor


@pytest.fixture
def mock_color_augmentation():
    """Factory for mock color augmentation compose."""

    def factory(side_effect=None):
        mock = MagicMock(spec=A.Compose)
        mock.transforms = [MagicMock()]
        if side_effect:
            mock.side_effect = side_effect
        else:
            mock.side_effect = lambda image: {"image": image * 1.1}
        return mock

    return factory


@pytest.fixture
def mock_spatial_augmentation():
    """Factory for mock spatial augmentation compose."""

    def factory(side_effect=None):
        mock = MagicMock(spec=A.Compose)
        mock.transforms = [MagicMock()]
        if side_effect:
            mock.side_effect = side_effect
        else:
            mock.side_effect = lambda image: {"image": image + 0.1}
        return mock

    return factory


@pytest.fixture
def camera_metadata_factory() -> Callable[..., dict[str, CameraMetadata]]:
    """Factory for per-camera metadata dicts."""

    def factory(
        cameras: dict[str, tuple[int, int, int]] | None = None,
    ) -> dict[str, CameraMetadata]:
        if cameras is None:
            cameras = {"left": (3, 224, 224)}
        result = {}
        for camera_key, (channels, height, width) in cameras.items():
            if channels == 1:
                result[camera_key] = DepthCameraMetadata(
                    camera_key=camera_key,
                    dtype="float32",
                    image_height=height,
                    image_width=width,
                )
            else:
                result[camera_key] = RGBCameraMetadata(
                    camera_key=camera_key,
                    dtype="uint8",
                    image_height=height,
                    image_width=width,
                )
        return result

    return factory


class TestImageProcessorInitialization:
    def test_no_augmentations(self):
        processor = ImageProcessor(train=True)
        assert not processor.use_color
        assert not processor.use_spatial
        assert processor.photometric_transform is None
        assert processor.spatial_transform is None

    @pytest.mark.parametrize("train", [True, False])
    def test_color_augmentation_only_applied_in_train(
        self,
        mock_color_augmentation,
        train: bool,
    ):
        mock_color = mock_color_augmentation()
        processor = ImageProcessor(color_augmentation=mock_color, train=train)
        assert processor.use_color == train

    @pytest.mark.parametrize(
        "cameras, expected_camera_count",
        [
            ({"left": (3, 224, 224)}, 1),
            ({"left": (3, 256, 256), "right": (3, 128, 128)}, 2),
            ({"left": (3, 224, 224), "depth": (1, 224, 224)}, 2),
        ],
    )
    def test_builds_per_camera_resize_transforms(
        self,
        camera_metadata_factory: Callable[..., dict[str, CameraMetadata]],
        cameras: dict[str, tuple[int, int, int]],
        expected_camera_count: int,
    ):
        metadata = camera_metadata_factory(cameras=cameras)
        processor = ImageProcessor(camera_metadata=metadata, train=True)
        assert len(processor._camera_resize) == expected_camera_count

    def test_single_channel_cameras_tracked(
        self,
        camera_metadata_factory: Callable[..., dict[str, CameraMetadata]],
    ):
        metadata = camera_metadata_factory(
            cameras={"left": (3, 224, 224), "depth": (1, 224, 224)}
        )
        processor = ImageProcessor(camera_metadata=metadata, train=True)
        assert "left" in processor._rgb_cameras
        assert "depth" not in processor._rgb_cameras


class TestNormalizeImageTensor:
    def test_scales_by_max_pixel_value(self):
        image = torch.tensor([[[0, 128, 255]]], dtype=torch.uint8)

        result = ImageProcessor.normalize_image_tensor(
            image=image,
            max_pixel_value=255.0,
        )

        assert result.dtype == torch.float32
        torch.testing.assert_close(
            result,
            torch.tensor([[[0.0, 128.0 / 255.0, 1.0]]]),
        )

    def test_uint16_casts_without_scaling(self):
        image = torch.tensor([[[0, 255, 1024]]], dtype=torch.uint16)

        result = ImageProcessor.normalize_image_tensor(image=image)

        assert result.dtype == torch.float32
        torch.testing.assert_close(
            result,
            torch.tensor([[[0.0, 255.0, 1024.0]]]),
        )

    def test_float_above_one_scales_by_max_pixel_value(self):
        image = torch.tensor([[[0.0, 128.0, 255.0]]])

        result = ImageProcessor.normalize_image_tensor(
            image=image,
            max_pixel_value=255.0,
        )

        torch.testing.assert_close(
            result,
            torch.tensor([[[0.0, 128.0 / 255.0, 1.0]]]),
        )

    def test_float_above_one_preserved_by_default(self):
        image = torch.tensor([[[0.0, 128.0, 255.0]]])

        result = ImageProcessor.normalize_image_tensor(image=image)

        torch.testing.assert_close(result, image)


class TestProcess:
    @pytest.mark.parametrize(
        "camera_key, channels, is_rgb",
        [
            ("left", 3, True),
            ("depth", 1, False),
        ],
    )
    def test_output_is_float32_tensor_with_correct_channel_order(
        self,
        synthetic_rgb_images: Callable[..., np.ndarray],
        synthetic_depth_images: Callable[..., np.ndarray],
        camera_metadata_factory: Callable[..., dict[str, CameraMetadata]],
        camera_key: str,
        channels: int,
        is_rgb: bool,
    ):
        metadata = camera_metadata_factory(cameras={camera_key: (channels, 64, 64)})
        processor = ImageProcessor(camera_metadata=metadata, train=False)
        if is_rgb:
            images = synthetic_rgb_images(num_timesteps=3, height=64, width=64)
        else:
            images = synthetic_depth_images(num_timesteps=3, height=64, width=64)
        result = processor.process(images=images, camera_key=camera_key)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        if is_rgb:
            assert result.shape == (3, channels, 64, 64)  # (T, C, H, W)
        else:
            assert result.shape == (3, 1, 64, 64)  # (T, 1, H, W)

    def test_depth_4d_hwc_converted_to_nchw(
        self,
        rng: np.random.Generator,
        camera_metadata_factory: Callable[..., dict[str, CameraMetadata]],
    ):
        metadata = camera_metadata_factory(cameras={"depth": (1, 64, 64)})
        processor = ImageProcessor(camera_metadata=metadata, train=False)
        depth_hwc = rng.uniform(0.5, 5.0, (3, 64, 64, 1)).astype(np.float32)
        result = processor.process(images=depth_hwc, camera_key="depth")
        assert result.shape == (3, 1, 64, 64)
        expected_values = torch.from_numpy(np.moveaxis(depth_hwc, -1, 1))
        assert torch.allclose(result, expected_values)

    def test_depth_3d_gets_channel_dim_added(
        self,
        rng: np.random.Generator,
        camera_metadata_factory: Callable[..., dict[str, CameraMetadata]],
    ):
        metadata = camera_metadata_factory(cameras={"depth": (1, 64, 64)})
        processor = ImageProcessor(camera_metadata=metadata, train=False)
        depth_3d = rng.uniform(0.5, 5.0, (3, 64, 64)).astype(np.float32)
        result = processor.process(images=depth_3d, camera_key="depth")
        assert result.shape == (3, 1, 64, 64)
        expected_values = torch.from_numpy(depth_3d[:, None])
        assert torch.allclose(result, expected_values)

    def test_rgb_values_normalized_to_0_1(
        self,
        camera_metadata_factory: Callable[..., dict[str, CameraMetadata]],
    ):
        metadata = camera_metadata_factory(cameras={"left": (3, 4, 4)})
        processor = ImageProcessor(camera_metadata=metadata, train=False)
        images = np.full((2, 4, 4, 3), 255, dtype=np.uint8)
        result = processor.process(images=images, camera_key="left")
        assert torch.allclose(result, torch.ones_like(result))

    def test_max_pixel_value_normalizes_single_channel_camera(self):
        metadata = {
            "left": CameraMetadata(
                camera_key="left",
                dtype="uint8",
                channels=1,
                image_height=2,
                image_width=2,
                max_pixel_value=255.0,
            )
        }
        processor = ImageProcessor(camera_metadata=metadata, train=False)
        images = np.array(
            [
                [
                    [0, 128],
                    [255, 64],
                ]
            ],
            dtype=np.uint8,
        )

        result = processor.process(images=images, camera_key="left")

        expected = torch.tensor(
            [
                [
                    [
                        [0.0, 128.0 / 255.0],
                        [1.0, 64.0 / 255.0],
                    ]
                ]
            ]
        )
        torch.testing.assert_close(result, expected)

    def test_uint16_single_channel_values_cast_without_scaling(self):
        metadata = {
            "left": CameraMetadata(
                camera_key="left",
                dtype="uint16",
                channels=1,
                image_height=2,
                image_width=2,
            )
        }
        processor = ImageProcessor(camera_metadata=metadata, train=False)
        images = np.array(
            [
                [
                    [0, 255],
                    [1024, 4096],
                ]
            ],
            dtype=np.uint16,
        )

        result = processor.process(images=images, camera_key="left")

        expected = torch.tensor(
            [
                [
                    [
                        [0.0, 255.0],
                        [1024.0, 4096.0],
                    ]
                ]
            ]
        )
        torch.testing.assert_close(result, expected)

    def test_color_augmentation_applied_only_to_rgb(
        self,
        synthetic_rgb_images: Callable[..., np.ndarray],
        synthetic_depth_images: Callable[..., np.ndarray],
        mock_color_augmentation,
        camera_metadata_factory: Callable[..., dict[str, CameraMetadata]],
    ):
        mock_color = mock_color_augmentation()
        metadata = camera_metadata_factory(
            cameras={"left": (3, 64, 64), "depth": (1, 64, 64)}
        )
        processor = ImageProcessor(
            color_augmentation=mock_color, camera_metadata=metadata, train=True
        )
        rgb_images = synthetic_rgb_images(num_timesteps=2, height=64, width=64)
        depth_images = synthetic_depth_images(num_timesteps=2, height=64, width=64)
        processor.process(images=rgb_images, camera_key="left")
        color_calls_after_rgb = mock_color.call_count

        processor.process(images=depth_images, camera_key="depth")
        color_calls_after_depth = mock_color.call_count

        assert color_calls_after_rgb == 2
        assert color_calls_after_depth == 2  # no additional calls for depth

    def test_resize_applied_per_camera(
        self,
        rng: np.random.Generator,
        camera_metadata_factory: Callable[..., dict[str, CameraMetadata]],
    ):
        metadata = camera_metadata_factory(
            cameras={"left": (3, 32, 32), "right": (3, 64, 64)}
        )
        processor = ImageProcessor(camera_metadata=metadata, train=False)
        images = rng.integers(0, 255, (2, 100, 100, 3), dtype=np.uint8)
        left_result = processor.process(images=images.copy(), camera_key="left")
        right_result = processor.process(images=images.copy(), camera_key="right")
        assert left_result.shape == (2, 3, 32, 32)
        assert right_result.shape == (2, 3, 64, 64)

    def test_eval_mode_skips_augmentations_but_resizes(
        self,
        rng: np.random.Generator,
        mock_color_augmentation,
        mock_spatial_augmentation,
        camera_metadata_factory: Callable[..., dict[str, CameraMetadata]],
    ):
        mock_color = mock_color_augmentation()
        mock_spatial = mock_spatial_augmentation()
        metadata = camera_metadata_factory(cameras={"left": (3, 32, 32)})
        processor = ImageProcessor(
            color_augmentation=mock_color,
            spatial_augmentation=mock_spatial,
            camera_metadata=metadata,
            train=False,
        )
        images = rng.integers(0, 255, (2, 100, 100, 3), dtype=np.uint8)
        result = processor.process(images=images, camera_key="left")
        assert result.shape == (2, 3, 32, 32)
        mock_color.assert_not_called()
        mock_spatial.assert_not_called()

    def test_spatial_augmentation_applied_to_depth(
        self,
        synthetic_depth_images: Callable[..., np.ndarray],
        mock_spatial_augmentation,
        camera_metadata_factory: Callable[..., dict[str, CameraMetadata]],
    ):
        mock_spatial = mock_spatial_augmentation()
        metadata = camera_metadata_factory(cameras={"depth": (1, 64, 64)})
        processor = ImageProcessor(
            spatial_augmentation=mock_spatial, camera_metadata=metadata, train=True
        )
        depth_images = synthetic_depth_images(num_timesteps=2, height=64, width=64)
        processor.process(images=depth_images, camera_key="depth")
        assert mock_spatial.call_count == 2

    def test_process_without_resize_skips_resize(
        self,
        rng: np.random.Generator,
    ):
        processor = ImageProcessor(train=False)
        images = rng.integers(0, 255, (2, 32, 32, 3), dtype=np.uint8)
        result = processor.process(images=images, camera_key="left")
        assert result.shape == (2, 3, 32, 32)

    def test_augmentation_order_is_resize_color_spatial(
        self,
        rng: np.random.Generator,
        mock_color_augmentation,
        mock_spatial_augmentation,
        camera_metadata_factory: Callable[..., dict[str, CameraMetadata]],
    ):
        call_order = []
        mock_color = mock_color_augmentation(
            side_effect=lambda image: (call_order.append("color"), {"image": image})[1]
        )
        mock_spatial = mock_spatial_augmentation(
            side_effect=lambda image: (
                call_order.append("spatial"),
                {"image": image},
            )[1]
        )
        metadata = camera_metadata_factory(cameras={"left": (3, 64, 64)})
        processor = ImageProcessor(
            color_augmentation=mock_color,
            spatial_augmentation=mock_spatial,
            camera_metadata=metadata,
            train=True,
        )
        images = rng.integers(0, 255, (1, 64, 64, 3), dtype=np.uint8)
        processor.process(images=images, camera_key="left")
        assert call_order == ["color", "spatial"]


@pytest.mark.integration
class TestRealHydraConfigIntegration:
    def test_real_augmentations_with_camera_metadata(
        self,
        synthetic_rgb_images: Callable[..., np.ndarray],
        camera_metadata_factory: Callable[..., dict[str, CameraMetadata]],
    ):
        images = synthetic_rgb_images(num_timesteps=3)
        color_config = OmegaConf.create(
            {
                "_target_": "albumentations.Compose",
                "transforms": [
                    {
                        "_target_": "albumentations.ColorJitter",
                        "brightness": 0.2,
                        "contrast": 0.2,
                        "saturation": 0.2,
                        "hue": 0.1,
                        "p": 1.0,
                    },
                ],
            }
        )
        color_config = hydra.utils.instantiate(color_config)
        metadata = camera_metadata_factory(cameras={"left": (3, 64, 64)})
        processor = ImageProcessor(
            color_augmentation=color_config,
            camera_metadata=metadata,
            train=True,
        )
        result = processor.process(images=images, camera_key="left")
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == images.shape[0]  # same temporal length
        assert result.shape[1] == 3  # channels first after reorder
