"""Tests for versatil.data.augmentation.augmentation_pipeline module."""

from collections.abc import Callable
from unittest.mock import MagicMock, call, patch

import albumentations as A
import hydra.utils
import numpy as np
import pytest
from omegaconf import OmegaConf

from versatil.data.augmentation.augmentation_pipeline import AugmentationPipeline


@pytest.fixture
def mock_color_augmentation():
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
def mock_resize_transform():
    def factory(side_effect=None, return_value=None):
        mock_resize = MagicMock(spec=A.Resize)
        mock_instance = MagicMock()
        if side_effect:
            mock_instance.side_effect = side_effect
        elif return_value:
            mock_instance.return_value = return_value
        else:
            mock_instance.side_effect = lambda **kwargs: {"image": kwargs.get("image")}
        mock_resize.return_value = mock_instance
        return mock_resize

    yield factory


class TestAugmentationPipelineInitialization:
    def test_init_no_augmentations(self):
        pipeline = AugmentationPipeline(train=True)

        assert not pipeline.use_color
        assert not pipeline.use_spatial
        assert not pipeline.use_resize
        assert pipeline.photometric_transform is None
        assert pipeline.spatial_transform is None
        assert pipeline.resize_transform_rgb is None
        assert pipeline.resize_transform_depth is None

    def test_init_with_color_train(self, mock_color_augmentation):
        mock_color = mock_color_augmentation()
        pipeline = AugmentationPipeline(color_augmentation=mock_color, train=True)
        assert pipeline.use_color
        assert pipeline.photometric_transform == mock_color

    def test_init_with_color_eval(self, mock_color_augmentation):
        pipeline = AugmentationPipeline(
            color_augmentation=mock_color_augmentation(), train=False
        )

        assert not pipeline.use_color
        assert pipeline.photometric_transform is None

    def test_init_with_spatial_train(self, mock_spatial_augmentation):
        mock_spatial = mock_spatial_augmentation()
        pipeline = AugmentationPipeline(spatial_augmentation=mock_spatial, train=True)
        assert pipeline.use_spatial
        assert pipeline.spatial_transform == mock_spatial

    @patch("versatil.data.augmentation.augmentation_pipeline.A.Resize")
    def test_init_with_resize(self, mock_resize_class):
        mock_rgb_resize = MagicMock()
        mock_depth_resize = MagicMock()
        mock_resize_class.side_effect = [mock_rgb_resize, mock_depth_resize]

        pipeline = AugmentationPipeline(target_height=224, target_width=224, train=True)

        assert pipeline.use_resize
        assert pipeline.resize_transform_rgb == mock_rgb_resize
        assert pipeline.resize_transform_depth == mock_depth_resize
        mock_resize_class.assert_has_calls(
            [
                call(height=224, width=224, interpolation=1, p=1.0),
                call(height=224, width=224, interpolation=0, p=1.0),
            ]
        )


class TestApplyRGBAugmentations:
    def test_apply_rgb_no_transforms(
        self,
        synthetic_rgb_images: Callable[..., np.ndarray],
    ):
        images = synthetic_rgb_images(num_timesteps=3)
        pipeline = AugmentationPipeline(train=True)

        result = pipeline.apply_rgb_augmentations(images)

        np.testing.assert_array_equal(result, images)

    def test_apply_rgb_with_resize(
        self,
        synthetic_rgb_images: Callable[..., np.ndarray],
        mock_resize_transform,
    ):
        images = synthetic_rgb_images(num_timesteps=3)
        resized_frame = np.ones((224, 224, 3), dtype=np.float32) * 0.5
        mock_class = mock_resize_transform(return_value={"image": resized_frame})
        mock_instance = mock_class.return_value

        with patch(
            "versatil.data.augmentation.augmentation_pipeline.A.Resize", new=mock_class
        ):
            pipeline = AugmentationPipeline(
                target_height=224, target_width=224, train=True
            )
            result = pipeline.apply_rgb_augmentations(images)

        expected = np.full((3, 224, 224, 3), 0.5, dtype=np.float32)
        assert result.shape == (3, 224, 224, 3)
        np.testing.assert_allclose(result, expected)
        assert mock_instance.call_count == 3

    def test_apply_rgb_with_color(
        self,
        mock_color_augmentation,
        synthetic_rgb_images: Callable[..., np.ndarray],
    ):
        images = synthetic_rgb_images(num_timesteps=3)
        modified_frame = images[0] * 1.1
        mock_color = mock_color_augmentation()
        pipeline = AugmentationPipeline(color_augmentation=mock_color, train=True)

        result = pipeline.apply_rgb_augmentations(images)

        assert result.shape == images.shape
        np.testing.assert_allclose(result[0], modified_frame)
        assert mock_color.call_count == 3

    def test_apply_rgb_with_spatial(
        self,
        mock_spatial_augmentation,
        synthetic_rgb_images: Callable[..., np.ndarray],
    ):
        images = synthetic_rgb_images(num_timesteps=3)
        modified_frame = images[0] + 0.1
        mock_spatial = mock_spatial_augmentation()
        pipeline = AugmentationPipeline(spatial_augmentation=mock_spatial, train=True)

        result = pipeline.apply_rgb_augmentations(images)

        assert result.shape == images.shape
        np.testing.assert_allclose(result[0], modified_frame)
        assert mock_spatial.call_count == 3

    def test_apply_rgb_augmentation_order(
        self,
        mock_resize_transform,
        mock_color_augmentation,
        mock_spatial_augmentation,
        synthetic_rgb_images: Callable[..., np.ndarray],
    ):
        images = synthetic_rgb_images(num_timesteps=3)
        mock_class = mock_resize_transform(
            side_effect=lambda **kwargs: {"image": kwargs["image"] + 0.1}
        )
        mock_instance = mock_class.return_value
        mock_color = mock_color_augmentation(
            side_effect=lambda image: {"image": image * 1.1}
        )
        mock_spatial = mock_spatial_augmentation(
            side_effect=lambda image: {"image": image + 0.1}
        )
        with patch(
            "versatil.data.augmentation.augmentation_pipeline.A.Resize", new=mock_class
        ):
            pipeline = AugmentationPipeline(
                color_augmentation=mock_color,
                spatial_augmentation=mock_spatial,
                target_height=64,
                target_width=64,
                train=True,
            )
            result = pipeline.apply_rgb_augmentations(images)

        # Expected: original +0.1 (resize) *1.1 (color) +0.1 (spatial)
        expected = ((images + 0.1) * 1.1) + 0.1
        np.testing.assert_allclose(result, expected)
        assert mock_instance.call_count == 3
        assert mock_color.call_count == 3
        assert mock_spatial.call_count == 3


class TestApplyDepthAugmentations:
    def test_apply_depth_no_transforms(
        self,
        synthetic_depth_images: Callable[..., np.ndarray],
    ):
        images = synthetic_depth_images(num_timesteps=3)
        pipeline = AugmentationPipeline(train=True)

        result = pipeline.apply_depth_augmentations(images)

        np.testing.assert_array_equal(result, images)

    def test_apply_depth_with_resize(
        self,
        synthetic_depth_images: Callable[..., np.ndarray],
        mock_resize_transform,
    ):
        images = synthetic_depth_images(num_timesteps=3)
        resized_frame = np.ones((128, 128), dtype=np.float32) * 5.0
        mock_class = mock_resize_transform(return_value={"image": resized_frame})
        mock_instance = mock_class.return_value

        with patch(
            "versatil.data.augmentation.augmentation_pipeline.A.Resize", new=mock_class
        ):
            pipeline = AugmentationPipeline(
                target_height=128, target_width=128, train=True
            )
            result = pipeline.apply_depth_augmentations(images)

        assert result.shape == (3, 128, 128)
        np.testing.assert_allclose(result, np.stack([resized_frame] * 3))
        assert mock_instance.call_count == 3

    def test_apply_depth_with_spatial(
        self,
        mock_spatial_augmentation,
        synthetic_depth_images: Callable[..., np.ndarray],
    ):
        images = synthetic_depth_images(num_timesteps=3)
        modified_frame = images[0] + 0.1
        mock_spatial = mock_spatial_augmentation(
            side_effect=lambda image: {"image": image + 0.1}
        )
        pipeline = AugmentationPipeline(spatial_augmentation=mock_spatial, train=True)
        result = pipeline.apply_depth_augmentations(images)
        assert result.shape == images.shape
        np.testing.assert_allclose(result[0], modified_frame)
        assert mock_spatial.call_count == 3

    def test_apply_depth_no_color_augmentation(
        self,
        synthetic_depth_images: Callable[..., np.ndarray],
        mock_color_augmentation,
    ):
        images = synthetic_depth_images(num_timesteps=3)
        mock_color = mock_color_augmentation()
        pipeline = AugmentationPipeline(color_augmentation=mock_color, train=True)
        result = pipeline.apply_depth_augmentations(images)
        mock_color.assert_not_called()
        np.testing.assert_array_equal(result, images)

    def test_apply_depth_augmentation_order(
        self,
        mock_resize_transform,
        mock_spatial_augmentation,
        synthetic_depth_images: Callable[..., np.ndarray],
    ):
        images = synthetic_depth_images(num_timesteps=3)
        mock_resize = mock_resize_transform(
            side_effect=lambda **kwargs: {"image": kwargs["image"] + 0.1}
        )
        mock_spatial = mock_spatial_augmentation(
            side_effect=lambda image: {"image": image - 0.05}
        )
        mock_instance_resize = mock_resize.return_value
        with patch(
            "versatil.data.augmentation.augmentation_pipeline.A.Resize", new=mock_resize
        ):
            pipeline = AugmentationPipeline(
                spatial_augmentation=mock_spatial,
                target_height=64,
                target_width=64,
                train=True,
            )
            result = pipeline.apply_depth_augmentations(images)
        # Expected: (original +0.1) -0.05
        expected = (images + 0.1) - 0.05
        np.testing.assert_allclose(result, expected)
        assert mock_instance_resize.call_count == 3
        assert mock_spatial.call_count == 3


class TestIntegration:
    def test_full_pipeline_workflow_rgb(
        self,
        synthetic_rgb_images: Callable[..., np.ndarray],
        mock_color_augmentation,
        mock_spatial_augmentation,
    ):
        images = synthetic_rgb_images(num_timesteps=3)
        mock_color = mock_color_augmentation(
            side_effect=lambda image: {"image": image * 1.1}
        )
        mock_spatial = mock_spatial_augmentation(
            side_effect=lambda image: {"image": image - 0.05}
        )

        pipeline = AugmentationPipeline(
            color_augmentation=mock_color, spatial_augmentation=mock_spatial, train=True
        )

        result = pipeline.apply_rgb_augmentations(images)
        expected = (images * 1.1) - 0.05
        np.testing.assert_allclose(result, expected)
        assert mock_color.call_count == 3
        assert mock_spatial.call_count == 3

    def test_full_pipeline_workflow_depth(
        self,
        synthetic_depth_images: Callable[..., np.ndarray],
        mock_spatial_augmentation,
    ):
        images = synthetic_depth_images(num_timesteps=3)
        mock_spatial = mock_spatial_augmentation(
            side_effect=lambda image: {"image": image - 0.05}
        )
        pipeline = AugmentationPipeline(spatial_augmentation=mock_spatial, train=True)
        result = pipeline.apply_depth_augmentations(images)

        expected = images - 0.05
        np.testing.assert_allclose(result, expected)
        assert mock_spatial.call_count == 3

    def test_eval_mode_disables_training_augmentations(self):
        pipeline = AugmentationPipeline(
            color_augmentation=MagicMock(),
            spatial_augmentation=MagicMock(),
            train=False,
        )

        assert not pipeline.use_color
        assert not pipeline.use_spatial

    def test_full_pipeline_with_resize(
        self,
        synthetic_rgb_images: Callable[..., np.ndarray],
        mock_resize_transform,
    ):
        images = synthetic_rgb_images(num_timesteps=3)
        mock_resize = mock_resize_transform(
            side_effect=lambda **kwargs: {"image": kwargs["image"][::2, ::2]}
        )
        with patch(
            "versatil.data.augmentation.augmentation_pipeline.A.Resize", new=mock_resize
        ):
            pipeline = AugmentationPipeline(
                target_height=32, target_width=32, train=True
            )
            rgb_result = pipeline.apply_rgb_augmentations(images)
        expected_rgb = np.stack([frame[::2, ::2] for frame in images])
        np.testing.assert_allclose(rgb_result, expected_rgb, rtol=1e-5)


@pytest.mark.integration
class TestRealHydraConfigIntegration:
    def test_real_color_augmentation_pipeline(
        self,
        synthetic_rgb_images: Callable[..., np.ndarray],
    ):
        images = synthetic_rgb_images(num_timesteps=3)
        config = OmegaConf.create(
            {
                "_target_": "albumentations.Compose",
                "transforms": [
                    {
                        "_target_": "albumentations.ColorJitter",
                        "brightness": 0.3,
                        "contrast": 0.4,
                        "saturation": 0.5,
                        "hue": 0.1,
                        "p": 0.5,
                    },
                    {
                        "_target_": "albumentations.RandomBrightnessContrast",
                        "brightness_limit": 0.4,
                        "contrast_limit": 0.4,
                        "p": 0.6,
                    },
                ],
            }
        )
        config = hydra.utils.instantiate(config)
        pipeline = AugmentationPipeline(color_augmentation=config, train=True)

        assert pipeline.photometric_transform is not None
        assert callable(pipeline.photometric_transform)

        result = pipeline.apply_rgb_augmentations(images)
        assert result.shape == images.shape

    def test_real_spatial_augmentation_pipeline(
        self,
        synthetic_rgb_images: Callable[..., np.ndarray],
    ):
        images = synthetic_rgb_images(num_timesteps=3)
        config = OmegaConf.create(
            {
                "_target_": "albumentations.Compose",
                "transforms": [
                    {
                        "_target_": "albumentations.GaussianBlur",
                        "blur_limit": (3, 7),
                        "p": 0.5,
                    },
                    {
                        "_target_": "albumentations.CoarseDropout",
                        "max_holes": 8,
                        "max_height": 8,
                        "max_width": 8,
                        "p": 0.3,
                    },
                ],
            }
        )
        config = hydra.utils.instantiate(config)
        pipeline = AugmentationPipeline(spatial_augmentation=config, train=True)

        assert pipeline.spatial_transform is not None
        assert callable(pipeline.spatial_transform)

        result = pipeline.apply_rgb_augmentations(images)
        assert result.shape == images.shape

    def test_real_both_augmentation_pipelines(
        self,
        synthetic_rgb_images: Callable[..., np.ndarray],
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

        spatial_config = OmegaConf.create(
            {
                "_target_": "albumentations.Compose",
                "transforms": [
                    {
                        "_target_": "albumentations.GaussianBlur",
                        "blur_limit": (3, 5),
                        "p": 1.0,
                    },
                ],
            }
        )
        spatial_config = hydra.utils.instantiate(spatial_config)
        color_config = hydra.utils.instantiate(color_config)
        pipeline = AugmentationPipeline(
            color_augmentation=color_config,
            spatial_augmentation=spatial_config,
            train=True,
        )

        assert callable(pipeline.photometric_transform)
        assert callable(pipeline.spatial_transform)

        result = pipeline.apply_rgb_augmentations(images)
        assert result.shape == images.shape
        assert not np.allclose(result, images)
