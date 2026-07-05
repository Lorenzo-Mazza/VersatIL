"""Tests for versatil.configs.data.augmentations module."""

import warnings

import pytest
from hydra.utils import instantiate
from omegaconf import MISSING

from versatil.configs.data.augmentations import (
    AugmentationConfig,
    AugmentationPipelineConfig,
    CenterCropConfig,
    CLAHEConfig,
    CoarseDropoutConfig,
    ColorJitterConfig,
    GaussianBlurConfig,
    ImageCompressionConfig,
    RandomBrightnessContrastConfig,
    RandomGammaConfig,
    RandomShadowConfig,
    RandomSunFlareConfig,
    RotateConfig,
    ShiftScaleRotateConfig,
)


@pytest.mark.unit
class TestAugmentationConfig:
    def test_target_defaults_to_missing(self):
        config = AugmentationConfig()
        assert config._target_ == MISSING

    @pytest.mark.parametrize("probability", [0.5, 1.0])
    def test_stores_probability(self, probability):
        config = AugmentationConfig(p=probability)
        assert config.p == probability


@pytest.mark.unit
class TestColorJitterConfig:
    def test_target_points_to_albumentations(self):
        config = ColorJitterConfig()
        assert config._target_ == "albumentations.ColorJitter"

    @pytest.mark.parametrize("brightness", [0.2, 0.4])
    @pytest.mark.parametrize("hue", [0.05, 0.15])
    def test_stores_configuration(self, brightness, hue):
        config = ColorJitterConfig(brightness=brightness, hue=hue)
        assert config.brightness == brightness
        assert config.hue == hue


@pytest.mark.unit
class TestRandomSunFlareConfig:
    def test_target_points_to_albumentations(self):
        config = RandomSunFlareConfig()
        assert config._target_ == "albumentations.RandomSunFlare"


@pytest.mark.unit
class TestRandomBrightnessContrastConfig:
    def test_target_points_to_albumentations(self):
        config = RandomBrightnessContrastConfig()
        assert config._target_ == "albumentations.RandomBrightnessContrast"


@pytest.mark.unit
class TestRandomGammaConfig:
    def test_target_points_to_albumentations(self):
        config = RandomGammaConfig()
        assert config._target_ == "albumentations.RandomGamma"

    def test_gamma_limit_default(self):
        config = RandomGammaConfig()
        assert config.gamma_limit == (80, 120)


@pytest.mark.unit
class TestCLAHEConfig:
    def test_target_points_to_albumentations(self):
        config = CLAHEConfig()
        assert config._target_ == "albumentations.CLAHE"

    @pytest.mark.parametrize("clip_limit", [2.0, 6.0])
    def test_stores_clip_limit(self, clip_limit):
        config = CLAHEConfig(clip_limit=clip_limit)
        assert config.clip_limit == clip_limit


@pytest.mark.unit
class TestRandomShadowConfig:
    def test_target_points_to_albumentations(self):
        config = RandomShadowConfig()
        assert config._target_ == "albumentations.RandomShadow"


@pytest.mark.unit
class TestImageCompressionConfig:
    def test_target_points_to_albumentations(self):
        config = ImageCompressionConfig()
        assert config._target_ == "albumentations.ImageCompression"

    @pytest.mark.parametrize("quality_range", [(30, 80), (70, 100)])
    def test_stores_quality_range(self, quality_range):
        config = ImageCompressionConfig(quality_range=quality_range)
        assert config.quality_range == quality_range


@pytest.mark.unit
class TestGaussianBlurConfig:
    def test_target_points_to_albumentations(self):
        config = GaussianBlurConfig()
        assert config._target_ == "albumentations.GaussianBlur"


@pytest.mark.unit
class TestCoarseDropoutConfig:
    def test_target_points_to_albumentations(self):
        config = CoarseDropoutConfig()
        assert config._target_ == "albumentations.CoarseDropout"


@pytest.mark.unit
class TestShiftScaleRotateConfig:
    def test_target_points_to_albumentations(self):
        config = ShiftScaleRotateConfig()
        assert config._target_ == "albumentations.ShiftScaleRotate"

    def test_rotate_limit_default_is_zero(self):
        config = ShiftScaleRotateConfig()
        assert config.rotate_limit == (0, 0)


@pytest.mark.unit
class TestCenterCropConfig:
    def test_target_points_to_albumentations(self):
        config = CenterCropConfig()
        assert config._target_ == "albumentations.CenterCrop"

    def test_dimensions_required(self):
        config = CenterCropConfig()
        assert config.height == MISSING
        assert config.width == MISSING


@pytest.mark.unit
class TestRotateConfig:
    def test_target_points_to_albumentations(self):
        config = RotateConfig()
        assert config._target_ == "albumentations.Rotate"

    @pytest.mark.parametrize("limit", [(-5, 5), (-10, 10)])
    def test_stores_rotation_limit(self, limit):
        config = RotateConfig(limit=limit)
        assert config.limit == limit


@pytest.mark.unit
class TestAugmentationPipelineConfig:
    def test_target_points_to_compose(self):
        config = AugmentationPipelineConfig()
        assert config._target_ == "albumentations.Compose"

    def test_transforms_default_to_empty_list(self):
        config = AugmentationPipelineConfig()
        assert config.transforms == []


@pytest.mark.unit
class TestAugmentationInstantiation:
    def test_color_jitter_instantiates(self):
        config = ColorJitterConfig()
        instance = instantiate(config)
        assert type(instance).__name__ == "ColorJitter"

    def test_random_brightness_contrast_instantiates(self):
        config = RandomBrightnessContrastConfig()
        instance = instantiate(config)
        assert type(instance).__name__ == "RandomBrightnessContrast"

    def test_random_gamma_instantiates(self):
        config = RandomGammaConfig()
        instance = instantiate(config)
        assert type(instance).__name__ == "RandomGamma"

    def test_clahe_instantiates(self):
        config = CLAHEConfig()
        instance = instantiate(config)
        assert type(instance).__name__ == "CLAHE"

    def test_random_shadow_instantiates(self):
        config = RandomShadowConfig()
        instance = instantiate(config)
        assert type(instance).__name__ == "RandomShadow"

    def test_gaussian_blur_instantiates(self):
        config = GaussianBlurConfig()
        instance = instantiate(config)
        assert type(instance).__name__ == "GaussianBlur"

    def test_coarse_dropout_instantiates_with_configured_ranges(self):
        config = CoarseDropoutConfig(
            num_holes_range=(2, 4),
            hole_height_range=(3, 5),
            hole_width_range=(6, 7),
        )
        instance = instantiate(config)
        assert type(instance).__name__ == "CoarseDropout"
        assert instance.num_holes_range == (2, 4)
        assert instance.hole_height_range == (3, 5)
        assert instance.hole_width_range == (6, 7)

    def test_shift_scale_rotate_instantiates(self):
        config = ShiftScaleRotateConfig()
        instance = instantiate(config)
        assert type(instance).__name__ == "ShiftScaleRotate"

    def test_center_crop_instantiates_with_dimensions(self):
        config = CenterCropConfig(height=224, width=224)
        instance = instantiate(config)
        assert type(instance).__name__ == "CenterCrop"

    def test_rotate_instantiates(self):
        config = RotateConfig()
        instance = instantiate(config)
        assert type(instance).__name__ == "Rotate"

    def test_image_compression_instantiates_with_configured_range(self):
        config = ImageCompressionConfig(quality_range=(30, 90))
        instance = instantiate(config)
        assert type(instance).__name__ == "ImageCompression"
        assert instance.quality_range == (30, 90)

    def test_pipeline_instantiates_with_transforms(self):
        config = AugmentationPipelineConfig(
            transforms=[ColorJitterConfig(), GaussianBlurConfig()]
        )
        instance = instantiate(config)
        assert type(instance).__name__ == "Compose"
        assert len(instance.transforms) == 2


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        ColorJitterConfig(),
        RandomSunFlareConfig(),
        RandomBrightnessContrastConfig(),
        RandomGammaConfig(),
        CLAHEConfig(),
        RandomShadowConfig(),
        ImageCompressionConfig(),
        GaussianBlurConfig(),
        CoarseDropoutConfig(),
        ShiftScaleRotateConfig(),
        CenterCropConfig(height=224, width=224),
        RotateConfig(),
    ],
    ids=lambda config: type(config).__name__,
)
def test_augmentation_config_arguments_are_accepted_by_albumentations(config):
    # Albumentations warns and silently ignores unknown arguments, so any
    # warning here means a configured value would not reach the transform.
    # The ShiftScaleRotate deprecation is exempt: its arguments are honored,
    # it just recommends Affine as a replacement.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", message="ShiftScaleRotate is a special case")
        instantiate(config)
