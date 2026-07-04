"""Augmentation configuration module for data preprocessing.

This module defines configuration classes for various image augmentation strategies that can be instantiated at runtime using Hydra.
"""

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING


@dataclass
class AugmentationConfig:
    """Base configuration for augmentation transforms.

    Attributes:
        _target_: Import path instantiated by Hydra.
        p: Probability of applying the transform.
    """

    _target_: str = MISSING
    p: float = 1.0


# Color augmentations


@dataclass
class ColorJitterConfig(AugmentationConfig):
    """Random brightness, contrast, saturation, and hue jitter.

    Attributes:
        _target_: Import path instantiated by Hydra.
        brightness: Brightness jitter range.
        contrast: Contrast jitter range.
        saturation: Saturation jitter range.
        hue: Hue jitter range.
        p: Probability of applying the transform.
    """

    _target_: str = "albumentations.ColorJitter"
    brightness: float = 0.3
    contrast: float = 0.4
    saturation: float = 0.5
    hue: float = 0.1
    p: float = 0.5


@dataclass
class RandomSunFlareConfig(AugmentationConfig):
    """Simulated sun-flare artifacts in the upper image region.

    Attributes:
        _target_: Import path instantiated by Hydra.
        flare_roi: Region of the image where the flare can appear, as fractions.
        src_color: RGB color of the flare.
        p: Probability of applying the transform.
    """

    _target_: str = "albumentations.RandomSunFlare"
    flare_roi: tuple[float, float, float, float] = (0, 0, 1, 0.5)
    src_color: tuple[int, int, int] = (255, 255, 255)
    p: float = 0.6


@dataclass
class RandomBrightnessContrastConfig(AugmentationConfig):
    """Random brightness and contrast shifts.

    Attributes:
        _target_: Import path instantiated by Hydra.
        brightness_limit: Maximum brightness shift as a fraction.
        contrast_limit: Maximum contrast shift as a fraction.
        p: Probability of applying the transform.
    """

    _target_: str = "albumentations.RandomBrightnessContrast"
    brightness_limit: float = 0.4
    contrast_limit: float = 0.4
    p: float = 0.6


@dataclass
class RandomGammaConfig(AugmentationConfig):
    """Random gamma correction within the configured limits.

    Attributes:
        _target_: Import path instantiated by Hydra.
        gamma_limit: Gamma correction range in percent.
        p: Probability of applying the transform.
    """

    _target_: str = "albumentations.RandomGamma"
    gamma_limit: tuple[int, int] = (80, 120)
    p: float = 0.3


@dataclass
class CLAHEConfig(AugmentationConfig):
    """Contrast-limited adaptive histogram equalization.

    Attributes:
        _target_: Import path instantiated by Hydra.
        clip_limit: Contrast-limiting threshold for histogram equalization.
        p: Probability of applying the transform.
    """

    _target_: str = "albumentations.CLAHE"
    clip_limit: float = 4.0
    p: float = 0.3


@dataclass
class RandomShadowConfig(AugmentationConfig):
    """Random polygonal shadows cast over the image.

    Attributes:
        _target_: Import path instantiated by Hydra.
        p: Probability of applying the transform.
    """

    _target_: str = "albumentations.RandomShadow"
    p: float = 0.4


@dataclass
class ImageCompressionConfig(AugmentationConfig):
    """Lossy compression artifacts within a random quality range.

    Attributes:
        _target_: Import path instantiated by Hydra.
        quality_lower: Lower bound of the JPEG/WebP quality range.
        quality_upper: Upper bound of the JPEG/WebP quality range.
        compression_type: Compression codec, jpeg or webp.
        p: Probability of applying the transform.
    """

    _target_: str = "albumentations.ImageCompression"
    quality_lower: int = 50
    quality_upper: int = 100
    compression_type: str = "jpeg"
    p: float = 0.2


# Spatial augmentations (compatible with depth images)


@dataclass
class GaussianBlurConfig(AugmentationConfig):
    """Gaussian blur with a random kernel size.

    Attributes:
        _target_: Import path instantiated by Hydra.
        blur_limit: Kernel size range for the blur.
        p: Probability of applying the transform.
    """

    _target_: str = "albumentations.GaussianBlur"
    blur_limit: tuple[int, int] = (3, 7)
    p: float = 0.5


@dataclass
class CoarseDropoutConfig(AugmentationConfig):
    """Random rectangular occlusion holes.

    Attributes:
        _target_: Import path instantiated by Hydra.
        max_holes: Maximum number of dropped rectangular regions.
        max_height: Maximum height of a dropped region in pixels.
        max_width: Maximum width of a dropped region in pixels.
        p: Probability of applying the transform.
    """

    _target_: str = "albumentations.CoarseDropout"
    max_holes: int = 8
    max_height: int = 8
    max_width: int = 8
    p: float = 0.3


@dataclass
class ShiftScaleRotateConfig(AugmentationConfig):
    """Random shift and scale; rotation stays disabled to keep kinematics consistent.

    Attributes:
        _target_: Import path instantiated by Hydra.
        rotate_limit: Rotation range in degrees.
        scale_limit: Scaling range as a fraction.
        shift_limit: Translation range as a fraction of the image size.
        p: Probability of applying the transform.
    """

    _target_: str = "albumentations.ShiftScaleRotate"
    rotate_limit: tuple[float, float] = (
        0,
        0,
    )  # No rotation here; handled separately because it requires to be consistent with robot kinematics.
    scale_limit: tuple[float, float] = (-0.5, 0.6)
    shift_limit: tuple[float, float] = (-0.0625, 0.0625)
    p: float = 0.5


@dataclass
class CenterCropConfig(AugmentationConfig):
    """Center crop augmentation that preserves aspect ratio.

    Attributes:
        _target_: Import path instantiated by Hydra.
        height: Will be set from DataConfig.
        width: Will be set from DataConfig.
        p: Always apply if enabled.
    """

    _target_: str = "albumentations.CenterCrop"
    height: int = MISSING  # Will be set from DataConfig
    width: int = MISSING  # Will be set from DataConfig
    p: float = 1.0  # Always apply if enabled


@dataclass
class RotateConfig(AugmentationConfig):
    """Rotation augmentation that requires special handling due to paired rotation of robot actions.

    Attributes:
        _target_: Import path instantiated by Hydra.
        limit: Rotation range in degrees.
        interpolation: cv2.INTER_LINEAR.
        p: Probability of applying the transform.
    """

    _target_: str = "albumentations.Rotate"
    limit: tuple[float, float] = (-5, 5)
    interpolation: int = 1  # cv2.INTER_LINEAR
    p: float = 0.5


@dataclass
class AugmentationPipelineConfig:
    """Configuration for augmentation pipeline.

    Attributes:
        _target_: Import path instantiated by Hydra.
        transforms: Augmentation transforms applied in order.
    """

    _target_: str = "albumentations.Compose"
    transforms: list[Any] = field(default_factory=list)
