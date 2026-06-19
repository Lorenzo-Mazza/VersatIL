"""Augmentation configuration module for data preprocessing.

This module defines configuration classes for various image augmentation strategies that can be instantiated at runtime using Hydra.
"""

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING


@dataclass
class AugmentationConfig:
    """Base configuration for augmentation transforms."""

    _target_: str = MISSING
    p: float = 1.0


# Color augmentations


@dataclass
class ColorJitterConfig(AugmentationConfig):
    """Random brightness, contrast, saturation, and hue jitter."""

    _target_: str = "albumentations.ColorJitter"
    brightness: float = 0.3
    contrast: float = 0.4
    saturation: float = 0.5
    hue: float = 0.1
    p: float = 0.5


@dataclass
class RandomSunFlareConfig(AugmentationConfig):
    """Simulated sun-flare artifacts in the upper image region."""

    _target_: str = "albumentations.RandomSunFlare"
    flare_roi: tuple[float, float, float, float] = (0, 0, 1, 0.5)
    src_color: tuple[int, int, int] = (255, 255, 255)
    p: float = 0.6


@dataclass
class RandomBrightnessContrastConfig(AugmentationConfig):
    """Random brightness and contrast shifts."""

    _target_: str = "albumentations.RandomBrightnessContrast"
    brightness_limit: float = 0.4
    contrast_limit: float = 0.4
    p: float = 0.6


@dataclass
class RandomGammaConfig(AugmentationConfig):
    """Random gamma correction within the configured limits."""

    _target_: str = "albumentations.RandomGamma"
    gamma_limit: tuple[int, int] = (80, 120)
    p: float = 0.3


@dataclass
class CLAHEConfig(AugmentationConfig):
    """Contrast-limited adaptive histogram equalization."""

    _target_: str = "albumentations.CLAHE"
    clip_limit: float = 4.0
    p: float = 0.3


@dataclass
class RandomShadowConfig(AugmentationConfig):
    """Random polygonal shadows cast over the image."""

    _target_: str = "albumentations.RandomShadow"
    p: float = 0.4


@dataclass
class ImageCompressionConfig(AugmentationConfig):
    """Lossy compression artifacts within a random quality range."""

    _target_: str = "albumentations.ImageCompression"
    quality_lower: int = 50
    quality_upper: int = 100
    compression_type: str = "jpeg"
    p: float = 0.2


# Spatial augmentations (compatible with depth images)


@dataclass
class GaussianBlurConfig(AugmentationConfig):
    """Gaussian blur with a random kernel size."""

    _target_: str = "albumentations.GaussianBlur"
    blur_limit: tuple[int, int] = (3, 7)
    p: float = 0.5


@dataclass
class CoarseDropoutConfig(AugmentationConfig):
    """Random rectangular occlusion holes."""

    _target_: str = "albumentations.CoarseDropout"
    max_holes: int = 8
    max_height: int = 8
    max_width: int = 8
    p: float = 0.3


@dataclass
class ShiftScaleRotateConfig(AugmentationConfig):
    """Random shift and scale; rotation stays disabled to keep kinematics consistent."""

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
    """Center crop augmentation that preserves aspect ratio."""

    _target_: str = "albumentations.CenterCrop"
    height: int = MISSING  # Will be set from DataConfig
    width: int = MISSING  # Will be set from DataConfig
    p: float = 1.0  # Always apply if enabled


@dataclass
class RotateConfig(AugmentationConfig):
    """Rotation augmentation that requires special handling due to paired rotation of robot actions."""

    _target_: str = "albumentations.Rotate"
    limit: tuple[float, float] = (-5, 5)
    interpolation: int = 1  # cv2.INTER_LINEAR
    p: float = 0.5


@dataclass
class AugmentationPipelineConfig:
    """Configuration for augmentation pipeline."""

    _target_: str = "albumentations.Compose"
    transforms: list[Any] = field(default_factory=list)
