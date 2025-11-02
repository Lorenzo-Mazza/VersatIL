"""Augmentation configuration module for data preprocessing.

This module defines configuration classes for various image augmentation strategies that can be instantiated at runtime using Hydra.
"""

from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class AugmentationConfig:
    """Base configuration for augmentation transforms."""
    _target_: str = MISSING
    p: float = 1.0


# Color augmentations

@dataclass
class ColorJitterConfig(AugmentationConfig):
    _target_: str = "albumentations.ColorJitter"
    brightness: float = 0.3
    contrast: float = 0.4
    saturation: float = 0.5
    hue: float = 0.1
    p: float = 0.5


@dataclass
class RandomSunFlareConfig(AugmentationConfig):
    _target_: str = "albumentations.RandomSunFlare"
    flare_roi: tuple[float, float, float, float] = (0, 0, 1, 0.5)
    src_color: tuple[int, int, int] = (255, 255, 255)
    p: float = 0.6


@dataclass
class RandomBrightnessContrastConfig(AugmentationConfig):
    _target_: str = "albumentations.RandomBrightnessContrast"
    brightness_limit: float = 0.4
    contrast_limit: float = 0.4
    p: float = 0.6


@dataclass
class RandomGammaConfig(AugmentationConfig):
    _target_: str = "albumentations.RandomGamma"
    gamma_limit: tuple[int, int] = (80, 120)
    p: float = 0.3


@dataclass
class CLAHEConfig(AugmentationConfig):
    _target_: str = "albumentations.CLAHE"
    clip_limit: float = 4.0
    p: float = 0.3


@dataclass
class RandomShadowConfig(AugmentationConfig):
    _target_: str = "albumentations.RandomShadow"
    p: float = 0.4


@dataclass
class ImageCompressionConfig(AugmentationConfig):
    _target_: str = "albumentations.ImageCompression"
    quality_lower: int = 50
    quality_upper: int = 100
    compression_type: str = "jpeg"
    p: float = 0.2


# Spatial augmentations (compatible with depth images)

@dataclass
class GaussianBlurConfig(AugmentationConfig):
    _target_: str = "albumentations.GaussianBlur"
    blur_limit: tuple[int, int] = (3, 7)
    p: float = 0.5


@dataclass
class CoarseDropoutConfig(AugmentationConfig):
    _target_: str = "albumentations.CoarseDropout"
    max_holes: int = 8
    max_height: int = 8
    max_width: int = 8
    p: float = 0.3


@dataclass
class ShiftScaleRotateConfig(AugmentationConfig):
    _target_: str = "albumentations.ShiftScaleRotate"
    rotate_limit: tuple[float, float] = (0, 0)  # No rotation here; handled separately because it requires to be consistent with robot kinematics.
    scale_limit: tuple[float, float] = (-0.5, 0.6)
    shift_limit: tuple[float, float] = (-0.0625, 0.0625)
    p: float = 0.5


@dataclass
class CenterCropConfig(AugmentationConfig):
    """Center crop augmentation that preserves aspect ratio."""
    _target_: str = "albumentations.CenterCrop"
    height: int = MISSING  # Will be set from DataConfig
    width: int = MISSING   # Will be set from DataConfig
    p: float = 1.0  # Always apply if enabled


@dataclass
class RotateConfig(AugmentationConfig):
    """Rotation augmentation that requires special handling due to paired rotation of robot actions."""
    _target_: str = "albumentations.Rotate"
    limit: tuple[float, float] = (-5, 5)
    interpolation: int = 1  # cv2.INTER_LINEAR
    p: float = 0.5


# Pipeline configurations

@dataclass
class AugmentationPipelineConfig:
    """Configuration for augmentation pipeline."""
    _target_: str = "albumentations.Compose"
    transforms: list[AugmentationConfig] = field(default_factory=list)


@dataclass
class ColorAugmentationPipeline(AugmentationPipelineConfig):
    """Pipeline for color augmentations."""
    transforms: list[AugmentationConfig] = field(default_factory=lambda: [
        ColorJitterConfig(),
        RandomSunFlareConfig(),
        RandomBrightnessContrastConfig(),
        RandomGammaConfig(),
        CLAHEConfig(),
        RandomShadowConfig(),
        ImageCompressionConfig(),
    ])


@dataclass
class SpatialAugmentationPipeline(AugmentationPipelineConfig):
    """Pipeline for spatial augmentations."""
    transforms: list[AugmentationConfig] = field(default_factory=lambda: [
        GaussianBlurConfig(),
        CoarseDropoutConfig(),
        ShiftScaleRotateConfig(),
    ])


@dataclass
class LightColorAugmentationPipeline(AugmentationPipelineConfig):
    """Light color augmentation pipeline."""
    transforms: list[AugmentationConfig] = field(default_factory=lambda: [
        RandomBrightnessContrastConfig(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        RandomGammaConfig(gamma_limit=(90, 110), p=0.3),
    ])


@dataclass
class LightSpatialAugmentationPipeline(AugmentationPipelineConfig):
    """Light spatial augmentation pipeline."""
    transforms: list[AugmentationConfig] = field(default_factory=lambda: [
        GaussianBlurConfig(blur_limit=(3, 5), p=0.3),
        ShiftScaleRotateConfig(scale_limit=(-0.1, 0.1), shift_limit=(-0.05, 0.05), p=0.3),
    ])


@dataclass
class StrongColorAugmentationPipeline(AugmentationPipelineConfig):
    """Strong color augmentation pipeline."""
    transforms: list[AugmentationConfig] = field(default_factory=lambda: [
        ColorJitterConfig(brightness=0.4, contrast=0.5, saturation=0.6, hue=0.2, p=0.7),
        RandomSunFlareConfig(p=0.8),
        RandomBrightnessContrastConfig(brightness_limit=0.5, contrast_limit=0.5, p=0.7),
        RandomGammaConfig(gamma_limit=(70, 130), p=0.5),
        CLAHEConfig(clip_limit=6.0, p=0.5),
        RandomShadowConfig(p=0.6),
        ImageCompressionConfig(quality_lower=30, p=0.3),
    ])


@dataclass
class StrongSpatialAugmentationPipeline(AugmentationPipelineConfig):
    """Strong spatial augmentation pipeline."""
    transforms: list[AugmentationConfig] = field(default_factory=lambda: [
        GaussianBlurConfig(blur_limit=(3, 9), p=0.6),
        CoarseDropoutConfig(max_holes=12, max_height=12, max_width=12, p=0.5),
        ShiftScaleRotateConfig(scale_limit=(-0.6, 0.8), shift_limit=(-0.1, 0.1), p=0.6),
    ])
