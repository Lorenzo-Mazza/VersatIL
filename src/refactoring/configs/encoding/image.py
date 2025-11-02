from dataclasses import dataclass

from omegaconf import MISSING

from refactoring.configs import EncoderConfig
from refactoring.models.encoding.encoders.constants import (
    FeatureExtractionMethod,
    PoolingMethod,
)


@dataclass
class ImageEncoderConfig(EncoderConfig):
    """Abstract base config for image encoders (CNN or ViT)."""
    _target_: str = MISSING
    backbone: str = MISSING


@dataclass
class CNNEncoderConfig(ImageEncoderConfig):
    """CNN-based image encoder configuration."""
    _target_: str = "refactoring.models.encoding.image.cnn.CNNEncoder"
    pooling_method: str = PoolingMethod.SPATIAL_SOFTMAX.value
    use_group_norm: bool = True


@dataclass
class ViTEncoderConfig(ImageEncoderConfig):
    """ViT-based image encoder configuration."""
    _target_: str = MISSING
    feature_method: str = FeatureExtractionMethod.AVERAGE_PATCH_TOKENS.value


