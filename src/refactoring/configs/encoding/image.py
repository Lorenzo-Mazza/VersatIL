from dataclasses import dataclass

from omegaconf import MISSING

from refactoring.configs import EncoderConfig
from refactoring.models.encoding.encoders.constants import (
    PoolingMethod, BatchNormHandling,
)


@dataclass
class ImageEncoderConfig(EncoderConfig):
    """Abstract base config for image encoders (CNN or ViT)."""
    _target_: str = MISSING
    backbone: str = MISSING


@dataclass
class CNNEncoderConfig(ImageEncoderConfig):
    """CNN-based image encoder configuration."""
    _target_: str = "refactoring.models.encoding.encoders.rgb.cnn.CNNEncoder"
    pooling_method: str = PoolingMethod.NONE.value
    batch_norm_handling: str = BatchNormHandling.FROZEN.value


@dataclass
class ConditionalCNNEncoder:
    _target_: str = "refactoring.models.encoding.encoders.rgb.conditional_cnn.ConditionalCNNEncoder"
    condition_key: str = MISSING
    condition_dim: int = MISSING
    pooling_method: str = PoolingMethod.NONE.value
    batch_norm_handling: str = BatchNormHandling.FROZEN.value


@dataclass
class ViTEncoderConfig(ImageEncoderConfig):
    """ViT-based image encoder configuration."""
    _target_: str = MISSING
    feature_method: str = PoolingMethod.AVERAGE.value
    pooling_method: str = PoolingMethod.NONE.value


