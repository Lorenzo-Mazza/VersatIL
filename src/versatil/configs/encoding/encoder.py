"""Configuration classes for observation encoders of different data modalities."""

from dataclasses import dataclass, field

from omegaconf import MISSING

from versatil.data.constants import Cameras
from versatil.models.encoding.encoders.constants import (
    BatchNormHandling,
    LanguageEncoderType,
    PoolingMethod,
    SwinBackboneType,
)
from versatil.models.layers.activation import ActivationFunction


@dataclass
class EncoderConfig:
    """Base encoder configuration."""

    _target_: str = MISSING
    input_keys: list[str] = MISSING
    pretrained: bool = False
    frozen: bool = False
    model_dtype: str | None = "${experiment.precision}"


@dataclass
class DepthCNNEncoderConfig(EncoderConfig):
    """Depth CNN encoder configuration."""

    _target_: str = "versatil.models.encoding.encoders.depth.cnn.DepthCNNEncoder"
    backbone: str = MISSING
    batch_norm_handling: str = BatchNormHandling.FROZEN.value
    image_height: int = MISSING
    image_width: int = MISSING
    pooling_method: str = PoolingMethod.NONE.value


@dataclass
class DFormerEncoderConfig(EncoderConfig):
    """DFormer RGB+Depth encoder configuration."""

    _target_: str = (
        "versatil.models.encoding.encoders.cross_modal.rgbd.dformerv2.DFormerEncoder"
    )
    input_keys: list[str] = field(
        default_factory=lambda: [Cameras.LEFT.value, Cameras.DEPTH.value]
    )
    variant: str = "S"
    checkpoint_path: str | None = None
    pooling_method: str = PoolingMethod.NONE.value


@dataclass
class GeometricRGBDEncoderConfig(EncoderConfig):
    """Geometric RGB+Depth encoder configuration."""

    _target_: str = "versatil.models.encoding.encoders.cross_modal.rgbd.geometric_rgbd.GeometricRGBDEncoder"
    input_keys: list[str] = field(
        default_factory=lambda: [Cameras.LEFT.value, Cameras.DEPTH.value]
    )
    embedding_dimension: int = 512
    num_heads: int = 8
    ffn_dimension: int = 2048
    patch_size: int = 16
    pooling_method: str = PoolingMethod.AVERAGE.value


@dataclass
class ProprioEncoderConfig(EncoderConfig):
    """State encoder configuration for proprioceptive data."""

    _target_: str = (
        "versatil.models.encoding.encoders.proprioceptive.base.ProprioceptiveEncoder"
    )
    output_dim: int = 128
    hidden_dims: list[int] | None = None
    activation: str = ActivationFunction.RELU.value
    dropout: float = 0.1


@dataclass
class TwoTowerVLMEncoderConfig(EncoderConfig):
    """Two-tower VLM encoder configuration."""

    _target_: str = "versatil.models.encoding.encoders.cross_modal.vision_language.two_tower_vlm.TwoTowerVLMEncoder"
    model_name: str = MISSING
    pooling_method: str = PoolingMethod.NONE.value


@dataclass
class PaliGemmaEncoderConfig(EncoderConfig):
    """PaliGemma VLM encoder configuration."""

    _target_: str = "versatil.models.encoding.encoders.cross_modal.vision_language.paligemma.PaliGemmaEncoder"
    model_name: str = MISSING
    use_embeddings_only: bool = False
    max_text_length: int | None = None


@dataclass
class SmolVLMEncoderConfig(EncoderConfig):
    """SmolVLM/Idefics3 VLM encoder configuration."""

    _target_: str = "versatil.models.encoding.encoders.cross_modal.vision_language.smolvlm.SmolVLMEncoder"
    model_name: str = MISSING
    use_embeddings_only: bool = False
    max_text_length: int | None = None


@dataclass
class SwinEncoderConfig(EncoderConfig):
    """Swin Transformer image encoder configuration."""

    _target_: str = "versatil.models.encoding.encoders.rgb.swin.SwinEncoder"
    backbone: str = SwinBackboneType.SWIN_TINY.value
    pooling_method: str = PoolingMethod.AVERAGE.value


@dataclass
class LanguageEncoderConfig:
    """Language encoder configuration.
    It doesn't inherit from EncoderConfig because its input key is fixed, i.e. `SampleKey.TOKENIZED_OBSERVATIONS`
    """

    _target_: str = (
        "versatil.models.encoding.encoders.language.language.LanguageEncoder"
    )
    model_name: str = LanguageEncoderType.BERT_BASE.value
    pooling_method: str = PoolingMethod.NONE.value
    pretrained: bool = False
    frozen: bool = False
    max_token_len: int = 128
    use_embeddings_only: bool = False


@dataclass
class ImageEncoderConfig(EncoderConfig):
    """Abstract base config for image encoders (CNN or ViT)."""

    _target_: str = MISSING
    backbone: str = MISSING


@dataclass
class CNNEncoderConfig(ImageEncoderConfig):
    """CNN-based image encoder configuration."""

    _target_: str = "versatil.models.encoding.encoders.rgb.cnn.CNNEncoder"
    pooling_method: str = PoolingMethod.NONE.value
    batch_norm_handling: str = BatchNormHandling.FROZEN.value


@dataclass
class ConditionalCNNEncoderConfig(CNNEncoderConfig):
    """Language-conditioned CNN encoder configuration."""

    _target_: str = (
        "versatil.models.encoding.encoders.rgb.conditional_cnn.ConditionalCNNEncoder"
    )
    condition_key: str = MISSING
    condition_dim: int = MISSING
    pooling_method: str = PoolingMethod.NONE.value
    batch_norm_handling: str = BatchNormHandling.FROZEN.value


@dataclass
class ViTEncoderConfig(ImageEncoderConfig):
    """ViT-based image encoder configuration."""

    _target_: str = MISSING
    pooling_method: str = PoolingMethod.NONE.value
