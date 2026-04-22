"""Configuration classes for observation encoders of different data modalities.

``input_keys`` exposes only user-facing observation keys (camera names,
proprioceptive keys). Language is excluded: the tokenizer rewrites it to
``SampleKey.TOKENIZED_OBSERVATIONS`` during preprocessing, and encoders
that consume language (LanguageEncoder, VLMs) bind to that internal key
automatically. VLM configs therefore list only their vision keys;
the tokenized text is routed to the language tower without user config.

LanguageEncoderConfig does not inherit from EncoderConfig because
LanguageEncoder's constructor does not accept ``input_keys`` — its
input specification is tied to the tokenized-observations key.
"""

from dataclasses import dataclass, field

from omegaconf import MISSING

from versatil.data.constants import Cameras
from versatil.models.encoding.encoders.constants import (
    BatchNormHandling,
    LanguageEncoderType,
    PoolingMethod,
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
class SpatialDepthEncoderConfig(EncoderConfig):
    """Spatial depth encoder configuration for backbones producing (B, C, H, W) feature maps."""

    _target_: str = (
        "versatil.models.encoding.encoders.depth.spatial.SpatialDepthEncoder"
    )
    backbone: str = MISSING
    batch_norm_handling: str = BatchNormHandling.FROZEN.value
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
    """Two-tower VLM encoder configuration.

    Note: its input_keys should only include vision keys; the tokenized text is routed to the language
        tower automatically via the fixed key SampleKey.TOKENIZED_OBSERVATIONS, so it doesn't
    """

    _target_: str = "versatil.models.encoding.encoders.cross_modal.vision_language.two_tower_vlm.TwoTowerVLMEncoder"
    model_name: str = MISSING
    pooling_method: str = PoolingMethod.NONE.value


@dataclass
class PaliGemmaEncoderConfig(EncoderConfig):
    """PaliGemma VLM encoder configuration.

    Note: its input_keys should only include vision keys; the tokenized text is routed automatically via the
        fixed key SampleKey.TOKENIZED_OBSERVATIONS.
    """

    _target_: str = "versatil.models.encoding.encoders.cross_modal.vision_language.paligemma.PaliGemmaEncoder"
    model_name: str = MISSING
    use_embeddings_only: bool = False
    max_text_length: int | None = None


@dataclass
class SmolVLMEncoderConfig(EncoderConfig):
    """SmolVLM/Idefics3 VLM encoder configuration.

    Note: its input_keys should only include vision keys; the tokenized text is routed automatically via the
        fixed key SampleKey.TOKENIZED_OBSERVATIONS.
    """

    _target_: str = "versatil.models.encoding.encoders.cross_modal.vision_language.smolvlm.SmolVLMEncoder"
    model_name: str = MISSING
    use_embeddings_only: bool = False
    max_text_length: int | None = None


@dataclass
class LanguageEncoderConfig:
    """Language encoder configuration.

    Note:
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
    model_dtype: str | None = "${experiment.precision}"


@dataclass
class ImageEncoderConfig(EncoderConfig):
    """Abstract base config for image encoders."""

    _target_: str = MISSING
    backbone: str = MISSING


@dataclass
class SpatialRGBEncoderConfig(ImageEncoderConfig):
    """Spatial RGB encoder configuration for backbones producing (B, C, H, W) feature maps."""

    _target_: str = "versatil.models.encoding.encoders.rgb.spatial.SpatialRGBEncoder"
    pooling_method: str = PoolingMethod.NONE.value
    batch_norm_handling: str = BatchNormHandling.FROZEN.value


@dataclass
class ConditionalCNNEncoderConfig(SpatialRGBEncoderConfig):
    """Feature-conditioned CNN encoder configuration.

    Note: this vision encoder receives as conditioning an encoded feature from
        another unconditional encoder in the pipeline. Conditional encoders are always
        run after conditional encoders, and their condition_key must be the output key
        of the desired unconditional encoder's feature.
    """

    _target_: str = (
        "versatil.models.encoding.encoders.rgb.conditional_cnn.ConditionalCNNEncoder"
    )
    condition_key: str = MISSING
    condition_dim: int = MISSING
    pooling_method: str = PoolingMethod.NONE.value
    batch_norm_handling: str = BatchNormHandling.FROZEN.value


@dataclass
class FlatRGBEncoderConfig(ImageEncoderConfig):
    """Flat RGB encoder configuration for backbones producing (B, S, D) token sequences."""

    _target_: str = "versatil.models.encoding.encoders.rgb.flat.FlatRGBEncoder"
    pooling_method: str = PoolingMethod.NONE.value
