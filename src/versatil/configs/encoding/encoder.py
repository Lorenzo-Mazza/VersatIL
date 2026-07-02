"""Configuration classes for observation encoders of different data modalities.

``input_keys`` exposes only user-facing observation keys (camera names,
proprioceptive keys). Language is excluded: the tokenizer rewrites it to
``SampleKey.TOKENIZED_OBSERVATIONS`` during preprocessing, and language/VLM
encoders bind to that internal key automatically. VLM encoder configs therefore
list only their vision keys; the tokenized text is routed to the language tower
without user config.

LanguageEncoderConfig does not inherit from EncoderConfig because
LanguageEncoder's constructor does not accept ``input_keys`` — its
input specification is tied to the tokenized-observations key.
"""

from dataclasses import dataclass, field

from omegaconf import MISSING

from versatil.configs.adaptation import LoRAAdaptationConfig
from versatil.data.constants import Cameras
from versatil.models.encoding.encoders.constants import (
    BatchNormHandling,
    DinoV2SigLIPBackboneType,
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
    intermediate_layer_index: int | None = None
    lora_config: LoRAAdaptationConfig | None = None


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
class VLMEncoderConfig(EncoderConfig):
    """VLM encoder configuration for image-text embedding models.

    Note: its input_keys should only include vision keys; the tokenized text is routed to the language
        tower automatically via the fixed key SampleKey.TOKENIZED_OBSERVATIONS, so it doesn't
    """

    _target_: str = "versatil.models.encoding.encoders.cross_modal.vision_language.vlm_encoder.VLMEncoder"
    model_name: str = MISSING
    pooling_method: str = PoolingMethod.NONE.value
    lora_config: LoRAAdaptationConfig | None = None


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
    lora_config: LoRAAdaptationConfig | None = None
    max_token_len: int = 128
    use_embeddings_only: bool = False
    model_dtype: str | None = "${experiment.precision}"
    trust_remote_code: bool = False


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
    intermediate_layer_index: int | None = None
    lora_config: LoRAAdaptationConfig | None = None


@dataclass
class ConditionalCNNEncoderConfig(ImageEncoderConfig):
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
    lora_config: LoRAAdaptationConfig | None = None


@dataclass
class FlatRGBEncoderConfig(ImageEncoderConfig):
    """Flat RGB encoder configuration for backbones producing (B, S, D) token sequences."""

    _target_: str = "versatil.models.encoding.encoders.rgb.flat.FlatRGBEncoder"
    pooling_method: str = PoolingMethod.NONE.value
    image_size: int | None = None
    intermediate_layer_index: int | None = None
    lora_config: LoRAAdaptationConfig | None = None


@dataclass
class DinoV2SigLIPRGBEncoderConfig(ImageEncoderConfig):
    """DINOv2+SigLIP RGB encoder configuration for fused patch-token sequences."""

    _target_: str = (
        "versatil.models.encoding.encoders.rgb.dinov2_siglip.DinoV2SigLIPRGBEncoder"
    )
    backbone: str = DinoV2SigLIPBackboneType.DINOV2_SIGLIP_VIT_SO_224PX.value
    lora_config: LoRAAdaptationConfig | None = None
