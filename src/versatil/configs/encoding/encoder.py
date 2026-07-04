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
from versatil.models.encoding.encoders.cross_modal.rgbd.dformerv2 import (
    DFormerPretrainedWeights,
    DFormerVariant,
)
from versatil.models.layers.activation import ActivationFunction


@dataclass
class EncoderConfig:
    """Base encoder configuration.

    Attributes:
        _target_: Import path instantiated by Hydra.
        input_keys: Observation keys consumed as inputs.
        pretrained: Whether to use pretrained weights.
        frozen: Whether to freeze encoder weights.
        model_dtype: Precision string from experiment config (e.g. ``"bf16-mixed"``).
    """

    _target_: str = MISSING
    input_keys: list[str] = MISSING
    pretrained: bool = False
    frozen: bool = False
    model_dtype: str | None = "${experiment.precision}"


@dataclass
class SpatialDepthEncoderConfig(EncoderConfig):
    """Spatial depth encoder configuration for backbones producing (B, C, H, W) feature maps.

    Attributes:
        _target_: Import path instantiated by Hydra.
        backbone: timm backbone name producing spatial feature maps.
        batch_norm_handling: BatchNorm strategy: keep, freeze, or replace.
        pooling_method: Spatial pooling applied to the feature map, or null to keep it.
        intermediate_layer_index: Backbone stage the features are taken from, or null
            for the last.
        lora_config: LoRA adaptation settings, or null to fine-tune directly.
    """

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
    """DFormer RGB+Depth encoder configuration.

    Attributes:
        _target_: Import path instantiated by Hydra.
        input_keys: Input keys for RGB and depth.
        variant: Model variant (S/B/L).
        pretrained_weights: Which checkpoint family to download from
            https://huggingface.co/bbynku/DFormerv2 when ``pretrained`` is set: the
            ImageNet backbone or the NYU/SUNRGBD finetuned models.
        pooling_method: Feature pooling method (spatial_softmax or global_average).
        lora_config: Optional LoRA adapter configuration applied to the stage linears.
    """

    _target_: str = (
        "versatil.models.encoding.encoders.cross_modal.rgbd.dformerv2.DFormerEncoder"
    )
    input_keys: list[str] = field(
        default_factory=lambda: [Cameras.LEFT.value, Cameras.DEPTH.value]
    )
    variant: str = DFormerVariant.SMALL.value
    pretrained_weights: str = DFormerPretrainedWeights.IMAGENET.value
    pooling_method: str = PoolingMethod.NONE.value
    lora_config: LoRAAdaptationConfig | None = None


@dataclass
class GeometricRGBDEncoderConfig(EncoderConfig):
    """Geometric RGB+Depth encoder configuration.

    Attributes:
        _target_: Import path instantiated by Hydra.
        input_keys: Input keys for RGB and depth observations.
        embedding_dimension: Dimension of patch embeddings and attention.
        number_of_heads: Number of attention heads.
        ffn_dimension: Hidden dimension of the feed-forward network.
        patch_size: Size of image patches for the patch embedding.
        pooling_method: Feature pooling method applied after attention.
    """

    _target_: str = "versatil.models.encoding.encoders.cross_modal.rgbd.geometric_rgbd.GeometricRGBDEncoder"
    input_keys: list[str] = field(
        default_factory=lambda: [Cameras.LEFT.value, Cameras.DEPTH.value]
    )
    embedding_dimension: int = 512
    number_of_heads: int = 8
    ffn_dimension: int = 2048
    patch_size: int = 16
    pooling_method: str = PoolingMethod.AVERAGE.value


@dataclass
class ProprioEncoderConfig(EncoderConfig):
    """State encoder configuration for proprioceptive data.

    Attributes:
        _target_: Import path instantiated by Hydra.
        output_dim: Output feature dimension.
        hidden_dimensions: Hidden layer dimensions. If None or [], creates simple linear
            layer. If [128], creates one hidden layer. If [256, 128], creates two hidden
            layers.
        activation: Activation function from ActivationFunction enum.
        dropout: Dropout rate between layers.
    """

    _target_: str = (
        "versatil.models.encoding.encoders.proprioceptive.base.ProprioceptiveEncoder"
    )
    output_dim: int = 128
    hidden_dimensions: list[int] | None = None
    activation: str = ActivationFunction.RELU.value
    dropout: float = 0.1


@dataclass
class VLMEncoderConfig(EncoderConfig):
    """VLM encoder configuration for image-text embedding models.

    Note: its input_keys should only include vision keys; the tokenized text is routed to the language
        tower automatically via the fixed key SampleKey.TOKENIZED_OBSERVATIONS, so it doesn't

    Attributes:
        _target_: Import path instantiated by Hydra.
        model_name: HuggingFace model identifier for the VLM.
        pooling_method: Feature pooling strategy for vision and language outputs.
        lora_config: Optional LoRA adapter configuration.
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

    Attributes:
        _target_: Import path instantiated by Hydra.
        model_name: Model identifier from LanguageEncoderType.
        pooling_method: How to extract features from transformer output.
        pretrained: Whether to use pretrained weights.
        frozen: Whether to freeze backbone weights.
        lora_config: Optional LoRA adapter configuration.
        max_token_len: Maximum token sequence length for the encoder.
        use_embeddings_only: If True, use only the pretrained token embedding layer.
        model_dtype: Precision string from experiment config (e.g. ``"bf16-mixed"``).
        trust_remote_code: Whether to allow HuggingFace models that ship custom modeling
            code (e.g. nvidia/llama-nemotron-embed).
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
    """Abstract base config for image encoders.

    Attributes:
        _target_: Import path instantiated by Hydra.
        backbone: Backbone name.
    """

    _target_: str = MISSING
    backbone: str = MISSING


@dataclass
class SpatialRGBEncoderConfig(ImageEncoderConfig):
    """Spatial RGB encoder configuration for backbones producing (B, C, H, W) feature maps.

    Attributes:
        _target_: Import path instantiated by Hydra.
        pooling_method: Spatial pooling applied to the feature map, or null to keep it.
        batch_norm_handling: BatchNorm strategy: keep, freeze, or replace.
        intermediate_layer_index: Backbone stage the features are taken from, or null
            for the last.
        lora_config: LoRA adaptation settings, or null to fine-tune directly.
    """

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

    Attributes:
        _target_: Import path instantiated by Hydra.
        condition_key: Key for the conditioning feature tensor.
        conditioning_dimension: Dimensionality of the conditioning feature.
        pooling_method: Feature pooling strategy.
        batch_norm_handling: How to handle batch normalization layers.
        lora_config: Optional PEFT LoRA adapter configuration.
    """

    _target_: str = (
        "versatil.models.encoding.encoders.rgb.conditional_cnn.ConditionalCNNEncoder"
    )
    condition_key: str = MISSING
    conditioning_dimension: int = MISSING
    pooling_method: str = PoolingMethod.NONE.value
    batch_norm_handling: str = BatchNormHandling.FROZEN.value
    lora_config: LoRAAdaptationConfig | None = None


@dataclass
class FlatRGBEncoderConfig(ImageEncoderConfig):
    """Flat RGB encoder configuration for backbones producing (B, S, D) token sequences.

    Attributes:
        _target_: Import path instantiated by Hydra.
        pooling_method: Feature pooling strategy for patch tokens. Defaults to CLS token
            selection.
        image_size: Optional image size passed to timm during backbone construction.
        intermediate_layer_index: Optional intermediate layer index for feature
            extraction. Negative values index from the end.
        lora_config: Optional PEFT LoRA adapter configuration.
    """

    _target_: str = "versatil.models.encoding.encoders.rgb.flat.FlatRGBEncoder"
    pooling_method: str = PoolingMethod.NONE.value
    image_size: int | None = None
    intermediate_layer_index: int | None = None
    lora_config: LoRAAdaptationConfig | None = None


@dataclass
class DinoV2SigLIPRGBEncoderConfig(ImageEncoderConfig):
    """DINOv2+SigLIP RGB encoder configuration for fused patch-token sequences.

    Attributes:
        _target_: Import path instantiated by Hydra.
        backbone: DINOv2+SigLIP paired backbone identifier.
        lora_config: Optional LoRA adapter configuration for the timm towers.
    """

    _target_: str = (
        "versatil.models.encoding.encoders.rgb.dinov2_siglip.DinoV2SigLIPRGBEncoder"
    )
    backbone: str = DinoV2SigLIPBackboneType.DINOV2_SIGLIP_VIT_SO_224PX.value
    lora_config: LoRAAdaptationConfig | None = None
