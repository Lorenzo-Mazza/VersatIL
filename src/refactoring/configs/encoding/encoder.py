"""Configuration classes for observation encoders of different data modalities."""
from dataclasses import dataclass, field

from omegaconf import MISSING

from refactoring.data.constants import Cameras
from refactoring.models.encoding.encoders.constants import (
    LanguageEncoderType,
    PoolingMethod,
)
from refactoring.models.layers.activation import ActivationFunction


@dataclass
class EncoderConfig:
    """Base encoder configuration."""

    _target_: str = MISSING
    input_keys: list[str] = MISSING
    pretrained: bool = False
    frozen: bool = False


@dataclass
class DepthCNNEncoderConfig(EncoderConfig):
    """Depth CNN encoder configuration."""

    _target_: str = "refactoring.models.encoding.encoders.depth.cnn.DepthCNNEncoder"
    backbone: str = MISSING
    use_group_norm: bool = True
    image_height: int = MISSING
    image_width: int = MISSING
    pooling_method: str = PoolingMethod.NONE.value


@dataclass
class DFormerEncoderConfig(EncoderConfig):
    """DFormer RGB+Depth encoder configuration."""

    _target_: str = (
        "refactoring.models.encoding.encoders.depth.dformerv2.DFormerEncoder"
    )
    input_keys: list[str] = field(
        default_factory=lambda: [Cameras.LEFT.value, Cameras.DEPTH.value]
    )
    variant: str = "S"
    checkpoint_path: str | None = None
    pooling_method: str = PoolingMethod.NONE.value


@dataclass
class LightGeometricEncoderConfig(EncoderConfig):
    """Geometric RGB+Depth encoder configuration."""

    _target_: str = "refactoring.models.encoding.encoders.depth.light_geometric.LightGeometricEncoder"
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
        "refactoring.models.encoding.encoders.proprioceptive.base.ProprioceptiveEncoder"
    )
    output_dim: int = 128
    hidden_dims: list[int] | None = None
    activation: str = ActivationFunction.RELU.value
    dropout: float = 0.1


@dataclass
class VLMEncoderConfig(EncoderConfig):
    """Vision-Language Model encoder configuration."""

    _target_: str = "refactoring.models.encoding.encoders.multimodal.vlm.VLMEncoder"
    model_name: str = MISSING
    pooling_method: str = PoolingMethod.NONE.value


# These two configs don't inherit from EncoderConfig because their input keys are fixed, i.e. `TOKENIZED_OBSERVATIONS_KEY`


@dataclass
class LanguageEncoderConfig:
    """Language encoder configuration."""

    _target_: str = (
        "refactoring.models.encoding.encoders.language.language.LanguageEncoder"
    )
    model_name: str = LanguageEncoderType.BERT_BASE.value
    pooling_method: str = PoolingMethod.NONE.value
    pretrained: bool = False
    frozen: bool = False
    max_token_len: int = 128


@dataclass
class EmbedderConfig:
    """Embedding layer encoder configuration.

    Simple encoder consisting of a single nn.Embedding layer that converts token IDs
    into dense embeddings. Useful for tokenized observations or actions from the data pipeline.

    Output shape: (max_token_len, embedding_dim)
    """

    _target_: str = "refactoring.models.encoding.encoders.language.embedder.Embedder"
    embedding_dim: int = MISSING
    max_token_len: int = MISSING
