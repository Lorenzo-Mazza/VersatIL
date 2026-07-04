"""Configuration for feature fusion modules."""

from dataclasses import dataclass

from omegaconf import MISSING

from versatil.models.encoding.fusion.constants import ConcatDimension
from versatil.models.layers.activation import ActivationFunction


@dataclass
class FusionConfig:
    """A fusion module that combines multiple input features into one representation.

    Attributes:
        _target_: Import path instantiated by Hydra.
        input_features: Feature names consumed by the fusion stage.
        output_name: Name registered for the fused feature.
        hidden_dimension: Hidden layer width.
    """

    _target_: str = MISSING
    input_features: list[str] = MISSING
    output_name: str = MISSING
    hidden_dimension: int = MISSING


@dataclass
class ConcatFusionConfig(FusionConfig):
    """Configuration for feature concatenation fusion."""

    _target_: str = "versatil.models.encoding.fusion.concat.ConcatFusion"


@dataclass
class AttentionFusionConfig(FusionConfig):
    """Configuration for cross-attention feature fusion.

    Attributes:
        _target_: Import path instantiated by Hydra.
        number_of_heads: Number of attention heads.
        dropout: Dropout rate for attention weights.
        input_feature_query: Name of the feature to use as query in cross-attention. If
            None, uses the first feature.
    """

    _target_: str = "versatil.models.encoding.fusion.attention.AttentionFusion"
    number_of_heads: int = 8
    dropout: float = 0.1
    input_feature_query: str | None = None


@dataclass
class MLPFusionConfig(FusionConfig):
    """Configuration for MLP-based feature fusion.

    Attributes:
        _target_: Import path instantiated by Hydra.
        mlp_hidden_dims: List of hidden layer dimensions for the MLP.
        activation_name: Name of the activation function to use in the MLP.
        dropout: Dropout rate for the MLP.
    """

    _target_: str = "versatil.models.encoding.fusion.mlp.MLPFusion"
    mlp_hidden_dims: list[int] = MISSING
    activation_name: str = ActivationFunction.GELU.value
    dropout: float = 0.1


@dataclass
class SpatialFusionConfig(FusionConfig):
    """Configuration for spatial concatenation of feature maps.

    Attributes:
        _target_: Import path instantiated by Hydra.
        concat_dim: Dimension along which spatial features are concatenated.
    """

    _target_: str = "versatil.models.encoding.fusion.spatial.SpatialFusion"
    concat_dim: str = ConcatDimension.WIDTH.value
