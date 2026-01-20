"""Configuration for feature fusion modules."""
from dataclasses import dataclass

from omegaconf import MISSING

from versatil.models.encoding.fusion.constants import ConcatDimension
from versatil.models.layers.activation import ActivationFunction


@dataclass
class FusionConfig:
    """A fusion module that combines multiple input features into one representation."""

    _target_: str = MISSING
    input_features: list[str] = MISSING
    output_name: str = MISSING
    hidden_dim: int = MISSING


@dataclass
class ConcatFusionConfig(FusionConfig):
    _target_: str = "versatil.models.encoding.fusion.concat.ConcatFusion"


@dataclass
class AttentionFusionConfig(FusionConfig):
    _target_: str = "versatil.models.encoding.fusion.attention.AttentionFusion"
    num_heads: int = 8
    dropout: float = 0.1
    input_feature_query: str | None = None


@dataclass
class MLPFusionConfig(FusionConfig):
    _target_: str = "versatil.models.encoding.fusion.mlp.MLPFusion"
    mlp_hidden_dims: list[int] = MISSING
    activation_name: str = ActivationFunction.GELU.value
    dropout: float = 0.1


@dataclass
class SpatialFusionConfig(FusionConfig):
    _target_: str = "versatil.models.encoding.fusion.spatial.SpatialFusion"
    concat_dim: str = ConcatDimension.WIDTH.value
