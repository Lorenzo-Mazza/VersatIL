"""Configuration for feature fusion modules."""
from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class FusionModule:
    """A fusion module that combines multiple input features into one representation."""
    _target_: str = MISSING
    input_features: list[str] = MISSING
    output_name: str = MISSING


@dataclass
class ConcatFusionModule(FusionModule):
    _target_: str = "refactoring.models.encoding.fusion.concat.ConcatFusion"


@dataclass
class AttentionFusionModule(FusionModule):
    _target_: str = "refactoring.models.encoding.fusion.attention.AttentionFusion"
    num_heads: int = 8
    dropout: float = 0.1
