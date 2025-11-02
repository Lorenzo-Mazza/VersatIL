"""Configuration for input modalities encoding and feature fusion pipeline."""
from dataclasses import dataclass

from omegaconf import MISSING

from refactoring.configs.encoding.encoder import EncoderConfig
from refactoring.configs.encoding.fusion import FusionModule


@dataclass
class EncodingPipelineConfig:
    """Pipeline that encodes inputs and fuses them hierarchically."""
    #: Encoders for different input modalities. Items are keyed by encoder name, output feature names are prefixed with this name.
    #: EncoderType.value -> EncoderConfig
    encoders: dict[str, EncoderConfig] = MISSING
    #: Fusion modules that combine features together
    fusion_stages: list[FusionModule] = MISSING
