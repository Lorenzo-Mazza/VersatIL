"""Configuration for input modalities encoding and feature fusion pipeline."""
from dataclasses import dataclass

from omegaconf import MISSING

from refactoring.configs.encoding.encoder import EncoderConfig
from refactoring.configs.encoding.fusion import FusionModule


@dataclass
class EncodingPipelineConfig:
    """Pipeline that encodes inputs and fuses them hierarchically."""
    _target_: str = "refactoring.models.encoding.pipeline.EncodingPipeline"
    encoders: dict[str, EncoderConfig] = MISSING
    fusion_stages: list[FusionModule] | None = None
