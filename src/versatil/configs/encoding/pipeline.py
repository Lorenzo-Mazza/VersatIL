"""Configuration for input modalities encoding and feature fusion pipeline."""

from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING


@dataclass
class EncodingPipelineConfig:
    """Pipeline that encodes inputs and fuses them hierarchically."""

    _target_: str = "versatil.models.encoding.pipeline.EncodingPipeline"
    encoders: dict[str, Any] = MISSING
    observation_space: Any = MISSING
    fusion_stages: list[Any] | None = None
