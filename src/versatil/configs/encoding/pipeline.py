"""Configuration for input modalities encoding and feature fusion pipeline."""

from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING

from versatil.configs.data.task import ObservationSpaceConfig


@dataclass
class EncodingPipelineConfig:
    """Pipeline that encodes inputs and fuses them hierarchically.

    Attributes:
        _target_: Import path instantiated by Hydra.
        encoders: Dictionary of instantiated encoders keyed by name.
        observation_space: Observation space with task metadata.
        fusion_stages: List of instantiated fusion modules.
    """

    _target_: str = "versatil.models.encoding.pipeline.EncodingPipeline"
    encoders: dict[str, Any] = MISSING
    observation_space: ObservationSpaceConfig = "${task.observation_space}"
    fusion_stages: list[Any] | None = None
