"""Inference configuration."""
from dataclasses import dataclass


@dataclass
class InferenceConfig:
    """Inference configuration."""

    temporal_agg: bool = True
    update_rate_hz: float = (
        3.0  # Frequency at which to update the policy during inference
    )
    rotate_images: bool = False # Whether to rotate images of 180 degrees
