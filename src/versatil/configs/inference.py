"""Inference configuration."""

from dataclasses import dataclass


@dataclass
class InferenceConfig:
    """Inference configuration."""

    temporal_agg: bool = True
    action_execution_horizon: int | None = (
        None  # Actions to execute per chunk (None = prediction_horizon)
    )
    update_rate_hz: float = (
        3.0  # Frequency at which to update the policy during inference
    )
    rotate_images: bool = False  # Whether to rotate images of 180 degrees
