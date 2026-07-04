"""Deployment endpoint configuration."""

from dataclasses import dataclass, field

from omegaconf import MISSING

from versatil.configs.inference_client import InferenceClientConfig
from versatil.training.constants import CheckpointFilename


@dataclass
class DeploymentConfig:
    """Configuration for the real-time deployment endpoint.

    Policy behavior comes from the checkpoint; these fields only describe the
    deployment environment and the action-execution strategy.
    """

    checkpoint_path: str = MISSING
    checkpoint_name: str = CheckpointFilename.DEFAULT_CHECKPOINT.value
    device: str | None = None  # None selects cuda when available, else cpu.
    max_steps: int = 1_000_000
    compile_model: bool = True
    client: InferenceClientConfig = field(default_factory=InferenceClientConfig)
