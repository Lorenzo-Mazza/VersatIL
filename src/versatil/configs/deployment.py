"""Deployment endpoint configuration."""

from dataclasses import dataclass

from omegaconf import MISSING

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
    model_server_address: str = "127.0.0.1"
    model_server_port: int = 5555
    temporal_aggregation: bool = False
    action_execution_horizon: int | None = None  # None executes the full chunk.
    update_rate_hz: float | None = None  # Action send rate; None for simulation.
    max_steps: int = 1_000_000
    temporal_max_timesteps: int = 800
    timing_log: bool = False
    compile_model: bool = True
    # Per-request transport timeout in seconds; None blocks indefinitely.
    request_timeout_seconds: float | None = None
