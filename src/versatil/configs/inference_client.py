"""Shared configuration for the socket inference client."""

from dataclasses import dataclass

from tso_robotics_sockets import CompressionType


@dataclass
class InferenceClientConfig:
    """Settings for a client driving an environment server.

    Used by the deployment endpoint and by online-inference explanations,
    which run the same client loop.

    Attributes:
        model_server_address: Environment server address.
        model_server_port: Environment server port.
        temporal_aggregation: Whether predicted chunks are ensembled.
        action_execution_horizon: Actions executed per chunk; None executes
            the full prediction horizon.
        update_rate_hz: Action send rate; None for simulation.
        temporal_max_timesteps: Timesteps tracked by temporal aggregation.
        timing_log: Whether the client logs per-step timing.
        compression_type: Wire compression for camera observations.
        request_timeout_seconds: Per-request transport timeout; None blocks
            indefinitely.
    """

    model_server_address: str = "127.0.0.1"
    model_server_port: int = 5555
    temporal_aggregation: bool = False
    action_execution_horizon: int | None = None
    update_rate_hz: float | None = None
    temporal_max_timesteps: int = 800
    timing_log: bool = False
    compression_type: str = CompressionType.RAW.value
    request_timeout_seconds: float | None = None
