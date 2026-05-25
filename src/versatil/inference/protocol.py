"""Transport and inference protocols for the inference package."""

from typing import Protocol, runtime_checkable

import torch

from versatil.configs import MainConfig
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.tokenization.tokenizer import Tokenizer


class ObservationTransport(Protocol):
    """Receives raw observations from the environment."""

    def receive(self, requested_keys: list[str], compression_type: str) -> dict:
        """Receive one observation packet from the environment."""
        ...

    def register(self, client_name: str) -> dict:
        """Register a client with the environment server."""
        ...

    def close(self) -> None:
        """Close transport resources."""
        ...


class ActionTransport(Protocol):
    """Sends actions to the environment."""

    def send(self, actions: dict, action_metadata: dict) -> dict:
        """Send one action packet to the environment."""
        ...


@runtime_checkable
class PolicyInference(Protocol):
    """Protocol for policy loaders used by InferenceClient.

    Both PolicyLoader and CompressedPolicyLoader satisfy this protocol,
    enabling the InferenceClient to work with either float or compressed
    policies transparently.
    """

    @property
    def device(self) -> torch.device:
        """Device where policy inference runs."""
        ...

    @property
    def observation_space(self) -> ObservationSpace:
        """Observation space expected by the policy."""
        ...

    @property
    def action_space(self) -> ActionSpace:
        """Action space produced by the policy."""
        ...

    @property
    def prediction_horizon(self) -> int:
        """Number of future action steps predicted per policy call."""
        ...

    @property
    def observation_horizon(self) -> int:
        """Number of observation steps consumed per policy call."""
        ...

    @property
    def denoising_thresholds(self) -> dict[str, float]:
        """Per-action denoising thresholds loaded from the checkpoint."""
        ...

    @property
    def depth_clamp_range(self) -> tuple[float, float] | None:
        """Optional min/max range for clamping depth observations."""
        ...

    @property
    def checkpoint_path(self) -> str:
        """Path to the loaded checkpoint."""
        ...

    @property
    def client_identifier(self) -> str:
        """Stable identifier used when registering with a server."""
        ...

    @property
    def config(self) -> MainConfig:
        """Loaded training configuration."""
        ...

    @property
    def tokenizer(self) -> Tokenizer | None:
        """Tokenizer saved with the checkpoint, if present."""
        ...

    def run_inference(
        self, obs_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Run policy inference on preprocessed observations."""
        ...
