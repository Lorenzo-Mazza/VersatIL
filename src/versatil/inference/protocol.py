"""Transport and inference protocols for the inference package."""

from typing import Protocol

import torch

type InferenceObservationValue = torch.Tensor | str | list[str] | list[list[str]]


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


class OnlineExplanationSource(Protocol):
    """Converts model-ready online inference windows into explanations."""

    def explain_observation_batch(
        self,
        observation: dict[str, InferenceObservationValue],
        display_observation: dict[str, torch.Tensor],
        environment_indices: list[int],
        timestep: int,
    ) -> None:
        """Explain one ready inference batch."""
        ...
