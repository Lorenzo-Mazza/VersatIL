"""Transport and inference protocols for the inference package."""

from typing import Protocol


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
