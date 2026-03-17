"""Transport protocols for inference client communication."""

from typing import Protocol


class ObservationTransport(Protocol):
    """Receives raw observations from the environment."""

    def receive(self, requested_keys: list[str], compression_type: str) -> dict: ...

    def register(self, client_name: str) -> dict: ...

    def close(self) -> None: ...


class ActionTransport(Protocol):
    """Sends actions to the environment."""

    def send(self, actions: dict, action_metadata: dict) -> dict: ...
