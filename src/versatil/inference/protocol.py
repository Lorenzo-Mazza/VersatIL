"""Transport and inference protocols for the inference package."""

from typing import Protocol, runtime_checkable

import torch

from versatil.configs import MainConfig
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.tokenization.tokenizer import Tokenizer


class ObservationTransport(Protocol):
    """Receives raw observations from the environment."""

    def receive(self, requested_keys: list[str], compression_type: str) -> dict: ...

    def register(self, client_name: str) -> dict: ...

    def close(self) -> None: ...


class ActionTransport(Protocol):
    """Sends actions to the environment."""

    def send(self, actions: dict, action_metadata: dict) -> dict: ...


@runtime_checkable
class PolicyInference(Protocol):
    """Protocol for policy loaders used by InferenceClient.

    Both PolicyLoader and CompressedPolicyLoader satisfy this protocol,
    enabling the InferenceClient to work with either float or compressed
    policies transparently.
    """

    @property
    def device(self) -> torch.device: ...

    @property
    def observation_space(self) -> ObservationSpace: ...

    @property
    def action_space(self) -> ActionSpace: ...

    @property
    def prediction_horizon(self) -> int: ...

    @property
    def observation_horizon(self) -> int: ...

    @property
    def denoising_thresholds(self) -> dict[str, float]: ...

    @property
    def depth_clamp_range(self) -> tuple[float, float] | None: ...

    @property
    def checkpoint_path(self) -> str: ...

    @property
    def config(self) -> MainConfig: ...

    @property
    def tokenizer(self) -> Tokenizer | None: ...

    def run_inference(
        self, obs_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]: ...
