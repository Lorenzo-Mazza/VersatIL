"""Base runtime wrapper for policy inference."""

import logging
import secrets
from abc import ABC, abstractmethod

import torch

from versatil.checkpoint_loading.base import BaseCheckpointLoader
from versatil.configs import MainConfig
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.tokenization.tokenizer import Tokenizer
from versatil.models.policy import Policy


def initialize_inference_seed(seed: int | None) -> int:
    """Initialize PyTorch's global RNG for an inference runtime.

    Args:
        seed: Explicit seed for reproducible inference. ``None`` draws a fresh
            seed from operating-system entropy.

    Returns:
        The seed applied to the PyTorch RNG.
    """
    resolved_seed = secrets.randbits(63) if seed is None else seed
    torch.manual_seed(seed=resolved_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed=resolved_seed)
    logging.info(f"Inference random seed: {resolved_seed}")
    return resolved_seed


class PolicyRuntime(ABC):
    """Base class for policy inference runtimes."""

    def __init__(
        self,
        checkpoint_loader: BaseCheckpointLoader,
        client_identifier: str,
    ) -> None:
        """Initialize runtime metadata delegation."""
        self.checkpoint_loader = checkpoint_loader
        self._client_identifier = client_identifier
        self._policy = checkpoint_loader.policy
        self._seed: int | None = None

    @abstractmethod
    def run_inference(
        self,
        obs_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Run inference on preprocessed observations."""

    @property
    def policy(self) -> Policy:
        """Get the runtime policy module."""
        return self._policy

    @property
    def device(self) -> torch.device:
        """Device where policy inference runs."""
        return self.checkpoint_loader.device

    @property
    def checkpoint_path(self) -> str:
        """Path to the loaded checkpoint."""
        return self.checkpoint_loader.checkpoint_path

    @property
    def client_identifier(self) -> str:
        """Stable identifier used when registering with a server."""
        return self._client_identifier

    @property
    def seed(self) -> int | None:
        """Resolved inference seed, if this runtime initializes one."""
        return self._seed

    @property
    def config(self) -> MainConfig:
        """Loaded training configuration."""
        return self.checkpoint_loader.config

    @property
    def tokenizer(self) -> Tokenizer | None:
        """Tokenizer saved with the checkpoint, if present."""
        return self.checkpoint_loader.tokenizer

    @property
    def observation_space(self) -> ObservationSpace:
        """Observation space expected by the policy."""
        return self.checkpoint_loader.observation_space

    @property
    def action_space(self) -> ActionSpace:
        """Action space produced by the policy."""
        return self.checkpoint_loader.action_space

    @property
    def prediction_horizon(self) -> int:
        """Number of future action steps predicted per policy call."""
        return self.checkpoint_loader.prediction_horizon

    @property
    def observation_horizon(self) -> int:
        """Number of observation steps consumed per policy call."""
        return self.checkpoint_loader.observation_horizon

    @property
    def denoising_thresholds(self) -> dict[str, float]:
        """Per-action denoising thresholds loaded from the checkpoint."""
        return self.checkpoint_loader.denoising_thresholds

    @property
    def depth_clamp_ranges(self) -> dict[str, tuple[float, float]]:
        """Per-camera min/max ranges for clamping depth observations."""
        return self.checkpoint_loader.depth_clamp_ranges
