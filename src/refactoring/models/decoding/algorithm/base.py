"""Base class for action-decoding algorithms (BC, Diffusion, FlowMatching, etc.)."""

import abc
from abc import abstractmethod

import torch
import torch.nn as nn

from refactoring.models.decoding.decoders.base import ActionDecoder


class DecodingAlgorithm(nn.Module, abc.ABC):
    """Base class for decoding algorithms.

    Algorithms define the learning paradigm (behavioral cloning, diffusion, flow matching, etc.).
    Pure algorithms should not contain latent variable logic - use VariationalAlgorithm wrapper
    to add variational inference capabilities to any algorithm.

    Examples:
        # Pure algorithm (deterministic)
        algorithm = BehavioralCloning()

        # Algorithm with variational inference
        algorithm = VariationalAlgorithm(
            base_algorithm=BehavioralCloning(),
            posterior_encoder=VAETransformerEncoder(...),
            prior=GaussianPrior(...)  # or DiffusionPrior(...) for learned prior
        )
    """

    def __init__(self):
        """Initialize decoding algorithm."""
        super().__init__()

    @abstractmethod
    def forward(
            self,
            network: ActionDecoder,
            features: dict[str, torch.Tensor],
            actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass during training.

        Args:
            network: The action decoder network module
            features: Dictionary of encoded features from the encoding pipeline.
            actions: Optional dictionary of ground truth actions, if architecture requires it, e.g. ACT.

        Returns:
            Decoder output dictionary containing predictions and any algorithm-specific outputs.
        """
        raise NotImplementedError


    @abstractmethod
    def predict(
            self,
            network: ActionDecoder,
            features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Inference/prediction pass.

        Args:
            network: The action decoder network module
            features: Dict of encoded features from encoding pipeline

        Returns:
            Decoder output dictionary containing predictions and any algorithm-specific outputs.
        """
        raise NotImplementedError
