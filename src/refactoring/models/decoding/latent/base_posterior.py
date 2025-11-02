"""Base classes for the posterior action encoder."""

import abc
import torch
import torch.nn as nn


class LatentActionEncoder(nn.Module, abc.ABC):
    """Abstract base class for encoding actions into latent representations, i.e. by modeling the latent posterior `q(z|a,s)`.

    Latent action encoders transform action sequences into lower-dimensional
    latent embeddings that capture action variability and execution style.
    They are used by decoding algorithms (e.g., BehavioralCloning with VAE)
    to model multi-modal action distributions.

    Design:
        - Supports both action-only and action+observation conditioning
        - Returns dictionary with LATENT_KEY + algorithm-specific auxiliary outputs
        - Provides sample_prior() for inference (when actions unavailable)

    Example (VAE):
        Returns {LATENT_KEY: z, MU_KEY: mu, LOGVAR_KEY: logvar}
    """

    def __init__(self, latent_dim: int, output_dim: int, device: str):
        """Initialize latent action encoder.

        Args:
            latent_dim: Dimension of latent embedding
            device: Device to place encoder on
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.device = torch.device(device)

    @abc.abstractmethod
    def encode(
        self,
        actions: dict[str, torch.Tensor],
        observations: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode actions (and optionally observations) into latent space.

        Args:
            actions: Dictionary of action tensors (e.g., position, orientation, gripper)
                Shape: (B, horizon, action_dim) for each action component
            observations: Optional observation features for conditional encoding
                Shape depends on observation type (e.g., (B, obs_dim) for flat features)

        Returns:
            Dictionary containing at minimum:
                - LATENT_KEY: Latent embedding (B, latent_dim)
                Plus algorithm-specific outputs (e.g., mu, logvar for VAE)
        """
        raise NotImplementedError("encode() must be implemented by subclasses.")


    def forward(
        self,
        actions: dict[str, torch.Tensor],
        observations: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass: encode actions into a latent representation."""
        return self.encode(actions, observations)


