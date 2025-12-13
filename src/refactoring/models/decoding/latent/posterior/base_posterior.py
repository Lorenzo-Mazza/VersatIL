"""Base classes for the posterior action encoder."""

import abc
import torch
import torch.nn as nn


class PosteriorLatentEncoder(nn.Module, abc.ABC):
    """Abstract base class for posterior encoders, used for modeling the conditional posterior `q_\phi(z|a,s)`.

    Posterior encoders learn lower-dimensional latent embeddings conditioned on privileged
     information such as expert actions (a) and optionally observations (s),
     in order to learn a latent representation of the target action multi-modality and execution style.
     They are trained with variational inference to learn a conditional latent distribution that is close to
     a prior probability p(z) (which can also be learned, ref. latent/prior package) .

    Design:
        - Supports both action-only and action+observation conditioning
        - Returns dictionary with LATENT_KEY + algorithm-specific auxiliary outputs
        - Provides sample_prior() for inference (when actions unavailable)

    """

    def __init__(self, latent_dimension: int, device: str):
        """Initialize posterior encoder."""
        super().__init__()
        self.latent_dimension = latent_dimension
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


