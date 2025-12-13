"""Abstract base class for latent prior networks."""
import abc

import torch
from torch import nn as nn


class PriorLatentEncoder(nn.Module, abc.ABC):
    """Abstract base class for prior parametrizations over a latent space z, which can be either learned (through a NN)
      or fixed.

    Latent priors model the conditional distribution p(z|s) where z is a latent variable
    and s is an optional conditioning of observations (if the prior distribution is learned).

    Design:
        - forward() takes target latents and conditioning, returns predictions and targets
        - sample_prior() generates latent samples for inference
        - Loss computation happens in separate loss modules (not in forward())

    """

    def __init__(self, latent_dimension: int, device: str):
        """Initialize latent prior.

        Args:
            latent_dimension: Dimension of latent variable z
            device: Device to place prior on
        """
        super().__init__()
        self.latent_dimension = latent_dimension
        self.device = torch.device(device)

    @abc.abstractmethod
    def forward(
        self,
        target_latents: torch.Tensor,
        observations: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute prior predictions for training.

        This method takes clean latent samples (typically from a posterior
        encoder) and conditioning, and returns predictions and targets for
        loss computation. The actual loss is computed externally.

        Args:
            target_latents: Clean latent samples, shape (B, latent_dim)
                These are sampled from the approximate posterior q_phi(z|a,s) and
                should be detached to prevent gradients flowing twice to the approximate posterior.
            observations: Dictionary of conditioning features, typically the observations at current state.

        Returns:
            Dictionary containing predictions and targets for loss computation.
            Keys depend on the specific prior type (e.g., "predicted_prior", "target_prior").
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample_prior(
        self,
        batch_size: int,
        observations: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Sample latent variable from learned prior p(z|s).

        During inference, we don't have ground-truth actions (and thus no
        posterior samples), so we sample directly from the (conditional) prior.

        Args:
            batch_size: Number of samples to generate
            observations: Optional dictionary of conditioning features

        Returns:
            Sampled latent embeddings, shape (batch_size, embedding_dim)
        """
        raise NotImplementedError
