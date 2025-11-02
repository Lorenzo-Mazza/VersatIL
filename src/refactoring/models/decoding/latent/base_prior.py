"""Abstract base class for learned latent priors."""
import abc

import torch
from torch import nn as nn


class LatentPrior(nn.Module, abc.ABC):
    """Abstract base class for learned priors over latent spaces.

    Latent priors model the distribution p(z|s) where z is a latent variable
    and s is optional conditioning (e.g., state/observation features).

    Design:
        - forward() takes target latents and conditioning, returns predictions and targets
        - sample_prior() generates latent samples for inference
        - Loss computation happens in separate loss modules (not in forward())

    """

    def __init__(self, latent_dim: int, output_dim: int, device: str):
        """Initialize latent prior.

        Args:
            latent_dim: Dimension of latent variable z
            output_dim: Output embedding dimension
            device: Device to place prior on
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.device = torch.device(device)

    @abc.abstractmethod
    def forward(
        self,
        target_latents: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute prior predictions for training.

        This method takes clean latent samples (typically from a posterior
        encoder) and conditioning, and returns predictions and targets for
        loss computation. The actual loss is computed externally.

        Args:
            target_latents: Clean latent samples, shape (B, latent_dim)
                These are typically sampled from the posterior q(z|a,s) and
                should be DETACHED to prevent gradients flowing to posterior.
            conditioning: Conditioning features (state), shape (B, conditioning_dim)

        Returns:
            Dictionary containing predictions and targets for loss computation.
            Keys depend on the specific prior type (e.g., "predicted_prior", "target_prior").
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample_prior(
        self,
        batch_size: int,
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample latent variable from learned prior p(z|s).

        During inference, we don't have ground-truth actions (and thus no
        posterior samples), so we sample directly from the prior.

        Args:
            batch_size: Number of samples to generate
            conditioning: Optional conditioning features (state), shape (B, conditioning_dim)

        Returns:
            Sampled latent embeddings, shape (batch_size, embedding_dim)
        """
        raise NotImplementedError
