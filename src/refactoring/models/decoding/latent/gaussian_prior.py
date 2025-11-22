"""Simple Gaussian prior for variational models.

This module implements a standard Gaussian N(0, I) prior for latent variable models.
Unlike learned priors (e.g., DiffusionPrior), this prior requires no training and
simply samples from a standard normal distribution.

This is the default prior used when no learned prior is specified, providing
a simple baseline for variational models.
"""

import torch

from refactoring.models.decoding.latent import LatentPrior


class GaussianPrior(LatentPrior):
    """Standard Gaussian N(0, I) prior for latent variable models.

    Args:
        latent_dimension: Dimension of latent variable z
        device: Device to place prior on
    """

    def __init__(
        self,
        latent_dimension: int,
        device: str,
        infer_constant_prior: bool = False,
    ):
        """Initialize Gaussian prior."""
        super().__init__(latent_dimension=latent_dimension, device=device)
        self.infer_constant_prior = infer_constant_prior
        self.device = device
        self.to(torch.device(device))

    def sample_prior(
        self,
        batch_size: int,
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample latent variable from standard multivariate (z dimensional) Gaussian N(0, I).

        Args:
            batch_size: Number of samples to generate
            conditioning: Optional conditioning features (ignored for Gaussian prior)

        Returns:
            Sampled latent z vector of dimension (batch_size, latent_dim)
        """
        if self.infer_constant_prior:
            # Use constant zero latent for prior (like in ACT)
            return torch.zeros(batch_size, self.latent_dimension, device=self.device)
        else:
            # Sample from standard normal N(0, I)
            return torch.randn(batch_size, self.latent_dimension, device=self.device)



    def forward(
        self,
        target_latents: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """No-op forward pass for Gaussian prior (no training needed).

        For a standard Gaussian prior, there is no training objective since
        the prior distribution is fixed. This method returns an empty dictionary
        to indicate no prior loss should be computed."""
        return {}
