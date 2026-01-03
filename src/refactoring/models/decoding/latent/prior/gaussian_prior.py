"""Fixed Gaussian prior for variational inference.

This module implements a standard Gaussian N(0, I) prior for latent variable models.
Unlike learned priors (e.g., DiffusionPrior), this prior requires no training and
simply samples from a standard normal distribution.

This is the default prior used when no learned prior is specified, providing
the traditional approach for imposing a gaussian distribution for the inference model (the approximated posterior q_\phi(z|x)).
"""

import torch

from refactoring.models.decoding.constants import (
    PRIOR_MU_KEY,
    PRIOR_LOGVAR_KEY,
    PRIOR_LATENT_KEY,
)
from refactoring.models.decoding.latent import PriorLatentEncoder


class GaussianPrior(PriorLatentEncoder):
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
        observations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample latent variable from standard multivariate (z dimensional) Gaussian N(0, I).

        Args:
            batch_size: Number of samples to generate
            observations: Optional conditioning features (ignored for Gaussian prior)

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
        observations: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for a fixed Gaussian prior, returning zero mu and unit logvar."""
        mu = torch.zeros_like(target_latents, device=self.device)
        logvar = torch.zeros_like(target_latents, device=self.device)
        z = torch.randn(mu.size(0), self.latent_dimension, device=self.device)
        return {
            PRIOR_MU_KEY: mu,
            PRIOR_LOGVAR_KEY: logvar,
            PRIOR_LATENT_KEY: z,
        }
